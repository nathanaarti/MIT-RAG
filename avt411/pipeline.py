"""The request pipeline: the cheapest reliable path to a grounded answer.

    question
      │ 1. input screening ─────────────── block ──▶ canned refusal          (0 LLM calls)
      │ 2. answer cache ─────────────────── hit ───▶ cached answer           (0 LLM calls)
      │ 3. KG tier (authorship) ─────────── hit ───▶ computed answer         (0 LLM calls)
      │ 4. curated FAQ ──────────────────── hit ───▶ stored answer           (0 LLM calls, 1 embedding)
      │ 5. scope gate ───────────────────── off-domain ─▶ decline            (0 LLM calls)
      │ 6. router ──┬─ standard: retrieve + 1 answer call
      │             │      └─ answer says evidence insufficient ─▶ agent, seeded with that evidence
      │             └─ agent: plan → tools → answer (vague / multi-paper / author+content / follow-up)
      │ 7. output guard (inside the answer step) → answer cache → request log
      ▼
    answer + route + cost

Every request writes one JSON line to logs/requests.jsonl (route, reasons, LLM calls,
tokens, billed $, latency), so the routing can be audited in production.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config
from . import guardrails as guard
from .agent import SECURE_LEAN, AgentConfig, SecureAgentRAG
from .answer import AnswerWriter
from .faq import FAQHit, FAQStore, KGAnswerer, is_context_dependent
from .knowledge_graph import GraphIndex, paper_key
from .metering import _METER, CostMeter
from .retrieval import HybridRetriever, tokenize
from .router import decide, needs_escalation
from .tools import CorpusTools

ERROR_REPLY = ("Sorry — I couldn't complete that request because a backend service failed. "
               "Please try again in a moment.")


@dataclass
class PipelineOptions:
    """Switches for the optimizations, so baselines are the same code with features off."""
    name: str = "routed"
    screen: bool = True                  # input screening (False only for the unhardened audit baseline)
    use_cache: bool = config.CACHE_ENABLED
    use_kg_tier: bool = config.FAQ_ENABLED
    use_faq: bool = config.FAQ_ENABLED
    use_scope_gate: bool = True
    use_router: bool = True              # False = every question goes to the agent (6.1 behaviour)
    escalate: bool = True                # standard answer says "insufficient" -> agent
    persona_filter_mode: str = config.PERSONA_FILTER_MODE
    standard_k: int = config.STANDARD_NUM_RETRIEVED


ROUTED = PipelineOptions()
AGENT_ONLY = PipelineOptions(name="agent_only", use_cache=False, use_kg_tier=False, use_faq=False,
                             use_scope_gate=False, use_router=False, escalate=False, persona_filter_mode="off")
STANDARD_ONLY = PipelineOptions(name="standard_only", use_cache=False, use_kg_tier=False, use_faq=False,
                                use_scope_gate=False, escalate=False, persona_filter_mode="off")


@dataclass
class PipelineResult:
    answer: str
    route: str
    reasons: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    blocked: bool = False
    history_user: str | None = None
    request_id: str = ""
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def agent_used(self) -> bool:
        return "agent" in self.route and "failed" not in self.route


class AnswerCache:
    """TTL + LRU cache for history-free questions (Checkpoint 6.1). Keyed on the SANITIZED
    question; only answers that raised no guard findings are stored, so a poisoned answer is
    never replayed to other users (per-tenant keys in a multi-tenant deployment)."""

    def __init__(self, max_items: int | None = None, ttl_s: float | None = None):
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items or config.CACHE_MAX_ITEMS
        self.ttl_s = ttl_s or config.CACHE_TTL_S

    @staticmethod
    def key(question: str) -> str:
        return " ".join(tokenize(question))

    def get(self, question: str) -> dict | None:
        k = self.key(question)
        with self._lock:
            hit = self._d.get(k)
            if not hit or time.time() - hit[0] > self.ttl_s:
                return None
            self._d.move_to_end(k)
            return hit[1]

    def put(self, question: str, value: dict) -> None:
        k = self.key(question)
        with self._lock:
            self._d[k] = (time.time(), value)
            self._d.move_to_end(k)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, index: GraphIndex, *, answer_llm, planner_llm,
                 persona_filter: guard.PersonaFilter | None = None, faq_store: FAQStore | None = None,
                 kg_complete: bool = True, options: PipelineOptions = ROUTED,
                 agent_cfg: AgentConfig = SECURE_LEAN, request_log: Path | None = config.REQUEST_LOG,
                 audit: guard.AuditLog | None = None, input_fn=input):
        self.retriever, self.index, self.options = retriever, index, options
        self.tools = CorpusTools(retriever, index)
        self.kg = KGAnswerer(self.tools, complete=kg_complete)
        self.faq = faq_store
        self.persona_filter = persona_filter
        self.audit = audit or guard.AuditLog(config.SECURITY_LOG)
        self.writer = AnswerWriter(answer_llm, index)
        self.agent = SecureAgentRAG(self.tools, agent_cfg, audit=self.audit, planner_llm=planner_llm,
                                    answer_llm=answer_llm, input_fn=input_fn)
        self.cache = AnswerCache()
        self.request_log = request_log
        self._log_lock = threading.Lock()

    # — the standard (1-call) path —
    def retrieve_standard(self, question: str) -> list[tuple[str, str]]:
        doc = self.tools.resolve_paper(question) if re.search(r"\d", question) else None
        named = doc and re.search(rf"(?<![\w-]){re.escape(paper_key(doc['source']).lower())}(?![\w-])", question.lower())
        hits = (self.retriever.search_in_source(question, doc["source"], self.options.standard_k) if named
                else self.retriever.search(question, self.options.standard_k))
        items = []
        # KG card (title, authors, summary) for the named paper or the top-2 retrieved papers:
        # chunks rarely carry the author list, and the card gives the model the exact title to
        # cite (security audit: a Spanish "who wrote…" question failed without it).
        top_sources = [doc["source"]] if named else list(dict.fromkeys(h.source for h in hits))[:2]
        for src in top_sources:
            pid = self.index.filename_to_paper_id.get(src)
            card = self.index.paper_card(pid, "NAMED PAPER" if named else "RETRIEVED PAPER") if pid else None
            if card:
                items.append(card)
        for h in hits:
            clean, _ = guard.sanitize_evidence(h.text)
            items.append((h.chunk_id, f"=== CHUNK {h.chunk_id} ===\n{clean}"))
        return items

    def _history_text(self, history: list[dict]) -> str:
        return "\n".join(f"{m['role'].capitalize()}: {m['content'][:1500]}" for m in history[-4:])

    def _run_agent(self, q: str, mode: str, scripted, history, seed=None) -> dict:
        return self.agent.run(q, mode=mode, scripted_replies=scripted, history=history, prescreened=True,
                              seed_evidence=dict(seed) if seed else None)

    def ask(self, question: str, *, history: list[dict] | None = None, mode: str = "batch",
            scripted_replies: list[str] | None = None) -> PipelineResult:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        meter = _METER.get()
        own_meter = meter is None
        token = _METER.set(CostMeter()) if own_meter else None
        meter = _METER.get()
        start = len(meter.calls)
        try:
            res = self._ask(rid, question, history or [], mode, scripted_replies)
        except Exception as e:  # never let a backend failure surface as a stack trace
            self.audit.event(rid, "pipeline_error", error=f"{type(e).__name__}: {e}")
            res = PipelineResult(ERROR_REPLY, "error", [f"{type(e).__name__}: {str(e)[:200]}"], request_id=rid)
        finally:
            calls = meter.calls[start:]
            if own_meter:
                _METER.reset(token)
        res.request_id = rid
        res.latency_s = round(time.perf_counter() - t0, 3)
        res.usage = {"llm_calls": len(calls), "planner_calls": sum(c["role"] == "planner" for c in calls),
                     "input_tokens": sum(c["input"] for c in calls), "output_tokens": sum(c["output"] for c in calls),
                     "cost_usd": round(sum(c["cost"] for c in calls), 6)}
        self._log(res, question)
        return res

    def _ask(self, rid: str, question: str, history: list[dict], mode: str, scripted) -> PipelineResult:
        o = self.options
        trace: list[dict] = []
        # 1. screening
        v = (guard.screen_input(question, self.persona_filter, o.persona_filter_mode) if o.screen
             else guard.InputVerdict("allow", question))
        events = []
        if v.findings and v.findings != ["normalized"]:
            self.audit.event(rid, "input_guard", pipeline=o.name, action=v.action, findings=v.findings, risk=v.risk,
                             removed=[r[:200] for r in v.removed])
            events.append(f"input_guard:{v.action}:" + ",".join(v.findings))
        if v.action == "block":
            return PipelineResult(v.reply, "blocked", v.findings, events=events, blocked=True,
                                  trace=[{"step": 0, "action": "input screening -> block", "outcome": ", ".join(v.findings)}])
        q = v.text
        base = dict(events=events, history_user=q)
        cacheable = o.use_cache and not history and mode != "chat" and not events
        # 2. cache
        if cacheable and (hit := self.cache.get(q)):
            return PipelineResult(hit["answer"], "cache", ["repeat question"], hit["sources"], **base,
                                  trace=[{"step": 0, "action": "answer cache hit", "outcome": "0 LLM calls"}])
        # 3-4. quick lookups
        dependent = is_context_dependent(q, bool(history))
        if not dependent and o.use_kg_tier:
            r = self.kg.answer(q)
            trace.append({"step": len(trace), "action": "kg tier", "outcome": r.kind if isinstance(r, FAQHit) else r.reason})
            if isinstance(r, FAQHit):
                return self._finish(q, PipelineResult(r.answer, "kg", [r.kind], r.sources, trace=trace, **base), cacheable)
        if not dependent and o.use_faq and self.faq is not None:
            r = self.faq.match(q)
            trace.append({"step": len(trace), "action": "faq lookup",
                          "outcome": f"hit {r.detail}" if isinstance(r, FAQHit) else f"{r.reason} {r.detail}"})
            if isinstance(r, FAQHit):
                return self._finish(q, PipelineResult(r.answer, "faq", [r.detail["entry"]], r.sources, trace=trace, **base),
                                    cacheable)
        # 5. scope gate (cheap: one cached embedding + one nearest-neighbour lookup)
        named_authors = len(self.kg._names(q)[0])
        # every turn is gated except a genuine follow-up ("how accurate was it?") or one naming an author
        if o.use_scope_gate and not dependent and not named_authors:
            sim = self.retriever.best_similarity(q)
            trace.append({"step": len(trace), "action": "scope gate", "outcome": f"best chunk similarity {sim:.3f}"})
            if sim < config.SCOPE_MIN_SIMILARITY:
                self.audit.event(rid, "out_of_scope", pipeline=o.name, similarity=round(sim, 4))
                return PipelineResult(guard.REFUSAL_SCOPE, "out_of_scope", [f"similarity {sim:.3f}"], trace=trace, **base)
        # 6. route
        decision = decide(q, self.index, bool(history), named_authors) if o.use_router else None
        reasons = decision.reasons if decision else ["router disabled"]
        if decision is None or decision.route == "agent":
            try:
                a = self._run_agent(q, mode, scripted, history)
                return self._finish(q, PipelineResult(a["answer"], "agent", reasons, self._ids(a["evidence"]),
                                                      events + a["events"], trace + a["trace"], history_user=q), cacheable)
            except Exception as e:
                self.audit.event(rid, "agent_failed", error=f"{type(e).__name__}: {e}")
                reasons = reasons + [f"agent failed: {type(e).__name__}"]
                route_prefix = "agent->standard_fallback"
        else:
            route_prefix = "standard"
        items = self.retrieve_standard(q)
        answer, findings = self.writer.write(q, items, self._history_text(history))
        trace.append({"step": len(trace), "action": f"standard path: retrieve {len(items)} block(s) + 1 answer call",
                      "outcome": "output guard: " + (",".join(findings) or "clean")})
        if findings:
            events = events + [f"output_guard:{','.join(findings)}"]
        res = PipelineResult(answer, route_prefix, reasons, self._ids([k for k, _ in items]), events, trace, history_user=q)
        if route_prefix == "standard" and o.escalate and needs_escalation(answer):
            try:
                a = self._run_agent(q, mode, scripted, history, seed=items)
                return self._finish(q, PipelineResult(a["answer"], "standard->agent", reasons + ["insufficient_evidence"],
                                                      self._ids(a["evidence"]), events + a["events"],
                                                      trace + a["trace"], history_user=q), cacheable)
            except Exception as e:
                self.audit.event(rid, "agent_failed", error=f"{type(e).__name__}: {e}")
                res.route, res.reasons = "standard->agent_failed", reasons + [f"agent failed: {type(e).__name__}"]
        return self._finish(q, res, cacheable)

    def _ids(self, keys) -> list[str]:
        from .answer import evidence_paper_ids
        return sorted(evidence_paper_ids(keys, self.index))

    def _finish(self, q: str, res: PipelineResult, cacheable: bool) -> PipelineResult:
        if cacheable and not any(e.startswith(("output_guard", "evidence_sanitized")) for e in res.events):
            self.cache.put(q, {"answer": res.answer, "sources": res.sources})
        return res

    def _log(self, res: PipelineResult, question: str) -> None:
        if not self.request_log:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": res.request_id,
               "pipeline": self.options.name, "route": res.route, "reasons": res.reasons, "agent_used": res.agent_used,
               **res.usage, "latency_s": res.latency_s, "sources": res.sources, "events": res.events,
               "question": guard.redact_secrets(question[:200])}
        with self._log_lock:
            self.request_log.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Session:
    """Multi-turn conversation. Blocked turns never enter memory and only the sanitized
    question is remembered (6.1 memory quarantine)."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.history: list[dict] = []

    def send(self, message: str, mode: str = "batch") -> PipelineResult:
        res = self.pipeline.ask(message, history=list(self.history), mode=mode)
        if res.history_user is not None and not res.blocked:
            self.history += [{"role": "user", "content": res.history_user}, {"role": "assistant", "content": res.answer}]
        return res


# ─── Factory: wire the real components ──────────────────────────────────────────
_STACK: dict = {}
_STACK_LOCK = threading.Lock()


def load_stack() -> dict:
    """Vector store, retriever, KG index, FAQ store and coverage — built once per process."""
    with _STACK_LOCK:
        if not _STACK:
            from .ingestion import MANIFEST_NAME, Manifest, file_sha256
            from .knowledge_graph import load_index
            from .llm import get_embeddings
            from .retrieval import open_vector_store
            emb = get_embeddings()
            db = open_vector_store(emb)
            retriever = HybridRetriever(db)
            if not retriever.n_chunks:
                raise SystemExit("[setup] the vector index is empty — run `python -m avt411 ingest` first.")
            index = load_index()
            missing_kg, _ = index.coverage(retriever.sources)
            manifest = Manifest(Path(config.CHROMA_DIR) / MANIFEST_NAME)
            hashes = ({s: e["sha256"] for s, e in manifest.papers.items()} if manifest.exists else
                      {p.name: file_sha256(p) for p in Path(config.PAPERS_DIR).glob("*.pdf")})
            from .faq import load_faq_store
            faq = load_faq_store(emb, doc_freq=retriever.doc_freq, n_chunks=retriever.n_chunks, current_hashes=hashes)
            if missing_kg:
                print(f"[setup] {len(missing_kg)} indexed paper(s) are not in the knowledge graph; the KG authorship "
                      f"tier is disabled until `ingest --extract-metadata` describes them: {sorted(missing_kg)[:5]}")
            if faq and faq.stale:
                print(f"[setup] {len(faq.stale)} FAQ entr(ies) disabled because their source paper changed: {faq.stale[:5]}")
            _STACK.update(db=db, retriever=retriever, index=index, faq=faq, kg_complete=not missing_kg, embeddings=emb)
    return _STACK


def build_pipeline(options: PipelineOptions = ROUTED, agent_cfg: AgentConfig = SECURE_LEAN,
                   answer_model: str | None = None, planner_model: str | None = None) -> RAGPipeline:
    from .llm import make_llm
    s = load_stack()
    answer_llm = make_llm(answer_model or config.ANSWER_MODEL, max_tokens=agent_cfg.answer_max_tokens)
    planner_llm = make_llm(planner_model or config.PLANNER_MODEL, max_tokens=agent_cfg.planner_max_tokens)
    pf = guard.PersonaFilter(lambda: make_llm(config.FILTER_MODEL, temperature=0, max_tokens=800))
    return RAGPipeline(s["retriever"], s["index"], answer_llm=answer_llm, planner_llm=planner_llm, persona_filter=pf,
                       faq_store=s["faq"], kg_complete=s["kg_complete"], options=options, agent_cfg=agent_cfg)


def result_dict(res: PipelineResult) -> dict:
    d = asdict(res)
    d["agent_used"] = res.agent_used
    return d
