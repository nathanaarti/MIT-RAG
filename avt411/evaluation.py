"""Evaluation: retrieval recall, FAQ/scope calibration, and quality-vs-cost comparisons.

  retrieval_eval   free (local BM25 + cached query embeddings): does top-k contain a chunk
                   from the paper(s) testInputs.json says answer the question?
  faq_eval         embeddings only: FAQ match precision/recall on paraphrases (should hit),
                   leave-one-out + near-miss + authorship questions (should miss).
  scope_eval       embeddings only: best-chunk similarity of every eval question vs off-domain probes.
  quality_eval     paid: answers testInputs (+ incomplete set) with several systems, grades
                   pass/fail with the Checkpoint 3.1 correctness prompt, and records route,
                   LLM calls, agent use, tokens, billed $, latency per answer.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import config
from . import guardrails as guard
from .knowledge_graph import paper_key
from .metering import metered
from .retrieval import normalize_scores, tokenize

JUDGE_PROMPT = (
    "You are grading a retrieval-augmented answer against reference grading notes.\n"
    "Return 'pass' if the response is factually consistent with the grading notes and "
    "captures their main point(s) — even if it omits some minor details or is worded "
    "differently. Return 'fail' only if the response contradicts the notes, is "
    "unsupported by them, or misses the central point.\n"
    "Response: {response}\nGrading Notes: {grading_notes}\n\n"
    "Reply with exactly one word: pass or fail."
)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


def load_cases(which: str) -> list[dict]:
    full = json.loads(config.TEST_INPUTS.read_text(encoding="utf-8"))
    rows = [{"question": r["question"], "grading_notes": r["grading_notes"], "sources": r["sources"], "row": i,
             "kind": "multi_source" if len(r["sources"]) > 1 else "single_source"} for i, r in enumerate(full)]
    if which == "full":
        return rows
    by_q = {r["question"]: r for r in rows}
    inc = json.loads(config.TEST_INPUTS_INCOMPLETE.read_text(encoding="utf-8"))
    cases = [{**c, "grading_notes": by_q[c["base_question"]]["grading_notes"],
              "sources": by_q[c["base_question"]]["sources"], "row": by_q[c["base_question"]]["row"]} for c in inc]
    return cases if which == "incomplete" else rows + cases


class Judge:
    def __init__(self, model: str | None = None):
        from .llm import make_llm
        self._llm = make_llm(model or config.JUDGE_MODEL, temperature=0)

    def choose(self, prompt: str, choices: list[str]) -> str:
        raw = (self._llm.invoke(prompt, config={"tags": ["role:judge"]}).content or "").lower()
        found = [(m.start(), c) for c in choices for m in [re.search(rf"\b{re.escape(c)}\b", raw)] if m]
        return min(found)[1] if found else choices[-1]

    def grade(self, response: str, grading_notes: str) -> str:
        return self.choose(JUDGE_PROMPT.format(response=response, grading_notes=grading_notes), ["pass", "fail"])


# ─── 1. Retrieval recall (free) ────────────────────────────────────────────────
def _legacy_3_1_topk(retriever, query: str, k: int, pool: int = 10) -> list[str]:
    """Checkpoint 3.1's fusion, reproduced for comparison: vector hits keyed by FILE name."""
    scores = retriever._bm25.get_scores(tokenize(query))
    bm_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
    bm = [(retriever._paths[i], scores[i]) for i in bm_idx]
    vec = [(d.metadata.get("source", "unknown"), s) for d, s in retriever._db.similarity_search_with_score(query, k=pool)]
    bm_n = {n: v for (n, _), v in zip(bm, normalize_scores([s for _, s in bm]))}
    vec_n = {n: v for (n, _), v in zip(vec, normalize_scores([s for _, s in vec], invert=True))}
    names = set(bm_n) | set(vec_n)
    return sorted(names, key=lambda n: 0.5 * bm_n.get(n, 0) + 0.5 * vec_n.get(n, 0), reverse=True)[:k]


def _gold_chunk(retriever, case: dict) -> str | None:
    """The chunk of the expected paper that best matches the reference answer (BM25 of the
    grading notes restricted to that paper) — a proxy label for 'the chunk with the answer'."""
    src = case["sources"][0]
    idx = [i for i, p in enumerate(retriever._paths) if p.startswith(src + "#")]
    if not idx:
        return None
    scores = retriever._bm25.get_scores(tokenize(case["grading_notes"]))
    return retriever._paths[max(idx, key=lambda i: scores[i])]


def retrieval_eval(retriever, ks=(4, 6, 8)) -> str:
    cases = load_cases("full")
    single = [c for c in cases if len(c["sources"]) == 1 and not re.search(r"\bauthor", c["question"], re.I)]
    gold = {c["question"]: _gold_chunk(retriever, c) for c in single}

    def fixed(pool):
        def fn(q, k):
            old = retriever.candidate_pool
            retriever.candidate_pool = pool
            try:
                return [h.chunk_id for h in retriever.search(q, k)]
            finally:
                retriever.candidate_pool = old
        return fn

    configs = {"legacy 3.1 fusion (pool 10, file-keyed vectors)": lambda q, k: _legacy_3_1_topk(retriever, q, k),
               "fixed fusion, pool 10": fixed(10), "fixed fusion, pool 20": fixed(20), "fixed fusion, pool 30": fixed(30),
               "bm25 only": lambda q, k: retriever.bm25_only(q, k), "vector only": lambda q, k: retriever.vector_only(q, k)}
    lines = ["## Retrieval on testInputs.json (no LLM calls)", "",
             f"paper hit@k: top-k holds a chunk from an expected source paper (all 70 rows). gold hit@k: top-k holds the "
             f"chunk of that paper that best matches the reference answer ({len(single)} single-paper fact rows).", "",
             "| Retriever | " + " | ".join(f"paper hit@{k}" for k in ks) + " | " + " | ".join(f"gold hit@{k}" for k in ks)
             + " | ms/query |", "|---|" + "---|" * (2 * len(ks) + 1)]
    for name, fn in configs.items():
        hits = {k: 0 for k in ks}
        ghits = {k: 0 for k in ks}
        t0 = time.perf_counter()
        for c in cases:
            got = fn(c["question"], max(ks))
            for k in ks:
                if {n.split("#")[0] for n in got[:k]} & set(c["sources"]):
                    hits[k] += 1
                if c["question"] in gold and gold[c["question"]] in got[:k]:
                    ghits[k] += 1
        ms = (time.perf_counter() - t0) * 1000 / len(cases)
        lines.append(f"| {name} | " + " | ".join(f"{hits[k]}/{len(cases)}" for k in ks) + " | "
                     + " | ".join(f"{ghits[k]}/{len(single)}" for k in ks) + f" | {ms:.0f} |")
    lines += ["", "The legacy row keys vector hits by file name, so its 'chunks' can be whole-paper labels; its gold "
                  "hits count only BM25-keyed chunks. Timing includes the query embedding (cached after first use)."]
    return "\n".join(lines) + "\n"


# ─── 2. FAQ calibration (embeddings only) ─────────────────────────────────────
def faq_eval(store, kg_answerer=None, thresholds=(0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88),
             margins=(0.0, 0.02, 0.04, 0.06)) -> tuple[str, dict]:
    data = json.loads(config.FAQ_PARAPHRASES.read_text(encoding="utf-8"))
    full = json.loads(config.TEST_INPUTS.read_text(encoding="utf-8"))
    inc = json.loads(config.TEST_INPUTS_INCOMPLETE.read_text(encoding="utf-8"))
    row_of = {r["question"]: i for i, r in enumerate(full)}
    ids = {e["id"] for e in store.entries}
    pos = [(p["question"], f"faq-{p['row']:03d}") for p in data["positives"] if f"faq-{p['row']:03d}" in ids]
    pos += [(c["question"], f"faq-{row_of[c['base_question']]:03d}") for c in inc
            if c["kind"] == "vague_reference" and f"faq-{row_of[c['base_question']]:03d}" in ids]
    # negatives: (question, entries to exclude from the store for this probe)
    neg = [(e["question"], {e["id"]}, "leave_one_out") for e in store.entries]
    neg += [(q, set(), "near_miss") for q in data["near_miss"]]
    neg += [(r["question"], set(), "authorship_row") for r in full if re.search(r"\bauthor", r["question"], re.I)]
    neg += [(c["question"], set(), f"incomplete_{c['kind']}") for c in inc if c["kind"] != "vague_reference"]

    def run(question, exclude):
        ranked = store.score(question, exclude)
        (s1, e1), s2 = ranked[0], ranked[1][0]
        missing = store.distinctive_terms(question) - store._terms[e1["id"]]
        return s1, s2, e1["id"], sorted(missing)

    pos_runs = [(q, want, *run(q, set())) for q, want in pos]
    neg_runs = [(q, kind, *run(q, ex)) for q, ex, kind in neg]
    grid = []
    for t in thresholds:
        for m in margins:
            tp = sum(1 for _, want, s1, s2, eid, miss in pos_runs if s1 >= t and s1 - s2 >= m and not miss and eid == want)
            wrong_pos = sum(1 for _, want, s1, s2, eid, miss in pos_runs if s1 >= t and s1 - s2 >= m and not miss and eid != want)
            fp = sum(1 for _, _, s1, s2, eid, miss in neg_runs if s1 >= t and s1 - s2 >= m and not miss)
            grid.append({"threshold": t, "margin": m, "recall": tp / len(pos_runs), "tp": tp,
                         "wrong_entry": wrong_pos, "false_matches": fp})
    safe = [g for g in grid if g["false_matches"] == 0 and g["wrong_entry"] == 0]
    best = max(safe, key=lambda g: (g["recall"], g["threshold"], g["margin"])) if safe else None
    lines = [f"## FAQ matcher calibration — {len(pos_runs)} should-hit paraphrases, {len(neg_runs)} should-miss probes", "",
             "| threshold | margin | recall (correct entry) | wrong entry | false matches |", "|---|---|---|---|---|"]
    for g in grid:
        lines.append(f"| {g['threshold']:.2f} | {g['margin']:.2f} | {g['tp']}/{len(pos_runs)} ({g['recall']:.0%}) | "
                     f"{g['wrong_entry']} | {g['false_matches']} |")
    lines += ["", f"Configured: threshold={store.min_similarity}, margin={store.min_margin}, "
                  f"distinctive-term df<{store.distinctive_df:.0%} of chunks."]
    if best:
        lines.append(f"Highest-recall setting with zero false or wrong matches: threshold={best['threshold']}, "
                     f"margin={best['margin']} -> recall {best['tp']}/{len(pos_runs)}.")
    lines += ["", "Should-miss probes with the highest similarity (closest calls):", ""]
    for q, kind, s1, s2, eid, miss in sorted(neg_runs, key=lambda r: -r[2])[:8]:
        lines.append(f"- {s1:.3f} (runner-up {s2:.3f}) {kind}: {q[:90]!r} -> nearest {eid}; distinctive terms missing: {miss}")
    lines += ["", "Should-hit paraphrases with the lowest similarity:", ""]
    for q, want, s1, s2, eid, miss in sorted(pos_runs, key=lambda r: r[2])[:8]:
        lines.append(f"- {s1:.3f} (runner-up {s2:.3f}) want {want}, nearest {eid}; missing {miss}: {q[:90]!r}")
    return "\n".join(lines) + "\n", {"grid": grid, "best": best}


def scope_eval(retriever) -> str:
    data = json.loads(config.FAQ_PARAPHRASES.read_text(encoding="utf-8"))
    ins = [c["question"] for c in load_cases("all")]
    ins_s = sorted((retriever.best_similarity(q), q) for q in ins)
    out_s = sorted(((retriever.best_similarity(q), q) for q in data["off_domain"]), reverse=True)
    lines = ["## Scope gate — best chunk cosine similarity", "",
             f"In-domain eval questions ({len(ins)}): min {ins_s[0][0]:.3f}, p5 {_pct([s for s, _ in ins_s], .05):.3f}, "
             f"median {statistics.median(s for s, _ in ins_s):.3f}",
             f"Off-domain probes ({len(out_s)}): max {out_s[0][0]:.3f}, median {statistics.median(s for s, _ in out_s):.3f}",
             f"Configured floor: {config.SCOPE_MIN_SIMILARITY} -> blocks {sum(s < config.SCOPE_MIN_SIMILARITY for s, _ in ins_s)} "
             f"in-domain, {sum(s < config.SCOPE_MIN_SIMILARITY for s, _ in out_s)}/{len(out_s)} off-domain", "",
             "Lowest in-domain:"] + [f"- {s:.3f} {q[:100]!r}" for s, q in ins_s[:5]] + ["", "Off-domain:"] + \
            [f"- {s:.3f} {q[:100]!r}" for s, q in out_s]
    return "\n".join(lines) + "\n"


# ─── 3. Offline guard checks (from 6.1) ───────────────────────────────────────
def guard_fp() -> str:
    qs = [c["question"] for c in load_cases("all")]
    changed = [(q, v) for q in qs for v in [guard.inspect_user_input(q)] if v.action != "allow" or v.text != q]
    blocked = [c for c in changed if c[1].action == "block"]
    lines = [f"Input guard over {len(qs)} legitimate eval questions: {len(blocked)} blocked, "
             f"{len(changed) - len(blocked)} altered, {len(qs) - len(changed)} untouched"]
    lines += [f"  {v.action:8s} {v.findings}  {q[:90]!r} -> {v.text[:90]!r}" for q, v in changed]
    return "\n".join(lines) + "\n"


def scan_corpus(retriever, index) -> str:
    hits = [(cid, f) for cid, text in zip(retriever._paths, retriever._contents) for f in [guard.sanitize_evidence(text)[1]] if f]
    hits += [(f"kg:{d['source']}", f) for d in index.papers for f in [guard.sanitize_evidence(d.get("text", ""))[1]] if f]
    n = len(retriever._contents) + len(index.papers)
    return (f"Indirect-injection scan: {len(hits)} of {n} chunks/records flagged ({len(hits) / max(1, n):.2%}).\n" +
            "".join(f"  {cid}: {f}\n" for cid, f in hits[:25]))


# ─── 4. Quality vs cost ───────────────────────────────────────────────────────
class Hybrid31:
    """The Checkpoint 3.1 single-call pipeline (k=4, no hardening) — the cost floor reference."""

    def __init__(self, retriever, llm):
        self.retriever, self.llm = retriever, llm

    def ask(self, question: str, **_):
        from langchain_core.messages import HumanMessage, SystemMessage
        from . import prompts
        hits = self.retriever.search(question, config.NUM_RETRIEVED)
        ctx = "\n\n---\n\n".join(f"[{h.chunk_id}]\n{h.text}" for h in hits)
        ans = self.llm.invoke([SystemMessage(content=prompts.BASE_ANSWER_SYSTEM),
                               HumanMessage(content=f"Context (paper excerpts):\n{ctx}\n\nQuestion: {question}")],
                              config={"tags": ["role:answer"]}).content
        from .pipeline import PipelineResult
        return PipelineResult(ans, "hybrid_3_1", sources=sorted({paper_key(h.source) for h in hits}))


def build_system(name: str):
    from .pipeline import AGENT_ONLY, ROUTED, STANDARD_ONLY, build_pipeline, load_stack
    from dataclasses import replace
    if name == "hybrid_3_1":
        from .llm import make_llm
        return Hybrid31(load_stack()["retriever"], make_llm(config.ANSWER_MODEL))
    opts = {"routed": replace(ROUTED, use_cache=False), "routed_no_faq": replace(ROUTED, name="routed_no_faq", use_cache=False,
                                                                                 use_faq=False),
            "agent_only": AGENT_ONLY, "standard_only": STANDARD_ONLY}[name]
    return build_pipeline(opts)


def answer_case(system, case: dict) -> dict:
    t0 = time.perf_counter()
    with metered() as m:
        try:
            reply = case.get("clarification")
            res = system.ask(case["question"], mode="scripted" if reply else "batch",
                             scripted_replies=[reply] if reply else None)
            answer, route, reasons, sources, trace = res.answer, res.route, res.reasons, res.sources, res.trace
        except Exception as e:
            answer, route, reasons, sources, trace = f"[error] {type(e).__name__}: {e}", "error", [], [], []
    t = m.totals()
    return {"answer": answer, "route": route, "reasons": ";".join(reasons), "sources": ";".join(sources),
            "agent_used": int(route in ("agent", "standard->agent", "agent->standard_fallback")),
            "llm_calls": t["calls"], "planner_calls": t.get("planner_calls", 0), "answer_calls": t.get("answer_calls", 0),
            "filter_calls": t.get("filter_calls", 0), "input_tokens": t["input"], "output_tokens": t["output"],
            "cost_usd": round(t["cost"], 6), "cost_estimated": t["estimated"],
            "latency_s": round(time.perf_counter() - t0, 2), "trace": trace}


def quality_eval(which: str, system_names: list[str], workers: int = 6, limit: int | None = None) -> Path:
    cases = load_cases(which)
    cases = cases[:limit] if limit else cases
    systems = {n: build_system(n) for n in system_names}
    judge = Judge()
    jobs = [(n, c) for c in cases for n in system_names]
    print(f"Quality eval: {len(cases)} question(s) x {len(system_names)} system(s) = {len(jobs)} answers ({workers} workers)")
    done, lock = [0], threading.Lock()

    def work(job):
        n, c = job
        r = answer_case(systems[n], c)
        r["score"] = judge.grade(r["answer"], c["grading_notes"]) if not r["answer"].startswith("[error]") else "fail"
        row = {"system": n, "kind": c["kind"], "row": c["row"], "question": c["question"],
               "clarification": c.get("clarification", ""), **r}
        with lock:
            done[0] += 1
            print(f"  [{done[0]:3d}/{len(jobs)}] {n:13s} {row['score']:4s} {row['route']:24s} calls={row['llm_calls']} "
                  f"${row['cost_usd']:.4f} {row['latency_s']:5.1f}s  {c['question'][:50]}")
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, jobs))
    out = config.OUTPUT_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"quality_{which}_{_stamp()}"
    fields = ["system", "kind", "row", "question", "clarification", "score", "route", "reasons", "agent_used", "llm_calls",
              "planner_calls", "answer_calls", "filter_calls", "input_tokens", "output_tokens", "cost_usd",
              "cost_estimated", "latency_s", "sources", "answer", "trace"]
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "trace": json.dumps(r["trace"], ensure_ascii=False, default=str)})
    md = summarize_quality(rows, system_names)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\nResults: {stem.with_suffix('.csv')}")
    return stem.with_suffix(".csv")


def summarize_quality(rows: list[dict], names: list[str]) -> str:
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    base = "agent_only" if "agent_only" in names else names[0]
    base_cost = statistics.mean(r["cost_usd"] for r in rows if r["system"] == base) or 1e-12
    out = ["## Quality vs cost", "",
           "Pass/fail by the Checkpoint 3.1 correctness judge (gpt-5.4-mini, temperature 0). Cost is what OpenRouter "
           "billed; judge calls excluded. Latency is end-to-end per question.", "",
           "| System | Pass | LLM calls/q | Agent runs | Planner calls/q | In tok/q | Out tok/q | $/q | vs " + base +
           " | $/1k q | p50 / p95 latency |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        rs = [r for r in rows if r["system"] == n]
        k = len(rs)
        cost = statistics.mean(r["cost_usd"] for r in rs)
        out.append(f"| {n} | {sum(r['score'] == 'pass' for r in rs)}/{k} | {statistics.mean(r['llm_calls'] for r in rs):.2f} | "
                   f"{sum(r['agent_used'] for r in rs)}/{k} | {statistics.mean(r['planner_calls'] for r in rs):.2f} | "
                   f"{statistics.mean(r['input_tokens'] for r in rs):,.0f} | {statistics.mean(r['output_tokens'] for r in rs):,.0f} | "
                   f"${cost:.4f} | {cost / base_cost:.2f}× | ${cost * 1000:.2f} | "
                   f"{_pct([r['latency_s'] for r in rs], .5):.1f}s / {_pct([r['latency_s'] for r in rs], .95):.1f}s |")
    out += ["", "Pass by question kind:", "", "| System | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
    for n in names:
        rs = [r for r in rows if r["system"] == n]
        out.append(f"| {n} | " + " | ".join(f"{sum(r['score'] == 'pass' for r in rs if r['kind'] == kd)}/"
                                            f"{sum(r['kind'] == kd for r in rs)}" for kd in kinds) + " |")
    for n in names:
        rs = [r for r in rows if r["system"] == n]
        routes: dict[str, list] = {}
        for r in rs:
            routes.setdefault(r["route"], []).append(r)
        if len(routes) > 1 or n.startswith("routed"):
            out += ["", f"Routes taken by {n}:", "", "| Route | Questions | Pass | LLM calls/q | $/q | mean latency |",
                    "|---|---|---|---|---|---|"]
            for route, rr in sorted(routes.items(), key=lambda kv: -len(kv[1])):
                out.append(f"| {route} | {len(rr)} | {sum(r['score'] == 'pass' for r in rr)}/{len(rr)} | "
                           f"{statistics.mean(r['llm_calls'] for r in rr):.2f} | ${statistics.mean(r['cost_usd'] for r in rr):.4f} | "
                           f"{statistics.mean(r['latency_s'] for r in rr):.1f}s |")
            faq_rows = [r for r in rs if r["route"] == "faq"]
            if faq_rows:
                non = [r for r in rs if r["route"] != "faq"]
                out.append(f"\nFAQ answers are the reference answers themselves, so their pass rate is not evidence of "
                           f"quality. Excluding the {len(faq_rows)} FAQ hits, {n} passed "
                           f"{sum(r['score'] == 'pass' for r in non)}/{len(non)}.")
    if base in names:
        for n in names:
            if n == base:
                continue
            a = {(r["question"], r["clarification"]): r for r in rows if r["system"] == base}
            b = {(r["question"], r["clarification"]): r for r in rows if r["system"] == n}
            gained = [k for k in a if k in b and a[k]["score"] == "fail" and b[k]["score"] == "pass"]
            lost = [k for k in a if k in b and a[k]["score"] == "pass" and b[k]["score"] == "fail"]
            out += ["", f"Paired {n} vs {base}: {len(gained)} question(s) gained, {len(lost)} lost."]
            out += [f"- lost: {q[:100]!r} (route {b[(q, c)]['route']})" for q, c in lost]
            out += [f"- gained: {q[:100]!r} (route {b[(q, c)]['route']})" for q, c in gained]
    return "\n".join(out) + "\n"
