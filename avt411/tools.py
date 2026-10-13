"""The agent's deterministic, read-only tools (Checkpoint 5.1 CorpusTools).

The LLM planner picks one; this code runs it. Each tool returns (evidence_blocks, outcome):
evidence_blocks maps a stable key to a labelled text block, outcome is a one-line summary the
planner sees. Changes from 5.1: tools use the new HybridRetriever API, paper references are
resolved against the registry (any file-naming scheme), and the planner catalog can be a
question-specific shortlist once the corpus outgrows the prompt.
"""
from __future__ import annotations

import difflib
import re

from . import config
from .knowledge_graph import GraphIndex, paper_key
from .retrieval import HybridRetriever, tokenize

KG_MAX_PAPERS = 8             # cap on papers a single KG lookup can add
RELATED_PER_PAPER = 4


def _words(text: str) -> set[str]:
    return set(tokenize(text))


class CorpusTools:
    def __init__(self, retriever: HybridRetriever, index: GraphIndex, fuzzy_names: bool = True):
        self.retriever = retriever
        self.index = index
        self.fuzzy_names = fuzzy_names
        self.papers = index.papers
        self._author_names = index.author_names
        self._last_names: dict[str, list[str]] = {}
        for node_id, name in self._author_names.items():
            self._last_names.setdefault(name.split()[-1].lower(), []).append(node_id)
        self._entity_lookup = self._build_entity_lookup(index.kg)
        self._paper_words = {d["source"]: _words(d["title"] + " " + " ".join(d.get("topics", []))) for d in self.papers}

    # — catalog shown to the planner so it can resolve vague paper references —
    def catalog(self, question: str | None = None, include_sources: set[str] | frozenset = frozenset()) -> str:
        papers = self.papers
        if question is not None and len(papers) > config.CATALOG_FULL_LIMIT:
            q = _words(question)
            ranked = sorted(papers, key=lambda d: len(q & self._paper_words[d["source"]]), reverse=True)
            keep = {d["source"] for d in ranked[:config.CATALOG_SHORTLIST]} | set(include_sources)
            papers = [d for d in papers if d["source"] in keep]
        return "\n".join(f"- {paper_key(d['source'])}: {d['title']}" for d in papers)

    def catalog_is_static(self) -> bool:
        return len(self.papers) <= config.CATALOG_FULL_LIMIT

    # — search: hybrid retrieval over every paper —
    def search(self, queries: list[str]) -> tuple[dict[str, str], str]:
        found: dict[str, str] = {}
        for q in queries:
            for h in self.retriever.search(q, config.NUM_RETRIEVED):
                found[h.chunk_id] = f"=== CHUNK {h.chunk_id} ===\n{h.text}"
        sources = sorted({paper_key(k.split("#")[0]) for k in found})
        return found, f"{len(found)} chunk(s) from {', '.join(sources) or 'nothing'}"

    # — paper resolution (catalog id, legacy number, TER, or title words) —
    def resolve_paper(self, ref: str) -> dict | None:
        ref_l = ref.lower().strip()
        for key in sorted(self.index.paper_by_key, key=len, reverse=True):
            if re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", ref_l):
                return self.index.paper_by_key[key]
        if re.search(r"\bter\b", ref_l) or "technical evaluation report" in ref_l:
            reports = [d for d in self.papers if self.index.doc_kind(d).startswith("Technical Evaluation")]
            if len(reports) == 1:
                return reports[0]
        m = re.search(r"411[-_ ]?(\d{1,2})\b", ref_l)
        if m:  # legacy shorthand for this corpus: "411-14" -> MP-AVT-411-14
            return self.index.paper_by_key.get(f"mp-avt-411-{int(m.group(1)):02d}")
        q = _words(ref)
        if not q:
            return None
        best = max(self.papers, key=lambda d: len(q & self._paper_words[d["source"]]))
        return best if len(q & self._paper_words[best["source"]]) / len(q) >= 0.5 else None

    def search_in_paper(self, paper_ref: str, queries: list[str]) -> tuple[dict[str, str], str]:
        doc = self.resolve_paper(paper_ref)
        if not doc:
            return {}, f"could not resolve paper {paper_ref!r} — check the catalog id"
        found: dict[str, str] = {}
        for q in queries:
            for h in self.retriever.search_in_source(q, doc["source"], config.NUM_RETRIEVED):
                found[h.chunk_id] = f"=== CHUNK {h.chunk_id} (paper-scoped search) ===\n{h.text}"
        return found, f"{len(found)} chunk(s) from {paper_key(doc['source'])} (\"{doc['title']}\")"

    # — find_by_author: exact or fuzzy author resolution, then AUTHOR_OF —
    def resolve_author(self, name: str) -> tuple[str, str] | None:
        """(node_id, how) for a possibly partial/misspelled/nickname author string."""
        n = name.lower().strip()
        if n in self.index.author_lookup:
            return self.index.author_lookup[n], "exact"
        exact = self.index.authors_mentioned(n)
        if exact:
            return next(iter(exact.values())), "exact"
        if not self.fuzzy_names:
            return None
        parts = n.replace(".", " ").split()
        if not parts:
            return None
        cands = self._last_names.get(parts[-1], [])
        if len(cands) > 1 and len(parts) > 1:
            cands = [c for c in cands if self._author_names[c].lower().startswith(parts[0][0])] or cands
        if len(cands) == 1:
            return cands[0], "last-name match"
        full = {v.lower(): k for k, v in self._author_names.items()}
        close = difflib.get_close_matches(n, list(full), n=1, cutoff=0.75)
        if close:
            return full[close[0]], "fuzzy full-name match"
        close = difflib.get_close_matches(parts[-1], list(self._last_names), n=1, cutoff=0.8)
        if close and len(self._last_names[close[0]]) == 1:
            return self._last_names[close[0]][0], "fuzzy last-name match"
        return None

    def papers_by_authors(self, author_ids: list[str]) -> list[str]:
        sets = [set(self.index.author_papers.get(a, [])) for a in author_ids]
        return sorted(set.intersection(*sets)) if sets else []

    def find_by_author(self, names: list[str]) -> tuple[dict[str, str], str]:
        resolved, notes = [], []
        for name in names:
            hit = self.resolve_author(name)
            if hit:
                resolved.append(hit[0])
                notes.append(f"{name!r} -> {self._author_names[hit[0]]} ({hit[1]})")
            else:
                notes.append(f"{name!r} -> no author match")
        if not resolved:
            return {}, "; ".join(notes)
        papers = self.papers_by_authors(resolved)
        who = " & ".join(self._author_names[a] for a in resolved)
        label = f"AUTHOR MATCH ({'co-authored by' if len(resolved) > 1 else 'papers by'} {who})"
        found = dict(filter(None, (self.index.paper_card(p, label) for p in papers[:KG_MAX_PAPERS])))
        return found, f"{'; '.join(notes)}; {len(papers)} paper(s)"

    # — find_by_topic: KG topic/method/tool/application nodes -> papers —
    @staticmethod
    def _build_entity_lookup(kg: dict) -> dict[str, str]:
        lookup = {}
        for n in kg["nodes"]:
            if n.get("type") in {"topic", "method", "tool", "application"}:
                for name in [n["name"], *n.get("aliases", [])]:
                    lookup[name.lower()] = n["id"]
        for canonical, aliases in kg.get("normalization", {}).get("concept_aliases", {}).items():
            node_id = lookup.get(canonical.lower())
            if node_id:
                lookup.update({a.lower(): node_id for a in aliases})
        return lookup

    def find_by_topic(self, term: str) -> tuple[dict[str, str], str]:
        t = term.lower().strip()
        ids = {self._entity_lookup[t]} if t in self._entity_lookup else set()
        if not ids and len(t) >= 3:
            ids = {nid for name, nid in self._entity_lookup.items() if len(name) >= 3 and (t in name or name in t)}
        if not ids:
            ids = {self._entity_lookup[c] for c in difflib.get_close_matches(t, list(self._entity_lookup), n=3, cutoff=0.8)}
        ids = set(sorted(ids)[:6])
        if not ids:
            return {}, f"no KG entity matches {term!r}"
        papers = {u for u, v in self.index.graph.edges() if v in ids and u in self.index.paper_by_id}
        names = ", ".join(self.index.graph.nodes[i].get("name", i) for i in ids)
        found = dict(filter(None, (self.index.paper_card(p, f"TOPIC MATCH ({names})") for p in sorted(papers)[:KG_MAX_PAPERS])))
        return found, f"entities [{names}] -> {len(papers)} paper(s)"

    # — related_papers: RELATED_BY_SHARED_ENTITIES —
    def related_papers(self, paper_ref: str) -> tuple[dict[str, str], str]:
        doc = self.resolve_paper(paper_ref)
        if not doc:
            return {}, f"could not resolve paper {paper_ref!r}"
        found = {}
        for pid, reason in self.index.related_papers(doc["id"])[:RELATED_PER_PAPER]:
            block = self.index.paper_card(pid, f"RELATED PAPER (via {reason.replace('_', ' ')})")
            if block:
                found[block[0]] = block[1]
        return found, f"{len(found)} paper(s) related to {paper_key(doc['source'])}"

    def run(self, plan: dict) -> tuple[dict[str, str], str]:
        a = plan["action"]
        if a == "search":
            return self.search(plan["queries"])
        if a == "search_in_paper":
            return self.search_in_paper(plan.get("paper", ""), plan.get("queries") or [plan.get("paper", "")])
        if a == "find_by_author":
            return self.find_by_author(plan["authors"])
        if a == "find_by_topic":
            return self.find_by_topic(plan["term"])
        if a == "related_papers":
            return self.related_papers(plan["paper"])
        return {}, f"unknown action {a!r}"


def describe_plan(plan: dict) -> str:
    import json
    args = {k: plan[k] for k in ("queries", "paper", "authors", "term") if plan.get(k)}
    return f"{plan['action']}({json.dumps(args, ensure_ascii=False)})"


TOOLS = frozenset({"search", "search_in_paper", "find_by_author", "find_by_topic", "related_papers"})
