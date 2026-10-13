"""The AVT-411 knowledge graph (Checkpoint 4.1) and the paper registry built from it.

The curated graph (avt411_knowledge_graph.json) covers the 25 test papers. Papers added
later are described in a supplement file (data/kg_supplement.json) that ingestion fills by
extracting title/authors/summary from each new PDF; both are merged at load time. The
registry exposes catalog ids for ANY file-naming scheme (the stem of the PDF file name),
replacing the hard-coded `MP-AVT-411-NN` patterns of earlier checkpoints, and reports
coverage so callers can refuse to claim a complete author list when a paper is missing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from . import config


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def paper_key(source: str) -> str:
    """Catalog id of a paper = its file name without '.pdf' (e.g. MP-AVT-411-14)."""
    return source[:-4] if source.lower().endswith(".pdf") else source


def load_knowledge_graph(path: Path | None = None, supplement: Path | None = None) -> dict:
    kg = json.loads(Path(path or config.KG_PATH).read_text(encoding="utf-8"))
    sup_path = Path(supplement or config.KG_SUPPLEMENT_PATH)
    if sup_path.exists():
        sup = json.loads(sup_path.read_text(encoding="utf-8"))
        have_docs = {d["source"] for d in kg["documents"]}
        have_nodes = {n["id"] for n in kg["nodes"]}
        kg["documents"] += [d for d in sup.get("documents", []) if d["source"] not in have_docs]
        kg["nodes"] += [n for n in sup.get("nodes", []) if n["id"] not in have_nodes]
        kg["edges"] += sup.get("edges", [])
    return kg


def build_graph(kg: dict) -> nx.DiGraph:
    """Directed graph over paper/author/topic/method/tool/application/organization nodes; the
    derived RELATED_BY_SHARED_ENTITIES links are added in both directions."""
    g = nx.DiGraph()
    for n in kg["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in kg["edges"]:
        g.add_edge(e["source"], e["target"], edge_type=e["type"], confidence=e.get("confidence"))
    for link in kg.get("derived_paper_links", []):
        for a, b in ((link["source"], link["target"]), (link["target"], link["source"])):
            g.add_edge(a, b, edge_type=link["type"], score=link.get("score"), reasons=link.get("reasons"))
    return g


class GraphIndex:
    """Lookups built once: papers by id/file, author name -> node id (alias-aware),
    author -> papers, and the catalog."""

    def __init__(self, kg: dict, graph: nx.DiGraph | None = None):
        self.kg = kg
        self.graph = graph if graph is not None else build_graph(kg)
        self.papers: list[dict] = kg["documents"]
        self.paper_by_id = {d["id"]: d for d in self.papers}
        self.paper_by_source = {d["source"]: d for d in self.papers}
        self.filename_to_paper_id = {d["source"]: d["id"] for d in self.papers}
        self.paper_by_key = {paper_key(d["source"]).lower(): d for d in self.papers}
        self.author_lookup = self._build_author_lookup(kg)
        self.author_names = {n["id"]: n["name"] for n in kg["nodes"] if n.get("type") == "author"}
        self.author_papers: dict[str, list[str]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "AUTHOR_OF":
                self.author_papers.setdefault(u, []).append(v)

    @staticmethod
    def _build_author_lookup(kg: dict) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for n in kg["nodes"]:
            if n.get("type") == "author":
                for name in [n["name"], *n.get("aliases", [])]:
                    lookup[name.lower()] = n["id"]
        for canonical, aliases in kg.get("normalization", {}).get("author_aliases", {}).items():
            node_id = lookup.get(canonical.lower())
            if node_id:
                lookup.update({a.lower(): node_id for a in aliases})
        return lookup

    # — registry —
    def catalog_ids(self) -> set[str]:
        return {paper_key(d["source"]) for d in self.papers}

    def coverage(self, indexed_sources: list[str] | set[str]) -> tuple[set[str], set[str]]:
        """(sources indexed but absent from the KG, KG papers not indexed)."""
        indexed = set(indexed_sources)
        return indexed - set(self.paper_by_source), set(self.paper_by_source) - indexed

    def doc_kind(self, doc: dict) -> str:
        node = self.graph.nodes.get(doc["id"], {})
        if node.get("document_type") == "technical_report" or doc["source"].startswith("RSM_"):
            return "Technical Evaluation Report (meeting summary, not a technical paper)"
        return "technical paper"

    def paper_card(self, paper_id: str, label: str) -> tuple[str, str] | None:
        doc = self.paper_by_id.get(paper_id)
        if not doc:
            return None
        return (f"kg:{paper_id}",
                f"=== KG {label}: \"{doc['title']}\" ({doc['source']}) ===\n"
                f"Type: {self.doc_kind(doc)}\nAuthors: {', '.join(doc.get('authors', []))}\n"
                f"Summary: {doc.get('text', '')}")

    def authors_mentioned(self, query: str) -> dict[str, str]:
        """{matched name: author node id} for known names/aliases in the query, longest first."""
        q = query.lower()
        found, seen_ids = {}, set()
        for name in sorted(self.author_lookup, key=len, reverse=True):
            node_id = self.author_lookup[name]
            if node_id not in seen_ids and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", q):
                found[name] = node_id
                seen_ids.add(node_id)
        return found

    def related_papers(self, paper_id: str) -> list[tuple[str, str]]:
        out = []
        if not self.graph.has_node(paper_id):
            return out
        for _, target, data in self.graph.out_edges(paper_id, data=True):
            if data.get("edge_type") == "RELATED_BY_SHARED_ENTITIES" and target in self.paper_by_id:
                reasons = data.get("reasons") or []
                out.append((target, reasons[0]["relation"] if reasons else "shared_entity"))
        return out


def load_index(path: Path | None = None, supplement: Path | None = None) -> GraphIndex:
    return GraphIndex(load_knowledge_graph(path, supplement))


# ─── Supplement: describe papers the curated graph does not cover ─────────────
METADATA_SYSTEM = (
    "You extract bibliographic metadata from the first pages of a research paper. Reply with "
    'ONLY a JSON object: {"title": "...", "authors": ["Full Name", ...], "summary": "one sentence", '
    '"topics": ["...", "..."]}. Use names exactly as printed; no affiliations, no titles (Dr.).'
)


def extract_metadata(text: str, llm) -> dict:
    """One LLM call per NEW paper at ingestion time (not per query)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    raw = llm.invoke([SystemMessage(content=METADATA_SYSTEM), HumanMessage(content=text[:6000])],
                     config={"tags": ["role:metadata"]}).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(raw)
    authors = [a.strip() for a in data.get("authors", []) if isinstance(a, str) and a.strip()]
    if not data.get("title") or not authors:
        raise ValueError("metadata extraction returned no title or authors")
    return {"title": str(data["title"]).strip(), "authors": authors,
            "summary": str(data.get("summary", "")).strip(), "topics": [str(t) for t in data.get("topics", [])][:8]}


def add_supplement_paper(source: str, meta: dict, index: GraphIndex, path: Path | None = None) -> None:
    """Record a new paper (document, paper node, author nodes, AUTHOR_OF edges) in the supplement.
    Existing author nodes are reused when the name (or an alias) is already known."""
    path = Path(path or config.KG_SUPPLEMENT_PATH)
    sup = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"documents": [], "nodes": [], "edges": []}
    sup["documents"] = [d for d in sup["documents"] if d["source"] != source]
    pid = f"paper_{_slug(paper_key(source))}"
    sup["nodes"] = [n for n in sup["nodes"] if n["id"] != pid]
    sup["edges"] = [e for e in sup["edges"] if e["target"] != pid]
    sup["documents"].append({"id": pid, "source": source, "title": meta["title"], "text": meta.get("summary", ""),
                             "topics": meta.get("topics", []), "authors": meta["authors"], "links": [],
                             "provenance": "extracted at ingestion"})
    sup["nodes"].append({"id": pid, "type": "paper", "document_type": "technical_paper", "title": meta["title"],
                         "source": source, "summary": meta.get("summary", "")})
    known_sup = {n["id"] for n in sup["nodes"]}
    for name in meta["authors"]:
        aid = index.author_lookup.get(name.lower()) or f"author_{_slug(name)}"
        if aid not in index.author_names and aid not in known_sup:
            sup["nodes"].append({"id": aid, "type": "author", "name": name, "aliases": []})
            known_sup.add(aid)
        sup["edges"].append({"id": f"sup_{aid}_{pid}", "source": aid, "target": pid, "type": "AUTHOR_OF",
                             "confidence": 0.9, "evidence": {"document": source, "basis": "LLM extraction"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sup, indent=1, ensure_ascii=False), encoding="utf-8")
