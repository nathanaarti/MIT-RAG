"""Hybrid (BM25 + vector) chunk retrieval — the Checkpoint 2.1/3.1 retriever.

Changes from 3.1, each made for the 25 -> 200+ paper deployment:
  * BM25 is built from the chunk texts already stored in the Chroma collection instead of
    re-extracting every PDF on every start (97 s for 25 papers; ~13 min at 200).
  * Vector hits are keyed by chunk id. 3.1 keyed them by file name, so a paper contributed
    at most one vector chunk and a chunk's BM25 and vector scores were never fused.
  * The per-retriever candidate pool is a setting (30, was 10) — fusion needs overlap
    between the two candidate lists, which shrinks as the corpus grows.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from . import config

_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def normalize_scores(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale to [0, 1]; invert=True turns a small distance into a high score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


def distance_to_cosine(distance: float) -> float:
    """Chroma's default space is squared L2; OpenAI embeddings are unit-length, so
    ||a-b||^2 = 2 - 2cos."""
    return 1.0 - distance / 2.0


def open_vector_store(embeddings, chroma_dir=None, collection: str | None = None) -> Chroma:
    return Chroma(persist_directory=str(chroma_dir or config.CHROMA_DIR), embedding_function=embeddings,
                  collection_name=collection or config.CHROMA_COLLECTION)


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float

    @property
    def source(self) -> str:
        return self.chunk_id.split("#")[0]


class HybridRetriever:
    """BM25 over every stored chunk + Chroma vector search, min-max fused per query."""

    def __init__(self, db: Chroma, candidate_pool: int | None = None):
        self._db = db
        self.candidate_pool = candidate_pool or config.CANDIDATE_POOL
        self.reload()

    def reload(self) -> None:
        """(Re)build the BM25 side from the vector store — call after ingestion."""
        got = self._db.get(include=["documents", "metadatas"])
        rows = sorted(
            ((m.get("chunk_id") or f"{m.get('source', 'unknown')}#{i:05d}", d)
             for i, (d, m) in enumerate(zip(got["documents"], got["metadatas"]))),
            key=lambda r: r[0])
        self._paths = [r[0] for r in rows]
        self._contents = [r[1] for r in rows]
        self._index_of = {cid: i for i, cid in enumerate(self._paths)}
        tokenized = [tokenize(t) for t in self._contents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self.doc_freq: Counter = Counter()
        for toks in tokenized:
            self.doc_freq.update(set(toks))
        self.n_chunks = len(self._paths)
        self.sources = sorted({p.split("#")[0] for p in self._paths})

    # — the two sides —
    def _bm25_top(self, query: str, k: int, restrict: list[int] | None = None) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        idx = restrict if restrict is not None else range(len(scores))
        top = sorted(idx, key=lambda i: scores[i], reverse=True)[:k]
        return [(self._paths[i], float(scores[i])) for i in top]

    def _vector_top(self, query: str, k: int, source: str | None = None) -> list[tuple[str, str, float]]:
        kwargs = {"filter": {"source": source}} if source else {}
        out = []
        for d, dist in self._db.similarity_search_with_score(query, k=k, **kwargs):
            cid = d.metadata.get("chunk_id") or d.metadata.get("source", "unknown")
            out.append((cid, d.page_content, float(dist)))
        return out

    def _fuse(self, bm: list[tuple[str, float]], vec: list[tuple[str, str, float]], k: int) -> list[Hit]:
        content: dict[str, str] = {}
        bm_norm = {cid: v for (cid, _), v in zip(bm, normalize_scores([s for _, s in bm]))}
        for cid, _ in bm:
            content[cid] = self._contents[self._index_of[cid]]
        vec_norm = {cid: v for (cid, _, _), v in zip(vec, normalize_scores([d for _, _, d in vec], invert=True))}
        for cid, text, _ in vec:
            content.setdefault(cid, text)
        fused = [Hit(cid, text, config.WEIGHT_BM25 * bm_norm.get(cid, 0.0) + config.WEIGHT_VECTOR * vec_norm.get(cid, 0.0))
                 for cid, text in content.items()]
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused[:k]

    # — public API —
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or config.NUM_RETRIEVED
        return self._fuse(self._bm25_top(query, self.candidate_pool),
                          self._vector_top(query, self.candidate_pool), k)

    def search_in_source(self, query: str, source: str, k: int | None = None) -> list[Hit]:
        """The same fusion restricted to one paper (Chroma metadata filter + BM25 subset)."""
        k = k or config.NUM_RETRIEVED
        restrict = [i for i, p in enumerate(self._paths) if p.startswith(source + "#")]
        return self._fuse(self._bm25_top(query, self.candidate_pool, restrict),
                          self._vector_top(query, self.candidate_pool, source), k)

    def best_similarity(self, query: str) -> float:
        """Cosine similarity of the closest chunk — the scope gate's relevance signal."""
        res = self._db.similarity_search_with_score(query, k=1)
        return distance_to_cosine(res[0][1]) if res else 0.0

    # 3.1-compatible accessors (used by evaluation and the retrieval-only baselines)
    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        return [(h.chunk_id, h.text, h.score) for h in self.search(query, k)]

    def bm25_only(self, query: str, k: int) -> list[str]:
        return [cid for cid, _ in self._bm25_top(query, k)]

    def vector_only(self, query: str, k: int) -> list[str]:
        return [cid for cid, _, _ in self._vector_top(query, k)]
