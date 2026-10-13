"""PDF ingestion: extract -> chunk -> embed -> Chroma, incrementally.

Checkpoint 3.1 built the index once ("delete chroma_db_papers/ to rebuild"), so adding one
paper meant re-embedding the whole corpus. Here a manifest (sha256 per PDF) next to the
index decides, per paper, whether it is new, changed, removed, or unchanged; only the first
three touch the index. Chunk ids are deterministic (`<file>#<nnn>`), so re-ingesting a paper
replaces exactly its own chunks. Each paper is committed separately, so an interrupted run
resumes where it stopped.

An index built by Checkpoint 3.1 (no manifest, random ids) is adopted in place: the
papers already in it are recorded as ingested, nothing is re-embedded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from . import config, guardrails

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding warnings flood the console
MANIFEST_NAME = "ingest_manifest.json"


def extract_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Overlapping character windows, broken on whitespace (unchanged from Checkpoint 3.1)."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.papers: dict[str, dict] = data.get("papers", {})
        self.exists = path.exists()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "papers": self.papers}, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class SyncReport:
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0
    flagged_chunks: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"new={len(self.new)} changed={len(self.changed)} removed={len(self.removed)} "
                f"unchanged={len(self.unchanged)} adopted={len(self.adopted)} failed={len(self.failed)} "
                f"chunks_added={self.chunks_added} flagged_chunks={len(self.flagged_chunks)}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sync_index(db, papers_dir: Path | None = None, *, index_dir: Path | None = None,
               force: set[str] | None = None, dry_run: bool = False,
               extractor: Callable[[Path], str] = extract_pdf_text,
               on_paper_indexed: Callable[[str, str], None] | None = None,
               log: Callable[[str], None] = print) -> SyncReport:
    """Bring the vector store in line with the PDFs in papers_dir. `db` is a langchain Chroma
    store persisted in index_dir. `on_paper_indexed(source, text)` runs after each new/changed
    paper is committed (used to extract KG metadata for papers the knowledge graph lacks)."""
    papers_dir = Path(papers_dir or config.PAPERS_DIR)
    manifest = Manifest(Path(index_dir or config.CHROMA_DIR) / MANIFEST_NAME)
    report = SyncReport()
    files = {p.name: p for p in sorted(papers_dir.glob("*.pdf"))}
    indexed = {m.get("source") for m in db.get(include=["metadatas"])["metadatas"]}
    force = force or set()

    if not manifest.exists and indexed:
        for src in sorted(indexed & set(files)):
            n = len(db.get(where={"source": src}, include=[])["ids"])
            manifest.papers[src] = {"sha256": file_sha256(files[src]), "chunks": n, "ingested_at": _now(), "adopted": True}
            report.adopted.append(src)
        log(f"[ingest] adopted existing index: {len(report.adopted)} paper(s) recorded, nothing re-embedded")

    for src in sorted(set(manifest.papers) | indexed):
        if src and src not in files:
            report.removed.append(src)

    todo: list[str] = []
    for src, path in files.items():
        entry = manifest.papers.get(src)
        if entry is None or src not in indexed:
            report.new.append(src)
            todo.append(src)
        elif src in force or entry.get("sha256") != file_sha256(path):
            report.changed.append(src)
            todo.append(src)
        else:
            report.unchanged.append(src)

    if dry_run:
        return report

    for src in report.removed:
        db.delete(where={"source": src})
        manifest.papers.pop(src, None)
        manifest.save()
        log(f"[ingest] removed {src}")

    for src in todo:
        path = files[src]
        try:
            text = extractor(path)
        except Exception as e:  # one unreadable PDF must not stop the batch
            report.failed[src] = f"{type(e).__name__}: {e}"
            log(f"[ingest] skip {src}: could not extract text ({e})")
            continue
        seen: set[str] = set()
        chunks = []
        for c in chunk_text(text):
            if c not in seen:  # drop exact duplicate windows (repeated boilerplate pages)
                seen.add(c)
                chunks.append(c)
        if not chunks:
            report.failed[src] = "no extractable text (scanned PDF?)"
            log(f"[ingest] skip {src}: no extractable text")
            continue
        ids = [f"{src}#{i:03d}" for i in range(len(chunks))]
        flagged = [cid for cid, c in zip(ids, chunks) if guardrails.sanitize_evidence(c)[1]]
        db.delete(where={"source": src})
        stem = Path(src).stem
        for start in range(0, len(chunks), config.EMBED_BATCH):
            sl = slice(start, start + config.EMBED_BATCH)
            db.add_texts(chunks[sl], ids=ids[sl],
                         metadatas=[{"source": src, "chunk_id": cid, "paper_id": stem, "chunk_index": i}
                                    for i, cid in enumerate(ids[sl], start=start)])
        manifest.papers[src] = {"sha256": file_sha256(path), "chunks": len(chunks), "ingested_at": _now(),
                                "flagged_chunks": flagged}
        manifest.save()
        report.chunks_added += len(chunks)
        report.flagged_chunks += flagged
        log(f"[ingest] indexed {src}: {len(chunks)} chunk(s)"
            + (f"; {len(flagged)} chunk(s) contain instruction-like text (redacted at query time): {flagged}"
               if flagged else ""))
        if on_paper_indexed:
            try:
                on_paper_indexed(src, text)
            except Exception as e:
                log(f"[ingest] metadata step failed for {src}: {e}")
    if report.adopted:
        manifest.save()
    return report
