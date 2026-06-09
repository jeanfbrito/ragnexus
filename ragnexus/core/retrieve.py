"""Hybrid retrieval: dense + BM25 fused with RRF (T6).

Runs a vector KNN search (sqlite-vec) and a keyword BM25 search (FTS5) over the
same corpus DB independently, then fuses the two ranked lists with Reciprocal Rank
Fusion. Each surviving chunk is reconstructed into a :class:`~ragnexus.contracts.Document`
from the ``vec_chunks`` auxiliary columns and paired with its fused RRF score.

Optional cross-encoder rerank is an MVP 2.0 concern.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec
from sqlite_vec import serialize_float32

from ..contracts import Document

# RRF smoothing constant (standard default).
RRF_K = 60

# vec_chunks aux columns, in select order, mapped to metadata reconstruction.
_VEC_COLUMNS = "chunk_id, text, source, corpus, page, section_path, kind"


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _fts_query(query: str) -> str:
    """Build an FTS5 MATCH string that can't trip the query grammar.

    FTS5 treats characters like ``( ) " * - :`` as operators, so a raw user query
    such as ``spark plug (M10)`` raises ``fts5: syntax error``. We tokenize to
    alphanumeric runs and wrap each token as an FTS5 string literal (double quotes,
    internal quotes doubled), OR-joined. Empty/punctuation-only queries yield ``""``.
    """
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return '""'
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def _vector_search(
    db: sqlite3.Connection, qvec: list[float], k: int
) -> list[int]:
    """Return chunk_ids ranked by ascending vector distance (nearest first)."""
    emb = serialize_float32(np.asarray(qvec, dtype=np.float32))
    rows = db.execute(
        """
        SELECT chunk_id
        FROM vec_chunks
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [emb, k],
    ).fetchall()
    return [int(r[0]) for r in rows]


def _keyword_search(db: sqlite3.Connection, query: str, k: int) -> list[int]:
    """Return chunk_ids ranked by BM25 (best first). Empty on no match."""
    match = _fts_query(query)
    rows = db.execute(
        """
        SELECT rowid
        FROM fts_chunks
        WHERE fts_chunks MATCH ?
        ORDER BY bm25(fts_chunks)
        LIMIT ?
        """,
        [match, k],
    ).fetchall()
    return [int(r[0]) for r in rows]


def _rrf_fuse(*ranked_lists: list[int]) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across ranked lists of chunk_ids.

    ``score(id) = Σ 1 / (RRF_K + rank)`` over each list the id appears in (rank is
    1-based). Returns ``(chunk_id, score)`` sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _reconstruct(db: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, Document]:
    """Fetch and rebuild Documents for the given chunk_ids from vec_chunks."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = db.execute(
        f"SELECT {_VEC_COLUMNS} FROM vec_chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()

    docs: dict[int, Document] = {}
    for chunk_id, text, source, corpus, page, section_path, kind in rows:
        meta: dict = {"chunk_id": int(chunk_id)}
        if source is not None:
            meta["source"] = source
        if corpus is not None:
            meta["corpus"] = corpus
        if page is not None:
            meta["page"] = int(page)
        if section_path is not None:
            meta["section_path"] = section_path
        if kind is not None:
            meta["kind"] = kind
        docs[int(chunk_id)] = {"text": text, "metadata": meta}  # type: ignore[typeddict-item]
    return docs


def search(
    db_path: Path,
    query: str,
    *,
    top_k: int = 5,
    embedder=None,
) -> list[tuple[Document, float]]:
    """Hybrid search a corpus index, returning ``(document, score)`` ranked desc.

    Embeds ``query`` (with the query-side prefix — ``is_query=True``), runs vector
    KNN and FTS5 BM25 over ``top_k * 4`` candidates each, RRF-fuses the two ranked
    lists, and returns the top ``top_k`` reconstructed documents with their fused
    scores. ``embedder`` may be injected (e.g. in tests); otherwise the default
    local embedder is built lazily so this is usable standalone.
    """
    if embedder is None:
        from ragnexus.core.embedding import get_embedder

        embedder = get_embedder()

    qvec = embedder.encode([query], is_query=True)[0]

    candidate_k = max(top_k * 4, top_k)
    db = _connect(db_path)
    try:
        vec_ranked = _vector_search(db, qvec, candidate_k)
        kw_ranked = _keyword_search(db, query, candidate_k)
        fused = _rrf_fuse(vec_ranked, kw_ranked)[:top_k]
        docs = _reconstruct(db, [cid for cid, _ in fused])
    finally:
        db.close()

    results: list[tuple[Document, float]] = []
    for chunk_id, score in fused:
        doc = docs.get(chunk_id)
        if doc is not None:
            results.append((doc, score))
    return results
