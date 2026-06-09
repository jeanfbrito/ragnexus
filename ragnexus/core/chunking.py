"""Section-aware chunking (T4) — generic, adapter-agnostic splitter.

The PDF adapter already produces well-sized, heading-prefixed chunks via Docling's
HybridChunker, so this stage is **idempotent for that path**: any Document already
within ``max_tokens`` passes through untouched. It only splits oversized Documents —
which is the path future adapters (web/markdown) that emit coarse blocks will lean on.

Invariants:
- Table blocks (``kind == "table"``) are NEVER split; an oversized table passes
  through whole (a split table row is useless for retrieval) and is logged.
- Metadata is carried forward onto every produced chunk.
- ``chunk_id`` is assigned sequentially across the entire returned list.
"""

from __future__ import annotations

import logging

from ..contracts import Document, DocMeta

logger = logging.getLogger(__name__)

# Cheap, dependency-free token estimate. English/Latin text averages ~4 chars/token;
# good enough to decide *whether* to split. Exact token counts are not needed here —
# the adapter already enforces a real token budget via the HybridChunker tokenizer.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def chunk_documents(
    docs: list[Document],
    *,
    max_tokens: int = 800,
    overlap: int = 100,
) -> list[Document]:
    """Split/group ``docs`` into retrieval-sized chunks.

    Contract: never splits a table; every output chunk carries its metadata and a
    sequential ``chunk_id``. Documents already within ``max_tokens`` pass through
    unchanged (idempotent).
    """
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    out: list[Document] = []
    for doc in docs:
        kind = doc["metadata"].get("kind", "prose")

        if kind == "table":
            # Tables are atomic. If oversized, keep whole and log — splitting rows
            # destroys the spec table that made us preserve it in the first place.
            if _estimate_tokens(doc["text"]) > max_tokens:
                logger.warning(
                    "Table chunk exceeds max_tokens=%d (~%d tokens); kept intact "
                    "(source=%s, section=%s).",
                    max_tokens,
                    _estimate_tokens(doc["text"]),
                    doc["metadata"].get("source", "?"),
                    doc["metadata"].get("section_path", ""),
                )
            out.append(doc)
            continue

        if _estimate_tokens(doc["text"]) <= max_tokens:
            out.append(doc)  # idempotent pass-through
            continue

        out.extend(_split_prose(doc, max_tokens=max_tokens, overlap=overlap))

    # Assign sequential chunk_id across the full returned list.
    for i, doc in enumerate(out):
        doc["metadata"]["chunk_id"] = i
    return out


def _split_prose(doc: Document, *, max_tokens: int, overlap: int) -> list[Document]:
    """Word-aware sliding-window split of an oversized prose Document.

    Splits on whitespace (keeps words intact) using a char budget derived from the
    token estimate, with ``overlap`` tokens of trailing context carried into the next
    window so cross-boundary facts stay retrievable. Metadata is copied to each piece.
    """
    words = doc["text"].split()
    if not words:
        return [doc]

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN

    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            pieces.append(" ".join(cur))
            # Build overlap tail from the end of the current window.
            tail: list[str] = []
            tail_len = 0
            for w in reversed(cur):
                t_add = len(w) + (1 if tail else 0)
                if tail_len + t_add > overlap_chars:
                    break
                tail.insert(0, w)
                tail_len += t_add
            cur = tail
            cur_len = tail_len
            add = len(word) + (1 if cur else 0)
        cur.append(word)
        cur_len += add
    if cur:
        pieces.append(" ".join(cur))

    base_meta: DocMeta = dict(doc["metadata"])  # type: ignore[assignment]
    base_meta.pop("chunk_id", None)  # reassigned by caller
    return [Document(text=p, metadata=dict(base_meta)) for p in pieces]  # type: ignore[arg-type]
