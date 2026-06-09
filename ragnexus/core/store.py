"""Per-corpus index store: sqlite-vec + FTS5 (T6).

One single-file SQLite DB per corpus (``~/.ragnexus/<name>.db``) holding a sqlite-vec
vector table mirrored by an FTS5 keyword index for hybrid retrieval.

The two tables are joined on ``chunk_id``:

* ``vec_chunks`` is a ``vec0`` virtual table — the embedding plus every metadata
  field carried as an *auxiliary* column (the ``+col`` syntax) so a KNN row can be
  reconstructed into a full :class:`~ragnexus.contracts.Document` without a second
  lookup.
* ``fts_chunks`` is an FTS5 virtual table whose ``rowid`` is set to the same
  ``chunk_id``. BM25 results therefore key straight back to the vector row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec
from sqlite_vec import serialize_float32

from ..contracts import Document


class Store:
    """Single-corpus hybrid index (vector + keyword)."""

    def __init__(self, db_path: Path, *, dim: int) -> None:
        self.db_path = Path(db_path)
        self.dim = dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(self.db_path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self._create_tables()

    def _create_tables(self) -> None:
        # Vector table: embedding + metadata aux columns (the leading `+` makes a
        # column auxiliary — stored alongside but not part of the ANN index).
        self.db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{self.dim}],
                +chunk_id INTEGER,
                +text TEXT,
                +source TEXT,
                +corpus TEXT,
                +page INTEGER,
                +section_path TEXT,
                +kind TEXT
            )
            """
        )
        # Keyword mirror: own-content FTS5; rowid is set to chunk_id at insert time
        # so the two tables join on chunk_id.
        self.db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(text)"
        )
        self.db.commit()

    def add(self, docs: list[Document], vectors: list[list[float]]) -> int:
        """Insert chunks into both ``vec_chunks`` and ``fts_chunks``.

        Uses ``metadata["chunk_id"]`` as the shared rowid for both tables. If a
        document lacks a ``chunk_id`` it is assigned sequentially, continuing past
        the current max id already in the store. Returns the number of rows added.
        """
        if len(docs) != len(vectors):
            raise ValueError(
                f"docs/vectors length mismatch: {len(docs)} != {len(vectors)}"
            )
        if not docs:
            return 0

        next_id = self._next_chunk_id()
        vec_rows: list[tuple] = []
        fts_rows: list[tuple[int, str]] = []

        for doc, vec in zip(docs, vectors):
            meta = doc.get("metadata") or {}
            chunk_id = meta.get("chunk_id")
            if chunk_id is None:
                chunk_id = next_id
                next_id += 1
            else:
                chunk_id = int(chunk_id)
                next_id = max(next_id, chunk_id + 1)

            text = doc["text"]
            emb = serialize_float32(np.asarray(vec, dtype=np.float32))
            vec_rows.append(
                (
                    emb,
                    chunk_id,
                    text,
                    meta.get("source"),
                    meta.get("corpus"),
                    meta.get("page"),
                    meta.get("section_path"),
                    meta.get("kind"),
                )
            )
            fts_rows.append((chunk_id, text))

        with self.db:  # transaction
            self.db.executemany(
                """
                INSERT INTO vec_chunks(
                    embedding, chunk_id, text, source, corpus, page, section_path, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                vec_rows,
            )
            self.db.executemany(
                "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)",
                fts_rows,
            )

        return len(docs)

    def _next_chunk_id(self) -> int:
        row = self.db.execute(
            "SELECT COALESCE(MAX(chunk_id), -1) FROM vec_chunks"
        ).fetchone()
        return int(row[0]) + 1

    def count(self) -> int:
        """Number of chunks in the vector table."""
        return int(self.db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0])

    def close(self) -> None:
        self.db.close()
