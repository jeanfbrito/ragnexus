"""Store + hybrid retrieve tests (T6).

No heavy model: synthetic fixed-dim vectors and an injected fake embedder so the
suite runs fully offline (no e5 download).
"""

from __future__ import annotations

import numpy as np
import pytest

from ragnexus.core import retrieve as retrieve_mod
from ragnexus.core.retrieve import _fts_query, _rrf_fuse, search
from ragnexus.core.store import Store

DIM = 8


def _doc(chunk_id: int, text: str, **meta) -> dict:
    md = {"chunk_id": chunk_id, "source": "/m.pdf", "corpus": "test"}
    md.update(meta)
    return {"text": text, "metadata": md}


def _unit(idx: int) -> list[float]:
    """One-hot vector of length DIM, normalized (already unit)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[idx % DIM] = 1.0
    return v.tolist()


class FakeEmbedder:
    """Returns a fixed vector regardless of input; records is_query usage."""

    model = "fake"
    dim = DIM

    def __init__(self, vec: list[float]):
        self._vec = vec
        self.last_is_query = None

    def encode(self, texts, is_query: bool = False):
        self.last_is_query = is_query
        return [list(self._vec) for _ in texts]


# --------------------------------------------------------------------------- #
# Store round-trip
# --------------------------------------------------------------------------- #
def test_store_roundtrip_and_count(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    docs = [_doc(i, f"chunk number {i}", page=i + 1, kind="prose") for i in range(5)]
    vectors = [_unit(i) for i in range(5)]
    added = store.add(docs, vectors)
    assert added == 5
    assert store.count() == 5
    store.close()


def test_store_persists_across_reopen(tmp_path):
    path = tmp_path / "c.db"
    store = Store(path, dim=DIM)
    store.add([_doc(0, "hello world")], [_unit(0)])
    store.close()

    again = Store(path, dim=DIM)
    assert again.count() == 1
    again.close()


def test_store_assigns_sequential_chunk_id_when_missing(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    docs = [{"text": "a", "metadata": {}}, {"text": "b", "metadata": {}}]
    store.add(docs, [_unit(0), _unit(1)])
    ids = sorted(
        r[0] for r in store.db.execute("SELECT chunk_id FROM vec_chunks").fetchall()
    )
    assert ids == [0, 1]
    store.close()


def test_store_length_mismatch_raises(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    with pytest.raises(ValueError):
        store.add([_doc(0, "x")], [])
    store.close()


# --------------------------------------------------------------------------- #
# KNN-only: nearest vector wins
# --------------------------------------------------------------------------- #
def test_knn_returns_nearest(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    docs = [_doc(i, f"text {i}") for i in range(5)]
    store.add(docs, [_unit(i) for i in range(5)])
    store.close()

    # Query is exactly the one-hot at index 3 -> chunk_id 3 must rank first.
    fake = FakeEmbedder(_unit(3))
    results = search(tmp_path / "c.db", "irrelevant", top_k=5, embedder=fake)
    assert results, "expected hybrid results"
    top_doc, _ = results[0]
    assert top_doc["metadata"]["chunk_id"] == 3


# --------------------------------------------------------------------------- #
# RRF fusion ordering on a crafted set
# --------------------------------------------------------------------------- #
def test_rrf_fuse_ordering():
    # id 2 appears high in both lists -> should beat ids unique to one list.
    vec_ranked = [1, 2, 3]
    kw_ranked = [4, 2, 5]
    fused = _rrf_fuse(vec_ranked, kw_ranked)
    order = [cid for cid, _ in fused]
    assert order[0] == 2  # ranked in both, highest combined score
    scores = dict(fused)
    # id 2: 1/(60+2) + 1/(60+2); id 1: 1/(60+1)
    assert scores[2] == pytest.approx(2 / 62)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[4] == pytest.approx(1 / 61)
    assert scores[2] > scores[1]


def test_rrf_fuse_empty():
    assert _rrf_fuse([], []) == []


# --------------------------------------------------------------------------- #
# FTS5 escaping / punctuation-heavy query
# --------------------------------------------------------------------------- #
def test_fts_query_escapes_special_chars():
    q = _fts_query("spark plug (M10)")
    assert q == '"spark" OR "plug" OR "M10"'


def test_fts_query_empty_is_safe():
    assert _fts_query("()-*:") == '""'
    assert _fts_query("") == '""'


def test_hybrid_search_with_punctuation_query_does_not_raise(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    docs = [
        _doc(0, "spark plug torque M10 x 1 15 N m", kind="table"),
        _doc(1, "oil filter replacement procedure", kind="prose"),
        _doc(2, "brake fluid capacity specification", kind="prose"),
    ]
    store.add(docs, [_unit(i) for i in range(3)])
    store.close()

    fake = FakeEmbedder(_unit(0))
    # punctuation-heavy query must not raise an FTS5 syntax error
    results = search(tmp_path / "c.db", "spark plug (M10) @ 15 N·m?", top_k=3, embedder=fake)
    assert results
    # Document reconstruction carries metadata back
    top_doc, score = results[0]
    assert top_doc["text"]
    assert top_doc["metadata"]["chunk_id"] == 0
    assert top_doc["metadata"]["kind"] == "table"
    assert score > 0


def test_search_uses_query_prefix(tmp_path):
    store = Store(tmp_path / "c.db", dim=DIM)
    store.add([_doc(0, "hello")], [_unit(0)])
    store.close()

    fake = FakeEmbedder(_unit(0))
    search(tmp_path / "c.db", "hello", top_k=1, embedder=fake)
    assert fake.last_is_query is True  # e5 query-prefix correctness


def test_default_embedder_built_lazily(tmp_path, monkeypatch):
    store = Store(tmp_path / "c.db", dim=DIM)
    store.add([_doc(0, "hello")], [_unit(0)])
    store.close()

    fake = FakeEmbedder(_unit(0))
    monkeypatch.setattr(
        "ragnexus.core.embedding.get_embedder", lambda *a, **k: fake, raising=False
    )
    results = search(tmp_path / "c.db", "hello", top_k=1)  # no embedder injected
    assert results
    assert fake.last_is_query is True
