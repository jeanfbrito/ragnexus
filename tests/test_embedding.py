"""Embedding layer tests (T5).

Two tiers:
* fast: prefix-policy helper + factory wiring, no model download.
* slow: real sentence-transformers e5-small encode (skipped if the model
  cannot be loaded, e.g. offline / not yet downloaded).
"""

from __future__ import annotations

import importlib

import pytest

from ragnexus.core.embedding import (
    DEFAULT_MODEL,
    Embedder,
    _apply_prefix,
    _prefix_for,
    get_embedder,
)


# --------------------------------------------------------------------------- #
# fast: prefix policy (no model load)
# --------------------------------------------------------------------------- #
def test_prefix_e5_query_and_passage():
    assert _prefix_for("intfloat/multilingual-e5-small", is_query=True) == "query: "
    assert _prefix_for("intfloat/multilingual-e5-small", is_query=False) == "passage: "
    # case-insensitive family detection
    assert _prefix_for("E5-LARGE", is_query=True) == "query: "


def test_prefix_bge_none():
    assert _prefix_for("BAAI/bge-small-en-v1.5", is_query=True) == ""
    assert _prefix_for("BAAI/bge-m3", is_query=False) == ""


def test_prefix_unknown_none():
    assert _prefix_for("some/random-model", is_query=True) == ""
    assert _prefix_for("some/random-model", is_query=False) == ""


def test_apply_prefix_e5():
    out_q = _apply_prefix("intfloat/multilingual-e5-small", ["a", "b"], is_query=True)
    assert out_q == ["query: a", "query: b"]
    out_p = _apply_prefix("intfloat/multilingual-e5-small", ["a"], is_query=False)
    assert out_p == ["passage: a"]


def test_apply_prefix_bge_unchanged():
    src = ["x", "y"]
    out = _apply_prefix("BAAI/bge-m3", src, is_query=True)
    assert out == src
    assert out is not src  # returns a copy, never aliases input


# --------------------------------------------------------------------------- #
# fast: factory / provider registry
# --------------------------------------------------------------------------- #
def test_get_embedder_default_model_is_multilingual():
    enc = get_embedder()
    assert enc.model == "intfloat/multilingual-e5-small"
    assert DEFAULT_MODEL == "intfloat/multilingual-e5-small"
    assert isinstance(enc, Embedder)


@pytest.mark.parametrize("provider", ["ollama", "openai", "voyage"])
def test_get_embedder_deferred_providers_raise(provider):
    with pytest.raises(NotImplementedError, match="MVP 2.0"):
        get_embedder(provider=provider)


def test_get_embedder_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown embedder provider"):
        get_embedder(provider="bogus")


# --------------------------------------------------------------------------- #
# slow: real model encode (skip if unavailable)
# --------------------------------------------------------------------------- #
def _model_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


slow_model = pytest.mark.skipif(
    not _model_available(),
    reason="sentence-transformers not installed; slow real-model test skipped",
)


@pytest.mark.slow
@slow_model
def test_real_e5_encode_dim_and_prefix(monkeypatch):
    enc = get_embedder()

    # Capture what actually reaches the underlying model to prove prefixing.
    seen: dict[str, list[str]] = {}

    real_load = enc._load  # type: ignore[attr-defined]

    def patched_load():
        st = real_load()
        orig_encode = st.encode

        def spy(texts, *a, **kw):
            seen["texts"] = list(texts)
            return orig_encode(texts, *a, **kw)

        monkeypatch.setattr(st, "encode", spy, raising=True)
        return st

    monkeypatch.setattr(enc, "_load", patched_load)

    try:
        q = enc.encode(["torque da vela de ignição"], is_query=True)
    except Exception as e:  # network / download failure -> skip, not fail
        pytest.skip(f"model could not be loaded/downloaded: {e}")

    assert enc.dim == 384
    assert len(q) == 1 and len(q[0]) == 384
    assert all(isinstance(x, float) for x in q[0])
    assert seen["texts"] == ["query: torque da vela de ignição"]

    p = enc.encode(["Spark plug torque is 15 N·m for M10x1."], is_query=False)
    assert seen["texts"][0].startswith("passage: ")
    assert len(p[0]) == 384

    # cosine similarity finite (vectors are normalized -> dot == cosine)
    cos = sum(a * b for a, b in zip(q[0], p[0]))
    assert -1.0001 <= cos <= 1.0001
    import math

    assert math.isfinite(cos)
