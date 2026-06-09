"""Embedding provider abstraction (T5).

Default provider is local sentence-transformers (``intfloat/multilingual-e5-small``),
in-process and offline. The default is **multilingual** because the corpus scope
includes a pt-BR manual and English-only models retrieve pt-BR badly.

Other providers (ollama/openai/voyage) slot in behind the ``Embedder`` protocol in
MVP 2.0 via the ``_PROVIDERS`` registry below.

Model-specific input prefixes
-----------------------------
Some embedding families require instruction prefixes or retrieval quality collapses:

* **e5** family (``multilingual-e5-*``, ``e5-*``): passages get ``"passage: "``,
  queries get ``"query: "``.
* **bge** family (``bge-m3``, ``bge-*-v1.5``): no prefix (older bge needed a
  "Represent..." instruction; v1.5+ and m3 do not).
* unknown: no prefix; the chosen policy is logged.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense, L2-normalized vectors (cosine-ready)."""

    model: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Batch-encode ``texts`` into vectors of length ``dim``.

        ``is_query`` selects the model-specific prefix policy (query vs passage).
        """
        ...


# --------------------------------------------------------------------------- #
# prefix policy
# --------------------------------------------------------------------------- #
def _prefix_for(model: str, is_query: bool) -> str:
    """Return the input prefix for ``model`` given query/passage intent.

    Family detection is by substring on the model name so new families are a
    one-line addition. Pure, model-free, and unit-testable without downloads.
    """
    name = model.lower()
    if "e5" in name:
        return "query: " if is_query else "passage: "
    if "bge" in name:
        return ""
    return ""


def _apply_prefix(model: str, texts: list[str], is_query: bool) -> list[str]:
    """Prepend the family-appropriate prefix to every text in ``texts``."""
    prefix = _prefix_for(model, is_query)
    if not prefix:
        return list(texts)
    return [prefix + t for t in texts]


def _log_prefix_policy(model: str) -> None:
    """Log which prefix policy applies to ``model`` (called once on load)."""
    name = model.lower()
    if "e5" in name:
        policy = "e5 (query:/passage: prefixes)"
    elif "bge" in name:
        policy = "bge (no prefix)"
    else:
        policy = "unknown family -> no prefix"
    logger.info("embedding prefix policy for %r: %s", model, policy)


# --------------------------------------------------------------------------- #
# local provider (sentence-transformers)
# --------------------------------------------------------------------------- #
class _LocalEmbedder:
    """sentence-transformers backed ``Embedder``.

    The heavy model is lazy-loaded on the first ``encode`` (or first ``dim``
    access). ``dim`` is read from the model, never hardcoded; for e5-small it
    resolves to 384.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._st = None  # type: ignore[var-annotated]
        self._dim: int | None = None
        _log_prefix_policy(model)

    def _load(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading sentence-transformers model %r", self.model)
            self._st = SentenceTransformer(self.model)
            self._dim = int(self._st.get_embedding_dimension())
        return self._st

    @property
    def dim(self) -> int:
        """Embedding dimension, read from the model (loads it on first access)."""
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        st = self._load()
        prefixed = _apply_prefix(self.model, texts, is_query)
        # ALWAYS normalize -> cosine == dot product downstream.
        vecs = st.encode(prefixed, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vecs]


# --------------------------------------------------------------------------- #
# provider registry / factory
# --------------------------------------------------------------------------- #
def _make_local(model: str) -> Embedder:
    return _LocalEmbedder(model)


def _deferred(provider: str):
    def _factory(_model: str) -> Embedder:
        raise NotImplementedError(f"provider {provider!r} deferred to MVP 2.0")

    return _factory


# Extension point for MVP 2.0: replace a value with a real factory
# ``Callable[[str], Embedder]`` (e.g. ``"ollama": _make_ollama``) and the new
# provider is selectable via config/CLI with no other change here.
_PROVIDERS: dict[str, "object"] = {
    "local": _make_local,
    "ollama": _deferred("ollama"),
    "openai": _deferred("openai"),
    "voyage": _deferred("voyage"),
}


def get_embedder(provider: str = "local", model: str = DEFAULT_MODEL) -> Embedder:
    """Factory: return an ``Embedder`` for the configured provider (T5).

    ``provider="local"`` is the only one wired in MVP 1.0; the others raise
    ``NotImplementedError`` until MVP 2.0.
    """
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"unknown embedder provider {provider!r}; "
            f"known: {sorted(_PROVIDERS)}"
        )
    return factory(model)  # type: ignore[operator]
