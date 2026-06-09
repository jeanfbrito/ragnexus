"""Corpus registry — ``~/.ragnexus/registry.json``.

Single source of truth for corpus lifecycle and freshness. Per-corpus settings
(embedder model, dim, counts, source hash) live here, not in ``config.toml``.

On-disk schema::

    {
      "corpora": {
        "<name>": {
          "name": "...",
          "source_path": "...",
          "file_hash": "sha256",
          "db_path": "~/.ragnexus/<name>.db",
          "chunks": 0,
          "vectors": 0,
          "embedder_model": "...",
          "dim": 0,
          "created_at": "ISO8601",
          "updated_at": "ISO8601"
        }
      }
    }

All writes are atomic (write tmp + ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import RAGNEXUS_HOME

REGISTRY_PATH: Path = RAGNEXUS_HOME / "registry.json"


def _registry_path(path: Path | None) -> Path:
    """Resolve the registry path at call time.

    Defaults are looked up from the module global on each call (not bound at def
    time) so that monkeypatching ``REGISTRY_PATH`` / ``RAGNEXUS_HOME`` works in tests
    and so a relocated home is always honoured.
    """
    return path if path is not None else REGISTRY_PATH


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    """UTC timestamp, ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, streamed in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slugify(filename: str) -> str:
    """Derive a default corpus name from a filename.

    Drops the extension, lowercases, replaces non-alphanumerics with hyphens, and
    collapses/trims runs of hyphens. ``"Ibex450 ... SM-WEB.pdf"`` -> ``"ibex450-sm-web"``.
    """
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "corpus"


def corpus_db_path(name: str, path: Path | None = None) -> Path:
    """Resolve the on-disk SQLite path for a corpus.

    Uses the registered ``db_path`` if the corpus exists, else the default
    ``~/.ragnexus/<name>.db``. Returned path is expanduser'd.
    """
    entry = get_corpus(name, path)
    if entry and entry.get("db_path"):
        return Path(entry["db_path"]).expanduser()
    return (RAGNEXUS_HOME / f"{name}.db").expanduser()


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def _load(path: Path | None = None) -> dict:
    path = _registry_path(path)
    if not path.exists():
        return {"corpora": {}}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("corpora", {})
    return data


def _save(data: dict, path: Path | None = None) -> None:
    """Atomically persist the registry (write tmp in same dir + replace)."""
    path = _registry_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".registry-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def add_corpus(entry: dict, path: Path | None = None) -> dict:
    """Register a corpus.

    ``entry`` must contain at least ``name``. Missing schema fields are filled with
    defaults; ``created_at``/``updated_at`` are set if absent. Returns the stored entry.
    """
    name = entry.get("name")
    if not name:
        raise ValueError("corpus entry requires a 'name'")

    now = _now()
    stored: dict = {
        "name": name,
        "source_path": entry.get("source_path", ""),
        "file_hash": entry.get("file_hash", ""),
        "db_path": entry.get("db_path", str(RAGNEXUS_HOME / f"{name}.db")),
        "chunks": entry.get("chunks", 0),
        "vectors": entry.get("vectors", 0),
        "embedder_model": entry.get("embedder_model", ""),
        "dim": entry.get("dim", 0),
        "created_at": entry.get("created_at", now),
        "updated_at": entry.get("updated_at", now),
    }
    data = _load(path)
    data["corpora"][name] = stored
    _save(data, path)
    return stored


def get_corpus(name: str, path: Path | None = None) -> dict | None:
    """Return the corpus entry, or ``None`` if not registered."""
    return _load(path)["corpora"].get(name)


def list_corpora(path: Path | None = None) -> list[dict]:
    """Return all corpus entries as a list (registration order preserved)."""
    return list(_load(path)["corpora"].values())


def remove_corpus(name: str, path: Path | None = None) -> bool:
    """Drop a corpus from the registry. Returns ``True`` if it existed.

    Does NOT delete the corpus ``.db`` file — that is the caller's (``clean``) job.
    """
    data = _load(path)
    if name in data["corpora"]:
        del data["corpora"][name]
        _save(data, path)
        return True
    return False


def update_corpus(name: str, path: Path | None = None, **fields) -> dict:
    """Patch fields on an existing corpus and bump ``updated_at``.

    Raises ``KeyError`` if the corpus is not registered. ``created_at`` cannot be
    overwritten through this path.
    """
    data = _load(path)
    corpora = data["corpora"]
    if name not in corpora:
        raise KeyError(name)
    entry = corpora[name]
    fields.pop("created_at", None)
    entry.update(fields)
    entry["updated_at"] = _now()
    _save(data, path)
    return entry
