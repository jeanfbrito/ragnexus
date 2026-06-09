"""Global configuration — ``~/.ragnexus/config.toml``.

Read with stdlib ``tomllib``, written with ``tomli-w``. Per-corpus overrides are NOT
stored here; the registry is the source of truth per corpus (see ``registry.py``).
This module only holds the global defaults and the resolved ``store_dir``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from ragnexus.core.embedding import DEFAULT_MODEL as _DEFAULT_EMBEDDER_MODEL

# Resolved at import: ~/.ragnexus/
RAGNEXUS_HOME: Path = Path("~/.ragnexus").expanduser()
CONFIG_PATH: Path = RAGNEXUS_HOME / "config.toml"

# Programmatic defaults — mirror the documented config.toml exactly.
DEFAULT_CONFIG: dict = {
    "embedder": {
        "provider": "local",
        "model": _DEFAULT_EMBEDDER_MODEL,
    },
    "llm": {
        "provider": "none",
    },
    "store": {
        "dir": "~/.ragnexus",
    },
}


@dataclass(frozen=True)
class Config:
    """Resolved global configuration."""

    embedder_provider: str
    embedder_model: str
    llm_provider: str
    store_dir: Path  # expanduser'd, absolute


def _merge_defaults(data: dict) -> dict:
    """Overlay loaded ``data`` onto ``DEFAULT_CONFIG`` (one level of nesting)."""
    merged: dict = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    for section, values in data.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
        else:
            merged[section] = values
    return merged


def _to_config(data: dict) -> Config:
    embedder = data.get("embedder", {})
    llm = data.get("llm", {})
    store = data.get("store", {})
    return Config(
        embedder_provider=embedder.get("provider", DEFAULT_CONFIG["embedder"]["provider"]),
        embedder_model=embedder.get("model", DEFAULT_CONFIG["embedder"]["model"]),
        llm_provider=llm.get("provider", DEFAULT_CONFIG["llm"]["provider"]),
        store_dir=Path(store.get("dir", DEFAULT_CONFIG["store"]["dir"])).expanduser(),
    )


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config from ``path``, falling back to defaults for anything missing.

    Does not create the file — call :func:`ensure_config` for that. A missing file
    yields the full default config.
    """
    if path.exists():
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    else:
        raw = {}
    return _to_config(_merge_defaults(raw))


def ensure_config(path: Path = CONFIG_PATH) -> Config:
    """Create ``path`` with defaults if missing, then return the loaded config.

    Also ensures the parent ``~/.ragnexus/`` directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as fh:
            tomli_w.dump(DEFAULT_CONFIG, fh)
    return load_config(path)
