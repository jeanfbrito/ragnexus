"""Foundation layer tests (T1/T2): contracts, config, registry.

These exercise only the fully-implemented foundation — no adapter/embed/store deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragnexus import config as config_mod
from ragnexus import registry as registry_mod
from ragnexus.config import Config, ensure_config, load_config
from ragnexus.contracts import Adapter, Document
from ragnexus.registry import (
    add_corpus,
    corpus_db_path,
    file_sha256,
    get_corpus,
    list_corpora,
    remove_corpus,
    slugify,
    update_corpus,
)


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #
def test_document_is_a_dict():
    doc: Document = {"text": "hello", "metadata": {"source": "/x.pdf", "page": 1}}
    assert doc["text"] == "hello"
    assert doc["metadata"]["page"] == 1


def test_adapter_protocol_runtime_checkable():
    class Dummy:
        name = "dummy"

        def ingest(self, path: str) -> list[Document]:
            return []

    assert isinstance(Dummy(), Adapter)
    assert not isinstance(object(), Adapter)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_load_config_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert isinstance(cfg, Config)
    assert cfg.embedder_provider == "local"
    assert cfg.embedder_model == "intfloat/multilingual-e5-small"
    assert cfg.llm_provider == "none"
    assert isinstance(cfg.store_dir, Path)
    assert cfg.store_dir.is_absolute()


def test_ensure_config_creates_and_roundtrips(tmp_path):
    path = tmp_path / "config.toml"
    cfg = ensure_config(path)
    assert path.exists()
    again = load_config(path)
    assert again == cfg


def test_config_partial_override_merges_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[embedder]\nmodel = "custom/model"\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.embedder_model == "custom/model"
    assert cfg.embedder_provider == "local"  # default retained
    assert cfg.llm_provider == "none"


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@pytest.fixture
def reg(tmp_path, monkeypatch):
    """Redirect registry + config home to a temp path."""
    home = tmp_path
    monkeypatch.setattr(config_mod, "RAGNEXUS_HOME", home)
    monkeypatch.setattr(registry_mod, "RAGNEXUS_HOME", home)
    monkeypatch.setattr(registry_mod, "REGISTRY_PATH", home / "registry.json")
    return home / "registry.json"


def test_slugify():
    assert slugify("Ibex450 ... SM-WEB.pdf") == "ibex450-sm-web"
    assert slugify("My File.PDF") == "my-file"
    assert slugify("___.pdf") == "corpus"


def test_file_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    # sha256("abc")
    assert file_sha256(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_registry_crud_roundtrip(reg):
    assert list_corpora(reg) == []

    stored = add_corpus({"name": "ibex450", "source_path": "/m.pdf", "chunks": 10}, reg)
    assert stored["name"] == "ibex450"
    assert stored["chunks"] == 10
    assert stored["created_at"]
    assert stored["updated_at"]
    assert stored["db_path"].endswith("ibex450.db")

    got = get_corpus("ibex450", reg)
    assert got == stored
    assert get_corpus("missing", reg) is None

    assert [c["name"] for c in list_corpora(reg)] == ["ibex450"]


def test_update_corpus_bumps_timestamp(reg):
    add_corpus({"name": "c1"}, reg)
    created = get_corpus("c1", reg)["created_at"]
    updated = update_corpus("c1", reg, chunks=42, dim=384)
    assert updated["chunks"] == 42
    assert updated["dim"] == 384
    assert updated["created_at"] == created  # immutable
    with pytest.raises(KeyError):
        update_corpus("ghost", reg, chunks=1)


def test_remove_corpus(reg):
    add_corpus({"name": "c1"}, reg)
    assert remove_corpus("c1", reg) is True
    assert remove_corpus("c1", reg) is False
    assert get_corpus("c1", reg) is None


def test_corpus_db_path_default_and_registered(reg):
    # unregistered -> default under home
    p = corpus_db_path("ghost")
    assert p.name == "ghost.db"
    # registered with custom db_path
    add_corpus({"name": "c1", "db_path": "~/.ragnexus/c1.db"}, reg)
    p2 = corpus_db_path("c1")
    assert p2.name == "c1.db"
    assert p2.is_absolute()
