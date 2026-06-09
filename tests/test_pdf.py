"""PDF adapter + generic chunking tests (T3/T4).

Fast tests cover dispatch and the idempotent splitter without touching Docling.
The real-PDF test is marked ``slow`` — first run downloads Docling models (minutes)
and converts the 288-page Ibex450 manual.
"""

from __future__ import annotations

import os

import pytest

from ragnexus.adapters.base import get_adapter
from ragnexus.adapters.pdf import PdfAdapter
from ragnexus.contracts import Document
from ragnexus.core.chunking import chunk_documents

IBEX450 = "/Users/jean/Downloads/Ibex450 (6AQ2-00XW02-13) SM-WEB.pdf"


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_get_adapter_pdf():
    a = get_adapter("/some/manual.PDF")
    assert isinstance(a, PdfAdapter)
    assert a.name == "pdf"


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="no adapter for .md"):
        get_adapter("/some/notes.md")


# --------------------------------------------------------------------------- #
# chunk_documents — idempotency & sizing
# --------------------------------------------------------------------------- #
def _doc(text: str, **meta) -> Document:
    return {"text": text, "metadata": {"source": "/x.pdf", **meta}}


def test_chunk_documents_passthrough_idempotent():
    docs = [
        _doc("short prose one", kind="prose", section_path="A > B", page=1),
        _doc("short prose two", kind="prose", section_path="A > C", page=2),
    ]
    out = chunk_documents(docs, max_tokens=800, overlap=100)
    assert len(out) == 2
    # text untouched
    assert [d["text"] for d in out] == ["short prose one", "short prose two"]
    # metadata carried forward + sequential chunk_id
    assert out[0]["metadata"]["section_path"] == "A > B"
    assert out[0]["metadata"]["page"] == 1
    assert [d["metadata"]["chunk_id"] for d in out] == [0, 1]


def test_chunk_documents_idempotent_second_pass():
    docs = [_doc("short prose", kind="prose", section_path="A")]
    once = chunk_documents(docs, max_tokens=800)
    twice = chunk_documents(once, max_tokens=800)
    assert len(once) == len(twice) == 1
    assert once[0]["text"] == twice[0]["text"]
    assert twice[0]["metadata"]["chunk_id"] == 0


def test_chunk_documents_splits_oversized_prose():
    # ~4 chars/token => 800 tokens ~= 3200 chars. Build ~8000 chars of words.
    big = " ".join(["word"] * 1600)  # ~8000 chars => ~2000 tokens
    out = chunk_documents([_doc(big, kind="prose", section_path="Big")], max_tokens=800, overlap=100)
    assert len(out) > 1
    # every piece within budget (cheap estimate: len/4)
    for d in out:
        assert (len(d["text"]) + 3) // 4 <= 800
        assert d["metadata"]["section_path"] == "Big"  # metadata carried
    # sequential ids
    assert [d["metadata"]["chunk_id"] for d in out] == list(range(len(out)))


def test_chunk_documents_overlap_present():
    big = " ".join(f"w{i}" for i in range(2000))
    out = chunk_documents([_doc(big, kind="prose")], max_tokens=400, overlap=100)
    assert len(out) > 1
    # consecutive windows should share tokens: the tail of chunk N is re-emitted as
    # the head of chunk N+1 (overlap window). Compare the full tail vs full head.
    first_tail = set(out[0]["text"].split()[-100:])
    second_head = set(out[1]["text"].split()[:100])
    assert first_tail & second_head, "expected overlap between consecutive chunks"


def test_chunk_documents_never_splits_table():
    big_table = " ".join(["cell"] * 4000)  # ~20000 chars, way over budget
    out = chunk_documents(
        [_doc(big_table, kind="table", section_path="Torque Specs")],
        max_tokens=800,
        overlap=100,
    )
    assert len(out) == 1  # kept whole despite being oversized
    assert out[0]["metadata"]["kind"] == "table"
    assert out[0]["text"] == big_table


def test_chunk_documents_validates_overlap():
    with pytest.raises(ValueError):
        chunk_documents([], max_tokens=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_documents([], max_tokens=100, overlap=-1)


def test_ingest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        PdfAdapter().ingest("/nope/does-not-exist.pdf")


# --------------------------------------------------------------------------- #
# real PDF — slow (Docling model download on first run)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(IBEX450), reason="Ibex450 manual not present")
def test_ingest_ibex450_real():
    docs = PdfAdapter().ingest(IBEX450)
    assert len(docs) > 0
    for d in docs:
        assert d["text"].strip()
        assert d["metadata"]["source"] == os.path.abspath(IBEX450)
        assert d["metadata"]["kind"] in ("prose", "table")

    # At least one table chunk somewhere in the manual.
    tables = [d for d in docs if d["metadata"]["kind"] == "table"]
    assert tables, "expected at least one table chunk"

    # Spot-check: a spark-plug torque chunk should be a table under a non-empty
    # section path, with the heading context inlined in the text.
    spark = [
        d
        for d in docs
        if "spark plug" in d["text"].lower() and "torque" in d["text"].lower()
    ]
    assert spark, "no spark-plug torque chunk found"
    table_spark = [d for d in spark if d["metadata"]["kind"] == "table"]
    assert table_spark, "spark-plug torque chunk was not classified as a table"
    sample = table_spark[0]
    assert sample["metadata"].get("section_path"), "spark-plug chunk missing section_path"
    assert sample["metadata"].get("page"), "spark-plug chunk missing page"
    # contextualize() should have inlined the heading breadcrumb into the text.
    head = sample["metadata"]["section_path"].split(" > ")[-1]
    assert head.lower() in sample["text"].lower()
