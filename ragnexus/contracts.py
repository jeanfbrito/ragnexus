"""Cross-module contracts — the spine every other module builds against.

These types are the stable seam between adapters, core (chunk/embed/store/retrieve),
the CLI, and the MCP server. Keep them minimal and final; downstream builders code
to these shapes.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable


class DocMeta(TypedDict, total=False):
    """Per-document metadata.

    All fields optional (``total=False``) so adapters can populate what they have
    and core stages enrich the rest (e.g. ``chunk_id`` is assigned at chunk time).
    """

    source: str          # absolute file path of the originating source
    corpus: str          # corpus name this document belongs to
    page: int            # 1-based page number
    section_path: str    # heading breadcrumb, e.g. "03 Technical Information > Torque Specs"
    kind: Literal["prose", "table"]
    chunk_id: int        # assigned by the chunking stage


class Document(TypedDict):
    """The unit that flows adapter -> chunk -> embed -> store -> retrieve."""

    text: str
    metadata: DocMeta


@runtime_checkable
class Adapter(Protocol):
    """Source extractor contract.

    An adapter turns a single source file into a list of ``Document``s. Core never
    knows the source type — a new source kind is a new adapter, nothing else changes.
    """

    name: str

    def ingest(self, path: str) -> list[Document]:
        """Parse ``path`` and return extracted documents (prose + table blocks)."""
        ...
