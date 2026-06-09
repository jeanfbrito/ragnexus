"""Adapter base — re-exports the ``Adapter`` protocol from contracts.

The contract itself lives in :mod:`ragnexus.contracts` so it has no dependency on the
adapters package. This module is the import site adapters extend, plus a tiny helper
for the registry-of-adapters pattern used by the CLI when more sources land.
"""

from __future__ import annotations

import os

from ..contracts import Adapter, Document, DocMeta

__all__ = ["Adapter", "Document", "DocMeta", "get_adapter"]


def get_adapter(path: str) -> Adapter:
    """Return the adapter that handles ``path`` (dispatch by extension).

    MVP only registers the PDF adapter. New source kinds add a branch here.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from .pdf import PdfAdapter

        return PdfAdapter()
    raise ValueError(f"no adapter for {ext}")
