"""PDF adapter (T3) — Docling extraction + HybridChunker chunking.

Docling owns *both* extraction and chunking here because ``HybridChunker`` operates
on the in-memory ``DoclingDocument`` (the structured tree with headings, tables, and
page provenance), not on serialized text. Splitting that across two stages would mean
re-parsing structure we already have, so the adapter emits one ``Document`` per
HybridChunker chunk with heading-path-prefixed text.

Guards:
- Text-less (scanned) PDFs are rejected — OCR is MVP 1.5, not implemented here.
- If Docling conversion raises, fall back to pymupdf page-text extraction (coarse
  per-page Documents) so a partially-supported PDF still indexes.
"""

from __future__ import annotations

import logging
import os

from ..contracts import Document, DocMeta

logger = logging.getLogger(__name__)

# Below this many extracted characters a "text-native" PDF is almost certainly a
# scan with no text layer (Docling returns near-empty structure). Tuned to be well
# under any real manual page count while catching fully-scanned docs.
_MIN_TEXT_CHARS = 200


class PdfAdapter:
    """Adapter for ``.pdf`` sources. Name ``"pdf"`` per the registry-of-adapters seam."""

    name = "pdf"

    def ingest(self, path: str) -> list[Document]:
        """Parse ``path`` into heading-aware, table-preserving ``Document``s."""
        abspath = os.path.abspath(path)
        if not os.path.isfile(abspath):
            raise FileNotFoundError(abspath)

        try:
            return self._ingest_docling(abspath)
        except RuntimeError:
            # Text-less guard is intentional — never fall through to pymupdf, which
            # would also yield empty text on a scan and silently index garbage.
            raise
        except Exception as exc:  # noqa: BLE001 - any Docling failure -> fallback
            logger.warning(
                "Docling conversion failed for %s (%s: %s); falling back to pymupdf.",
                abspath,
                type(exc).__name__,
                exc,
            )
            return self._ingest_pymupdf(abspath)

    # ----------------------------------------------------------------- docling #
    def _ingest_docling(self, abspath: str) -> list[Document]:
        from docling.chunking import HybridChunker
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = False  # text-native manuals; OCR is MVP 1.5
        opts.do_table_structure = True  # keep torque/spec tables intact

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts,
                )
            }
        )
        doc = converter.convert(abspath).document

        # Text-less guard: a scanned PDF (e.g. XRE300) has no text layer, so Docling
        # extracts ~nothing. Reject loudly rather than emit empty chunks.
        extracted = doc.export_to_text()
        if len(extracted.strip()) < _MIN_TEXT_CHARS:
            raise RuntimeError(
                "PDF appears to be scanned/text-less; OCR path (MVP 1.5) not yet "
                f"implemented: {abspath}"
            )

        chunker = HybridChunker(max_tokens=800)

        documents: list[Document] = []
        for chunk in chunker.chunk(doc):
            # contextualize() prepends the heading/section breadcrumb to the chunk
            # body — load-bearing for retrieval on queries like "spark plug torque".
            text = chunker.contextualize(chunk)
            meta = self._chunk_metadata(chunk, abspath)
            documents.append(Document(text=text, metadata=meta))

        return documents

    @staticmethod
    def _chunk_metadata(chunk, abspath: str) -> DocMeta:
        """Map a Docling chunk's ``meta`` onto our ``DocMeta`` shape.

        Docling 2.99 / docling-core 2.79 chunk meta exposes:
          - ``meta.headings``     : list[str] heading breadcrumb (may be None/empty)
          - ``meta.doc_items``    : list[DocItem]; each has ``.label`` and ``.prov``
          - ``DocItem.prov[]``    : ProvenanceItem with ``.page_no`` (1-based)
          - ``DocItem.label``     : DocItemLabel; ``TABLE`` marks table content
        There is no direct ``section_path`` accessor — it is assembled from headings.
        """
        from docling_core.types.doc.labels import DocItemLabel

        cmeta = chunk.meta

        headings = list(getattr(cmeta, "headings", None) or [])
        section_path = " > ".join(h.strip() for h in headings if h and h.strip())

        doc_items = list(getattr(cmeta, "doc_items", None) or [])

        # kind: "table" if the chunk is built from table doc_items, else "prose".
        is_table = any(
            getattr(it, "label", None) == DocItemLabel.TABLE for it in doc_items
        )

        # page: first page number across the chunk's provenance.
        page: int | None = None
        for it in doc_items:
            for prov in getattr(it, "prov", None) or []:
                pno = getattr(prov, "page_no", None)
                if pno is not None:
                    page = pno if page is None else min(page, pno)

        meta: DocMeta = {
            "source": abspath,
            "kind": "table" if is_table else "prose",
        }
        if section_path:
            meta["section_path"] = section_path
        if page is not None:
            meta["page"] = page
        return meta

    # ----------------------------------------------------------------- pymupdf #
    @staticmethod
    def _ingest_pymupdf(abspath: str) -> list[Document]:
        """Coarse fallback: one prose Document per page, re-chunked downstream.

        These carry ``page`` and ``source`` but no heading structure
        (``section_path=""``); ``chunk_documents`` will size them properly.
        """
        import fitz  # PyMuPDF

        documents: list[Document] = []
        total_chars = 0
        with fitz.open(abspath) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text") or ""
                total_chars += len(text.strip())
                if not text.strip():
                    continue
                meta: DocMeta = {
                    "source": abspath,
                    "page": page_index + 1,  # 1-based
                    "section_path": "",
                    "kind": "prose",
                }
                documents.append(Document(text=text, metadata=meta))

        if total_chars < _MIN_TEXT_CHARS:
            raise RuntimeError(
                "PDF appears to be scanned/text-less; OCR path (MVP 1.5) not yet "
                f"implemented: {abspath}"
            )
        return documents
