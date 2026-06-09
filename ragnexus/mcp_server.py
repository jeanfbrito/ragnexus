"""FastMCP stdio server (T9).

Exposes ``ragnexus_query(corpus, query)`` and ``ragnexus_list()`` as tools and
``ragnexus://doc/{name}/context`` as a resource, mirroring gitnexus naming for
muscle-memory parity. Started via ``ragnexus mcp``.

Design notes
------------
- Tools return JSON-serializable plain ``dict``/``list`` payloads — never raw
  ``Document`` tuples. Each search hit is flattened (text + metadata + score).
- The query embedder is built from the corpus's *recorded* ``embedder_model`` (not
  the global default) so the query-side vector matches the index dim/prefix policy.
- "Corpus not found" is returned as an error dict, never raised — a single bad
  call must not take down a long-lived stdio server.
"""

from __future__ import annotations

from fastmcp import FastMCP

from . import registry
from .core.embedding import get_embedder
from .core.retrieve import search as _search

mcp = FastMCP("ragnexus")


def _hit_to_dict(doc: dict, score: float) -> dict:
    """Flatten a ``(Document, score)`` pair into a JSON-able hit dict."""
    meta = doc.get("metadata") or {}
    return {
        "text": doc.get("text", ""),
        "score": float(score),
        "page": meta.get("page"),
        "section_path": meta.get("section_path"),
        "kind": meta.get("kind"),
        "source": meta.get("source"),
        "corpus": meta.get("corpus"),
        "chunk_id": meta.get("chunk_id"),
    }


@mcp.tool()
def ragnexus_query(corpus: str, query: str, top_k: int = 5) -> list[dict]:
    """Search an indexed document corpus.

    Returns ranked chunks with text, page, section_path, kind, and fused score.
    If the corpus is not registered, returns a single-element list containing an
    error dict rather than raising.
    """
    entry = registry.get_corpus(corpus)
    if entry is None:
        return [{"error": f"corpus {corpus!r} not found", "corpus": corpus}]

    model = entry.get("embedder_model") or None
    provider = "local"
    embedder = get_embedder(provider, model) if model else get_embedder(provider)

    db_path = registry.corpus_db_path(corpus)
    hits = _search(db_path, query, top_k=top_k, embedder=embedder)
    return [_hit_to_dict(doc, score) for doc, score in hits]


@mcp.resource("ragnexus://doc/{name}/context")
def doc_context(name: str) -> str:
    """Overview of a registered corpus: source, #chunks, model, freshness."""
    entry = registry.get_corpus(name)
    if entry is None:
        return f"corpus {name!r} not found"

    lines = [
        f"# Corpus: {entry.get('name', name)}",
        "",
        f"- source: {entry.get('source_path', '-')}",
        f"- chunks: {entry.get('chunks', 0)}",
        f"- vectors: {entry.get('vectors', 0)}",
        f"- embedder_model: {entry.get('embedder_model', '-')}",
        f"- dim: {entry.get('dim', 0)}",
        f"- created_at: {entry.get('created_at', '-')}",
        f"- updated_at: {entry.get('updated_at', '-')}",
        f"- db_path: {entry.get('db_path', '-')}",
    ]
    return "\n".join(lines)


@mcp.tool()
def ragnexus_list() -> list[dict]:
    """List all registered corpora with their core registry fields."""
    return [
        {
            "name": c.get("name", ""),
            "source_path": c.get("source_path", ""),
            "chunks": c.get("chunks", 0),
            "vectors": c.get("vectors", 0),
            "embedder_model": c.get("embedder_model", ""),
            "dim": c.get("dim", 0),
            "updated_at": c.get("updated_at", ""),
        }
        for c in registry.list_corpora()
    ]


def build_server() -> FastMCP:
    """Construct and return the FastMCP server instance (T9)."""
    return mcp


def run() -> None:
    """Run the MCP server over stdio (T9)."""
    mcp.run()


if __name__ == "__main__":
    run()
