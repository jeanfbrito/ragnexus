"""ragnexus CLI — Typer app.

Wires the end-to-end pipeline (T7-T9): ``analyze`` (adapter -> chunk -> embed ->
store -> register), ``search`` (hybrid retrieval), ``status``/``list``/``clean``
(introspection + teardown), and ``mcp`` (FastMCP stdio server).
"""

from __future__ import annotations

import os

import typer

from . import __version__
from .config import ensure_config
from .core.chunking import chunk_documents
from .core.embedding import get_embedder
from .core.retrieve import search as retrieve_search
from .core.store import Store
from .registry import (
    add_corpus,
    corpus_db_path,
    file_sha256,
    get_corpus,
    list_corpora,
    remove_corpus,
    slugify,
)

app = typer.Typer(
    name="ragnexus",
    help="Embed-anything knowledge engine — index a source once, query it from any agent.",
    no_args_is_help=True,
    add_completion=False,
)


def _fmt_size(num_bytes: int) -> str:
    """Human-readable byte size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to the source file to index."),
    name: str = typer.Option(None, "--name", help="Corpus name (default: slug of filename)."),
    embedder: str = typer.Option(
        None, "--embedder", help="Embedder model override (default: config embedder model)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing corpus of the same name (drops its index)."
    ),
) -> None:
    """Parse, chunk, embed, store and register a source (T3-T7)."""
    corpus_name = name or slugify(os.path.basename(path))

    existing = get_corpus(corpus_name)
    if existing is not None and not force:
        typer.echo(
            f"corpus {corpus_name!r} already registered. "
            "Re-run with --force to overwrite it (this drops the existing index)."
        )
        raise typer.Exit(code=1)
    if existing is not None and force:
        old_db = corpus_db_path(corpus_name)
        if old_db.exists():
            old_db.unlink()
        remove_corpus(corpus_name)
        typer.echo(f"--force: dropped existing corpus {corpus_name!r}")

    config = ensure_config()
    model = embedder or config.embedder_model
    provider = config.embedder_provider

    # 1. extract --------------------------------------------------------------
    typer.echo(f"[extract] {path}")
    from .adapters.base import get_adapter

    try:
        docs = get_adapter(path).ingest(path)
    except RuntimeError as exc:
        typer.echo(
            f"error: could not extract text from {path!r} ({exc}). "
            "This looks like a scanned / text-less PDF — OCR support arrives in "
            "MVP 1.5. For now, supply a text-native PDF.",
            err=True,
        )
        raise typer.Exit(code=1)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[extract] {len(docs)} document blocks")

    # 2. chunk ----------------------------------------------------------------
    chunks = chunk_documents(docs)
    typer.echo(f"[chunk] {len(chunks)} chunks")

    # 3. stamp corpus name into every chunk's metadata ------------------------
    for c in chunks:
        meta = c.setdefault("metadata", {})
        meta["corpus"] = corpus_name

    # 4. embed (slow) ---------------------------------------------------------
    typer.echo(f"[embed] encoding {len(chunks)} chunks with {model!r} (this is the slow part)...")
    emb = get_embedder(provider, model)
    vectors = emb.encode([c["text"] for c in chunks], is_query=False)
    typer.echo(f"[embed] {len(vectors)} vectors, dim={emb.dim}")

    # 5. store ----------------------------------------------------------------
    db = corpus_db_path(corpus_name)
    store = Store(db, dim=emb.dim)
    n = store.add(chunks, vectors)
    store.close()
    typer.echo(f"[store] {n} chunks written to {db}")

    # 6. register -------------------------------------------------------------
    add_corpus(
        {
            "name": corpus_name,
            "source_path": os.path.abspath(path),
            "file_hash": file_sha256(path),
            "db_path": str(db),
            "chunks": len(chunks),
            "vectors": n,
            "embedder_model": emb.model,
            "dim": emb.dim,
        }
    )
    typer.echo(
        f"[registered] {corpus_name}: {len(chunks)} chunks, {n} vectors, "
        f"model={emb.model}, dim={emb.dim}"
    )


@app.command(name="list")
def list_cmd() -> None:
    """List registered corpora."""
    corpora = list_corpora()
    if not corpora:
        typer.echo("no corpora registered")
        return

    rows = [
        (
            c.get("name", ""),
            str(c.get("chunks", 0)),
            c.get("embedder_model", "") or "-",
            c.get("updated_at", "") or "-",
        )
        for c in corpora
    ]
    headers = ("NAME", "CHUNKS", "MODEL", "UPDATED")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    typer.echo(fmt.format(*headers))
    for r in rows:
        typer.echo(fmt.format(*r))


@app.command()
def status(name: str = typer.Argument(..., help="Corpus name.")) -> None:
    """Show corpus details: chunk/vector counts, model, freshness (T8)."""
    entry = get_corpus(name)
    if entry is None:
        typer.echo(f"error: corpus {name!r} not found", err=True)
        raise typer.Exit(code=1)

    db = corpus_db_path(name)
    db_size = _fmt_size(db.stat().st_size) if db.exists() else "missing"

    typer.echo(f"corpus: {entry.get('name', name)}")
    typer.echo(f"  chunks:        {entry.get('chunks', 0)}")
    typer.echo(f"  vectors:       {entry.get('vectors', 0)}")
    typer.echo(f"  embedder:      {entry.get('embedder_model', '-')}")
    typer.echo(f"  dim:           {entry.get('dim', 0)}")
    typer.echo(f"  created_at:    {entry.get('created_at', '-')}")
    typer.echo(f"  updated_at:    {entry.get('updated_at', '-')}")
    typer.echo(f"  source:        {entry.get('source_path', '-')}")
    typer.echo(f"  file_hash:     {entry.get('file_hash', '-')}")
    typer.echo(f"  db_path:       {db}")
    typer.echo(f"  db_size:       {db_size}")


@app.command()
def clean(name: str = typer.Argument(..., help="Corpus name.")) -> None:
    """Remove a corpus from the registry and delete its index file (T8)."""
    entry = get_corpus(name)
    if entry is None:
        typer.echo(f"error: corpus {name!r} not found", err=True)
        raise typer.Exit(code=1)

    db = corpus_db_path(name)
    db_deleted = False
    if db.exists():
        db.unlink()
        db_deleted = True

    remove_corpus(name)
    typer.echo(
        f"removed corpus {name!r} from registry"
        + (f" and deleted index {db}" if db_deleted else f" (no index file at {db})")
    )


@app.command()
def search(
    name: str = typer.Argument(..., help="Corpus name to query."),
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "--top_k", "--top-k", help="Number of results to return."),
) -> None:
    """Hybrid (dense + BM25) search over a corpus (T6)."""
    entry = get_corpus(name)
    if entry is None:
        typer.echo(f"error: corpus {name!r} not found", err=True)
        raise typer.Exit(code=1)

    # Build the embedder from the corpus's recorded model so the query vector
    # matches what indexed it (dim + e5 prefix policy). Falling back to the
    # global default here would risk a dim/prefix mismatch.
    model = entry.get("embedder_model") or None
    emb = get_embedder("local", model) if model else get_embedder("local")

    hits = retrieve_search(corpus_db_path(name), query, top_k=top_k, embedder=emb)
    if not hits:
        typer.echo("no results")
        return

    for rank, (doc, score) in enumerate(hits, start=1):
        meta = doc.get("metadata") or {}
        snippet = " ".join(doc.get("text", "").split())[:200]
        typer.echo(f"#{rank}  score={score:.4f}  page={meta.get('page', '-')}  kind={meta.get('kind', '-')}")
        typer.echo(f"    section: {meta.get('section_path', '-')}")
        typer.echo(f"    {snippet}")
        typer.echo("")


@app.command()
def mcp() -> None:
    """Start the FastMCP stdio server (T9)."""
    from . import mcp_server

    mcp_server.run()


@app.command()
def version() -> None:
    """Print the ragnexus version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
