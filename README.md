# ragnexus

**Embed anything, query it from any agent.**

ragnexus is a fully local knowledge engine with a gitnexus-style developer experience:
run `ragnexus analyze <file>` once to index a document, then query it from the CLI or
from any AI agent through a built-in MCP server. No API keys, no daemons, no network —
everything runs in-process against a single-file index.

The current scope is **PDFs** — built and validated on real motorcycle service manuals —
but the core is deliberately source-agnostic, so other adapters (web, docs, code) slot in
without touching the retrieval pipeline.

```
  PDF ──▶ Docling ──▶ section-aware ──▶ e5 ──▶ sqlite-vec + FTS5 ──▶ CLI  ┐
        (layout +      chunks         embeddings   (hybrid index)         ├─▶ you / agents
         tables)    (heading path)    (384-dim)                    MCP ───┘
```

---

## Why

A service manual is hundreds of pages of torque specs, fluid capacities, and removal
procedures buried in tables and section hierarchies. Full-text search misses the
phrasing; pasting the whole PDF into an LLM is slow and lossy. ragnexus indexes the
document once, locally, and exposes precise hybrid retrieval to whatever is asking —
your terminal or an agent reasoning over the results.

- **Local & offline.** Embeddings run in-process via sentence-transformers. Zero keys.
- **Zero infrastructure.** Each corpus is one SQLite file under `~/.ragnexus/`.
- **Agent-native.** Ships an MCP server so any MCP-capable agent can query your corpora.
- **Layout-aware.** Docling preserves tables and reading order — torque tables survive intact.

---

## Quick start

Python ≥ 3.12 is required. A virtual environment is recommended.

```bash
git clone https://github.com/jeanfbrito/ragnexus
cd ragnexus
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

> **First run is heavy.** Dependencies include PyTorch (pulled in by Docling), so the
> install is sizeable. On first use, models download once from Hugging Face — the e5
> embedder (~470 MB) plus Docling's layout and table models — then stay cached. After
> that, ragnexus is fully offline.

Index a document, then search it:

```bash
# Index once (parse → chunk → embed → store → register).
ragnexus analyze "Ibex450 SM-WEB.pdf" --name ibex450

# Query it.
ragnexus search ibex450 "spark plug torque"
```

Inspect and manage your corpora:

```bash
ragnexus list              # all registered corpora
ragnexus status ibex450    # chunk/vector counts, model, freshness, db size
ragnexus clean ibex450     # drop from registry and delete the index file
```

### Commands

| Command | What it does |
|---|---|
| `analyze <path>` | Parse, chunk, embed, store and register a source. Flags: `--name`, `--embedder`, `--force`. |
| `search <corpus> <query>` | Hybrid (dense + BM25) search. `--top_k` (default 5). |
| `list` | List registered corpora. |
| `status <corpus>` | Show corpus details: chunks, vectors, model, timestamps, source, db size. |
| `clean <corpus>` | Remove a corpus from the registry and delete its index file. |
| `mcp` | Start the FastMCP stdio server. |
| `version` | Print the ragnexus version. |

---

## How it works

ragnexus is a five-stage pipeline behind a thin adapter seam. The core never knows it is
handling a PDF — a new source type is just a new adapter.

| Stage | Choice | Detail |
|---|---|---|
| **Extract** | [Docling](https://github.com/DS4SD/docling) (default backend) | Layout-aware parsing with tables preserved and a reading-order model that handles watermark overlays interleaved in the body text. |
| **Chunk** | Docling `HybridChunker` | Section-aware chunks; the full heading path is prepended to each chunk's text so context survives retrieval. Table blocks stay whole. |
| **Embed** | `intfloat/multilingual-e5-small` | 384-dim, multilingual (incl. Portuguese), runs in-process and offline. Applies e5 `query:` / `passage:` prefixes automatically. |
| **Store** | sqlite-vec + FTS5 | A single file per corpus at `~/.ragnexus/<name>.db`. Vector index and keyword index in the same database. |
| **Retrieve** | Hybrid search | Dense vector KNN + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF, k=60). |

A local registry at `~/.ragnexus/registry.json` tracks every indexed corpus — source
path, file hash, chunk and vector counts, embedder model, and timestamps — so `list`
and `status` work without re-scanning anything.

The embedder is provider-pluggable. Local sentence-transformers is wired and is the
default; `ollama`, `openai`, and `voyage` providers are stubbed for a future release and
are **not yet usable**.

### Performance: index once, query cheaply

Indexing is the one-time cost — a few minutes per manual on CPU, dominated by Docling
extraction and embedding (the manuals below each took roughly 2–6 minutes). Once the
embedder model is loaded, queries are answered locally against the embedded sqlite-vec
index, so repeated querying is cheap.

---

## Real-world results

ragnexus was built and validated against three real motorcycle service manuals, each
exercising a different ingestion path:

| Manual | Language | Pages | Chunks | Notes |
|---|---|---|---|---|
| CFMOTO Ibex450 | English | 288 | 678 | text-native |
| Honda CRF1100L Africa Twin | Portuguese (pt-BR) | 836 | 965 | text-native; diagonal watermark overlay handled by Docling's reading-order model |
| Honda XRE300 Sahara | Portuguese (pt-BR) | 440 | 716 | originally scanned images, pre-OCR'd with `ocrmypdf -l por` |

**Retrieval quality.** Across real queries on each manual — torque specs, fluid
capacities, removal procedures, in both English and Portuguese — **at least 4 of 5
returned the correct chunk in the top 3**. That held for the English manual, the pt-BR
manual, and the OCR'd scan alike.

For example:

```console
$ ragnexus search ibex450 "spark plug torque"
#1  score=...  page=30  kind=table
    section: 3.5.2 Tighten Torque Table for Engine
    ... M10×1 / 15 N·m ...
```

The spark-plug torque table chunk — `M10×1 / 15 N·m`, page 30, section
"3.5.2 Tighten Torque Table for Engine" — comes back in the top 3.

---

## Using it from agents (MCP)

ragnexus ships a [FastMCP](https://github.com/jlowin/fastmcp) server so any MCP-capable
agent can query your indexed corpora. Register it with Claude Code:

```bash
claude mcp add ragnexus -s user -- /path/to/ragnexus/.venv/bin/ragnexus mcp
```

The server exposes:

| Kind | Name | Purpose |
|---|---|---|
| Tool | `ragnexus_query(corpus, query, top_k)` | Hybrid search over a corpus; returns ranked chunks with metadata. |
| Tool | `ragnexus_list()` | List registered corpora. |
| Resource | `ragnexus://doc/{name}/context` | Pull contextual chunks for a corpus. |

The MCP tools return data and the calling agent reasons over it — ragnexus has no
internal LLM, by design.

---

## Requirements

- **Python ≥ 3.12.**
- **Install:** `pip install -e .` (use a venv).
- **Disk & first-run download:** dependencies include PyTorch via Docling (a sizeable
  install). Models — the e5 embedder (~470 MB) and Docling's layout/table models —
  download once from Hugging Face on first use, then cache locally. Subsequent runs are
  fully offline.

Core dependencies: `typer`, `fastmcp`, `sqlite-vec`, `sentence-transformers`, `docling`,
`pymupdf`.

---

## Roadmap

- **Auto-OCR for scanned PDFs** — detect a text-less PDF and OCR it automatically.
  Today you pre-OCR scans yourself (e.g. `ocrmypdf -l por`).
- **Standalone `ask` command** — retrieve plus LLM synthesis for terminal use without an
  agent — alongside embedder/LLM providers (`ollama`, `openai`, `voyage`).
- **Multi-document groups** and **more source adapters** (web, docx, code).

---

## License

MIT © 2026 Jean Brito. See [LICENSE](LICENSE).
