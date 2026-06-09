# ragnexus — Plan & Roadmap

> Embed-anything knowledge engine with a gitnexus-style DX. Index a source once,
> query it from any agent via MCP — or ask it directly from the CLI.
> **First corpus scope: PDFs** (motorcycle service manuals, e.g. CFMOTO Ibex450, 288p).

## Status (2026-06-09)

**MVP 1.0 — DONE.** All tasks T1–T10 complete. T10 retrieval verify passed:
ibex450 (EN) 5/5 · africatwin (pt-BR) 5/5 · xre300 (pt-BR, OCR'd) 4/5 — all ≥4/5 gate.
MCP server registered in Claude Code (user scope, `ragnexus mcp`, ✔ connected).
42 tests green. Three manuals indexed locally with `intfloat/multilingual-e5-small`.

Next: MVP 1.5 (auto OCR for scanned PDFs) or MVP 2.0 (standalone `ask` + providers).

---

## 1. Vision

`ragnexus analyze <file>` → source gets parsed, chunked, embedded, and registered.
After that, **any agent** (Claude Code, etc.) queries it through MCP tools, and a
human can `ragnexus ask` it straight from the terminal.

Same mental model as gitnexus:
- **MCP tools return data, the calling agent reasons** (gitnexus has no internal LLM).
- A local registry (`~/.ragnexus/registry.json`) tracks indexed corpora.
- Each corpus is a single-file embedded index (`~/.ragnexus/<corpus>.db`) — zero infra.

"Anything" is the long game; **PDF is the only adapter built now**. Core stays
source-agnostic so web/code/markdown adapters slot in later without touching it.

---

## 1b. Test corpus matrix

Real manuals used to build + verify. Each exercises a different path:

| Manual | Lang | Pages | Size | Type | Exercises | Milestone |
|---|---|---|---|---|---|---|
| CFMOTO **Ibex450** | EN | 288 | 57MB | text-native (InDesign) | clean baseline, torque tables | MVP 1.0 |
| Honda **CRF1100L Africa Twin** | pt-BR | 836 | 53MB | text-native (InDesign) | multilingual + scrambled reading order (diagonal watermark interleaved in body text → Docling reading-order model required) | MVP 1.0 |
| Honda **XRE300 Sahara 2024** | pt-BR | 440 | **502MB** | **fully scanned images** (ImageMagick/Neevia, one 1582×2048 RGB scan per page, **no text layer**) | OCR path, heavy-file path | **MVP 1.5 (OCR)** |

Paths:
- `/Users/jean/Downloads/Ibex450 (6AQ2-00XW02-13) SM-WEB.pdf`
- `/Users/jean/Downloads/20241021031526_MS CRF 1100L Africa Twin (2021_2024) 00X6B-MKS-003 (1).pdf`
- `/Users/jean/Downloads/Manual Serviços XRE 300 Sahara 2024.pdf`

## 2. Architecture

```
ragnexus/
  core/
    chunking.py      # structure-aware splitter (heading path + table blocks)
    embedding.py     # provider abstraction (local | ollama | openai | voyage)
    store.py         # sqlite-vec + FTS5 hybrid index, one file per corpus
    retrieve.py      # hybrid search (dense + BM25) + optional rerank
    registry.py      # ~/.ragnexus/registry.json  (corpus lifecycle, freshness)
    config.py        # ~/.ragnexus/config.toml  (defaults + per-corpus override)
  adapters/
    base.py          # Adapter contract: ingest(source) -> list[Document]
    pdf.py           # PDF adapter (ONLY adapter in MVP)
  cli.py             # Typer CLI
  mcp_server.py      # FastMCP server (stdio)
```

**Adapter contract** (the extensibility seam):
```python
class Document(TypedDict):
    text: str
    metadata: dict   # {source, page, section_path, kind: "prose"|"table", ...}

class Adapter(Protocol):
    def ingest(self, path: str) -> list[Document]: ...
```
Core never knows it's a PDF. New source = new adapter file. Nothing else changes.

### Stack
| Concern   | Choice                                   | Why |
|-----------|------------------------------------------|-----|
| Extract   | **Docling** primary, pymupdf fallback    | manuals are table-heavy (torque/fluid specs); Docling ~98% table accuracy → Markdown w/ tables intact; pymupdf for plain pages / fallback |
| Chunk     | section-aware, keep heading path + tables intact | torque-spec tables die under fixed-size splits |
| Embed     | sentence-transformers `multilingual-e5-small` (in-process, offline) | **multilingual** (pt-BR manual in scope), 384d, CPU-fast; needs `query:`/`passage:` prefixes. `bge-m3` = quality opt-in. |
| Store     | sqlite-vec + FTS5                         | single-file, embedded, hybrid vector+keyword |
| MCP       | FastMCP                                   | first-class Python MCP |
| CLI       | Typer                                     | clean subcommands |

### Locked decisions
- **Default fully local, zero keys.** Embeddings in-process (sentence-transformers).
- **Default embedder is multilingual** (`multilingual-e5-small`, 384d) — corpus scope includes a pt-BR manual. Embedder module must apply model-specific input prefixes (e5: `query:`/`passage:`); per-corpus embedder override in registry.
- **Heavy files (500MB+):** extraction must stream/batch (Docling page-by-page if possible), embedding in batches — no whole-doc-in-RAM assumptions.
- **Default `llm = none`.** Inside an agent the agent reasons; no LLM needed.
- Standalone CLI `ask` (its own LLM) is **MVP 2.0**, not 1.0.
- Providers are pluggable from day 1 (config), but only the local default is wired in MVP 1.0.
- Tool/resource naming mirrors gitnexus (`context`/`query`, `ragnexus://doc/{name}/context`)
  for muscle-memory parity.

---

## 3. MVP 1.0 — agent-first, fully local

**Goal:** `analyze` the Ibex450 manual, then retrieve correct chunks for a real query
(e.g. "spark plug torque spec") via CLI `search` AND via the MCP server.

Tasks (ordered, each with Definition of Done):

- [ ] **T1 — Project skeleton.** `pyproject.toml` (Typer, fastmcp, sqlite-vec,
  sentence-transformers, docling, pymupdf), package layout, `ragnexus` entrypoint.
  *DoD:* `pip install -e .` works; `ragnexus --help` lists commands.
- [ ] **T2 — Config + registry.** `config.py` (TOML defaults), `registry.py`
  (read/write `~/.ragnexus/registry.json`, corpus add/list/remove, file-hash freshness).
  *DoD:* round-trip a corpus entry; `ragnexus list` shows it.
- [ ] **T3 — PDF adapter (Docling).** `adapters/pdf.py`: run Docling → Markdown +
  structured tables; map headings → `section_path`, tables → `kind:"table"` blocks,
  prose → `kind:"prose"`; carry `page`. pymupdf fallback if Docling fails on a doc.
  *DoD:* Ibex450 → N documents; the spark-plug torque table emits as one intact
  table block with its `03 Technical Information` heading path.
- [ ] **T4 — Chunking.** Section-aware: group by heading, prepend heading path to each
  chunk, keep table blocks whole, cap size with overlap.
  *DoD:* no chunk splits a table row; every chunk carries its section path.
- [ ] **T5 — Embedding (local).** `embedding.py` provider interface + sentence-transformers
  `bge-small` impl; batch encode.
  *DoD:* embeds all Ibex450 chunks offline; vector dim logged.
- [ ] **T6 — Store + hybrid retrieve.** sqlite-vec table + FTS5 mirror; `retrieve.py`
  fuses dense + BM25 (RRF), returns chunks with metadata + score.
  *DoD:* `ragnexus search ibex450 "spark plug torque"` returns the M10×1 / 15 N·m chunk top-3.
- [ ] **T7 — `analyze` command.** Wire adapter→chunk→embed→store→register end-to-end
  with progress output; `--name --force`.
  *DoD:* `ragnexus analyze "Ibex450 ... SM-WEB.pdf"` completes, appears in `list`, `status` shows chunk count.
- [ ] **T8 — `status` / `list` / `clean`.** Corpus introspection + teardown.
  *DoD:* the three commands behave on the real corpus.
- [ ] **T9 — MCP server.** FastMCP exposing `ragnexus_query(corpus, query)` (tool) +
  `ragnexus://doc/{name}/context` (resource). `ragnexus mcp` starts stdio server.
  *DoD:* register in Claude Code, query the manual through MCP, get correct chunks.
- [ ] **T10 — Verify + README.** Run real queries on **both** manuals: EN (CFMOTO
  Ibex450) and **pt-BR (500MB elephant)** — torque specs, fluid capacities, a removal
  procedure, in both languages. Confirm heavy-file path doesn't OOM and pt-BR retrieval
  works with multilingual embedder. Document setup + usage.
  *DoD:* ≥4/5 queries per manual return the right chunk in top-3; 500MB manual indexes
  without OOM; README has install + analyze + MCP-register steps.

**MVP 1.0 exit criteria:** Ibex450 indexed locally; correct retrieval from both CLI
`search` and the MCP server inside Claude Code; zero API keys, zero daemons.

---

## 3b. MVP 1.5 — scanned / OCR path

Driven by the XRE300 corpus (440 scanned image pages, pt-BR, 502MB, no text layer).

- [ ] Detect text-less PDFs (no extractable text layer) → route to OCR path automatically.
- [ ] Docling OCR enabled (`do_ocr=True`) with a pt-capable engine (EasyOCR/Tesseract/RapidOCR); pick fastest acceptable on CPU.
- [ ] Heavy-file handling: page-batched OCR, progress, resumable analyze (don't redo 440 pages on crash).
- [ ] Verify: XRE300 indexes end-to-end; pt-BR queries return correct chunks despite OCR noise.
- *Note:* OCR of 440 scans on CPU is slow (many minutes+). Acceptable for index-once; surface progress.

## 4. MVP 2.0 — standalone `ask` + providers

- [ ] `ask` command: retrieve → synthesize an answer (for bare-terminal use, no agent).
- [ ] LLM providers: `ollama` (local llama), `anthropic`, `openai`. Config `[llm]`.
- [ ] Embedding providers beyond local: `ollama`, `openai`, `voyage` wired + selectable.
- [ ] `reanalyze` + `detect-changes`: re-index when the source file hash drifts.
- [ ] `doctor`: check deps, model downloads, ollama reachability, API keys.
- [ ] Cross-encoder reranking stage (top-20 → top-5) for higher precision.

---

## 5. Future / backlog

- **Groups** (multi-manual): `group create/add`, `group search` across many corpora.
- **More adapters:** web pages, markdown/docs, source code (gitnexus-ish), images (OCR).
- **`wiki`**: generate a browsable HTML/Markdown summary of a corpus (gitnexus wiki analog).
- **`toc` / `show --chunk`**: section map + single-chunk inspection.
- **Per-corpus tuning profiles** (chunk size, embedder) saved in the registry.
- **Eval harness:** a question→expected-chunk set per corpus to catch retrieval regressions.

---

## 6. Open questions

- Chunk size / overlap defaults for dense manuals? (start 800 tok / 100 overlap, tune in T10)
- ~~Keep pdftotext for tables, or jump to Docling?~~ **Resolved: Docling primary, pymupdf fallback.**
- Docling first-run model download (~hundreds MB) — bundle a `doctor`/warm-up step? (defer to 2.0 `doctor`)

## 7. Lessons

- **Docling backend: use the DEFAULT, not `PyPdfiumDocumentBackend`.** PyPdfium drops the
  leading characters of body lines on PDFs with an overlapping left-margin watermark
  (Honda CRF1100L: "Algumas"→"umas", "Informações"→"mações") → garbage chunks → pt-BR
  retrieval collapsed to 2/5. The default backend (full layout parse) extracts cleanly,
  and was *faster* in testing. OCR is gated by `do_ocr`, not the backend, so PyPdfium
  bought nothing. Fixed in `adapters/pdf.py`; africatwin went 2/5 → 5/5.
- **Diagnose per-corpus, don't trust an aggregate verdict.** A tester concluded "the
  multilingual embedder is broken" from africatwin 2/5 — but xre300 (also pt-BR, same
  embedder) scored 4/5. The cross-corpus contradiction located the real cause
  (per-file extraction corruption), not the embedder.
- **Watermark text still pollutes ~50% of africatwin chunks** ("PROIBIDA A REPRODUÇÃO E
  DISTRIBUIÇÃO") but it's constant noise → minimal ranking impact; deferred (add a
  repeated-line watermark stripper only if a future corpus needs it).
- Corpus name derivation: slug from filename vs required `--name`? (auto-slug, `--name` overrides)
