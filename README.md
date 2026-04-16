# PDF Parser Benchmark for RAG Pipelines

A reproducible benchmark comparing 8 PDF parsers on a real-world technical document
(OCP Java SE 17 Study Guide — 1000+ pages, code blocks, tables, multi-column layout).

Companion to the blog post: _"Which PDF Parser Should You Use for RAG? A Deep Dive"_

## Parsers evaluated

| Parser | Type | Code-aware | Notes |
|---|---|---|---|
| PyMuPDF | Text-layer | ✗ | Baseline |
| pdfplumber | Text-layer | ✗ | Best table extraction |
| Docling (IBM) | Layout model | ✓ | Markdown output |
| unstructured.io | Hybrid | Partial | Auto-strategy selection |
| Marker | Vision model | ✓ | Trained on technical docs |
| openparse | Structural | Partial | Bounding-box tree |
| LiteParse | TypeScript/Node | ✓ | Optional, requires Node ≥ 18 |
| LlamaParse | Cloud API | ✓ | Optional, requires API key |

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- NVIDIA GPU with CUDA 12.1+ (optional but recommended for Marker and Docling)
- Node.js ≥ 18 (optional, for LiteParse only)

## Quickstart

```bash
# 1. Clone
git clone https://github.com/medAli-ai/pdf-parser-benchmark.git
cd pdf-parser-benchmark

# 2. Create the virtual environment (uv reads .python-version automatically)
uv venv

# 3a. Install core + fast parsers (quick start, no model weights)
uv sync --extra parsers

# 3b. OR: install everything including Marker, Docling, unstructured
#     Expect 5-15 min on first run (model weight downloads)
uv sync --all-extras

# 4. Place the OCP Java 17 Study Guide PDF in data/
#    (Sybex ISBN 978-1-119-86465-8 — purchase from the publisher)
cp /path/to/ocp_java17_study_guide.pdf data/

# 5. (Optional) Add your LlamaParse API key
cp .env.example .env
# edit .env

# 6. Launch the notebook inside the managed venv
uv run jupyter notebook notebooks/benchmark.ipynb
```

## Structure

```
notebooks/benchmark.ipynb   ← main benchmark notebook
data/                       ← place your PDF here (gitignored)
benchmark_cache/            ← parser outputs cached here (gitignored)
benchmark_results/          ← charts and CSVs written here (gitignored)
liteparse_runner/           ← Node.js runner (scaffolded by Cell 1.4)
```

## Reproducibility

All results in the blog post were generated with:
- Python 3.12.3
- BAAI/bge-small-en-v1.5 embeddings (384-dim)
- Qdrant in-memory mode
- NVIDIA RTX 3060 6GB, CUDA 12.1, Ubuntu 22.04