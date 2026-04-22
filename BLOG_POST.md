# I Benchmarked 6 PDF Parsers for RAG and Was Wrong About Docling

*A deep dive into whether parser choice actually affects retrieval quality — and what I learned when the data pushed back on my hypothesis.*

---

## The question I thought had an obvious answer

I'm building [CodeMentor](https://github.com/medAli-ai/CodeMentor) — a Java tutoring app that uses RAG to answer student questions from programming textbooks. Right now it uses PyMuPDF because PyMuPDF is fast, reliable, and it was the first parser I reached for.

But the Java textbook I feed it is 1059 pages of mixed prose, code blocks, and tables. And PyMuPDF outputs raw text with no structural awareness. The code blocks get mashed together with surrounding paragraphs. Tables become whitespace-separated streams. It feels wasteful.

So I set out to answer a simple question: **if I swap PyMuPDF for a smarter parser, does my RAG pipeline actually retrieve better answers?**

My hypothesis going in: yes, obviously. A parser that preserves code blocks as discrete units and tables as structured data should produce cleaner chunks, which should produce better retrieval.

I spent three weeks building a benchmark to prove it. The data told a more interesting story.

---

## The setup

I compared **6 popular PDF parsers** across **12 parser/chunker combinations** using an 1059-page Java programming textbook:

- **PyMuPDF** — fast raw text extraction (C bindings)
- **pdfplumber** — raw text plus basic table detection
- **Docling** — IBM's ML-based parser that outputs full Markdown
- **Unstructured** — typed element extraction (NarrativeText, CodeSnippet, Table, etc.)
- **OpenParse** — structural node-based parsing
- **LiteParse** — LlamaIndex's spatial-text parser

Each parser's output went through two chunking strategies:

- **SemanticChunker** (from Chonkie) — embedding-based boundary detection
- **RecursiveCharacterTextSplitter** (from LangChain) — the classic heuristic splitter

All chunks got embedded with `BAAI/bge-small-en-v1.5` and indexed into Qdrant. The same config I use in CodeMentor production.

Then I ran **three layers of evaluation**:

1. **Chunk quality** — ROUGE-L, BLEU, semantic similarity against hand-picked reference passages
2. **Retrieval quality** — Precision@10, Recall@10 across 36 realistic student queries
3. **Per-category breakdown** — how each parser handles conceptual questions vs code questions vs table lookups

I'll walk through what each layer showed, and why the second one surprised me.

---

## Finding #1: Docling dominates chunk quality

The first layer is the purest test: given the parser's output, how intact is the content after chunking?

I picked 8 reference passages from the actual textbook — a functional interface definition, a code block, a table row, prose explanations — and measured how well each parser+chunker combination preserved them.

Here's the clean signal:

| Parser + Chunker           | ROUGE-L | BLEU  | Semantic Sim | Mid-sentence % |
|----------------------------|---------|-------|--------------|----------------|
| **docling + semantic**     | **0.512** | **0.388** | **0.873** | 23.8% |
| pymupdf + recursive        | 0.460   | 0.317 | 0.855       | 38.5% |
| openparse + recursive      | 0.439   | 0.243 | 0.864       | 31.0% |
| pdfplumber + semantic      | 0.388   | 0.233 | 0.860       | 32.9% |
| liteparse + semantic       | 0.379   | 0.240 | 0.862       | 34.2% |
| unstructured + semantic    | 0.349   | 0.197 | 0.824       | **10.9%** |
| docling + recursive        | 0.318   | 0.164 | 0.833       | **1.8%** |

Docling + SemanticChunker wins on lexical preservation (ROUGE-L, BLEU) *and* semantic preservation (cosine sim). Docling + RecursiveCharacterTextSplitter wins on chunk coherence (only 1.8% of chunks start mid-sentence — versus 38.5% for PyMuPDF+recursive).

This confirmed my hypothesis. **Better parser → better chunks.** Case closed, right?

Not quite. The setup that made Docling win needed one non-obvious piece.

---

## The Markdown insight I almost missed

Initially, Docling was losing to PyMuPDF. ROUGE-L of 0.23 versus 0.40. That made no sense — Docling is supposed to be the smart parser.

The problem: Docling outputs full Markdown with `##` headings, ` ``` ` code fences, and `|`-separated tables. But I was treating it like raw text and feeding the whole Markdown string into SemanticChunker, which split on sentence boundaries and mangled the structure.

The fix was [Chonkie's MarkdownChef](https://docs.chonkie.ai/oss/chefs/markdownchef). It parses Markdown into discrete components — code blocks, tables, images, and prose — as separate chunks. Code blocks don't get split. Tables don't get split. Only the prose goes through SemanticChunker.

```python
from chonkie import MarkdownChef, SemanticChunker

markdown_chef = MarkdownChef()

def chunk_markdown_parser(markdown_text: str) -> list[str]:
    doc = markdown_chef.process(markdown_file_path)
    
    chunks = []
    # Code blocks stay whole
    for code in doc.code:
        chunks.append(code.content)
    
    # Tables stay whole
    for table in doc.tables:
        chunks.append(table.content)
    
    # Only prose goes through SemanticChunker
    prose = "\n\n".join(c.text for c in doc.chunks)
    for chunk in semantic_chunker.chunk(prose):
        chunks.append(chunk.text)
    
    return chunks
```

This single change jumped Docling's ROUGE-L from **0.23 → 0.51**. More than doubled.

**The lesson:** *A parser's output format should drive your chunking strategy.* If your parser produces Markdown, use a Markdown-aware chunker. Treating Markdown as raw text wastes the structure the parser worked hard to recover.

This is the kind of thing I only discovered because the numbers didn't match my expectations.

---

## Finding #2: Chunk quality doesn't translate directly to retrieval

This is where my hypothesis started to crack.

I built a gold set of 36 realistic student queries — the kinds of things a Java learner actually types:

```python
"what is a lambda expression in java?"
"how do I loop through an arraylist"
"why am I getting a nullpointerexception?"
"when should I use an interface vs an abstract class?"
"what's covered in the lambdas chapter of the OCP exam?"
```

Categories: conceptual, how-to, code-specific, debugging, best practices, exam prep, conversational.

For each query, I retrieved top-10 chunks per parser/chunker combination and measured:

- **Precision@10** — of the 10 returned, how many are relevant?
- **Recall@10** — of all relevant chunks in the corpus, how many made top-10?

Relevance was defined as cosine similarity ≥ 0.75 to the query embedding (I'll discuss why this is imperfect in the methodology section).

Here's what the data showed:

| Parser + Chunker         | P@10  | R@10  | Best at      |
|--------------------------|-------|-------|--------------|
| **docling + semantic**   | **0.803** | 0.710 | Code queries (0.917) |
| docling + recursive      | 0.800 | 0.638 | Concept queries (0.838) |
| openparse + recursive    | 0.747 | 0.695 | — |
| liteparse + recursive    | 0.745 | 0.721 | — |
| unstructured + recursive | 0.739 | 0.707 | — |
| liteparse + semantic     | 0.739 | 0.703 | — |
| openparse + semantic     | 0.716 | 0.670 | — |
| pymupdf + recursive      | 0.711 | 0.736 | Exam prep (0.780) |
| pdfplumber + semantic    | 0.708 | 0.718 | — |
| pymupdf + semantic       | 0.703 | 0.714 | — |
| unstructured + semantic  | 0.650 | **0.798** | — |
| pdfplumber + recursive   | 0.647 | 0.715 | — |

Docling still wins. But notice the gap: the best parser (0.803) versus the worst (0.647) is only **15 percentage points**. And everyone is in a tight cluster at 0.70-0.80.

Meanwhile, chunk quality had docling at 0.51 ROUGE-L versus pdfplumber at 0.21 ROUGE-L. A gap of 2.5x collapsed to 1.24x at the retrieval stage.

**Why the gap shrinks:** the embedding model is robust. `BAAI/bge-small-en-v1.5` can still find relevant chunks even when the text has line breaks in weird places or tables are formatted poorly. It's not *as good* at it — Docling's cleaner text helps — but it's not catastrophic either.

**The practical implication:** if you're optimizing for retrieval only, the parser choice is less important than I thought. If you're optimizing for what the LLM sees in its context window (chunk quality), it matters a lot.

---

## The category breakdown is where it gets interesting

The aggregated numbers hide the real differences. Here's the per-category breakdown:

| Category      | Winner              | Score | Why it matters                                    |
|---------------|---------------------|-------|---------------------------------------------------|
| Conceptual    | docling + recursive | 0.838 | Markdown structure preserves definitions cleanly  |
| How-to        | docling + semantic  | 0.738 | Clean code blocks help with procedural queries    |
| Code-specific | docling + semantic  | 0.917 | MarkdownChef isolates code blocks as chunks       |
| Debugging     | docling + recursive | 0.920 | Error terminology is high-signal for embeddings   |
| Best practices| Many tied at 1.000  | 1.000 | Comparative "X vs Y" queries retrieve reliably    |
| Exam prep     | pymupdf + recursive | 0.780 | Raw text preserves chapter/TOC structure          |
| Conversational| All tied at 0.500   | 0.500 | RAG limitation, not a parser limitation           |

Three things jump out:

**1. Code-specific queries are where Docling earns its keep.** Getting a 0.917 on code queries versus pdfplumber's 0.833 is the single biggest per-category delta. If your RAG system answers a lot of "show me code for X" questions, Docling + SemanticChunker + MarkdownChef is the right stack.

**2. PyMuPDF wins exam prep queries.** This surprised me. Questions like "what topics does chapter 8 cover?" retrieve better from PyMuPDF because its raw text preserves the table of contents and chapter boundaries as-is. Docling's Markdown conversion reformats these and changes how they embed.

**3. Conversational queries score 0.500 across everyone.** Questions like "what about when the interface has a default method?" or "how does that compare to Python?" are context-dependent follow-ups. Without conversation history they're ambiguous. **No parser can retrieve what the query doesn't specify.** This is a RAG limitation, not a parser limitation.

---

## Two things I got wrong along the way

Writing the benchmark was messier than the results make it look. Two mistakes are worth flagging, because I've seen other benchmark posts make the same ones.

### Mistake 1: Snippet-based gold annotations

My first attempt at relevance judgment was snippet matching — for each gold query I wrote out expected text snippets and marked chunks as relevant if they contained any of them.

This was a disaster for two reasons:

**Broad snippets matched too much.** My snippet `"final"` for the "what does final do" query matched 334 chunks in PyMuPDF's output. Every chunk that mentioned the word "final" counted as relevant. Recall@3 was mathematically guaranteed to be near zero.

**Narrow snippets missed parser-specific formatting.** My snippet `"An interface is an abstract data type..."` didn't match because PyMuPDF extracted it as `"An interface is an abstract \ndata type..."` with a line break.

I ended up rebuilding the evaluation using embedding-based relevance: a chunk is relevant if its cosine similarity to the query embedding exceeds 0.75. No string matching. Parser-agnostic.

### Mistake 2: The self-reference problem

Once I switched to embedding-based relevance, I hit a different issue: I was using the same embedding model for both retrieval and relevance judgment.

This is tautological. Qdrant retrieves by cosine similarity to the query embedding. Relevance is defined as cosine similarity to the query embedding above a threshold. Precision@K is almost guaranteed to be high because the retrieval mechanism and the relevance definition are the same mechanism.

**The honest interpretation of my P@10 numbers:** they measure *agreement with the threshold cutoff*, not retrieval quality in some absolute sense. The ranking between parsers is still meaningful because the bias applies equally to all of them. But you shouldn't read "P@10 = 0.80" as "80% of retrievals are correct by some objective standard."

The proper fix is one of:

- **A stronger, independent embedding model** for relevance judgment (e.g., `bge-large` scoring results from `bge-small`)
- **LLM-as-judge** — for each retrieved chunk, ask GPT-4 or Claude "is this relevant to the query?"
- **Manual annotation** — expensive but unambiguous

I tried LLM-as-judge via Groq's free tier but hit rate limits on 63% of calls and had to abandon that approach for this post. For CodeMentor's production eval setup I'll probably use it — 30 requests per minute is fine for running against a curated golden set of 30 queries offline, just not for 120 calls during a benchmark.

---

## What I'd tell you if you asked "which parser should I use?"

Honestly, it depends on your use case. But here's my decision tree:

**Default recommendation: Docling + SemanticChunker + MarkdownChef.**

- Best overall retrieval quality (P@10 = 0.803)
- Best code block preservation (ROUGE-L = 0.51)
- Best on code-specific queries by a wide margin (0.917 vs ~0.85)
- Output format is reusable if you ever need Markdown elsewhere

**But use PyMuPDF if:**

- Your documents are simple prose with no code or tables
- You need blazing-fast parsing (4s vs 568s for Docling on 1000 pages)
- Your queries are mostly "navigate the document" style (TOC lookups, chapter summaries)
- You have less than 16GB RAM (Docling needs more)

**Use Unstructured if:**

- You need the highest code block preservation specifically (99.2% vs Docling's 92.5%)
- You want typed element output for downstream filtering
- You're OK with a slight hit on retrieval precision

**Skip:**

- **Marker** — crashed my 16GB machine twice on a 1000-page document. Probably fine on 32GB+ but I couldn't get it to run.
- **pdfplumber for RAG** — it's a great library for tabular data extraction, but for RAG the raw text quality isn't enough to justify the 114s parse time over PyMuPDF's 4s.

---

## The bigger lesson

Going into this benchmark I thought the result would be "use Docling." The actual result is messier and more interesting:

**The parser matters less than you think for retrieval, but more than you think for what the LLM sees.**

Modern embedding models are robust. They'll find relevant chunks even from sloppy parsers. So if you're only measuring retrieval metrics, you might conclude parsing doesn't matter much.

But once those chunks reach the LLM's context window, parsing quality determines whether the model gets a clean code block or a mangled sequence of whitespace. That affects faithfulness, not retrieval precision — a dimension my benchmark couldn't measure without LLM-as-judge, but one that matters in production.

For CodeMentor I'm migrating to Docling + SemanticChunker + MarkdownChef. Not because the retrieval numbers demanded it (the gap was only 15 points), but because my students get code examples in their chat interface. Mangled code blocks are visible to users in a way that mangled prose isn't.

---

## What I'd do differently next time

If I were rebuilding this benchmark from scratch:

1. **Start with LLM-as-judge evaluation** using a paid API or local Ollama without rate limits. My biggest regret is not measuring generation faithfulness end-to-end.
2. **Use an independent embedding model** for relevance judgment to avoid the self-reference issue.
3. **Include hybrid search** (dense + BM25 via Qdrant's fusion API) — code terminology like `ArrayList` or `@FunctionalInterface` is exactly what BM25 excels at.
4. **Test on multiple document types** — my results are specific to a programming textbook. Legal documents, medical literature, and financial reports would likely show different patterns.

---

## Code and results

Everything is open source:

- **Benchmark code:** [github.com/medAli-ai/pdf-parser-benchmark](https://github.com/medAli-ai/pdf-parser-benchmark)
- **CodeMentor (the project that sparked this):** [github.com/medAli-ai/CodeMentor](https://github.com/medAli-ai/CodeMentor)

The notebook is self-contained — clone, `uv sync`, drop a PDF in `data/`, and you can replicate everything. Cache persists between runs, so re-running is fast.

If you've run a similar benchmark or disagree with my methodology, I'd love to hear it. The best feedback I got while writing this post was "why are you using keywords for gold queries when real users ask questions?" That question rebuilt half my evaluation.

Thanks for reading.

---

*I'm [Mohamed Ali](https://github.com/medAli-ai) — building AI infrastructure and RAG systems. If this post was useful, [follow me on Medium](https://medium.com/@your-handle) for more deep dives on RAG engineering.*
