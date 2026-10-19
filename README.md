# groundtutor

Citation-grounded tutoring over your own lectures, slides, notes, and transcripts.

`groundtutor` answers from documents you add. Every factual sentence points at a page, a slide, or a timestamp, and each claim carries a quote that has to appear in that source. Claims the model cannot support are dropped, flagged, or cause the whole answer to be refused.

```python
from groundtutor import Tutor

tutor = Tutor(llm="openai:gpt-4o-mini", embedder="local")
tutor.add("lecture.pdf")
tutor.add("slides.pptx")
tutor.add_transcript("talk.vtt")

res = tutor.ask("What is backpropagation?", mode="strict")
print(res.answer)
res.citations
res.ungrounded

tutor.quiz(5)
tutor.flashcards(10)
```

## Install

```bash
pip install groundtutor
```

The core install reads PDF, PowerPoint, text, Markdown, WebVTT, and SRT. Embeddings and hosted indexes are extras:

| Extra | Install | What it adds |
| --- | --- | --- |
| `local` | `pip install "groundtutor[local]"` | `embedder="local"` (sentence-transformers) and the cross-encoder reranker |
| `youtube` | `pip install "groundtutor[youtube]"` | YouTube transcript downloads |
| `chroma` | `pip install "groundtutor[chroma]"` | Chroma vector store |
| `pgvector` | `pip install "groundtutor[pgvector]"` | Postgres pgvector store |

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for those providers. Ollama needs a local server at `http://127.0.0.1:11434` (override with `OLLAMA_HOST`).

`embedder="hash"` is a deterministic bag-of-words embedder for tests and smoke checks. It is not a semantic model.

## Grounding

1. Documents are split into chunks that keep their locator: page, slide, or start/end time, plus character offsets and the Markdown heading path. Transcript lines stay together until the chunk fills or a silence longer than five seconds starts a new segment.
2. Retrieval mixes BM25 with dense vectors and fuses the two lists with reciprocal rank fusion. An optional reranker can reorder the fused list.
3. The model must return JSON. Each claim cites source numbers like `[1]` and includes a quote copied from that source.
4. Quotes that are not in the cited chunk are rejected before any model is asked to judge them.
5. An entailment pass scores the remaining claims. The default judge is the same LLM that wrote the answer. `verifier="lexical"` uses term overlap instead, which is useful offline.

`tutor.ask(..., mode=...)` controls what happens after that pass:

| Mode | Behavior |
| --- | --- |
| `strict` | If any claim is unsupported, the answer is replaced with a refusal. |
| `flag` | The answer is kept. Unsupported claims are listed on `res.ungrounded`. This is the default. |
| `permissive` | Quotes still have to be real. The entailment judge does not run. |

If the best dense score and the best query-term coverage are both below their thresholds, `ask` does not call the model. The answer is exactly `I couldn't find this in your materials`, and `res.suggestions` lists the nearest sections.

Quizzes and flashcards use the same quote check and verifier. An item with no supporting quote is not returned, so you may get fewer than `n`.

## Providers

```python
tutor = Tutor(llm="openai:gpt-4o-mini", embedder="local")
tutor = Tutor(llm="anthropic:claude-3-5-haiku-latest", embedder="openai:text-embedding-3-small")
tutor = Tutor(llm="ollama:llama3.2", embedder="hash")
tutor = Tutor(llm="openai:gpt-4o-mini", embedder="local", store="chroma:./index")
tutor = Tutor(
    llm="openai:gpt-4o-mini",
    embedder="openai:text-embedding-3-small",
    store="pgvector:postgresql://localhost/tutor",
)
```

Any object with the right methods can stand in for a provider. An LLM needs `complete(messages, *, temperature, json_mode) -> str`. An embedder needs `embed_documents` and `embed_query`. A vector store needs `add` and `query`. A verifier needs `verify`.

```python
class EchoLLM:
    def complete(self, messages, *, temperature=0.0, json_mode=False) -> str:
        return messages[-1]["content"]
```

## Persistence

`save` and `load` round-trip the in-memory index so a new process can query without embedding the corpus again. They support the default memory store.

```python
tutor.save("tutor-index")
tutor = Tutor.load("tutor-index", llm="openai:gpt-4o-mini", embedder="local")
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest
```

Tests use a scripted LLM and the hash embedder, so they do not call a network or download a model.

## Limits

- PDF ingestion reads the text layer. A scanned page with no text is skipped, and a PDF with no extractable text raises `IngestError`.
- Character offsets refer to the normalized text of that page, slide, transcript, or notes file.
- The chunk catalog lives in the process. Chroma and pgvector hold vectors for that process; they are not a full document database.
- The hash embedder only ranks shared words. Use `local` or an OpenAI embedding model for paraphrases.
- A strict answer is only as reliable as the judge model. The quote check is deterministic; the entailment score is not.
- Markdown headings are ATX-style (`# Title`). Headings inside fenced code blocks are ignored.
