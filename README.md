# Multimodal AI Chat Assistant

<p align="center">
  <strong>A Gradio chat assistant with selectable LLM providers, tool calling, text-to-speech, and a document question-answering (RAG) pipeline.</strong>
</p>

<p align="center">
  Upload your own documents, ask questions about them, and get answers grounded in their content, optionally spoken aloud.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio" />
  <img src="https://img.shields.io/badge/RAG-Document_QA-7C3AED?style=for-the-badge" alt="RAG" />
  <img src="https://img.shields.io/badge/MCP-Tool_Calling-0F766E?style=for-the-badge" alt="MCP Tool Calling" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-16A34A?style=for-the-badge" alt="ChromaDB" />
</p>

---

## ✨ Features

- **Two LLM providers:** Switch between OpenAI GPT and Groq (`openai/gpt-oss-120b`) from a dropdown.
- **Tool calling:** The model decides when to use a tool, the app runs it, and the result goes back to the model for the final answer. The loop is capped at a fixed number of rounds and tool errors are returned to the model instead of crashing the turn.
  - **Calculator (MCP):** Addition, subtraction, multiplication and division (with a divide-by-zero check).
  - **Weather (MCP):** Geocodes a city and returns current temperature, humidity and wind speed via the Open-Meteo APIs.
  - **Document search (`search_kb`, local):** Searches the documents the user uploaded in their session.
- **MCP tool layer:** The calculator and weather tools run in a separate MCP server that the app connects to at startup and discovers tools from. New tools can be added to the server without changing the chat loop or the UI.
- **Document upload and RAG:** Upload one or more files in the UI. They are converted to text, chunked, enriched, embedded and stored in ChromaDB, then searched by the model through `search_kb`.
- **Per-session document isolation:** Each browser session only searches the documents it uploaded. The model cannot choose which documents to search.
- **Text-to-speech:** Replies can be spoken with Groq-hosted Orpheus TTS. A checkbox turns it off, markdown and citation brackets are stripped before speaking, and a TTS failure never loses the text reply.
- **Grounded answers:** When documents are loaded, the model is instructed to answer only from retrieved extracts, search again with different terms if needed, and say so when the answer isn't in the documents.

---

## ⚙️ How it works

### 💬 Chat flow

```text
User message (text box or Enter)
        |
        v
Provider selection (GPT or Groq)
        |
        v
Chat loop with tools (max rounds, errors returned to the model)
        |-- calculator    (MCP server)
        |-- get_weather   (MCP server)
        |-- search_kb     (local; only offered when the session has uploaded documents)
        |
        v
Text reply  -->  optional Groq Orpheus TTS  -->  text + audio in the UI
```

The tool list is built per request: the tools discovered from the MCP server are combined with the local `search_kb` tool. `search_kb` is registered only when the session has documents, and the session's document ids are bound by the app, not supplied by the model. That is why document search stays inside the app instead of running over MCP.

### 📥 Ingestion (on upload)

1. **Load:** The file is converted to markdown text (`rag/loader.py`). PDFs go through `pymupdf4llm`, which keeps headings and tables as markdown; other formats go through MarkItDown.
2. **Clean up plain text:** For text without markdown structure (typical of PDF extraction), numbered headings are detected, table-of-contents lines and bare page numbers are removed, and hard-wrapped lines are reflowed into paragraphs.
3. **Structure-aware chunking:** Text is split by headings and paragraphs. Each chunk carries its heading path. Oversized paragraphs are split by sentence.
   - **Tables:** Are kept whole where possible. Large tables are split by rows with the header repeated in every piece.
   - **Overlap:** Is added between neighbouring chunks within a section, and skipped around tables.
4. **LLM enrichment:** Each chunk gets a headline, a short summary, and, if it contains tabular data, one fact per line rebuilt from the table. The original text is never rewritten. Results are cached in `enrich_cache.json`, so re-ingesting unchanged chunks costs nothing, and a failed call falls back to the raw chunk.
5. **Embed and store:** Chunks are embedded in batches with `text-embedding-3-large` and stored in a persistent ChromaDB collection.

Each file gets a document id from its content hash, so re-uploading an identical file is skipped, and every chunk carries that `doc_id` in its metadata. Files with no extractable text (for example scans) produce a clear error in the upload status.

### 🔎 Retrieval

`search_kb(query, doc_ids)`:

1. Embeds the model-written query.
2. Retrieves the top candidates from ChromaDB, filtered to the session's documents.
3. Reranks them with an LLM. Returned ids are validated, and if reranking fails the vector order is used.
4. Returns the top results with file name and headline so the model can answer and mention its sources.

### 🔌 MCP tool layer

- `mcp_servers/utility_server.py` is an MCP server (FastMCP, stdio transport) that exposes the calculator and weather tools. Tool schemas are generated from the function signatures and docstrings.
- `mcp_client.py` starts the server as a subprocess when the app starts, lists its tools, converts their schemas to the chat-completions tool format, and forwards calls from the chat loop. It runs on its own background event loop so it works with Gradio's synchronous handlers.
- `tools.py` combines these discovered tools with the local, session-scoped `search_kb` tool.
- Tool errors come back to the model as text, so a failing tool doesn't break the conversation.

### 🧪 Evaluation path

`rag/evaluation.py` is a standalone question-answering path for testing retrieval outside the UI. It adds query rewriting, retrieval for both the original and rewritten question with merging, reranking, and grounded answer generation. Run it to check answer quality against known questions.

---

## 📁 Project structure

```text
.
├── app.py                 # Gradio UI: chat, provider dropdown, upload, speech toggle
├── config.py              # API clients, model names, paths, tuning constants, MCP server definitions, Chroma access
├── prompts.py             # System prompt and document-answering rules
├── llm.py                 # Provider clients and the tool-calling chat loop
├── tools.py               # Merges MCP tools with the session-scoped search_kb tool
├── mcp_client.py          # Connects to MCP servers and exposes their tools to the chat loop
├── speech.py              # Groq Orpheus TTS and speech-text cleanup
├── mcp_servers/
│   └── utility_server.py   # MCP server: calculator and weather tools
└── rag/
    ├── schemas.py         # Shared Result model
    ├── loader.py          # Document loading and conversion to text
    ├── chunking.py        # Text cleanup, heading/table-aware chunking, overlap
    ├── enrichment.py      # LLM headline/summary/table-facts enrichment with caching
    ├── ingestion.py       # ingest_file (uploads) and full index rebuild
    ├── retrieval.py       # Scoped vector search, reranking, search_kb
    └── evaluation.py      # Standalone RAG question answering for testing
```

---

## 🧰 Tech stack

| Component | Technology |
|:--|:--|
| Interface | Gradio |
| LLM providers | OpenAI GPT, Groq |
| Tool calling | Provider chat-completions tool calling |
| Tool servers | MCP Python SDK (FastMCP server, stdio client) |
| Weather data | Open-Meteo geocoding and forecast APIs |
| Text-to-speech | Groq-hosted Orpheus |
| Document conversion | pymupdf4llm for PDFs, MarkItDown for other formats |
| Chunk enrichment and reranking | LiteLLM with a small OpenAI model (configurable) |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector database | ChromaDB (persistent) |
| Sentence splitting | NLTK |
| Validation and retries | Pydantic, Tenacity |

---

## 🚀 Getting started

### Prerequisites

- Python 3.12
- An OpenAI API key and a Groq API key

### 📦 Install

```bash
git clone <your-repo-url>
cd <your-repo>
uv sync
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

`uv sync` creates the virtual environment and installs the dependencies declared in `pyproject.toml`.

### 🔑 Configure

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
```

Model names, chunk sizes, retrieval sizes and file paths are set in `config.py`.

### ▶️ Run

```bash
uv run app.py
```

The MCP utility server starts automatically as a subprocess when the app starts, so there is nothing else to run. Open the local address printed in the terminal (Gradio defaults to `http://127.0.0.1:7860`).

---

## 🖥️ Usage

1. Pick a model in the dropdown.
2. Optionally upload one or more documents. The status line shows `Indexed <file>` when each one is ready, or the reason it could not be indexed.
3. Ask questions in the chat, by pressing Enter or clicking Send.
4. Untick **Speak replies** if you don't want audio.
5. **Clear documents** stops the current session from searching its uploaded documents.

### 💡 Things to try

| Ask | What should happen |
|:--|:--|
| A question about an uploaded document | The model calls `search_kb` and answers from the extracts |
| Something the document doesn't contain | The model says it couldn't find it |
| "What is 24 times 6?" with a document loaded | The calculator is used, not document search |
| "What's the weather in Mumbai?" | The weather tool is used |
| The same question in a second browser tab with no uploads | Document search is not available there |

---

## ⚠️ Known limitations

- **Scanned and image-only files** have no extractable text and are rejected with an error. OCR is not implemented.
- **Equations and figures** in documents are not extracted, so questions about them cannot be answered.
- **Replacing a document:** Uploading an edited version of a file adds it as a new document. The old version stays in the index until it is deleted. "Clear documents" only removes it from the current session's scope.
- **Citations** appear in the model's wording only. The UI does not yet show the source passages.
- **TTS** converts the whole reply in a single request.
- **Ingestion speed** depends on the number of chunks, since each is enriched by an LLM call. Cached chunks are instant.

---

## 🗺️ Roadmap

- Speech-to-text input
- An evaluation set of questions across document types, and answer-quality tracking
- Document search as its own MCP service (needs ChromaDB in server mode and session scope injected by the app)
- More tools (as MCP tools where they are stateless): current date and time, list uploaded documents, document overviews, a table query tool for CSV and Excel files, web search, notes and reminders
- Neighbouring-chunk and section reads for wider context
- Hybrid (keyword plus vector) retrieval
- Document replacement and deletion in the UI
- OCR for scanned documents
- Splitting long replies for TTS
