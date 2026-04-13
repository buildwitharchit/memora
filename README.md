# Memora

Persistent multi-tier memory system for AI assistants. Memora gives LLMs the ability to remember conversations across sessions using a four-tier architecture inspired by human memory: sensory, short-term, semantic, and episodic.

## Architecture

Memora implements four memory tiers that work together on every message:

### Tier 1 -- Sensory Memory (Context Window)

A bounded buffer (`collections.deque`) holding the last N messages (default 8). This is what the LLM directly "sees" on each turn. It has no persistence -- it resets when a new session begins.

### Tier 2 -- Short-Term Memory (Compression Buffer)

When the sensory buffer overflows, the oldest batch of messages is sent to the LLM for summarization. The compressed summary is stored in Tier 3. This tier tracks compression metrics (original vs. summary token counts and compression ratio).

### Tier 3 -- Semantic Memory (ChromaDB)

A vector database of compressed conversation summaries. Each summary is embedded using `sentence-transformers` (all-MiniLM-L6-v2) and stored in ChromaDB. On every new message, the system retrieves the top-K most semantically similar past summaries and injects them into the LLM context.

### Tier 4 -- Episodic Memory (SQLite)

A structured, timestamped event log. After each LLM response, a utility model extracts the user's intent and the interaction outcome, storing them as a row in SQLite. This enables exact queries by date, session, or intent pattern -- complementing Tier 3's fuzzy semantic search.

### Message Flow

```
User sends a message
    |
    v
1. Message added to Tier 1 (sensory buffer)
2. Message embedded --> ChromaDB searched --> top-K summaries retrieved (Tier 3)
3. SQLite queried --> recent episodic entries retrieved (Tier 4)
4. Context assembled: [system prompt] + [summaries] + [episodic log] + [sensory buffer]
5. Context sent to LLM via OpenRouter
6. Response added to Tier 1
7. Intent/outcome extracted via utility LLM --> logged to SQLite (Tier 4)
8. If sensory buffer full --> oldest batch compressed --> stored in ChromaDB (Tier 3)
```

## Quick Start

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

### Local Setup

```bash
# Clone the repository.
git clone https://github.com/your-username/memora.git
cd memora

# Create and activate a virtual environment.
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies.
pip install -r requirements.txt

# Configure environment.
cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY.

# Run the application.
streamlit run ui/app.py
```

The app will be available at `http://localhost:8501`.

### Docker

```bash
# Build the image (pre-downloads the embedding model).
docker build -t memora .

# Run with your API key.
docker run -p 8501:8501 -e OPENROUTER_API_KEY=sk-or-your-key memora
```

### Docker Compose

```bash
# Set your API key in the environment.
export OPENROUTER_API_KEY=sk-or-your-key

# Start the service (data persists in a named volume).
docker-compose up
```

## Configuration

All settings are configured via environment variables. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `PRIMARY_MODEL` | `google/gemma-3-12b-it` | LLM for conversation responses |
| `UTILITY_MODEL` | `google/gemma-3-12b-it` | LLM for summarization and intent extraction |
| `SENSORY_MAX_MESSAGES` | `8` | Max messages in the sensory buffer before compression triggers |
| `COMPRESSION_BATCH_SIZE` | `4` | Number of messages compressed per batch |
| `SEMANTIC_TOP_K` | `3` | Number of similar summaries to retrieve |
| `EPISODIC_SNIPPET_SIZE` | `5` | Number of recent episodic entries injected into context |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `STORE_PATH` | `./store` | Root directory for ChromaDB and SQLite data |

## UI Overview

### Chat Tab

The main conversation interface. The left panel shows the chat; the right panel is a live memory inspector displaying the state of all four tiers in real time:

- **Tier 1 (Sensory):** Current buffer contents and fill level
- **Tier 2 (Short-Term):** Compression event log with token counts and ratios
- **Tier 3 (Semantic):** Number of stored summaries and last retrieved results
- **Tier 4 (Episodic):** Total entry count and recent interaction log

### Episodic Log Tab

A sortable, filterable table of every logged interaction. Filter by session ID or intent substring to explore behavioral patterns over time.

### Semantic Browser Tab

A manual similarity search interface. Enter any phrase to find the most relevant stored summaries with their distance scores. When no search query is entered, browse all stored summaries.

### Sidebar Controls

- **New Session:** Clears the sensory buffer and starts a fresh session ID. Persistent tiers (3 and 4) retain their data for cross-session recall.
- **Reset Buttons:** Individually reset Tier 2 (compression log), Tier 3 (all stored summaries), or Tier 4 (all episodic entries).

## Running Tests

All tests use mocks -- no API calls, model downloads, or external services required.

```bash
# Activate your virtual environment.
source .venv/bin/activate

# Run all tests with verbose output.
pytest tests/ -v
```

### Test Coverage

| Module | Tests | What's Verified |
|---|---|---|
| `config` | 6 | Settings defaults, validation, required fields |
| `llm_client` | 4 | OpenRouter delegation, model selection |
| `sensory_memory` | 9 | Buffer operations, overflow behavior, no silent drops |
| `semantic_memory` | 9 | ChromaDB operations, embedding, search, reset |
| `episodic_memory` | 12 | SQLite CRUD, filtering, ordering |
| `short_term_memory` | 6 | Compression flow, metric tracking |
| `context_builder` | 7 | Message ordering, optional tier injection |
| `memory_manager` | 11 | Full pipeline, compression triggers, error handling |

## Project Structure

```
memora/
  core/
    config.py               # Pydantic settings loaded from environment
    llm_client.py           # OpenRouter API wrapper (openai SDK)
    sensory_memory.py       # Tier 1 -- bounded message buffer (deque)
    short_term_memory.py    # Tier 2 -- compression buffer with LLM summarization
    semantic_memory.py      # Tier 3 -- ChromaDB vector store with sentence-transformers
    episodic_memory.py      # Tier 4 -- SQLite structured event log
    context_builder.py      # Assembles LLM prompt from all tiers
    memory_manager.py       # Central orchestrator wiring all tiers together
  ui/
    app.py                  # Streamlit entrypoint (tab router)
    components/
      chat_tab.py           # Chat interface + memory inspector sidebar
      episodic_tab.py       # Filterable episodic log table
      semantic_tab.py       # Manual ChromaDB similarity search
      memory_controls.py    # Session and tier reset buttons
  tests/
    conftest.py             # Shared pytest fixtures (mocked LLM, memory tiers)
    test_config.py
    test_llm_client.py
    test_sensory_memory.py
    test_short_term_memory.py
    test_semantic_memory.py
    test_episodic_memory.py
    test_context_builder.py
    test_memory_manager.py
  store/                    # Persistent data (ChromaDB, SQLite) -- gitignored
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```

## Design Decisions

**No maxlen on the deque.** Python's `deque(maxlen=N)` silently drops the oldest element on append. Memora uses an unbounded deque with explicit overflow handling so that messages are always routed through compression (Tier 2) before being discarded.

**Dual chat history.** The Streamlit display history (`st.session_state["chat_history"]`) is separate from the sensory buffer. Users see the full conversation in the UI while the LLM receives a compressed sliding window. Messages only leave the display when the user starts a new session.

**Token count approximation.** Compression ratios use `len(text.split())` as a word-level proxy for token counts. This avoids adding a tokenizer dependency for what is an advisory metric.

**All dependencies injectable.** Every core class accepts its dependencies via constructor parameters and falls back to real implementations when none are provided. This makes the test suite entirely mock-based with zero external calls.

**Semantic + Episodic separation.** ChromaDB handles fuzzy semantic retrieval ("what past discussions are relevant to this?"). SQLite handles exact structured queries ("show all interactions from session X" or "how many times did the user ask about recursion?"). Together they cover all retrieval patterns.
