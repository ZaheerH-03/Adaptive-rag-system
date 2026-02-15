# The Complete RAG Walkthrough: From Document to Answer 🎓

You wanted to understand *everything*. Here is the map of your codebase (Refactored).

## I. The Foundation (Defining Data)

### 1. `core/document.py` 📄
**Purpose**: Defines *what* our data looks like.
- **`DocUnit`**: Represents a raw piece of a file (e.g., "Page 1 of PDF"). It has a `text`, `filename`, and a `uuid` (unique ID).
- **`Chunk`**: Represents a processed piece of text. It has `text`, `metadata`, and a link to its parent (`parent_id`). It *also* has its own `uuid`.
- **KEY**: This is where `uuid.uuid4()` happens. This is the "birth certificate" of every piece of data.

### 2. `core/interfaces.py` 📐
**Purpose**: The "Blueprints". It doesn't do anything itself; it tells other files *how* to behave.
- **`Retriever`**: "Anyone who wants to be a Retriever MUST have a `retrieve(query)` method."
- **`Generator`**: "Anyone who wants to be an LLM MUST have a `generate(prompt)` method."
- **`Index`**: "Anyone who stores data MUST differ `add()` and `query()`."

### 3. `config.py` ⚙️
**Purpose**: The Control Panel.
- Has paths (`CHROMA_DIR`), model names (`LLAMA_MODEL_NAME`), and settings (chunk size `SEM_MAX_CHARS`).
- Change things here to affect the whole app.

---

## II. The Factory (Ingestion & Indexing)

### 4. `ingest/loaders.py` 🚚
**Purpose**: The Delivery Trucks.
- Functions like `load_pdf_units`, `load_docx_units`.
- **Action**: Reads a file from disk -> Returns a list of `DocUnit` objects.

### 5. `ingest/semantic_chunking.py` ✂️
**Purpose**: The Smart Cutter.
- **Action**: Takes `DocUnit`s (pages) -> Returns `Chunk`s (paragraphs).
- **Logic**: It compares the *meaning* of sentences. If the meaning changes drastically, it cuts.

### 6. `index/storage.py` 📦 (Moved from `rag/storage.py`)
**Purpose**: The Warehouse Manager (ChromaDB Wrapper).
- **Class**: `ChromaIndex`.
- **Action**: It talkes to the database. It takes documents and IDs and puts them into ChromaDB.
- **Why?**: It hides the ugly database code from the rest of the app.

### 7. `index/build_index.py` 🏗️
**Purpose**: The Factory Floor Manager (The Main Script).
- **Action**:
    1.  Calls **Loaders** to get pages.
    2.  Calls **Chunker** to get paragraphs.
    3.  Calls **Embedder** to turn text into numbers.
    4.  Calls **Storage** (`index.add`) to save it all.
- **Run this**: Only when you add new files.

---

## III. The Service (Retrieval & Answering)

### 8. `retrieval/retriever.py` 🐕 (Moved from `rag/retriever.py`)
**Purpose**: The Fetcher.
- **Class**: `StandardRetriever`.
- **Action**: Takes a user query ("What is OB?") -> embeddings -> Asks `ChromaIndex` for the top 5 matches.
- **Output**: A list of relevant chunks.

### 9. `generation/llm_wrapper.py` 🧠 (Moved from `rag/llm_wrapper.py`)
**Purpose**: The Brain (LLM).
- **Class**: `Llama32Local`.
- **Action**: Loads the Llama model. Takes a prompt string -> Generates a text answer.

### 10. `pipeline/engine.py` ⚙️ (Moved from `rag/engine.py`)
**Purpose**: The Coordinator.
- **Class**: `RAGEngine`.
- **Action**:
    1.  `retrieve()`: Calls the **Retriever**.
    2.  `build_prompt()`: Pastes the retrieved text into a "You are a teacher..." template.
    3.  `answer()`: Sends the prompt to the **Generator** (LLM).

### 11. `app.py` 🚀
**Purpose**: The Front Desk.
- **Action**:
    1.  Sets up the **Retriever** (Standard).
    2.  Sets up the **Generator** (Llama).
    3.  Creates the **Engine**.
    4.  Asks a question and prints the answer.

---

## Visual Flow
`User` -> `app.py` -> `RAGEngine` -> `Retriever`
                                  |
                                  v
                               `Index (Chroma)`
                                  |
                                  v
                               `Generator (LLM)` -> `Answer`
