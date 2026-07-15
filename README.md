# Droid Knowledge Assistant for Star Wars Lore

A local **Retrieval-Augmented Generation (RAG)** assistant that answers Star Wars lore questions using a curated knowledge base, semantic search, lexical reranking, and a droid-style response prompt.

This project was built as an exercise to practice:

- Retrieval-Augmented Generation (RAG)
- Prompt engineering
- Local LLM inference with Ollama
- Vector search with ChromaDB
- Embeddings with Sentence Transformers
- Knowledge ingestion and chunking pipelines
- Source-grounded chatbot responses

## License
IMPORTANTE NOTE:

This project code is intended for educational and recreational purposes.

Star Wars, Disney, Wookiepedia and Lucasfilm names and properties belong to their respective owners. 

---

## Demo

Example query:

```text
Who was Anakin's apprentice?
```

Example response:

```text
[GALACTIC ARCHIVE DROID ONLINE]

Archive response:

Anakin Skywalker's apprentice was Ahsoka Tano.

Confidence: Medium

Sources consulted:

- Title: Ahsoka Tano — https://starwars.fandom.com/wiki/Ahsoka_Tano
```

<img width="1104" height="760" alt="Droid_knowledge_assistant" src="https://github.com/user-attachments/assets/c575763c-4df4-4b64-bfef-1203f337dbce" />

## Features
- Local RAG assistant for Star Wars lore
- Hybrid retrieval:
  - semantic search with ChromaDB
  - lexical matching and reranking
  - relation-aware retrieval improvements
- Local LLM support through Ollama
- Optional OpenAI API support
- Prompt-based droid personality
- Source-aware answers
- Local vectorstore generation
- Reproducible ingestion and chunking pipeline

The project is designed to run locally and does not require a paid API key when using Ollama. Of course, if you have an API key you can also use it. c: 
---

## Repository Structure

```text
droid-knowledge-assistant/
├── data/
│   ├── manual/
│   ├── raw/              # ignored by Git
│   ├── processed/        # ignored by Git
│   └── vectorstore/      # ignored by Git
├── docs/
├── evaluations/
├── ingestion/
│   ├── fetch_wookieepedia_pages.py
│   ├── chunk_documents.py
│   ├── selected_pages.txt
│   └── selected_pages_full.txt
├── prompts/
│   ├── droid_rag_prompt.md
│   ├── system_prompt_final.md
│   └── system_prompt_v1.md
├── rag/
│   ├── build_vectorstore.py
│   ├── search_vectorstore.py
│   └── droid_rag_cli.py
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- Api key or Ollama installed locally 
- A local Ollama model.

The project was tested using:

```text
llama3.2:3b
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Entov/droid-knowledge-assistant.git
cd droid-knowledge-assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

For local Ollama usage, configure `.env` like this:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```
---

## Running the Full Pipeline

### 1. Fetch selected lore pages

```bash
python ingestion/fetch_wookieepedia_pages.py
```

This creates local raw and processed lore files.

### 2. Chunk the documents

```bash
python ingestion/chunk_documents.py
```

This creates chunked lore documents for retrieval.

### 3. Build the vectorstore

```bash
python -m rag.build_vectorstore
```

This creates a local ChromaDB vectorstore.

### 4. Run the droid assistant

```bash
python -m rag.droid_rag_cli
```

Example:

```text
Query: What is the Death Star? 
```

---

## Local LLM Usage with Ollama

This project can run without an OpenAI API key by using Ollama locally.

Start Ollama and make sure the API is available:

```bash
curl http://localhost:11434/api/tags
```

If running from WSL and Ollama is installed on Windows, use the Windows host IP:

```bash
WINDOWS_HOST=$(ip route | awk '/default/ {print $3}')
curl http://$WINDOWS_HOST:11434/api/tags
```

---

## Optional OpenAI Usage

The assistant can also be configured to use OpenAI models.

Example `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

---

## Data and Attribution

This repository does not include the generated raw, processed, or vectorstore data files.

The ingestion scripts are designed to fetch a small selected set of lore pages for educational purposes.

Original Star Wars lore text may come from Wookieepedia/Fandom pages selected by the user. When using or adapting this project, respect the terms of the source websites and provide attribution to the original pages.

---

## Limitations

- The assistant is only as complete as the selected knowledge base.
- A small local model may sometimes mix unrelated retrieved sources.
- The retriever can be improved with better query expansion and metadata filtering.
- Generated answers should be treated as experimental, not official Star Wars canon references.

---
