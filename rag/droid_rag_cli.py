import os
from pathlib import Path
from typing import Dict, List

import chromadb
import requests
from chromadb.config import Settings
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from rag.search_vectorstore import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTORSTORE_DIR,
    deduplicate_by_title,
    load_chunks,
    merge_candidates,
    rerank_candidates,
    retrieve_lexical_candidates,
    retrieve_semantic_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "droid_rag_prompt.md"
ENV_PATH = PROJECT_ROOT / ".env"

FINAL_CONTEXT_CHUNKS = 3

console = Console()


def load_environment() -> None:
    """Load environment variables from .env."""
    load_dotenv(dotenv_path=ENV_PATH)


def load_prompt(path: Path) -> str:
    """Load the droid system prompt from a Markdown file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            "Create prompts/droid_rag_prompt.md before running this script."
        )

    return path.read_text(encoding="utf-8")


def get_llm_provider() -> str:
    """Return the selected LLM provider."""
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def get_openai_model_name() -> str:
    """Read the OpenAI model name from .env."""
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def get_ollama_model_name() -> str:
    """Read the Ollama model name from .env."""
    return os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_ollama_base_url() -> str:
    """Read the Ollama local server URL from .env."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def load_retrieval_system():
    """Load embedding model, local chunks, and ChromaDB collection."""
    console.print("[bold]Loading embedding model...[/bold]")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    console.print("[bold]Loading local lore chunks...[/bold]")
    all_chunks = load_chunks()

    console.print("[bold]Connecting to ChromaDB vectorstore...[/bold]")
    chroma_client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = chroma_client.get_collection(COLLECTION_NAME)

    return embedding_model, all_chunks, collection


def retrieve_context(question: str, embedding_model, all_chunks, collection) -> List[Dict]:
    """Retrieve the most relevant context chunks using the hybrid retriever."""
    semantic_candidates = retrieve_semantic_candidates(
        collection,
        embedding_model,
        question,
    )

    lexical_candidates = retrieve_lexical_candidates(
        question,
        all_chunks,
    )

    candidates = merge_candidates(
        semantic_candidates,
        lexical_candidates,
    )

    reranked = rerank_candidates(
        question,
        candidates,
    )

    final_results = deduplicate_by_title(
        reranked,
        FINAL_CONTEXT_CHUNKS,
    )

    return final_results


def format_context(retrieved_chunks: List[Dict]) -> str:
    """Format retrieved chunks as context for the LLM."""
    context_blocks = []

    for index, item in enumerate(retrieved_chunks, start=1):
        metadata = item["metadata"]
        document = item["document"]

        title = metadata.get("title", "Unknown")
        source_url = metadata.get("source_url", "")
        continuity = metadata.get("continuity_hint", "Unknown")
        category = metadata.get("category_hint", "Unknown")
        chunk_id = metadata.get("chunk_id", "Unknown")

        block = f"""
[Source {index}]
Title: {title}
Chunk ID: {chunk_id}
Category: {category}
Continuity hint: {continuity}
URL: {source_url}

Text:
{document}
""".strip()

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def build_user_message(question: str, retrieved_chunks: List[Dict]) -> str:
    """Build the final user message with retrieved context."""
    context = format_context(retrieved_chunks)

    return f"""
User question:
{question}

Retrieved archive context:
{context}

Instructions:
- Answer in the same language as the user's question.
- Answer using only the retrieved archive context.
- Use only sources that directly answer the question.
- Do not combine unrelated sources.
- Prioritize Source 1 unless another source is clearly more relevant.
- Do not invent unsupported Star Wars lore.
- If the context is insufficient, say so.
""".strip()


def generate_with_openai(system_prompt: str, user_message: str) -> str:
    """Generate a response using OpenAI."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "pega_aqui_tu_api_key":
        raise RuntimeError(
            "OPENAI_API_KEY was not found or still has the placeholder value. "
            "Either add a valid OpenAI API key or set LLM_PROVIDER=ollama."
        )

    client = OpenAI(api_key=api_key)
    model_name = get_openai_model_name()

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        max_output_tokens=700,
    )

    return response.output_text


def generate_with_ollama(system_prompt: str, user_message: str) -> str:
    """Generate a response using a local Ollama model."""
    base_url = get_ollama_base_url()
    model_name = get_ollama_model_name()

    url = f"{base_url}/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    }

    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


def generate_droid_answer(
    provider: str,
    system_prompt: str,
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """Generate a droid-style answer using the selected provider."""
    user_message = build_user_message(question, retrieved_chunks)

    if provider == "ollama":
        return generate_with_ollama(system_prompt, user_message)

    if provider == "openai":
        return generate_with_openai(system_prompt, user_message)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. Use 'ollama' or 'openai'."
    )


def print_retrieval_debug(retrieved_chunks: List[Dict]) -> None:
    """Print compact retrieval information."""
    console.print("\n[bold cyan]Retrieved archive context:[/bold cyan]")

    for index, item in enumerate(retrieved_chunks, start=1):
        metadata = item["metadata"]
        final_score = item.get("final_score", 0)

        title = metadata.get("title", "Unknown")
        source_url = metadata.get("source_url", "")
        source_type = item.get("source_type", "unknown")

        console.print(
            f"[dim]{index}. {title} | score={final_score:.2f} "
            f"| source_type={source_type} | {source_url}[/dim]"
        )


def main() -> None:
    console.print("[bold]Booting Droid RAG Assistant...[/bold]\n")

    load_environment()

    provider = get_llm_provider()
    system_prompt = load_prompt(PROMPT_PATH)

    embedding_model, all_chunks, collection = load_retrieval_system()

    console.print("\n[green]Droid RAG Assistant online.[/green]")
    console.print(f"LLM provider: {provider}")
    console.print("Type a Star Wars lore question. Type 'exit' to quit.\n")

    while True:
        question = input("Query: ").strip()

        if question.lower() in ["exit", "quit", "salir"]:
            console.print("\n[bold][ARCHIVE DROID OFFLINE][/bold]")
            break

        if not question:
            console.print("[yellow]Please enter a question.[/yellow]")
            continue

        retrieved_chunks = retrieve_context(
            question=question,
            embedding_model=embedding_model,
            all_chunks=all_chunks,
            collection=collection,
        )

        print_retrieval_debug(retrieved_chunks)

        console.print("\n[bold yellow]Generating droid response...[/bold yellow]\n")

        try:
            answer = generate_droid_answer(
                provider=provider,
                system_prompt=system_prompt,
                question=question,
                retrieved_chunks=retrieved_chunks,
            )

            console.print(Panel(answer, title="Droid response"))

        except Exception as error:
            console.print("[red]The droid failed to generate a response.[/red]")
            console.print(f"[red]{error}[/red]")

        console.print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
