import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rich.console import Console
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "lore_chunks.json"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

COLLECTION_NAME = "star_wars_lore_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64

console = Console()


def load_chunks(path: Path):
    """Load chunked lore dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["chunks"]


def build_metadata(chunk):
    """Build metadata stored alongside each vector."""
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "title": chunk["title"],
        "category_hint": chunk.get("category_hint", "Unknown"),
        "continuity_hint": chunk.get("continuity_hint", "Unknown"),
        "source": chunk.get("source", "Unknown"),
        "source_url": chunk.get("source_url", ""),
        "chunk_index": chunk.get("chunk_index", 0),
    }


def main():
    console.print("[bold]Loading lore chunks...[/bold]")
    chunks = load_chunks(CHUNKS_PATH)
    console.print(f"Chunks loaded: {len(chunks)}")

    console.print(f"[bold]Loading embedding model:[/bold] {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    existing_collections = [collection.name for collection in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        console.print(f"[yellow]Deleting existing collection:[/yellow] {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Star Wars lore chunks for Droid Knowledge Assistant",
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
    )

    console.print("[yellow]Building embeddings and storing vectors...[/yellow]")

    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[start:start + BATCH_SIZE]

        ids = [chunk["chunk_id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [build_metadata(chunk) for chunk in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    console.print("\n[bold green]Vectorstore built successfully.[/bold green]")
    console.print(f"Collection name: {COLLECTION_NAME}")
    console.print(f"Vectorstore path: {VECTORSTORE_DIR}")
    console.print(f"Stored chunks: {collection.count()}")


if __name__ == "__main__":
    main()