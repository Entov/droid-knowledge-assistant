import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "lore_documents.json"
CHUNKS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lore_chunks.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

console = Console()


def load_documents(path: Path) -> Dict:
    """Load processed lore documents."""
    if not path.exists():
        raise FileNotFoundError(f"Processed documents file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    This first version uses characters instead of tokens to keep the
    implementation simple and transparent.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def build_chunks(dataset: Dict) -> List[Dict]:
    """Build chunk records from processed documents."""
    all_chunks = []

    for document_position, document in enumerate(dataset["documents"], start=1):
        document_id = document["document_id"]
        title = document["title"]
        text = document["text"]

        text_chunks = split_text_into_chunks(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for index, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = f"doc_{document_position:04d}_{document_id}_chunk_{index:04d}"

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": title,
                "category_hint": document.get("category_hint", "Unknown"),
                "continuity_hint": document.get("continuity_hint", "Unknown"),
                "source": document.get("source", "Unknown"),
                "source_url": document.get("source_url", ""),
                "retrieved_at": document.get("retrieved_at", ""),
                "chunk_index": index,
                "text": chunk_text,
                "character_count": len(chunk_text),
            }

            all_chunks.append(chunk_record)

    return all_chunks


def save_chunks(chunks: List[Dict], output_path: Path) -> None:
    """Save all chunks into a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project": "Droid Knowledge Assistant for Star Wars Lore",
        "description": "Chunked lore dataset for retrieval-augmented generation.",
        "source_dataset": str(DOCUMENTS_PATH),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def print_summary(chunks: List[Dict]) -> None:
    """Print a summary of the chunking process."""
    chunks_by_title = {}

    for chunk in chunks:
        title = chunk["title"]
        chunks_by_title[title] = chunks_by_title.get(title, 0) + 1

    table = Table(title="Lore Chunking Summary")
    table.add_column("Title", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for title, count in sorted(chunks_by_title.items()):
        table.add_row(title, str(count))

    console.print(table)
    console.print(f"\n[bold green]Chunk dataset saved to:[/bold green] {CHUNKS_OUTPUT_PATH}")
    console.print(f"[bold green]Total chunks:[/bold green] {len(chunks)}")


def main() -> None:
    console.print("[bold]Loading processed lore documents...[/bold]")

    dataset = load_documents(DOCUMENTS_PATH)

    console.print(f"Documents loaded: {dataset['document_count']}")
    console.print("[yellow]Building chunks...[/yellow]\n")

    chunks = build_chunks(dataset)

    save_chunks(chunks, CHUNKS_OUTPUT_PATH)
    print_summary(chunks)


if __name__ == "__main__":
    main()