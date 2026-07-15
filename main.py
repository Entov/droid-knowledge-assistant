import json
import re
from pathlib import Path


DATA_PATH = Path("data/manual/starter_lore.json")


def load_lore_database(path):
    """Load the local Star Wars lore database from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    """Convert text to lowercase and remove non-alphanumeric symbols."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúñü\s-]", " ", text)
    return text


def retrieve_relevant_entries(question, database, top_k=3):
    """
    Retrieve the most relevant lore entries using a simple keyword score.

    This is intentionally simple for the first version of the project.
    Later, this can be replaced with embeddings and vector search.
    """
    normalized_question = normalize_text(question)
    question_terms = set(normalized_question.split())

    scored_entries = []

    for entry in database:
        searchable_text = " ".join([
            entry.get("title", ""),
            entry.get("category", ""),
            entry.get("continuity", ""),
            entry.get("summary", ""),
            " ".join(entry.get("tags", []))
        ])

        normalized_entry = normalize_text(searchable_text)
        entry_terms = set(normalized_entry.split())

        score = len(question_terms.intersection(entry_terms))

        if score > 0:
            scored_entries.append((score, entry))

    scored_entries.sort(key=lambda item: item[0], reverse=True)

    return [entry for score, entry in scored_entries[:top_k]]


def generate_droid_response(question, retrieved_entries):
    """
    Generate a simple droid-style answer using retrieved context.
    This version does not call an LLM yet.
    """
    print("\n[GALACTIC ARCHIVE DROID ONLINE]")
    print(f"\nQuery received: {question}")

    if not retrieved_entries:
        print("\nArchive response:")
        print(
            "No sufficiently relevant record was found in my current local archive. "
            "Recommendation: expand the knowledge base or refine the query."
        )
        print("\nConfidence: Low")
        return

    print("\nArchive response:")

    for index, entry in enumerate(retrieved_entries, start=1):
        print(f"\nRecord {index}: {entry['title']}")
        print(f"Category: {entry['category']}")
        print(f"Continuity: {entry['continuity']}")
        print(f"Summary: {entry['summary']}")
        print(f"Source: {entry['source']}")

    print("\nConfidence: Based on local archive match")
    print("Note: This assistant currently uses a small manual knowledge base.")


def main():
    database = load_lore_database(DATA_PATH)

    print("[DROID KNOWLEDGE ASSISTANT FOR STAR WARS LORE]")
    print("Type your question. Type 'exit' to shut down the archive droid.\n")

    while True:
        question = input("User query: ")

        if question.lower().strip() in ["exit", "quit", "salir"]:
            print("\n[ARCHIVE DROID OFFLINE]")
            break

        retrieved_entries = retrieve_relevant_entries(question, database)
        generate_droid_response(question, retrieved_entries)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()