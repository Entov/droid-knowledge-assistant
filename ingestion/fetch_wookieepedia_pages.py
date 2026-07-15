import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


BASE_API_URL = "https://starwars.fandom.com/api.php"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_PAGES_PATH = PROJECT_ROOT / "ingestion" / "selected_pages.txt"
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "wookieepedia_pages"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "lore_documents.json"

REQUEST_DELAY_SECONDS = 1.0
MAX_CHARS_PER_DOCUMENT = 30000

console = Console()


def read_selected_pages(path: Path) -> List[str]:
    """Read manually selected Wookieepedia page titles from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"Selected pages file not found: {path}")

    pages = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            clean_line = line.strip()

            if clean_line and not clean_line.startswith("#"):
                pages.append(clean_line)

    return pages


def build_parse_params(page_title: str) -> Dict[str, str]:
    """Build parameters for a MediaWiki parse API request."""
    return {
        "action": "parse",
        "format": "json",
        "page": page_title,
        "prop": "text|displaytitle",
        "redirects": "1",
        "disabletoc": "1",
    }


def clean_html_to_text(html: str) -> str:
    """Convert parsed wiki HTML into clean plain text."""
    soup = BeautifulSoup(html, "lxml")

    content = soup.select_one(".mw-parser-output")

    if content is None:
        content = soup

    selectors_to_remove = [
        "script",
        "style",
        "noscript",
        "table",
        "sup.reference",
        ".reference",
        ".references",
        ".toc",
        ".portable-infobox",
        ".navbox",
        ".metadata",
        ".mw-editsection",
        ".printfooter",
        ".catlinks",
        ".gallery",
        ".thumb",
        ".wds-tabs",
        ".pi-navigation",
    ]

    for selector in selectors_to_remove:
        for element in content.select(selector):
            element.decompose()

    text_blocks = []

    for element in content.find_all(["p", "h2", "h3", "li"]):
        text = element.get_text(separator=" ", strip=True)

        if not text:
            continue

        lower_text = text.lower()

        stop_sections = [
            "appearances",
            "sources",
            "notes and references",
            "external links",
            "bibliography",
            "see also",
            "references",
        ]

        if lower_text in stop_sections:
            break

        text_blocks.append(text)

    cleaned_text = "\n".join(text_blocks)

    cleaned_text = "\n".join(
        line.strip()
        for line in cleaned_text.splitlines()
        if line.strip()
    )

    if len(cleaned_text) > MAX_CHARS_PER_DOCUMENT:
        cleaned_text = cleaned_text[:MAX_CHARS_PER_DOCUMENT]

    return cleaned_text


def fetch_page(page_title: str) -> Optional[Dict]:
    """
    Fetch a single page from Wookieepedia using the MediaWiki parse API.

    This script only fetches pages listed manually in selected_pages.txt.
    It does not crawl links or download images.
    """
    headers = {
        "User-Agent": (
            "DroidKnowledgeAssistant/0.1 "
            "(educational fan-made project; selected-page ingestion)"
        )
    }

    params = build_parse_params(page_title)

    try:
        response = requests.get(
            BASE_API_URL,
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        console.print(f"[red]Request failed for {page_title}: {error}[/red]")
        return None

    if "error" in data:
        error_info = data["error"]
        console.print(
            f"[yellow]API error for {page_title}: "
            f"{error_info.get('code', 'unknown')} - "
            f"{error_info.get('info', 'no details')}[/yellow]"
        )
        return None

    parse_data = data.get("parse", {})

    if not parse_data:
        console.print(f"[yellow]No parse data found for {page_title}[/yellow]")
        return None

    resolved_title = parse_data.get("title", page_title)
    html = parse_data.get("text", {}).get("*", "")

    if not html.strip():
        console.print(f"[yellow]Empty HTML for {page_title}[/yellow]")
        return None

    source_url = f"https://starwars.fandom.com/wiki/{resolved_title.replace(' ', '_')}"
    retrieved_at = datetime.now(timezone.utc).isoformat()

    raw_record = {
        "requested_title": page_title,
        "resolved_title": resolved_title,
        "source": "Wookieepedia / Fandom",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "raw_html": html,
    }

    return raw_record


def save_raw_record(record: Dict, output_dir: Path) -> None:
    """Save the raw page record as an individual JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_title = record["resolved_title"].replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"{safe_title}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def infer_basic_metadata(title: str, text: str) -> Dict[str, str]:
    """Infer simple category and continuity hints."""
    lower_title = title.lower()
    lower_text = text.lower()

    if "legends" in lower_text or "star wars legends" in lower_text:
        continuity_hint = "Possibly Legends or includes Legends material"
    else:
        continuity_hint = "Unknown / requires verification"

    category_hint = "Unknown"

    if any(word in lower_title for word in ["battle", "war", "siege", "purge"]):
        category_hint = "Event"
    elif any(word in lower_title for word in ["order", "alliance", "empire", "republic", "sith"]):
        category_hint = "Faction or Organization"
    elif any(word in lower_title for word in ["tatooine", "coruscant", "naboo", "mandalore", "mustafar", "kamino", "dagobah", "lothal", "alderaan", "geonosis"]):
        category_hint = "Planet"
    elif any(word in lower_title for word in ["lightsaber", "death star", "falcon", "blaster", "hyperspace"]):
        category_hint = "Technology or Concept"
    else:
        category_keywords = {
            "Character": [
                "jedi",
                "sith",
                "padawan",
                "bounty hunter",
                "senator",
                "general",
                "admiral",
                "mandalorian warrior",
            ],
            "Planet": [
                "planet",
                "world",
                "outer rim",
                "core worlds",
            ],
            "Faction or Organization": [
                "organization",
                "order",
                "alliance",
                "empire",
                "republic",
                "confederacy",
            ],
            "Event": [
                "battle",
                "war",
                "conflict",
                "purge",
                "siege",
            ],
            "Technology or Concept": [
                "weapon",
                "starship",
                "space station",
                "lightsaber",
                "energy field",
                "hyperspace",
            ],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in lower_text for keyword in keywords):
                category_hint = category
                break

    return {
        "category_hint": category_hint,
        "continuity_hint": continuity_hint,
    }


def build_processed_document(raw_record: Dict) -> Optional[Dict]:
    """Convert a raw page record into a cleaned processed document."""
    cleaned_text = clean_html_to_text(raw_record["raw_html"])

    if not cleaned_text.strip():
        console.print(f"[yellow]Empty cleaned text for {raw_record['resolved_title']}[/yellow]")
        return None

    metadata = infer_basic_metadata(raw_record["resolved_title"], cleaned_text)

    return {
        "document_id": raw_record["resolved_title"].replace(" ", "_"),
        "title": raw_record["resolved_title"],
        "source": raw_record["source"],
        "source_url": raw_record["source_url"],
        "retrieved_at": raw_record["retrieved_at"],
        "category_hint": metadata["category_hint"],
        "continuity_hint": metadata["continuity_hint"],
        "text": cleaned_text,
        "character_count": len(cleaned_text),
    }


def save_processed_documents(documents: List[Dict], output_path: Path) -> None:
    """Save all processed documents into a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project": "Droid Knowledge Assistant for Star Wars Lore",
        "description": "Controlled selected-page lore ingestion dataset.",
        "source": "Wookieepedia / Fandom",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "documents": documents,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def print_summary(documents: List[Dict]) -> None:
    """Print a summary table for the ingestion run."""
    table = Table(title="Wookieepedia Selected-Page Ingestion Summary")

    table.add_column("Title", style="cyan")
    table.add_column("Category Hint", style="magenta")
    table.add_column("Continuity Hint", style="green")
    table.add_column("Chars", justify="right")

    for document in documents:
        table.add_row(
            document["title"],
            document["category_hint"],
            document["continuity_hint"],
            str(document["character_count"]),
        )

    console.print(table)
    console.print(f"\n[bold green]Processed dataset saved to:[/bold green] {PROCESSED_OUTPUT_PATH}")


def main() -> None:
    pages = read_selected_pages(SELECTED_PAGES_PATH)

    console.print(f"[bold]Selected pages to ingest:[/bold] {len(pages)}")
    console.print("[yellow]Starting controlled selected-page ingestion with parse API...[/yellow]\n")

    processed_documents = []

    for page_title in tqdm(pages, desc="Fetching pages"):
        raw_record = fetch_page(page_title)

        if raw_record is None:
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        save_raw_record(raw_record, RAW_OUTPUT_DIR)

        processed_document = build_processed_document(raw_record)

        if processed_document is not None:
            processed_documents.append(processed_document)

        time.sleep(REQUEST_DELAY_SECONDS)

    save_processed_documents(processed_documents, PROCESSED_OUTPUT_PATH)
    print_summary(processed_documents)


if __name__ == "__main__":
    main()