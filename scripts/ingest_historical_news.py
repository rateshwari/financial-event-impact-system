
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup

INPUT_FILE = Path("data/raw/gdelt_candidates_june_2024.csv")
OUTPUT_FILE = Path("data/processed/news_extracted_june_2024.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "gdelt_record_date",
    "publisher",
    "article_url",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

def extract_article(session, record):
    result = {
        **record,
        "headline": "",
        "article_text": "",
        "publication_date": "",
        "publication_date_source": "missing",
        "http_status": "",
        "text_characters": 0,
        "extraction_status": "",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    url = record["article_url"].strip()

    if not url.startswith(("http://", "https://")):
        result["extraction_status"] = "invalid_url"
        return result

    try:
        response = session.get(url, timeout=20)
        result["http_status"] = response.status_code

        if response.status_code != 200:
            result["extraction_status"] = "http_error"
            return result

        html = response.text

        article = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        extracted_metadata = trafilatura.extract_metadata(html)

        headline = (
            extracted_metadata.title
            if extracted_metadata and extracted_metadata.title
            else (article.title if article else None)
        )

        body = article.text if article and article.text else ""

        # Prefer explicit publication metadata from the publisher.
        soup = BeautifulSoup(html, "html.parser")

        publication_date = ""
        publication_date_source = "missing"

        for property_name in (
            "article:published_time",
            "article:published",
        ):
            tag = soup.find("meta", property=property_name)
            if tag and tag.get("content", "").strip():
                publication_date = tag["content"].strip()
                publication_date_source = "publisher_metadata"
                break

        # Fall back to other common publisher metadata fields.
        if not publication_date:
            for attribute, field_name in (
                ("property", "og:published_time"),
                ("name", "pubdate"),
                ("name", "publishdate"),
                ("name", "date"),
                ("name", "parsely-pub-date"),
                ("itemprop", "datePublished"),
            ):
                tag = soup.find(
                    "meta", attrs={attribute: field_name}
                )
                if tag and tag.get("content", "").strip():
                    publication_date = tag["content"].strip()
                    publication_date_source = "publisher_metadata"
                    break

        # Use Trafilatura only if no publisher date was found.
        if not publication_date and extracted_metadata:
            if extracted_metadata.date:
                publication_date = extracted_metadata.date
                publication_date_source = "trafilatura_fallback"

        result["headline"] = headline or ""
        result["article_text"] = body
        result["text_characters"] = len(body)
        result["publication_date"] = publication_date
        result["publication_date_source"] = publication_date_source

        if not headline:
            result["extraction_status"] = "missing_headline"
        elif len(body) < 300:
            result["extraction_status"] = "short_text"
        else:
            result["extraction_status"] = "candidate"

    except requests.RequestException as exc:
        result["extraction_status"] = (
            f"request_error_{type(exc).__name__}"
        )
    except Exception as exc:
        result["extraction_status"] = (
            f"extraction_error_{type(exc).__name__}"
        )

    return result



def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing expected columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        records = list(reader)

    # Keep the earliest record for each URL, preserving source metadata.
    unique_records = []
    seen_urls = set()

    for record in records:
        url = record["article_url"].strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_records.append(record)

    print(f"Input records: {len(records)}")
    print(f"Unique URLs: {len(unique_records)}")
    print("Starting extraction. Requests are spaced by one second.")

    results = []

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for index, record in enumerate(unique_records, start=1):
            publisher = record.get("publisher", "unknown")
            print(
                f"[{index}/{len(unique_records)}] "
                f"{publisher}: {record['article_url']}"
            )

            result = extract_article(session, record)
            results.append(result)

            print(
                f"  -> {result['extraction_status']}; "
                f"{result['text_characters']} text characters"
            )

            if index < len(unique_records):
                time.sleep(1)

    if results:
        fieldnames = list(results[0].keys())

        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(results)

    counts = {}
    for result in results:
        status = result["extraction_status"]
        counts[status] = counts.get(status, 0) + 1

    print("\n--- Extraction summary ---")
    print(f"Processed: {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
