
import csv
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

INPUT = Path("data/processed/news_extracted_june_2024.csv")

with INPUT.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

candidates = [
    r for r in rows
    if r.get("extraction_status") == "candidate"
]

print("=== DATASET COUNTS ===")
print("All records:", len(rows))
print("Extraction candidates:", len(candidates))

print("\n=== PUBLISHER COUNTS ===")
for publisher, count in Counter(
    r.get("publisher", "") for r in candidates
).most_common(15):
    print(f"{publisher}: {count}")

print("\n=== PUBLICATION DATE COUNTS ===")
date_counts = Counter(
    r.get("publication_date", "")[:10]
    for r in candidates
)
print("Distinct publication dates:", len(date_counts))
for date, count in sorted(date_counts.items())[:10]:
    print(f"{date or 'MISSING'}: {count}")

print("\n=== EXACT DUPLICATE HEADLINES ===")
headline_counts = Counter(
    " ".join(r.get("headline", "").lower().split())
    for r in candidates
    if r.get("headline", "").strip()
)
duplicates = [
    (headline, count)
    for headline, count in headline_counts.items()
    if count > 1
]
print("Repeated headline groups:", len(duplicates))
for headline, count in sorted(
    duplicates, key=lambda item: item[1], reverse=True
)[:10]:
    print(f"{count} copies: {headline[:180]}")

print("\n=== TEXT LENGTH ===")
lengths = sorted(
    int(r.get("text_characters") or 0)
    for r in candidates
)
if lengths:
    print("Minimum:", lengths[0])
    print("Median:", lengths[len(lengths) // 2])
    print("Maximum:", lengths[-1])

print("\n=== LANGUAGE / SCRIPT SAMPLE ===")
for r in candidates[:20]:
    headline = r.get("headline", "")
    has_devanagari = any(
        "\u0900" <= ch <= "\u097f" for ch in headline
    )
    has_gujarati = any(
        "\u0a80" <= ch <= "\u0aff" for ch in headline
    )
    language_hint = (
        "Devanagari" if has_devanagari
        else "Gujarati" if has_gujarati
        else "Other/Latin"
    )
    print(f"{language_hint}: {headline[:120]}")

print("\nAudit complete.")
