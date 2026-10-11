
import csv
from collections import Counter
from pathlib import Path

path = Path("data/processed/news_extracted_june_2024.csv")

with path.open(encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

print("Total records:", len(rows))
print("Extraction statuses:")
for status, count in Counter(
    row["extraction_status"] for row in rows
).most_common():
    print(f"  {status}: {count}")

candidates = [
    row for row in rows
    if row["extraction_status"] == "candidate"
]

dated = [
    row for row in candidates
    if row.get("publication_date", "").strip()
]

print("\nCandidate articles:", len(candidates))
print("Candidates with publication dates:", len(dated))
print(
    "Publication-date coverage:",
    f"{len(dated) / len(candidates) * 100:.1f}%"
    if candidates else "N/A"
)

print("\n--- First 10 extraction candidates ---")
for i, row in enumerate(candidates[:10], start=1):
    print(f"\n{i}. {row['headline']}")
    print("Publisher:", row["publisher"])
    print("GDELT timestamp:", row["gdelt_record_date"])
    print("Publication date:", row["publication_date"] or "MISSING")
    print("Text characters:", row["text_characters"])
    print("URL:", row["article_url"])
    print("Text preview:", row["article_text"][:250].replace("\n", " "))
