
import csv
from pathlib import Path

INPUT = Path("data/processed/news_extracted_june_2024.csv")

with INPUT.open(encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

print("Articles whose extracted publication date is outside June 2024:\n")

count = 0

for row in rows:
    if row.get("extraction_status") != "candidate":
        continue

    publication_date = row.get("publication_date", "")[:10]

    if not publication_date.startswith("2024-06-"):
        count += 1
        print(f"\nRecord {count}")
        print("Headline:", row.get("headline", ""))
        print("Publisher:", row.get("publisher", ""))
        print("GDELT timestamp:", row.get("gdelt_record_date", ""))
        print("Extracted publication date:", publication_date or "MISSING")
        print("URL:", row.get("article_url", ""))

print(f"\nTotal date mismatches: {count}")
