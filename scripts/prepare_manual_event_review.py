import csv
from pathlib import Path

SOURCE = Path("data/processed/company_news_candidates.csv")
OUTPUT = Path("data/processed/manual_event_review.csv")

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys()) if rows else []

extra_fields = [
    "relevance_decision",
    "verified_target_ticker",
    "event_category",
    "event_description",
    "duplicate_cluster_id",
    "timestamp_confidence",
    "review_notes",
]

for field in extra_fields:
    if field not in fieldnames:
        fieldnames.append(field)

for row in rows:
    row.setdefault("relevance_decision", "")
    row.setdefault("verified_target_ticker", "")
    row.setdefault("event_category", "")
    row.setdefault("event_description", "")
    row.setdefault("duplicate_cluster_id", "")
    row.setdefault("timestamp_confidence", "")
    row.setdefault("review_notes", "")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Articles prepared for manual review:", len(rows))
print("Review file:", OUTPUT)
print("Original candidate file unchanged.")
