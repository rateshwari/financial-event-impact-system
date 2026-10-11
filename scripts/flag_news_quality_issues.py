
import csv
from collections import Counter
from pathlib import Path

INPUT = Path("data/processed/news_extracted_june_2024.csv")
OUTPUT = Path("data/processed/news_quality_review.csv")

with INPUT.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

candidates = [
    row for row in rows
    if row.get("extraction_status") == "candidate"
]

headline_counts = Counter(
    " ".join(row.get("headline", "").lower().split())
    for row in candidates
    if row.get("headline", "").strip()
)

for row in candidates:
    issues = []
    publication_date = row.get("publication_date", "")[:10]
    text_length = int(row.get("text_characters") or 0)
    headline = " ".join(row.get("headline", "").lower().split())

    if not publication_date:
        issues.append("missing_publication_date")
    elif not publication_date.startswith("2024-06-"):
        issues.append("publication_date_outside_june_2024")

    if text_length > 20000:
        issues.append("unusually_long_extraction")

    if headline and headline_counts[headline] > 1:
        issues.append("duplicate_exact_headline")

    row["quality_flags"] = ";".join(issues) if issues else "none"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = list(rows[0].keys()) + ["quality_flags"]

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

flagged = [
    row for row in candidates
    if row["quality_flags"] != "none"
]

flag_counts = Counter()
for row in candidates:
    for flag in row["quality_flags"].split(";"):
        if flag and flag != "none":
            flag_counts[flag] += 1

print("Candidate records:", len(candidates))
print("Records with at least one flag:", len(flagged))
print("\nQuality flags:")
for flag, count in flag_counts.most_common():
    print(f"{flag}: {count}")

print(f"\nSaved review file: {OUTPUT}")
print("\nFirst 10 flagged records:")
for row in flagged[:10]:
    print(
        row.get("publication_date", "MISSING"),
        "|", row.get("headline", "")[:100],
        "|", row["quality_flags"],
    )
