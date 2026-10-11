import csv
from collections import Counter
from pathlib import Path

SOURCE = Path("data/processed/news_extracted_june_2024.csv")
OUTPUT = Path("data/processed/news_review_report.csv")

LIVE_PATTERNS = (
    "live updates",
    "live update",
    "live blog",
    "market highlights",
    "share market highlights",
    "stock market highlights",
    "stock market live",
    "share market live",
    "closing bell",
)

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

headline_counts = Counter(
    row.get("headline", "").strip().casefold()
    for row in rows
    if row.get("headline", "").strip()
)

review_rows = []

for row in rows:
    headline = row.get("headline", "").strip()
    headline_lower = headline.casefold()
    body_length = len(row.get("article_text", "").strip())
    publication_date = row.get("publication_date", "").strip()

    flags = []

    if body_length > 20000:
        flags.append("unusually_long")

    if any(pattern in headline_lower for pattern in LIVE_PATTERNS):
        flags.append("possible_live_update_page")

    if headline and headline_counts[headline_lower] > 1:
        flags.append("duplicate_exact_headline")

    if not publication_date:
        flags.append("missing_publication_date")
    elif not publication_date.startswith("2024-06-"):
        flags.append("outside_june_2024_or_needs_date_review")

    reviewed = dict(row)
    reviewed["review_flags"] = ";".join(flags)
    reviewed["review_recommendation"] = (
        "manual_review" if flags else "eligible_for_relevance_screen"
    )
    review_rows.append(reviewed)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if review_rows:
    fieldnames = list(review_rows[0].keys())
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

print("Total records:", len(rows))
print("Records flagged:", sum(bool(r["review_flags"]) for r in review_rows))

flag_counts = Counter()
for row in review_rows:
    for flag in row["review_flags"].split(";"):
        if flag:
            flag_counts[flag] += 1

print("\nFlag counts:")
for flag, count in sorted(flag_counts.items()):
    print(f"  {flag}: {count}")

print("\nSaved:", OUTPUT)
