import csv
import re
from collections import Counter
from pathlib import Path

SOURCE = Path("data/processed/news_review_report.csv")
OUTPUT = Path("data/processed/company_news_candidates.csv")

COMPANIES = {
    "HDFCBANK": [
        r"\bhdfc bank\b",
        r"\bhdfc\b",
    ],
    "ICICIBANK": [
        r"\bicici bank\b",
        r"\bicici\b",
    ],
    "INFY": [
        r"\binfosys\b",
    ],
    "RELIANCE": [
        r"\breliance industries\b",
        r"\breliance industries limited\b",
        r"\breliance\b",
    ],
    "TCS": [
        r"\btata consultancy services\b",
        r"\btcs\b",
    ],
}

compiled = {
    ticker: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for ticker, patterns in COMPANIES.items()
}

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

results = []

for row in rows:
    if row.get("extraction_status") != "candidate":
        continue

    headline = row.get("headline", "")
    body = row.get("article_text", "")
    text = f"{headline}\n{body}"

    matched = []
    for ticker, patterns in compiled.items():
        if any(pattern.search(text) for pattern in patterns):
            matched.append(ticker)

    if not matched:
        continue

    publication_date = row.get("publication_date", "").strip()
    date_present = bool(publication_date)
    in_june = publication_date.startswith("2024-06-")

    reviewed = dict(row)
    reviewed["matched_tickers"] = ";".join(matched)
    reviewed["company_match_count"] = len(matched)
    reviewed["has_publication_date"] = date_present
    reviewed["publication_date_in_june"] = in_june
    reviewed["candidate_screen_status"] = (
        "needs_manual_review"
        if row.get("review_flags")
        else "screened_candidate"
    )
    results.append(reviewed)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if results:
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

print("Input records:", len(rows))
print("Company-matching extraction candidates:", len(results))
print("Matches by ticker:")
counts = Counter(
    ticker
    for row in results
    for ticker in row["matched_tickers"].split(";")
)
for ticker, count in sorted(counts.items()):
    print(f"  {ticker}: {count}")

print(
    "Candidates with dates:",
    sum(row["has_publication_date"] == True for row in results),
)
print(
    "Candidates dated in June 2024:",
    sum(row["publication_date_in_june"] == True for row in results),
)
print("Saved:", OUTPUT)
