
import csv
from pathlib import Path

path = Path("data/processed/manual_event_review.csv")

keywords = (
    "exit poll", "election result", "election results",
    "sensex", "nifty", "market crash", "market mayhem",
    "stock market", "india vix", "lok sabha"
)

with path.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

selected = [
    row for row in rows
    if row.get("publication_date", "").startswith(
        ("2024-06-03", "2024-06-04")
    )
    and any(
        word in (
            row.get("headline", "") + " " +
            row.get("article_text", "")[:1500]
        ).lower()
        for word in keywords
    )
]

selected.sort(key=lambda r: (
    r.get("publication_date", ""),
    r.get("headline", "")
))

print("Selected articles:", len(selected))

for i, row in enumerate(selected[:12], start=1):
    print(f"\n{'=' * 60}")
    print(f"{i}. {row.get('headline', '')}")
    print("Published:", row.get("publication_date", ""))
    print("Publisher:", row.get("publisher", ""))
    print("Ticker matches:", row.get("matched_tickers", ""))
    print("Article text:")
    print((row.get("article_text", "") or "")[:1200])
    print("URL:", row.get("article_url", ""))
