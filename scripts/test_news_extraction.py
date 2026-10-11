
import csv
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

URLS = [
    (
        "Economic Times",
        "https://economictimes.indiatimes.com/markets/stocks/"
        "stock-watch/stock-market-update-mining-stocks-down-as-market-falls/"
        "articleshow/110705050.cms",
    ),
    (
        "Livemint",
        "https://www.livemint.com/market/stock-market-news/"
        "nifty-bank-falls-by-5-000-points-sbi-hdfc-bank-icici-bank-among-top-laggards-"
        "11717498043244.html",
    ),
    (
        "The Hindu",
        "https://www.thehindu.com/business/markets/"
        "bloodbath-on-d-street-on-vote-counting-day-sensex-suffers-worst-single-day-"
        "retreat-in-4-years/article68251060.ece",
    ),
    (
        "Financial Express",
        "https://www.financialexpress.com/market/"
        "washout-session-for-markets-nifty-ends-below-22000-and-sensex-plunges-4390-"
        "points-nifty-bank-and-midcap-100-tank-almost-8-3513745/",
    ),
    (
        "Moneycontrol",
        "https://www.moneycontrol.com/news/india/"
        "netherlands-emerges-as-indias-3rd-largest-export-destination-in-2023-24-12740776.html",
    ),
]

OUTPUT = Path("data/processed/news_extraction_sample.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
}

rows = []

with requests.Session() as session:
    session.headers.update(headers)

    for publisher, url in URLS:
        print(f"\nProcessing: {publisher}")

        row = {
            "publisher": publisher,
            "url": url,
            "headline": "",
            "article_text": "",
            "http_status": "",
            "extraction_status": "",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            response = session.get(url, timeout=20)
            row["http_status"] = response.status_code

            if response.status_code != 200:
                row["extraction_status"] = "http_error"
            else:
                html = response.text
                extracted = trafilatura.bare_extraction(
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=True,
                )
                metadata = trafilatura.extract_metadata(html)

                headline = (
                    metadata.title
                    if metadata and metadata.title
                    else (extracted.title if extracted else None)
                )
                body = extracted.text if extracted and extracted.text else ""

                row["headline"] = headline or ""
                row["article_text"] = body

                if headline and len(body) >= 300:
                    row["extraction_status"] = "candidate"
                else:
                    row["extraction_status"] = "needs_review"

        except requests.RequestException as exc:
            row["extraction_status"] = (
                f"request_error: {type(exc).__name__}"
            )

        rows.append(row)
        print(
            f"Status: {row['extraction_status']} | "
            f"Headline: {row['headline'] or 'MISSING'} | "
            f"Text characters: {len(row['article_text'])}"
        )

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} records to {OUTPUT}")
