
import csv
from pathlib import Path

SOURCE = Path("data/processed/manual_event_review.csv")
OUTPUT = Path("data/processed/event_review_batch1.csv")

# Match articles by URL fragments rather than their row positions.
labels = {
    "ahead-of-market-10-things": (
        "relevant", "MARKET", "analyst_commentary",
        "Election-result outlook; retain as contextual coverage.",
        "ELECTION_CONTEXT_2024", "medium",
        "Contains observed June 3 market figures and forecasts."
    ),
    "another-1000-point-rally": (
        "exclude", "", "analyst_commentary",
        "Forecast of a possible future Nifty rally.",
        "", "medium",
        "Prediction, not an independently observed event."
    ),
    "fiis-buy-shares-worth-rs-6851": (
        "relevant", "MARKET", "market_wide",
        "June 3 market rally and reported institutional net purchases.",
        "EXIT_POLL_RALLY_2024-06-03", "high",
        "Broad market report; not a company-specific event."
    ),
    "half-of-sensex-stocks-one-third": (
        "exclude", "", "analyst_commentary",
        "Technical-market commentary on stocks below their 200-day average.",
        "", "medium",
        "Background technical analysis, not the target event."
    ),
    "lok-sabha-election-2024-india-stocks-hit-record": (
        "relevant", "MARKET", "market_wide",
        "Indian market rally following exit-poll projections.",
        "EXIT_POLL_RALLY_2024-06-03", "high",
        "Broad market event; Reliance keyword match is not a company event."
    ),
    "moneycontrol-selects-top-stories": (
        "exclude", "", "irrelevant_or_uncertain",
        "Roundup containing multiple unrelated stories.",
        "", "medium",
        "Do not treat the roundup as a standalone event."
    ),
    "reliancefinancials-push-sensex-nifty": (
        "relevant", "MARKET", "market_wide",
        "June 3 market rally following exit-poll projections.",
        "EXIT_POLL_RALLY_2024-06-03", "high",
        "Reliance is mentioned as a market contributor, but this is a market-wide event."
    ),
    "sbi-seventh-company-8-lakh-crore": (
        "review", "SBIN", "corporate_event",
        "SBI crosses the reported ₹8 lakh crore market-capitalisation milestone.",
        "SBI_MCAP_MILESTONE_2024-06-03", "high",
        "SBI is outside the current five-stock study universe."
    ),
    "stock-market-update-fertilisers-stocks-up": (
        "exclude", "", "market_wide",
        "Fertiliser-stock performance during the broad market rally.",
        "", "high",
        "No identifiable event for the selected target companies."
    ),
    "stock-market-update-mining-stocks-up": (
        "exclude", "", "market_wide",
        "Mining-stock performance during the broad market rally.",
        "", "high",
        "No identifiable event for the selected target companies."
    ),
    "stock-market-update-nifty-it-index-advances": (
        "exclude", "", "market_wide",
        "IT-sector performance during the broad market rally.",
        "", "high",
        "Market summary; not an independent Infosys or TCS event."
    ),
    "stock-market-update-power-stocks-up": (
        "exclude", "", "market_wide",
        "Power-stock performance during the broad market rally.",
        "", "high",
        "Market summary; no direct event for the selected companies."
    ),
}

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

reviewed = []
unmatched = []

for row in rows:
    url = row.get("article_url", "")
    match = next(
        (key for key in labels if key in url),
        None
    )
    if not match:
        continue

    (
        decision, ticker, category, description,
        cluster, confidence, notes
    ) = labels[match]

    row["relevance_decision"] = decision
    row["verified_target_ticker"] = ticker
    row["event_category"] = category
    row["event_description"] = description
    row["duplicate_cluster_id"] = cluster
    row["timestamp_confidence"] = confidence
    row["review_notes"] = notes

    reviewed.append(row)

if len(reviewed) != len(labels):
    found = {
        key for key in labels
        if any(key in r.get("article_url", "") for r in reviewed)
    }
    unmatched = sorted(set(labels) - found)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
if reviewed:
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviewed)

print("Articles labeled:", len(reviewed))
print("Expected:", len(labels))
print("Unmatched URL patterns:", unmatched)
print("Output:", OUTPUT)
print("Source file unchanged.")
