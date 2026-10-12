"""Generate an event-study eligibility report without editing source files."""

from pathlib import Path
import pandas as pd

from financial_events.eligibility import (
    evaluate_event_eligibility,
    clean,
)

BASE = Path("data/processed")
REGISTRY_PATH = BASE / "event_registry_updated.csv"
TIMING_PATH = BASE / "event_timing_audit.csv"
MAPPING_PATH = BASE / "article_event_mapping.csv"
RETURNS_PATH = BASE / "daily_returns.csv"
REVIEW_PATH = BASE / "event_eligibility_review.csv"
OUTPUT_PATH = BASE / "event_eligibility_report.csv"


def require_columns(frame, required, label):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


registry = pd.read_csv(REGISTRY_PATH, dtype={"event_id": "string"})
timing = pd.read_csv(TIMING_PATH, dtype={"event_id": "string"})
mapping = pd.read_csv(MAPPING_PATH, dtype={"event_id": "string"})
review = pd.read_csv(REVIEW_PATH, dtype={"event_id": "string"})
returns = pd.read_csv(RETURNS_PATH, parse_dates=["date"])

require_columns(
    registry,
    ["event_id", "event_date", "event_category", "target_ticker",
     "event_name", "event_status"],
    "Registry",
)
require_columns(
    timing,
    ["event_id", "first_public_timestamp", "timestamp_timezone",
     "timestamp_source_url", "timestamp_evidence", "timestamp_confidence",
     "event_date_basis", "primary_analysis_eligible", "exclusion_reason"],
    "Timing audit",
)
require_columns(mapping, ["article_url", "event_id", "article_role"], "Mapping")
require_columns(returns, ["date", "ticker", "daily_return"], "Returns")
require_columns(
    review,
    ["event_id", "confounder_review_status", "confounder_notes",
     "reviewer_decision", "reviewed_at"],
    "Eligibility review",
)

for label, frame in [
    ("registry", registry),
    ("timing audit", timing),
    ("eligibility review", review),
]:
    if frame["event_id"].isna().any() or frame["event_id"].duplicated().any():
        raise ValueError(f"{label}: event IDs must be present and unique.")

if mapping["article_url"].isna().any() or mapping["article_url"].duplicated().any():
    raise ValueError("Mapping has missing or duplicate article URLs.")
if returns["date"].isna().any():
    raise ValueError("Returns contain missing dates.")
if returns.duplicated(["date", "ticker"]).any():
    raise ValueError("Returns contain duplicate date/ticker observations.")

event_ids = set(registry["event_id"])
if set(timing["event_id"]) != event_ids:
    raise ValueError("Registry and timing audit event IDs do not match.")
if set(review["event_id"]) != event_ids:
    raise ValueError("Registry and eligibility review event IDs do not match.")

registry["event_date"] = pd.to_datetime(
    registry["event_date"], errors="coerce"
).dt.normalize()
if registry["event_date"].isna().any():
    raise ValueError("Registry contains invalid event dates.")

returns["date"] = returns["date"].dt.normalize()
trading_dates = set(returns["date"].dropna().unique())
available_tickers = set(returns["ticker"].dropna().astype(str))
timing_by_id = timing.set_index("event_id")
review_by_id = review.set_index("event_id")

rows = []
for _, event in registry.iterrows():
    event_id = event["event_id"]
    audit = timing_by_id.loc[event_id]
    manual = review_by_id.loc[event_id]

    mapped_count = int(
        (
            mapping["event_id"].eq(event_id)
            & mapping["article_role"].eq("event_evidence")
        ).sum()
    )

    decision = evaluate_event_eligibility(
        event=event.to_dict(),
        audit=audit.to_dict(),
        manual=manual.to_dict(),
        mapped_evidence_count=mapped_count,
        event_date_is_trading_date=event["event_date"] in trading_dates,
        available_tickers=available_tickers,
    )

    rows.append({
        "event_id": event_id,
        "event_date": event["event_date"].date().isoformat(),
        "event_category": event["event_category"],
        "target_ticker": event["target_ticker"],
        "event_name": event["event_name"],
        "mapped_evidence_articles": mapped_count,
        "exact_trading_date": event["event_date"] in trading_dates,
        "timestamp_confidence": audit["timestamp_confidence"],
        "confounder_review_status": manual["confounder_review_status"],
        "reviewer_decision": manual["reviewer_decision"],
        **decision,
    })

result = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("Events evaluated:", len(result))
print("\nEligibility decisions:")
print(result["eligibility_decision"].value_counts().to_string())
print("\nDetailed report:")
print(result[
    ["event_id", "event_date", "mapped_evidence_articles",
     "eligibility_decision", "exclusion_reasons", "pending_reasons"]
].to_string(index=False))
print(f"\nSaved: {OUTPUT_PATH}")
print("Source datasets were not modified.")
