"""Create a separate structured event-eligibility review template."""

from pathlib import Path
import pandas as pd

REGISTRY = Path("data/processed/event_registry_updated.csv")
OUTPUT = Path("data/processed/event_eligibility_review.csv")

events = pd.read_csv(REGISTRY, dtype={"event_id": "string"})

required = {
    "event_id",
    "event_date",
    "event_category",
    "target_ticker",
    "event_name",
}
missing = required - set(events.columns)
if missing:
    raise ValueError(f"Registry missing columns: {sorted(missing)}")

if events["event_id"].isna().any() or events["event_id"].duplicated().any():
    raise ValueError("Event IDs must be present and unique.")

review = events[
    ["event_id", "event_date", "event_category", "target_ticker", "event_name"]
].copy()

review["confounder_review_status"] = "not_reviewed"
review["confounder_notes"] = ""
review["reviewer_decision"] = "pending_review"
review["reviewed_at"] = ""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
review.to_csv(OUTPUT, index=False)

print("Events in review template:", len(review))
print("\nInitial decisions:")
print(review["reviewer_decision"].value_counts().to_string())
print(f"\nSaved: {OUTPUT}")
print("The event registry was not modified.")
