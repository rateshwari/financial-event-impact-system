"""Build a separate, auditable article-to-event mapping proposal."""

from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/manual_event_review_merged.csv")
REGISTRY = Path("data/processed/event_registry_updated.csv")
OUTPUT = Path("data/processed/article_event_mapping.csv")

RALLY_CLUSTER = "EXIT_POLL_RALLY_2024-06-03"
CONTEXT_CLUSTER = "ELECTION_CONTEXT_2024"
SBI_CLUSTER = "SBI_MCAP_MILESTONE_2024-06-03"
RALLY_EVENT_ID = "EVT_2024_06_03_001"

df = pd.read_csv(INPUT)
registry = pd.read_csv(REGISTRY, dtype={"event_id": "string"})

required = {
    "article_url",
    "headline",
    "relevance_decision",
    "duplicate_cluster_id",
    "event_category",
    "publication_date",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Input is missing columns: {sorted(missing)}")

if df["article_url"].isna().any() or df["article_url"].duplicated().any():
    raise ValueError("Article URLs must be present and unique.")

if RALLY_EVENT_ID not in set(registry["event_id"]):
    raise ValueError(f"Event {RALLY_EVENT_ID} is absent from the registry.")

result = df[
    [
        "article_url",
        "headline",
        "publisher",
        "publication_date",
        "relevance_decision",
        "event_category",
        "duplicate_cluster_id",
    ]
].copy()

result["event_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
result["article_role"] = "unassigned"
result["mapping_confidence"] = "not_assessed"
result["mapping_notes"] = "No mapping assigned by the current rules."

rally = (
    result["relevance_decision"].eq("relevant")
    & result["duplicate_cluster_id"].eq(RALLY_CLUSTER)
)
context = (
    result["relevance_decision"].eq("relevant")
    & result["duplicate_cluster_id"].eq(CONTEXT_CLUSTER)
)
sbi = result["duplicate_cluster_id"].eq(SBI_CLUSTER)

result.loc[rally, "event_id"] = RALLY_EVENT_ID
result.loc[rally, "article_role"] = "event_evidence"
result.loc[rally, "mapping_confidence"] = "provisional"
result.loc[rally, "mapping_notes"] = (
    "Assigned to one provisional market-wide rally cluster; "
    "publication-time and confounder review remains open."
)

result.loc[context, "article_role"] = "contextual"
result.loc[context, "mapping_confidence"] = "provisional"
result.loc[context, "mapping_notes"] = (
    "Market outlook/background coverage; not a separate primary event."
)

result.loc[sbi, "article_role"] = "unassigned"
result.loc[sbi, "mapping_confidence"] = "low"
result.loc[sbi, "mapping_notes"] = (
    "Requires review; SBI is outside the predefined five-stock universe."
)

if result["article_url"].duplicated().any():
    raise ValueError("Output contains duplicate article URLs.")

if result.loc[rally, "event_id"].nunique(dropna=True) != 1:
    raise ValueError("Rally articles do not map to exactly one event.")

if not result.loc[context, "event_id"].isna().all():
    raise ValueError("Context articles must not receive a primary event ID.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("Input articles:", len(result))
print("Unique article URLs:", result["article_url"].nunique())
print("\nArticle-role counts:")
print(result["article_role"].value_counts(dropna=False).to_string())
print("\nMapped event counts:")
print(result.groupby("event_id", dropna=False).size().to_string())
print(f"\nSaved mapping proposal: {OUTPUT}")
print("Registry and source review files were not modified.")
