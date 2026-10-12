"""Build a reproducible, auditable article-to-event mapping proposal."""

from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/manual_event_review_merged.csv")
REGISTRY = Path("data/processed/event_registry_updated.csv")
OUTPUT = Path("data/processed/article_event_mapping.csv")

RALLY_CLUSTER = "EXIT_POLL_RALLY_2024-06-03"
CONTEXT_CLUSTER = "ELECTION_CONTEXT_2024"
SBI_CLUSTER = "SBI_MCAP_MILESTONE_2024-06-03"
JUNE4_CLUSTER = "ELECTION_RESULT_SELL_OFF_2024-06-04"

RALLY_EVENT_ID = "EVT_2024_06_03_001"
JUNE4_EVENT_ID = "EVT_2024_06_04_001"

df = pd.read_csv(INPUT, dtype={"article_url": "string"})
registry = pd.read_csv(REGISTRY, dtype={"event_id": "string"})

required = {
    "article_url", "headline", "publisher", "publication_date",
    "relevance_decision", "duplicate_cluster_id", "event_category",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Input is missing columns: {sorted(missing)}")

if df["article_url"].isna().any() or df["article_url"].duplicated().any():
    raise ValueError("Article URLs must be present and unique.")

registered_ids = set(registry["event_id"].dropna())
for event_id in (RALLY_EVENT_ID, JUNE4_EVENT_ID):
    if event_id not in registered_ids:
        raise ValueError(f"Event {event_id} is absent from the registry.")

result = df[
    [
        "article_url", "headline", "publisher", "publication_date",
        "relevance_decision", "event_category", "duplicate_cluster_id",
    ]
].copy()

result["event_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
result["article_role"] = "unassigned"
result["mapping_confidence"] = "not_assessed"
result["mapping_notes"] = "No mapping assigned by the current rules."

relevant = result["relevance_decision"].eq("relevant")

rally = relevant & result["duplicate_cluster_id"].eq(RALLY_CLUSTER)
context = relevant & result["duplicate_cluster_id"].eq(CONTEXT_CLUSTER)
sbi = result["duplicate_cluster_id"].eq(SBI_CLUSTER)
june4 = relevant & result["duplicate_cluster_id"].eq(JUNE4_CLUSTER)

# June 3 rally: retain existing provisional assignment.
result.loc[rally, "event_id"] = RALLY_EVENT_ID
result.loc[rally, "article_role"] = "event_evidence"
result.loc[rally, "mapping_confidence"] = "provisional"
result.loc[rally, "mapping_notes"] = (
    "Assigned to one provisional market-wide rally cluster; "
    "publication-time and confounder review remains open."
)

# June 3 background articles remain contextual, not primary event evidence.
result.loc[context, "article_role"] = "contextual"
result.loc[context, "mapping_confidence"] = "provisional"
result.loc[context, "mapping_notes"] = (
    "Market outlook/background coverage; not a separate primary event."
)

# SBI milestone remains unassigned.
result.loc[sbi, "article_role"] = "unassigned"
result.loc[sbi, "mapping_confidence"] = "low"
result.loc[sbi, "mapping_notes"] = (
    "Requires review; SBI is outside the predefined five-stock universe."
)

# June 4: explicitly distinguish contemporaneous evidence from corroboration.
june4_evidence = june4 & result["publisher"].eq("businesstoday.in")
june4_context = june4 & result["publisher"].eq("thehindu.com")

result.loc[june4, "event_id"] = JUNE4_EVENT_ID
result.loc[june4, "mapping_confidence"] = "provisional"
result.loc[june4, "mapping_notes"] = (
    "Provisional June 4 election-counting market episode. "
    "First public-information timestamp and confounders remain unresolved."
)

result.loc[june4_evidence, "article_role"] = "event_evidence"
result.loc[june4_evidence, "mapping_notes"] = (
    "Contemporaneous event-evidence candidate. Publication timestamp does "
    "not establish the first public-information time."
)

result.loc[june4_context, "article_role"] = "contextual"
result.loc[june4_context, "mapping_notes"] = (
    "Corroborating coverage of the same June 4 episode, not an independent "
    "event. Publication timestamp does not establish the first public time."
)

# Fail closed if a new article enters this cluster without a reviewed role.
unclassified_june4 = june4 & ~(
    june4_evidence | june4_context
)
if unclassified_june4.any():
    headlines = result.loc[unclassified_june4, "headline"].tolist()
    raise ValueError(
        "June 4 articles need explicit role review before mapping: "
        f"{headlines}"
    )

if result["article_url"].duplicated().any():
    raise ValueError("Output contains duplicate article URLs.")

assigned_ids = set(result["event_id"].dropna())
unknown_ids = assigned_ids - registered_ids
if unknown_ids:
    raise ValueError(
        "Mapping contains event IDs absent from the registry: "
        f"{sorted(unknown_ids)}"
    )

if result.loc[rally, "event_id"].nunique(dropna=True) != 1:
    raise ValueError("Rally articles do not map to exactly one event.")

if not result.loc[context, "event_id"].isna().all():
    raise ValueError("Context articles must not receive a primary event ID.")

if result.loc[june4, "event_id"].nunique(dropna=True) != 1:
    raise ValueError("June 4 articles do not map to exactly one event.")

if result.loc[june4_context, "event_id"].isna().any():
    raise ValueError("June 4 contextual article must retain episode linkage.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("Input articles:", len(result))
print("Unique article URLs:", result["article_url"].nunique())
print("\nArticle-role counts:")
print(result["article_role"].value_counts(dropna=False).to_string())
print("\nMapped event counts:")
print(result.groupby("event_id", dropna=False).size().to_string())
print("\nJune 4 assignments:")
print(
    result.loc[
        june4,
        ["publisher", "article_role", "event_id", "mapping_confidence"],
    ].to_string(index=False)
)
print(f"\nSaved mapping proposal: {OUTPUT}")
print("Event eligibility is not established by article mapping alone.")
