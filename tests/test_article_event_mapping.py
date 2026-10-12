from pathlib import Path

import pandas as pd


DATA = Path("data/processed")
MAPPING_PATH = DATA / "article_event_mapping.csv"
SOURCE_PATH = DATA / "manual_event_review_merged.csv"
REGISTRY_PATH = DATA / "event_registry_updated.csv"

RALLY_CLUSTER = "EXIT_POLL_RALLY_2024-06-03"
CONTEXT_CLUSTER = "ELECTION_CONTEXT_2024"
SBI_CLUSTER = "SBI_MCAP_MILESTONE_2024-06-03"
RALLY_EVENT_ID = "EVT_2024_06_03_001"


def load_files():
    return (
        pd.read_csv(MAPPING_PATH),
        pd.read_csv(SOURCE_PATH),
        pd.read_csv(REGISTRY_PATH),
    )


def test_mapping_preserves_all_source_articles():
    mapping, source, _ = load_files()
    assert len(mapping) == len(source)
    assert mapping["article_url"].is_unique
    assert set(mapping["article_url"]) == set(source["article_url"])


def test_rally_articles_map_to_one_provisional_event():
    mapping, _, registry = load_files()
    rally = mapping[mapping["duplicate_cluster_id"] == RALLY_CLUSTER]

    assert len(rally) == 3
    assert rally["event_id"].notna().all()
    assert set(rally["event_id"]) == {RALLY_EVENT_ID}
    assert set(rally["article_role"]) == {"event_evidence"}
    assert RALLY_EVENT_ID in set(registry["event_id"])


def test_context_articles_have_no_primary_event_id():
    mapping, _, _ = load_files()
    context = mapping[mapping["duplicate_cluster_id"] == CONTEXT_CLUSTER]

    assert len(context) == 2
    assert context["event_id"].isna().all()
    assert set(context["article_role"]) == {"contextual"}


def test_sbi_milestone_remains_unassigned():
    mapping, _, _ = load_files()
    sbi = mapping[mapping["duplicate_cluster_id"] == SBI_CLUSTER]

    assert len(sbi) == 1
    assert sbi["event_id"].isna().all()
    assert set(sbi["article_role"]) == {"unassigned"}


def test_all_assigned_event_ids_exist_in_registry():
    mapping, _, registry = load_files()
    assigned = set(mapping["event_id"].dropna())
    registered = set(registry["event_id"].dropna())

    assert assigned <= registered
