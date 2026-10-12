"""Pure decision rules for financial event-study eligibility."""

import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ALLOWED_CATEGORIES = {"market_wide", "corporate_event"}
ALLOWED_TICKERS = {
    "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "RELIANCE.NS", "TCS.NS", "MARKET",
}
REVIEW_STATUSES = {
    "not_reviewed",
    "reviewed_with_concerns",
    "reviewed_no_major_confounder",
}
REVIEW_DECISIONS = {"pending_review", "eligible", "excluded"}
UNRESOLVED = {"", "nan", "none", "nat", "unresolved", "unknown"}


def clean(value):
    """Convert missing values to an empty string and trim text."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def evaluate_event_eligibility(
    event,
    audit,
    manual,
    mapped_evidence_count,
    event_date_is_trading_date,
    available_tickers,
):
    """Return a decision and reasons for one event.

    Explicit exclusions take precedence over pending-review conditions.
    Eligibility requires complete timing evidence, an in-scope target,
    mapped event evidence, and a completed review without major concerns.
    """
    pending = []
    exclusions = []

    category = clean(event.get("event_category"))
    ticker = clean(event.get("target_ticker"))

    if category not in ALLOWED_CATEGORIES:
        exclusions.append(f"Unsupported event category: {category}")

    if ticker not in ALLOWED_TICKERS:
        exclusions.append(f"Target outside predefined universe: {ticker}")
    elif ticker != "MARKET" and ticker not in available_tickers:
        exclusions.append(f"No return series for target: {ticker}")

    if ticker == "MARKET" and category != "market_wide":
        pending.append("Review market target/category consistency.")

    if not event_date_is_trading_date:
        pending.append("Event date is not an observed trading date.")

    if mapped_evidence_count < 1:
        pending.append("No article mapped as event evidence.")

    confidence = clean(audit.get("timestamp_confidence")).lower()
    timestamp = clean(audit.get("first_public_timestamp"))
    timezone = clean(audit.get("timestamp_timezone"))
    source_url = clean(audit.get("timestamp_source_url"))
    evidence = clean(audit.get("timestamp_evidence"))
    date_basis = clean(audit.get("event_date_basis")).lower()
    timing_decision = clean(
        audit.get("primary_analysis_eligible")
    ).lower()
    timing_exclusion = clean(audit.get("exclusion_reason"))

    if confidence != "high":
        pending.append("Timestamp confidence is not high.")

    if (
        timestamp.lower() in UNRESOLVED
        or timestamp.lower().startswith(("unresolved", "unknown"))
    ):
        pending.append("First-public timestamp is unresolved.")
    elif pd.isna(pd.to_datetime(timestamp, errors="coerce", utc=True)):
        pending.append("First-public timestamp is not a valid datetime.")
    if timezone.lower() in UNRESOLVED:
        pending.append("Timestamp timezone is missing.")
    else:
        try:
            declared_timezone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            pending.append("Timestamp timezone is invalid.")
        else:
            parsed_timestamp = pd.to_datetime(
                timestamp, errors="coerce", utc=False
            )

            if not pd.isna(parsed_timestamp):
                timestamp_datetime = parsed_timestamp.to_pydatetime()

                if (
                    timestamp_datetime.tzinfo is None
                    or timestamp_datetime.utcoffset() is None
                ):
                    pending.append(
                        "First-public timestamp must include an explicit UTC offset."
                    )
                else:
                    declared_offset = timestamp_datetime.astimezone(
                        declared_timezone
                    ).utcoffset()

                    if timestamp_datetime.utcoffset() != declared_offset:
                        pending.append(
                            "Timestamp UTC offset does not match the recorded timezone."
                        )

    if source_url.lower() in UNRESOLVED:
        pending.append("Timestamp source URL is missing.")
    if evidence.lower() in UNRESOLVED:
        pending.append("Timestamp evidence is missing.")
    if date_basis in UNRESOLVED or date_basis == "unverified":
        pending.append("Event-date basis is unverified.")

    if timing_decision == "excluded":
        exclusions.append(
            timing_exclusion or "Timing audit explicitly excludes event."
        )
    elif timing_decision != "eligible":
        pending.append("Timing audit has not approved primary eligibility.")

    confounder_status = clean(
        manual.get("confounder_review_status")
    ).lower()
    confounder_notes = clean(manual.get("confounder_notes"))
    reviewer_decision = clean(manual.get("reviewer_decision")).lower()
    reviewed_at = clean(manual.get("reviewed_at"))

    if confounder_status not in REVIEW_STATUSES:
        pending.append("Invalid or missing confounder-review status.")
    elif confounder_status == "not_reviewed":
        pending.append("Confounder review has not been completed.")
    elif not confounder_notes:
        pending.append("Confounder-review notes are missing.")

    if confounder_status == "reviewed_with_concerns":
        pending.append("Confounder review identifies unresolved concerns.")

    if reviewer_decision not in REVIEW_DECISIONS:
        pending.append("Invalid or missing reviewer decision.")
    elif reviewer_decision == "pending_review":
        pending.append("Reviewer decision is pending.")
    elif reviewer_decision == "excluded":
        exclusions.append("Eligibility review explicitly excludes event.")

    if reviewer_decision in {"eligible", "excluded"} and not reviewed_at:
        pending.append("Completed reviewer decision has no review date.")

    if exclusions:
        decision = "excluded"
    elif pending:
        decision = "pending_review"
    elif (
        reviewer_decision == "eligible"
        and confounder_status == "reviewed_no_major_confounder"
    ):
        decision = "eligible"
    else:
        decision = "pending_review"
        pending.append(
            "Eligibility requires approval and completed confounder review "
            "without major unresolved concerns."
        )

    return {
        "eligibility_decision": decision,
        "exclusion_reasons": " | ".join(dict.fromkeys(exclusions)),
        "pending_reasons": " | ".join(dict.fromkeys(pending)),
    }
