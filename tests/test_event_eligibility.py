from financial_events.eligibility import evaluate_event_eligibility


def make_case():
    event = {
        "event_id": "SYNTHETIC_001",
        "event_category": "corporate_event",
        "target_ticker": "INFY.NS",
    }
    audit = {
        "timestamp_confidence": "high",
        "first_public_timestamp": "2025-01-15T09:00:00+05:30",
        "timestamp_timezone": "Asia/Kolkata",
        "timestamp_source_url": "https://example.com/article",
        "timestamp_evidence": "Verified original publication metadata.",
        "event_date_basis": "first_public_disclosure",
        "primary_analysis_eligible": "eligible",
        "exclusion_reason": "",
    }
    manual = {
        "confounder_review_status": "reviewed_no_major_confounder",
        "confounder_notes": "Reviewed relevant releases; none identified.",
        "reviewer_decision": "eligible",
        "reviewed_at": "2025-01-20",
    }
    kwargs = {
        "event": event,
        "audit": audit,
        "manual": manual,
        "mapped_evidence_count": 1,
        "event_date_is_trading_date": True,
        "available_tickers": {"INFY.NS", "TCS.NS"},
    }
    return kwargs


def test_complete_in_scope_event_can_be_eligible():
    result = evaluate_event_eligibility(**make_case())
    assert result["eligibility_decision"] == "eligible"
    assert result["pending_reasons"] == ""
    assert result["exclusion_reasons"] == ""


def test_missing_timestamp_keeps_event_pending():
    case = make_case()
    case["audit"]["timestamp_confidence"] = "partial"

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "Timestamp confidence is not high." in result["pending_reasons"]


def test_unresolved_confounder_concerns_keep_event_pending():
    case = make_case()
    case["manual"]["confounder_review_status"] = "reviewed_with_concerns"
    case["manual"]["confounder_notes"] = "Potential overlapping macroeconomic news."

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "unresolved concerns" in result["pending_reasons"]


def test_out_of_scope_ticker_is_excluded():
    case = make_case()
    case["event"]["target_ticker"] = "SBIN"

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "excluded"
    assert "outside predefined universe" in result["exclusion_reasons"]


def test_missing_event_evidence_keeps_event_pending():
    case = make_case()
    case["mapped_evidence_count"] = 0

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "No article mapped as event evidence." in result["pending_reasons"]


def test_malformed_timestamp_keeps_event_pending():
    case = make_case()
    case["audit"]["first_public_timestamp"] = "not-a-real-timestamp"

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "timestamp" in result["pending_reasons"].lower()


def test_unresolved_timestamp_cannot_pass_with_high_confidence():
    case = make_case()
    case["audit"]["first_public_timestamp"] = (
        "Unresolved — earliest disclosure not verified"
    )

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "timestamp" in result["pending_reasons"].lower()


def test_missing_review_date_keeps_event_pending():
    case = make_case()
    case["manual"]["reviewed_at"] = ""

    result = evaluate_event_eligibility(**case)

    assert result["eligibility_decision"] == "pending_review"
    assert "review date" in result["pending_reasons"].lower()
