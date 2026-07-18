from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import legal_event


def test_event_pressure_respects_recorded_scenario_probabilities_and_pair_exposure() -> None:
    event = legal_event.validate_events(legal_event.synthetic_events(), legal_event.load_schema())[0]
    pressure = legal_event.event_pressure(event)
    enacted_impact = 0.70 + 0.20 * 0.35 - 0.10 * 0.50
    assert pressure[legal_event.PAIRS.index("EURUSD")] == pytest.approx(0.8 * enacted_impact)
    assert pressure[legal_event.PAIRS.index("USDJPY")] == pytest.approx(-0.4 * enacted_impact)
    assert np.count_nonzero(pressure) == 2


def test_event_study_starts_strictly_after_known_time() -> None:
    events = legal_event.validate_events(legal_event.synthetic_events(), legal_event.load_schema())
    times, prices = legal_event.synthetic_input()
    study, _summary = legal_event.run_event_study(
        times, prices, events, legal_event.EventStudyConfig(horizon_steps=20, pre_event_baseline_steps=2))
    assert not study.empty
    assert (study["prediction_time"] > events[0]["known_at"]).all()
    assert (study["target_time"] > study["prediction_time"]).all()


def test_abnormal_return_uses_scaled_pre_event_expected_return() -> None:
    events = legal_event.validate_events(legal_event.synthetic_events(), legal_event.load_schema())
    times = np.arange(20, dtype=np.int64) * legal_event.DT_NS
    # Event known at minute 3 starts at minute 4.  With two pre-event and four
    # post-event minutes, constant 1bp/min drift implies a 4bp expected return.
    log_prices = np.arange(20, dtype=np.float64)[:, None] * 0.01
    log_prices = np.repeat(log_prices, len(legal_event.PAIRS), axis=1)
    eurusd = legal_event.PAIRS.index("EURUSD")
    log_prices[8, eurusd] += 0.02
    study, _summary = legal_event.run_event_study(
        times, log_prices, events, legal_event.EventStudyConfig(horizon_steps=4, pre_event_baseline_steps=2))
    row = study.loc[study["pair"] == "EURUSD"].iloc[0]
    assert row["pre_event_log_return"] == pytest.approx(0.02)
    assert row["expected_post_event_log_return"] == pytest.approx(0.04)
    assert row["post_event_log_return"] == pytest.approx(0.06)
    assert row["baseline_adjusted_abnormal_log_return"] == pytest.approx(0.02)


def test_assessment_provenance_is_hashed_sealed_and_append_only() -> None:
    schema = legal_event.load_schema()
    event = legal_event.synthetic_events()[0]
    validated = legal_event.validate_events([event], schema)[0]
    assert validated["assessment_sha256"] == legal_event.canonical_assessment_hash(validated)

    hindsight = legal_event.synthetic_events()[0]
    hindsight["sealed_before_market_data_through"] = "1970-01-01T00:04:00Z"
    hindsight["assessment_sha256"] = legal_event.canonical_assessment_hash(hindsight)
    with pytest.raises(legal_event.ContractError, match="future market data"):
        legal_event.validate_events([hindsight], schema)

    tampered = legal_event.synthetic_events()[0]
    tampered["pair_exposures"]["EURUSD"] = 0.1
    with pytest.raises(legal_event.ContractError, match="immutable assessment content"):
        legal_event.validate_events([tampered], schema)

    parent = legal_event.synthetic_events()[0]
    child = copy.deepcopy(parent)
    child["event_id"] = "synthetic-rule-002"
    child["source_document_id"] = "synthetic-primary-002"
    child["published_at"] = "1970-01-01T00:04:00Z"
    child["known_at"] = "1970-01-01T00:04:00Z"
    child["assessment_created_at"] = "1970-01-01T00:04:00Z"
    child["sealed_before_market_data_through"] = "1970-01-01T00:04:00Z"
    child["parent_assessment_sha256"] = parent["assessment_sha256"]
    child["assessment_sha256"] = legal_event.canonical_assessment_hash(child)
    ledger = legal_event.validate_events([parent, child], schema)
    assert ledger[1]["parent_assessment_sha256"] == parent["assessment_sha256"]


def test_future_citation_and_conflicting_duplicate_are_rejected() -> None:
    schema = legal_event.load_schema()
    first = legal_event.synthetic_events()[0]
    future = copy.deepcopy(first)
    future["event_id"] = "synthetic-rule-002"
    future["source_document_id"] = "synthetic-primary-002"
    future["published_at"] = "1970-01-01T00:10:00Z"
    future["known_at"] = "1970-01-01T00:10:00Z"
    first["citations"] = [future["source_document_id"]]
    with pytest.raises(legal_event.ContractError, match="not known"):
        legal_event.validate_events([first, future], schema)

    duplicate = legal_event.synthetic_events()[0]
    conflict = copy.deepcopy(duplicate)
    conflict["pair_exposures"]["EURUSD"] = -0.8
    conflict["assessment_sha256"] = legal_event.canonical_assessment_hash(conflict)
    with pytest.raises(legal_event.ContractError, match="conflicting duplicate"):
        legal_event.validate_events([duplicate, conflict], schema)


def test_gap_crossing_event_window_is_not_scored() -> None:
    events = legal_event.validate_events(legal_event.synthetic_events(), legal_event.load_schema())
    times, prices = legal_event.synthetic_input()
    # The event is known at minute 3 and starts at minute 4; create an observed
    # gap inside its 20-minute target window.
    times[15:] += legal_event.DT_NS
    study, summary = legal_event.run_event_study(
        times, prices, events, legal_event.EventStudyConfig(horizon_steps=20, pre_event_baseline_steps=2))
    assert study.empty
    assert summary["event_pair_observations"] == 0


def test_missing_primary_source_corpus_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(legal_event.ContractError, match="no legal-event corpus"):
        legal_event.load_events(tmp_path / "absent.jsonl")
