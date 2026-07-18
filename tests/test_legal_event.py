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
