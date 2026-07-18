"""Causal legal/regulatory FX event-study research engine.

The engine validates timestamped primary-source legal-event records, enforces
citation-time causality, translates recorded scenario probabilities and FX
exposures into diagnostic event-pressure vectors, and measures only subsequent
gap-safe canonical-BID event-window responses.  It is not legal advice, an NLP
legal interpreter, an outcome forecaster, a valuation engine, or an execution
system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import (ContractError as SharedContractError, canonical_pair_order,
                       contiguous_60s, validate_generated_manifest)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"
EVENT_SCHEMA_PATH = ROOT / "config" / "legal-event-schema.json"
DEFAULT_EVENT_PATH = ROOT / "data_events" / "legal_events.jsonl"
PAIRS = canonical_pair_order(ROOT)
N_PAIRS = len(PAIRS)
DT_NS = 60_000_000_000
VERSION = "legal-event-arena-0.1.0"
SCENARIO_IMPACT = {
    "withdrawn": 0.0,
    "enacted": 1.0,
    "diluted": 0.35,
    "injunction": -0.50,
    "appeal_success": -0.60,
    "aggressive_enforcement": 1.25,
    "loophole": -0.25,
}


class ContractError(RuntimeError):
    """Raised when event lineage, timing, or canonical-data contracts fail."""


@dataclass(frozen=True)
class EventStudyConfig:
    horizon_steps: int = 60
    pre_event_baseline_steps: int = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def parse_utc(value: object, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} is not a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ContractError(f"{field} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def load_schema(path: Path = EVENT_SCHEMA_PATH) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"tracked legal event schema is missing: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise ContractError(f"tracked legal event schema is unreadable: {path}") from exc
    if schema.get("schema_version") != "fxsim-legal-event-v1":
        raise ContractError("unsupported legal event schema_version")
    for field in ("legal_stages", "scenarios", "required_event_fields"):
        if not isinstance(schema.get(field), list) or not all(isinstance(item, str) for item in schema[field]):
            raise ContractError(f"legal event schema has invalid {field}")
    if set(schema["scenarios"]) != set(SCENARIO_IMPACT):
        raise ContractError("legal event schema scenarios do not match the fixed scenario-impact table")
    return schema


def canonical_event_hash(event: dict[str, Any]) -> str:
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_events(events: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate immutable known-time records and their citation DAG."""
    if not events:
        raise ContractError("event corpus contains no records")
    by_id: dict[str, dict[str, Any]] = {}
    document_times: dict[str, pd.Timestamp] = {}
    normalized: list[dict[str, Any]] = []
    allowed_stages = set(schema["legal_stages"])
    allowed_scenarios = set(schema["scenarios"])
    required = set(schema["required_event_fields"])
    for raw in events:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ContractError("legal event record lacks required fields")
        event = dict(raw)
        event_id = event["event_id"]
        document_id = event["source_document_id"]
        if not isinstance(event_id, str) or not event_id or not isinstance(document_id, str) or not document_id:
            raise ContractError("event_id and source_document_id must be nonempty strings")
        if event["legal_stage"] not in allowed_stages:
            raise ContractError(f"event {event_id} has unsupported legal_stage")
        published_at = parse_utc(event["published_at"], "published_at")
        known_at = parse_utc(event["known_at"], "known_at")
        if published_at > known_at:
            raise ContractError(f"event {event_id} has known_at before published_at")
        probabilities = event["scenario_probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != allowed_scenarios:
            raise ContractError(f"event {event_id} must provide every fixed scenario probability")
        values = np.asarray(list(probabilities.values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-9):
            raise ContractError(f"event {event_id} scenario probabilities must be finite, nonnegative, and sum to one")
        exposures = event["pair_exposures"]
        if not isinstance(exposures, dict) or not exposures or not set(exposures).issubset(PAIRS):
            raise ContractError(f"event {event_id} pair_exposures must name canonical pairs")
        exposure_values = np.asarray(list(exposures.values()), dtype=np.float64)
        if not np.isfinite(exposure_values).all() or np.any(np.abs(exposure_values) > 1.0):
            raise ContractError(f"event {event_id} pair exposures must be finite values in [-1, 1]")
        citations = event.get("citations", [])
        if not isinstance(citations, list) or not all(isinstance(citation, str) and citation for citation in citations):
            raise ContractError(f"event {event_id} citations must be a string list")
        if document_id in citations:
            raise ContractError(f"event {event_id} cannot cite its own source document")
        event["published_at"] = published_at
        event["known_at"] = known_at
        event["citations"] = citations
        event["content_sha256"] = canonical_event_hash(raw)
        if event_id in by_id:
            if by_id[event_id]["content_sha256"] != event["content_sha256"]:
                raise ContractError(f"conflicting duplicate event_id: {event_id}")
            continue
        if document_id in document_times and document_times[document_id] != known_at:
            raise ContractError(f"source document {document_id} has conflicting known_at values")
        by_id[event_id] = event
        document_times[document_id] = known_at
        normalized.append(event)
    for event in normalized:
        for citation in event["citations"]:
            if citation not in document_times:
                raise ContractError(f"event {event['event_id']} cites unknown document {citation}")
            if document_times[citation] > event["known_at"]:
                raise ContractError(f"event {event['event_id']} cites a document not known at event time")
    return sorted(normalized, key=lambda event: (event["known_at"], event["event_id"]))


def load_events(path: Path) -> list[dict[str, Any]]:
    schema = load_schema()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ContractError(
            f"no legal-event corpus at {path}; supply timestamped primary-source JSONL records"
        ) from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON at event corpus line {line_number}") from exc
        records.append(value)
    return validate_events(records, schema)


def event_pressure(event: dict[str, Any]) -> np.ndarray:
    """Scenario-weighted directional diagnostic pressure in canonical pair order."""
    expected_impact = sum(float(event["scenario_probabilities"][scenario]) * multiplier
                          for scenario, multiplier in SCENARIO_IMPACT.items())
    pressure = np.zeros(N_PAIRS, dtype=np.float64)
    for index, pair in enumerate(PAIRS):
        pressure[index] = expected_impact * float(event["pair_exposures"].get(pair, 0.0))
    return pressure


def _strict_contiguous_window(times: np.ndarray, start: int, end: int) -> bool:
    return 0 <= start < end < len(times) and bool(np.all(np.diff(times[start:end + 1]) == DT_NS))


def run_event_study(times: np.ndarray, log_prices: np.ndarray, events: list[dict[str, Any]],
                    config: EventStudyConfig = EventStudyConfig()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure post-known-time canonical responses without a price-side signal loop."""
    if times.ndim != 1 or log_prices.shape != (len(times), N_PAIRS):
        raise ContractError("event study requires synchronous ten-pair log prices")
    if not np.all(np.diff(times) > 0) or not np.isfinite(log_prices).all():
        raise ContractError("event study prices must be finite and strictly increasing")
    if config.horizon_steps < 1 or config.pre_event_baseline_steps < 1:
        raise ContractError("event-study horizons must be positive")
    records: list[dict[str, Any]] = []
    for event in events:
        # Strictly after known_at: never let a bar that closed at the same instant
        # carry a document whose intra-bar availability is unknown.
        start = int(np.searchsorted(times, event["known_at"].value, side="right"))
        end = start + config.horizon_steps
        baseline_start = start - config.pre_event_baseline_steps
        if not _strict_contiguous_window(times, baseline_start, end):
            continue
        pressure = event_pressure(event)
        post_return = log_prices[end] - log_prices[start]
        baseline_return = log_prices[start] - log_prices[baseline_start]
        # The pre-event window is a simple per-pair drift estimator.  Scale it
        # to the post horizon before computing abnormal return; cross-sectional
        # demeaning alone cannot make pre_event_baseline_steps an expected-return
        # model.  Factor/synthetic-control alternatives remain future contracts.
        expected_return = (config.horizon_steps / config.pre_event_baseline_steps) * baseline_return
        abnormal_return = post_return - expected_return
        for pair_index, pair in enumerate(PAIRS):
            if pressure[pair_index] == 0.0:
                continue
            predicted_sign = int(np.sign(pressure[pair_index]))
            realized_sign = int(np.sign(abnormal_return[pair_index]))
            records.append({
                "event_id": event["event_id"],
                "source_document_id": event["source_document_id"],
                "jurisdiction": event["jurisdiction"],
                "authority": event["authority"],
                "legal_stage": event["legal_stage"],
                "published_at": event["published_at"],
                "known_at": event["known_at"],
                "prediction_time": pd.Timestamp(int(times[start]), unit="ns", tz="UTC"),
                "target_time": pd.Timestamp(int(times[end]), unit="ns", tz="UTC"),
                "pair": pair,
                "scenario_weighted_pressure": float(pressure[pair_index]),
                "predicted_direction": predicted_sign,
                "post_event_log_return": float(post_return[pair_index]),
                "pre_event_log_return": float(baseline_return[pair_index]),
                "expected_post_event_log_return": float(expected_return[pair_index]),
                "baseline_adjusted_abnormal_log_return": float(abnormal_return[pair_index]),
                "direction_match": int(predicted_sign == realized_sign) if realized_sign != 0 else np.nan,
                "event_content_sha256": event["content_sha256"],
            })
    study = pd.DataFrame(records)
    scored = study[study["direction_match"].notna()] if not study.empty else study
    summary: dict[str, Any] = {
        "version": VERSION,
        "interpretation": (
            "timestamped legal/regulatory scenario event study; no legal advice, outcome forecast, "
            "execution, PnL, causation, or trade recommendation"
        ),
        "pair_scope": list(PAIRS),
        "causality": {
            "event_decision_time": "first observed canonical bar strictly after immutable known_at",
            "citation_rule": "every cited source document must have known_at <= citing event known_at",
            "target": f"post-known-time {config.horizon_steps}-minute gap-safe response",
            "event_data_required": "timestamped primary-source corpus with recorded scenario probabilities and pair exposures",
        },
        "event_count": len(events),
        "event_pair_observations": int(len(study)),
        "direction_match_rate": float(scored["direction_match"].mean()) if len(scored) else None,
        "promotion_status": (
            "BLOCKED_NO_VALIDATED_EVENT_CORPUS_OR_EXECUTION_DATA: descriptive event study only; "
            "no asset-selection, causal, or tradability claim"
        ),
    }
    return study, summary


def load_common_log_prices(max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows < 2_000:
        raise ContractError("max_rows must be at least 2,000")
    validate_generated_manifest(ROOT)
    reference = pd.read_parquet(CANONICAL_DIR / f"{PAIRS[0]}.parquet", columns=["timestamp"])
    if len(reference) < max_rows + 2:
        raise ContractError("canonical reference series is shorter than max_rows")
    start = reference.iloc[-max_rows - 1]["timestamp"]
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet", columns=["timestamp", "close"],
                                filters=[("timestamp", ">=", start)])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(frame, on="timestamp", how="inner",
                                                            validate="one_to_one")
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").iloc[-max_rows:].reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    prices = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if len(times) < 2_000 or not np.all(np.diff(times) > 0) or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ContractError("joined canonical event-study input is invalid")
    return times, np.log(prices)


def write_result(study: pd.DataFrame, summary: dict[str, Any], out_dir: Path, events_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    study_path = out_dir / "legal_event_study.parquet"
    summary_path = out_dir / "legal_event_summary.json"
    study.to_parquet(study_path, index=False, compression="zstd")
    summary["outputs"] = {"event_study": str(study_path.relative_to(ROOT)).replace("\\", "/")}
    summary["source_hashes"] = {
        "pipeline/legal_event.py": sha256_file(Path(__file__).resolve()),
        "event_corpus": sha256_file(events_path),
        **{f"data_canonical/{pair}.parquet": sha256_file(CANONICAL_DIR / f"{pair}.parquet")
           for pair in PAIRS},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def synthetic_events() -> list[dict[str, Any]]:
    scenarios = {scenario: 0.0 for scenario in SCENARIO_IMPACT}
    scenarios["enacted"] = 0.70
    scenarios["diluted"] = 0.20
    scenarios["injunction"] = 0.10
    return [{
        "event_id": "synthetic-rule-001",
        "source_document_id": "synthetic-primary-001",
        "jurisdiction": "TEST",
        "authority": "TEST_AUTHORITY",
        "published_at": "1970-01-01T00:03:00Z",
        "known_at": "1970-01-01T00:03:00Z",
        "legal_stage": "final_rule",
        "citations": [],
        "scenario_probabilities": scenarios,
        "pair_exposures": {"EURUSD": 0.8, "USDJPY": -0.4},
    }]


def synthetic_input(rows: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260718)
    returns = rng.normal(0.0, 1.0e-4, size=(rows, N_PAIRS))
    times = np.arange(rows, dtype=np.int64) * DT_NS
    return times, np.cumsum(returns, axis=0)


def self_check() -> dict[str, Any]:
    schema = load_schema()
    events = validate_events(synthetic_events(), schema)
    times, prices = synthetic_input()
    study, summary = run_event_study(times, prices, events, EventStudyConfig(horizon_steps=20,
                                                                              pre_event_baseline_steps=2))
    bad_citation = synthetic_events()[0]
    future_document = json.loads(json.dumps(bad_citation))
    future_document["event_id"] = "synthetic-rule-002"
    future_document["source_document_id"] = "synthetic-primary-002"
    future_document["published_at"] = "1970-01-01T00:10:00Z"
    future_document["known_at"] = "1970-01-01T00:10:00Z"
    bad_citation["citations"] = [future_document["source_document_id"]]
    try:
        validate_events([bad_citation, future_document], schema)
        future_citation_rejected = False
    except ContractError:
        future_citation_rejected = True
    first_prediction = study["prediction_time"].min() if not study.empty else pd.NaT
    return {
        "passed": bool(not study.empty and first_prediction > events[0]["known_at"]
                       and future_citation_rejected and summary["event_pair_observations"] == len(study)),
        "event_pair_observations": int(len(study)),
        "prediction_strictly_after_known_at": bool(first_prediction > events[0]["known_at"]),
        "future_citation_rejected": future_citation_rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENT_PATH)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    try:
        events = load_events(args.events.resolve())
        times, prices = load_common_log_prices(args.max_rows)
        study, summary = run_event_study(times, prices, events)
        write_result(study, summary, args.out_dir.resolve(), args.events.resolve())
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except (ContractError, SharedContractError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
