# Data Pipeline — Ingestion Spec (OQ-11)

| | |
|---|---|
| **Version** | 1.0.0 (= `PIPELINE_VERSION` in `engine/data/ingestion/ingest.py`) |
| **Status** | **OQ-11 RESOLVED** — idempotent sort/dedupe/validate ingestion implemented, run, and verified; canonical stream is `data/canonical/<PAIR>.parquet` + `data/canonical/manifest.json`. |
| **Owner** | sim-datapipe |
| **Date** | 2026-07-17 |
| **Upstream** | `data/raw/dukascopy/<PAIR>.csv` (raw; HC-3: never assumed sorted; not a valid input to any other agent) |
| **Downstream** | canonical stream only — schema §5 step 0 |

**OQ-10 remains OPEN**: cross-pair/cross-year volume normalization (broker-relative tick volume) is deliberately NOT part of ingestion. The canonical stream carries raw Dukascopy volume untouched. OQ-10 stays blocked until sim-dynamics picks a mass proxy (OQ-1); normalization will then ship as a separate versioned derivation, not a mutation of the canonical stream.

---

## 1. Ingestion steps (exact, in order)

Run: `python engine\data\ingestion\ingest.py` (all 10 pairs, tracked index order from `engine/config/instruments.json` per schema §2). `--pairs P1 P2 ...` ingests a subset and merges entries into the existing manifest.

Per pair `P`:

1. **Hash source.** SHA256 of the raw file bytes `data/raw/dukascopy/P.csv` → `source_sha256`.
2. **Read.** `pandas.read_csv` with explicit dtypes: `open/high/low/close/volume: float64`, `timestamp/symbol: str`. Hard error if the column set/order is not exactly `timestamp,open,high,low,close,volume,symbol`.
3. **Parse timestamps.** `pd.to_datetime(..., format="%Y-%m-%d %H:%M:%S%z", utc=True, errors="raise")`. Strict: any row deviating from this format aborts the pair. No coercion, no `NaT`.
4. **Symbol check.** Every `symbol` value must equal `P` (hard error otherwise); column then dropped — the canonical stream is per-pair, symbol is carried by filename and parquet metadata.
5. **Stable sort** by timestamp (`kind="stable"`). Stability is load-bearing: among equal timestamps, original file order is preserved.
6. **Dedupe.** Drop rows with a duplicate timestamp, `keep="first"`. Because the sort is stable and HC-3 appends come *later* in the file, "first" = the original download's row. Dropped rows whose `(open,high,low,close,volume)` differ from the kept row are counted as `duplicate_conflicts` and logged — never merged, never averaged.
7. **Validate** (§2 below).
8. **Hash canonical data** → `data_sha256` (§4 below).
9. **Write output** atomically (tmp file + `os.replace`) to `data/canonical/P.parquet`, columns `timestamp (timestamp[us, UTC]), open, high, low, close, volume` (all prices/volume `float64`), zstd compression, with parquet key-value metadata `fxsim.pipeline_version`, `fxsim.pair`, `fxsim.source_sha256`, `fxsim.data_sha256`.
10. **Write manifest entry** (§5).

If `pyarrow` is not importable the script attempts `pip install pyarrow`; if that fails it falls back to `data/canonical/P.csv.gz` (gzip `mtime=0` for byte-determinism) and records `output_format: "csv.gz"` in the manifest. The 2026-07-17 production run used parquet (pyarrow 24.0.0).

## 2. Validation rules

| # | Rule | Severity | Action |
|---|---|---|---|
| V1 | Timestamps strictly increasing after dedupe | hard | abort pair, nonzero exit |
| V2 | No NaN in open/high/low/close/volume | hard | abort pair |
| V3 | All timestamps in `[2015-01-01T00:00Z, 2025-01-01T00:00Z)` | hard | abort pair |
| V4 | OHLC sanity: `low ≤ open`, `low ≤ close`, `open ≤ high`, `close ≤ high`, `low ≤ high` | soft | count + log offending bars (first 5 sampled into manifest). **Rows are kept unmodified — no clipping, no repair.** |
| V5 | Timestamps minute-aligned (`epoch_us % 60e6 == 0`) | soft (informational) | count + log |
| V6 | `volume ≥ 0` | soft (informational) | count + log |

Hard failures leave the previous canonical output for that pair untouched (atomic replace never ran) and the process exits nonzero.

## 3. Gap policy

**Gaps are preserved, never filled, never interpolated.** A missing minute in raw stays a missing row in canonical. No resampling, no forward-fill, no synthetic bars, no weekend bridging. Downstream owns gap semantics: `Δt`/`g_gap` bookkeeping is schema §4.2, and integration across gaps is **OQ-8 (sim-integrator)**. Any downstream code that needs a dense minute grid must build it itself, explicitly, on its own version stamp — silent interpolation inside the pipeline is treated as a defect.

## 4. Identity: canonical data hash (`fxsim-canonical-v1`)

Idempotency is defined on **data content**, not on parquet bytes (parquet bytes are deterministic for a fixed pyarrow version, and were byte-identical across back-to-back runs on pyarrow 24.0.0, but this is not the contract).

```
data_sha256 = SHA256(
    utf8("fxsim-canonical-v1|<PAIR>|<rows_out>|")
  ‖ int64-LE bytes of UTC epoch SECONDS of every bar timestamp, ascending
  ‖ float64-LE (IEEE-754) raw bytes of column open   (same row order)
  ‖ float64-LE raw bytes of column high
  ‖ float64-LE raw bytes of column low
  ‖ float64-LE raw bytes of column close
  ‖ float64-LE raw bytes of column volume
)
```

Raw IEEE-754 bytes were chosen over any text serialization to eliminate float-formatting ambiguity; the hash is recomputable with numpy alone. **Contract: re-running ingestion on byte-identical raw input yields identical per-pair manifest entries** (every field; only the top-level `generated_utc` may differ). Verified 2026-07-17: two full runs, `pairs` blocks byte-identical.

## 5. Manifest format (`data/canonical/manifest.json`)

Top level: `pipeline_version`, `canonical_hash_spec` (the definition string), `generated_utc` (only non-deterministic field), `environment` (python/pandas/numpy/pyarrow versions — lineage, not identity), `raw_dir`, `instrument_index_order` (copied from tracked `engine/config/instruments.json` and validated against it), `pairs`.

Per-pair entry (all fields deterministic given raw bytes + pipeline version):

| Field | Meaning |
|---|---|
| `index` | frozen instrument index (schema §2) |
| `source_file`, `source_sha256` | raw CSV path (repo-relative, posix) and its byte hash |
| `rows_in`, `rows_out`, `duplicates_dropped` | `rows_out = rows_in − duplicates_dropped`, always |
| `duplicate_conflicts` | dropped duplicates whose OHLCV differed from the kept row |
| `ohlc_violations`, `ohlc_violation_samples` | V4 count + up to 5 offending ISO timestamps |
| `misaligned_timestamps`, `negative_volume_rows` | V5/V6 counts |
| `ts_first`, `ts_last` | first/last bar close time, ISO-8601 UTC |
| `data_sha256` | canonical identity (§4) |
| `output_format`, `output_file`, `output_bytes` | delivery artifact |

## 6. How sim-redteam re-derives and verifies

All steps independent of `ingest.py` internals:

1. `source_sha256`: hash `data/raw/dukascopy/<PAIR>.csv` bytes; must match manifest. If it does not, the raw file changed after ingestion (e.g. HC-3 append) — the canonical stream is stale and must be re-ingested; that is a detection, not a failure of this spec.
2. Re-run `python engine\data\ingestion\ingest.py` (or an independent reimplementation of §1) and confirm the per-pair entries are identical to the committed manifest.
3. `data_sha256` from delivered parquet, without ingest.py:
   ```python
   import hashlib, numpy as np, pyarrow.parquet as pq
   t = pq.read_table("data/canonical/EURUSD.parquet")
   h = hashlib.sha256(f"fxsim-canonical-v1|EURUSD|{t.num_rows}|".encode())
   h.update((t.column("timestamp").to_numpy().astype("<i8") // 1_000_000).astype("<i8").tobytes())
   for c in ["open", "high", "low", "close", "volume"]:
       h.update(t.column(c).to_numpy().astype("<f8").tobytes())
   assert h.hexdigest() == manifest["pairs"]["EURUSD"]["data_sha256"]
   ```
4. Cross-check parquet embedded metadata (`fxsim.*` keys) against the manifest entry.
5. Property checks on the delivered stream: strictly increasing unique timestamps, no NaN, range `[2015-01-01, 2025-01-01)`, `rows_out` matches, `ts_first`/`ts_last` match.

## 7. Production run record (2026-07-17, v1.0.0)

Raw data quality was high: zero duplicate timestamps, zero NaN, zero OHLC violations, zero misaligned timestamps, zero negative volumes across all 10 pairs (see `data/canonical/manifest.json` for the authoritative per-pair record, including all hashes). `rows_in == rows_out` everywhere — the dedupe/sort machinery is armed for HC-3 append events, and was exercised against synthetic unsorted/duplicated/conflicting input during acceptance (sorted correctly, dropped 2 dups keep-first, flagged 1 payload conflict and 1 OHLC violation without mutating data).

Gap magnitudes (informational — gaps preserved as-is): bar count vs. the full wall-clock minute grid over each pair's span is ~70.8–70.9% (weekends alone remove ~28.6%, so intra-session coverage is ~99%, consistent with schema §1). USDCNH is lower at 68.6% (thinner offshore-CNH sessions, mostly early years). First bars: USDCAD starts 2015-01-01 22:04Z and GBPJPY 22:01Z (missing opening minutes in raw); all other pairs 22:00Z; all pairs end 2024-12-31 21:59Z.

## 8. Open items owned here

| ID | Status | Note |
|---|---|---|
| OQ-11 | **RESOLVED** — one line: raw CSVs are stable-sorted, deduped keep-first, hard-validated (monotonic/NaN/range) with soft-logged OHLC violations, gaps untouched, delivered as per-pair zstd parquet whose identity is the `fxsim-canonical-v1` content hash recorded in `manifest.json`; re-runs on identical input reproduce identical manifest entries. | Schema §8 row may be flipped by the schema owner citing this doc. |
| OQ-10 | **OPEN** | Volume normalization across pairs/years. Blocked on sim-dynamics' mass-proxy choice (OQ-1). Canonical stream intentionally ships raw broker-relative volume; do not use it as mass (schema §4.1). |
