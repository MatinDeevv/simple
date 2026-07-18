"""
ingest.py -- idempotent ingestion of raw Dukascopy 1-min bid-bar CSVs into the
canonical per-pair stream (OQ-11, sim-datapipe).

Contract (see docs/datapipe.md and docs/state-schema.md HC-3, section 5 step 0):
  * Raw CSVs in dukascopy_data/ are NEVER assumed sorted (HC-3: downloader
    re-runs append unsorted chunks). Raw CSVs are not a valid downstream input.
  * This script: reads raw -> stable-sorts by timestamp -> drops exact
    duplicate timestamps (keep first occurrence in file order) -> validates ->
    writes data_canonical/<PAIR>.parquet + data_canonical/manifest.json.
  * Gaps are PRESERVED. No filling, no interpolation, ever. Gap semantics are
    owned downstream (OQ-8, sim-integrator).
  * OHLC sanity violations are LOGGED and COUNTED, never silently fixed.
  * Idempotent: re-running on byte-identical input produces identical per-pair
    manifest entries. Identity of the canonical data is defined by
    `data_sha256` (spec below), NOT by parquet file bytes.

Canonical data hash (canonical_hash_spec = "fxsim-canonical-v1"):
  sha256 over the concatenation of:
    1. UTF-8 header bytes:  "fxsim-canonical-v1|<PAIR>|<rows_out>|"
    2. int64 little-endian raw bytes of UTC epoch *seconds* of every bar
       timestamp, rows in ascending timestamp order
    3. float64 (IEEE-754) little-endian raw bytes of columns open, high, low,
       close, volume -- each full column in that fixed order, same row order.
  This is reproducible with numpy alone and has no float-formatting ambiguity.

Run:  python pipeline\\ingest.py            (all 10 pairs)
      python pipeline\\ingest.py --pairs EURUSD GBPUSD
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import canonical_pair_order

# --------------------------------------------------------------------------
# Constants (frozen)
# --------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"
CANONICAL_HASH_SPEC = (
    "fxsim-canonical-v1: sha256( utf8('fxsim-canonical-v1|<PAIR>|<rows_out>|')"
    " + int64-LE bytes of UTC epoch seconds (ascending)"
    " + float64-LE bytes of columns open, high, low, close, volume in that"
    " order, same row order )"
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dukascopy_data"
OUT_DIR = ROOT / "data_canonical"

# The tracked configuration, not generated data, defines this load-bearing order.
PAIRS = list(canonical_pair_order(ROOT))

TS_FORMAT = "%Y-%m-%d %H:%M:%S%z"          # strict; any deviant row raises
TS_MIN = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")   # inclusive
TS_MAX = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")   # exclusive

NUM_COLS = ["open", "high", "low", "close", "volume"]
RAW_DTYPES = {
    "timestamp": "str",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "symbol": "str",
}

# --------------------------------------------------------------------------
# Output backend: pyarrow parquet, with mandated fallback chain
# --------------------------------------------------------------------------


def _resolve_backend() -> str:
    """Return 'parquet' if pyarrow is importable (installing it if needed),
    else 'csv.gz'. Fallback is reported loudly, not silently."""
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except ImportError:
        pass
    log("SETUP", "pyarrow not importable; attempting `pip install pyarrow`")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyarrow"],
            check=True, capture_output=True, text=True, timeout=600,
        )
        import pyarrow  # noqa: F401
        return "parquet"
    except Exception as exc:  # install refused / offline / still broken
        log("SETUP", f"pyarrow unavailable ({exc!r}); FALLING BACK to gzipped "
                     "CSV output. Report this.")
        return "csv.gz"


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


class ValidationError(RuntimeError):
    """Hard validation failure -- ingestion of the pair is aborted."""


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def epoch_us(ts: pd.Series) -> np.ndarray:
    """tz-aware UTC pandas Series -> int64 numpy array of epoch microseconds."""
    naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.to_numpy(dtype="datetime64[us]").astype("<i8")


def canonical_data_hash(pair: str, df: pd.DataFrame) -> str:
    """Implements CANONICAL_HASH_SPEC. df must already be sorted/deduped."""
    h = hashlib.sha256()
    h.update(f"fxsim-canonical-v1|{pair}|{len(df)}|".encode("utf-8"))
    secs = epoch_us(df["timestamp"]) // 1_000_000
    h.update(np.ascontiguousarray(secs, dtype="<i8").tobytes())
    for col in NUM_COLS:
        h.update(np.ascontiguousarray(df[col].to_numpy(), dtype="<f8").tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Per-pair ingestion
# --------------------------------------------------------------------------


def read_raw(pair: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    expected_cols = list(RAW_DTYPES.keys())
    if list(df.columns) != expected_cols:
        raise ValidationError(
            f"{pair}: column mismatch: {list(df.columns)} != {expected_cols}")
    # Strict UTC parse. Any row that is not exactly 'YYYY-mm-dd HH:MM:SS+zz:zz'
    # raises -- no coercion, no NaT smuggling.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format=TS_FORMAT, utc=True, errors="raise")
    if not (df["symbol"] == pair).all():
        bad = df.loc[df["symbol"] != pair, "symbol"].unique()[:5]
        raise ValidationError(f"{pair}: foreign symbol values in file: {bad}")
    return df.drop(columns=["symbol"])


def sort_and_dedupe(pair: str, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Stable sort by timestamp, then drop duplicate timestamps keeping the
    FIRST occurrence in original file order (stable sort preserves file order
    among equal timestamps; under HC-3, appended re-download chunks come later
    in the file, so 'first' = original download).

    Returns (deduped df, n_dropped, n_conflicts) where n_conflicts counts
    dropped rows whose (open,high,low,close,volume) payload differs from the
    kept row at the same timestamp. Conflicts are logged, never merged."""
    df = df.sort_values("timestamp", kind="stable", ignore_index=True)
    drop_mask = df["timestamp"].duplicated(keep="first")
    n_dropped = int(drop_mask.sum())
    n_conflicts = 0
    if n_dropped:
        kept = (df[~drop_mask].set_index("timestamp"))[NUM_COLS]
        dropped = df[drop_mask]
        ref = kept.loc[dropped["timestamp"]].to_numpy()
        n_conflicts = int((~np.isclose(
            dropped[NUM_COLS].to_numpy(), ref, rtol=0.0, atol=0.0,
            equal_nan=True)).any(axis=1).sum())
        log(pair, f"duplicate timestamps dropped: {n_dropped} "
                  f"(payload-conflicting: {n_conflicts})")
        if n_conflicts:
            log(pair, "WARNING: conflicting duplicate payloads kept-first; "
                      "no merging performed")
    return df[~drop_mask].reset_index(drop=True), n_dropped, n_conflicts


def validate(pair: str, df: pd.DataFrame) -> dict:
    """Hard failures raise ValidationError. Soft findings are counted/logged
    and returned; rows are never mutated or dropped here."""
    ts = df["timestamp"]

    # 1. strictly increasing after dedupe (hard)
    if not (ts.is_monotonic_increasing and ts.is_unique):
        raise ValidationError(f"{pair}: timestamps not strictly increasing "
                              "after sort+dedupe")

    # 2. no NaN in OHLCV (hard)
    nan_counts = df[NUM_COLS].isna().sum()
    if int(nan_counts.sum()) > 0:
        raise ValidationError(f"{pair}: NaN in OHLCV: {nan_counts.to_dict()}")

    # 3. timestamp range [2015-01-01, 2025-01-01) UTC (hard)
    out_of_range = int(((ts < TS_MIN) | (ts >= TS_MAX)).sum())
    if out_of_range:
        raise ValidationError(
            f"{pair}: {out_of_range} timestamps outside "
            f"[{TS_MIN.isoformat()}, {TS_MAX.isoformat()})")

    # 4. OHLC sanity: low <= open,close <= high and low <= high (soft: log)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    viol_mask = (l > o) | (l > c) | (h < o) | (h < c) | (l > h)
    n_viol = int(viol_mask.sum())
    samples = []
    if n_viol:
        idx = np.flatnonzero(viol_mask)[:5]
        for i in idx:
            samples.append(ts.iloc[int(i)].isoformat())
            log(pair, "OHLC violation at "
                f"{ts.iloc[int(i)].isoformat()}: o={float(o[i])!r} "
                f"h={float(h[i])!r} l={float(l[i])!r} c={float(c[i])!r}")
        log(pair, f"OHLC sanity violations: {n_viol} (logged, NOT fixed)")

    # 5. informational: minute alignment, negative volume (soft: log)
    eus = epoch_us(ts)
    n_misaligned = int((eus % 60_000_000 != 0).sum())
    if n_misaligned:
        log(pair, f"WARNING: {n_misaligned} timestamps not minute-aligned")
    n_negvol = int((df["volume"].to_numpy() < 0).sum())
    if n_negvol:
        log(pair, f"WARNING: {n_negvol} rows with negative volume")

    return {
        "ohlc_violations": n_viol,
        "ohlc_violation_samples": samples,
        "misaligned_timestamps": n_misaligned,
        "negative_volume_rows": n_negvol,
    }


def write_canonical(pair: str, df: pd.DataFrame, backend: str,
                    meta: dict) -> tuple[Path, int]:
    """Atomic write (tmp + os.replace). Canonical column order:
    timestamp (timestamp[us, UTC]), open, high, low, close, volume."""
    out = df[["timestamp"] + NUM_COLS].copy()
    out["timestamp"] = out["timestamp"].dt.as_unit("us")

    if backend == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = OUT_DIR / f"{pair}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pandas(out, preserve_index=False)
        md = dict(table.schema.metadata or {})
        md.update({
            b"fxsim.pipeline_version": PIPELINE_VERSION.encode(),
            b"fxsim.pair": pair.encode(),
            b"fxsim.source_sha256": meta["source_sha256"].encode(),
            b"fxsim.data_sha256": meta["data_sha256"].encode(),
        })
        table = table.replace_schema_metadata(md)
        pq.write_table(table, tmp, compression="zstd")
    else:  # csv.gz fallback -- mtime=0 so output bytes are deterministic
        path = OUT_DIR / f"{pair}.csv.gz"
        tmp = path.with_suffix(".gz.tmp")
        csv_text = out.assign(
            timestamp=out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        ).to_csv(index=False, lineterminator="\n")
        with open(tmp, "wb") as fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
                gz.write(csv_text.encode("utf-8"))
    os.replace(tmp, path)
    return path, path.stat().st_size


def ingest_pair(pair: str, index: int, backend: str) -> dict:
    src = RAW_DIR / f"{pair}.csv"
    if not src.exists():
        raise ValidationError(f"{pair}: missing raw file {src}")
    log(pair, f"source={src.name} sha256...")
    source_sha = sha256_file(src)

    df = read_raw(pair, src)
    rows_in = len(df)

    df, n_dropped, n_conflicts = sort_and_dedupe(pair, df)
    rows_out = len(df)

    soft = validate(pair, df)
    data_sha = canonical_data_hash(pair, df)

    meta = {"source_sha256": source_sha, "data_sha256": data_sha}
    out_path, out_bytes = write_canonical(pair, df, backend, meta)

    entry = {
        "index": index,
        "source_file": f"dukascopy_data/{pair}.csv",
        "source_sha256": source_sha,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "duplicates_dropped": n_dropped,
        "duplicate_conflicts": n_conflicts,
        **soft,
        "ts_first": df["timestamp"].iloc[0].isoformat(),
        "ts_last": df["timestamp"].iloc[-1].isoformat(),
        "data_sha256": data_sha,
        "output_format": backend,
        "output_file": f"data_canonical/{out_path.name}",
        "output_bytes": out_bytes,
    }
    log(pair, f"rows_in={rows_in} rows_out={rows_out} dropped={n_dropped} "
              f"ohlc_viol={soft['ohlc_violations']} -> {out_path.name} "
              f"({out_bytes/1e6:.1f} MB)")
    return entry


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def write_manifest(entries: dict[str, dict]) -> Path:
    """Full-run manifest. If a partial run (--pairs) updates a subset, prior
    entries for other pairs are preserved. Per-pair entries contain only
    deterministic fields; `generated_utc` (top level) is the only field
    expected to differ between identical re-runs."""
    path = OUT_DIR / "manifest.json"
    prior: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("pipeline_version") == PIPELINE_VERSION:
                prior = old.get("pairs", {})
        except (json.JSONDecodeError, OSError):
            log("MANIFEST", "existing manifest unreadable; rewriting fresh")

    merged = {p: (entries.get(p) or prior.get(p)) for p in PAIRS}
    merged = {p: e for p, e in merged.items() if e is not None}

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "canonical_hash_spec": CANONICAL_HASH_SPEC,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": _pyarrow_version(),
        },
        "raw_dir": "dukascopy_data",
        "instrument_index_order": PAIRS,
        "pairs": merged,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _pyarrow_version() -> str:
    try:
        import pyarrow
        return pyarrow.__version__
    except ImportError:
        return "unavailable"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="subset of pairs (default: all 10, frozen order)")
    args = ap.parse_args(argv)

    todo = args.pairs if args.pairs else PAIRS
    unknown = [p for p in todo if p not in PAIRS]
    if unknown:
        log("FATAL", f"unknown pairs {unknown}; valid: {PAIRS}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = _resolve_backend()
    log("SETUP", f"pipeline_version={PIPELINE_VERSION} backend={backend} "
                 f"raw={RAW_DIR} out={OUT_DIR}")

    entries: dict[str, dict] = {}
    failures: list[str] = []
    for pair in PAIRS:              # always iterate in frozen index order
        if pair not in todo:
            continue
        try:
            entries[pair] = ingest_pair(pair, PAIRS.index(pair), backend)
        except ValidationError as exc:
            log("FAIL", str(exc))
            failures.append(pair)

    if entries:
        mpath = write_manifest(entries)
        log("MANIFEST", str(mpath))

    # summary table
    if entries:
        hdr = (f"{'pair':<8}{'rows_in':>10}{'rows_out':>10}{'dups':>7}"
               f"{'conflicts':>10}{'ohlc_viol':>10}{'out_MB':>8}")
        print("\n" + hdr)
        print("-" * len(hdr))
        for p in PAIRS:
            if p in entries:
                e = entries[p]
                print(f"{p:<8}{e['rows_in']:>10}{e['rows_out']:>10}"
                      f"{e['duplicates_dropped']:>7}{e['duplicate_conflicts']:>10}"
                      f"{e['ohlc_violations']:>10}"
                      f"{e['output_bytes']/1e6:>8.1f}")
        print()

    if failures:
        log("FATAL", f"validation failed for: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
