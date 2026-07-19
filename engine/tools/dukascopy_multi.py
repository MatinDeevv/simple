"""
Multi-process Dukascopy Forex Downloader (official library + proxy rotation)
==============================================================================
Uses the official `dukascopy-python` package. This workload is CPU-bound
(pandas parsing/bar-building inside the library), so it uses
ProcessPoolExecutor instead of threads — real parallel CPU usage across
cores, not GIL-limited.

Install:
    pip install dukascopy-python pandas requests tqdm

Run:
    python3 dukascopy_multi.py

Resumable: progress is checkpointed to OUTPUT_DIR/.progress.json after every
chunk. Killing the script and re-running it picks up where it left off
instead of re-downloading everything.

Retries: failed chunks are automatically re-queued and retried in rounds
(with backoff) until every chunk succeeds — the script does not stop until
all data is downloaded.

Tuning:
    MAX_WORKERS should be roughly your CPU core count (physical cores,
    not logical/hyperthreaded — test both, hyperthreading gains are
    inconsistent for this kind of workload). Check with:
        python3 -c "import os; print(os.cpu_count())"
    This workload is a mix of network wait (downloading .bin files) and
    CPU parsing, so a modest oversubscription (1.5x cores) usually helps
    throughput more than it hurts.
"""

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import logging
import os
import time

import pandas as pd
import requests
import dukascopy_python
from tqdm import tqdm
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_NZD_USD,
    INSTRUMENT_FX_CROSSES_USD_CNH,
    INSTRUMENT_FX_CROSSES_EUR_GBP,
    INSTRUMENT_FX_CROSSES_EUR_JPY,
    INSTRUMENT_FX_CROSSES_GBP_JPY,
)

# =================================================================
# CONFIG — edit these
# =================================================================
SYMBOLS = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD,   # your favorite
    "USDCAD": INSTRUMENT_FX_MAJORS_USD_CAD,
    "USDCNH": INSTRUMENT_FX_CROSSES_USD_CNH,
    "USDCHF": INSTRUMENT_FX_MAJORS_USD_CHF,
    "EURGBP": INSTRUMENT_FX_CROSSES_EUR_GBP,
    "EURJPY": INSTRUMENT_FX_CROSSES_EUR_JPY,
    "GBPJPY": INSTRUMENT_FX_CROSSES_GBP_JPY,
}

START = datetime(2015, 1, 1)      # 10 years
END = datetime(2025, 1, 1)

INTERVAL = dukascopy_python.INTERVAL_MIN_1     # swap to INTERVAL_TICK for raw ticks (much bigger/slower)
OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

CHUNK_DAYS = 30
MAX_WORKERS = max(int((os.cpu_count() or 8) * 1.5), 4)
OUTPUT_DIR = "./data/raw/dukascopy"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, ".progress.json")
MAX_RETRIES_PER_CHUNK = 3

# retry rounds: after a full pass, failed chunks are re-queued and retried
# with this backoff (seconds) between rounds, up to MAX_RETRY_ROUNDS times.
RETRY_BACKOFF_SECONDS = 10
MAX_RETRY_ROUNDS = 30

# ---- Proxy pool ----
# Fill with your own proxies, format: "http://user:pass@host:port"
# Leave empty to run without proxies (direct connection) — start here,
# this feed has no auth/rate-limit wall, so proxies are usually unnecessary.
PROXIES = [
    # "http://user:pass@proxy1.example.com:8000",
    # "http://user:pass@proxy2.example.com:8000",
]
PROXY_MAX_RETRIES = 4
REQUEST_TIMEOUT = 20


# =================================================================
# Per-process setup — runs once in EACH worker process, not the parent.
# ProcessPoolExecutor pickles arguments to send to workers, so this
# initializer configures each worker's own copy of requests/logging
# rather than relying on state built in the main process.
# =================================================================
def _init_worker(proxies):
    global _proxy_cycle, _proxy_lock_dummy, _original_get

    import threading
    _proxy_cycle = itertools.cycle(proxies) if proxies else None
    _proxy_lock_dummy = threading.Lock()  # single-threaded per process, but harmless
    _original_get = requests.get

    def _proxied_get(*args, **kwargs):
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        if not proxies:
            return _original_get(*args, **kwargs)

        last_exc = None
        for attempt in range(PROXY_MAX_RETRIES):
            proxy = next(_proxy_cycle)
            kwargs["proxies"] = {"http": proxy, "https": proxy}
            try:
                resp = _original_get(*args, **kwargs)
                if resp.status_code == 200:
                    return resp
                last_exc = RuntimeError(f"status {resp.status_code} via {proxy}")
            except requests.exceptions.RequestException as e:
                last_exc = e
            time.sleep(0.5 * (attempt + 1))
        raise last_exc

    requests.get = _proxied_get

    # Library resets this logger to INFO on every fetch() call, so a plain
    # setLevel() gets silently undone — disabling it survives that reset
    # and removes a real source of I/O overhead per hourly sub-request.
    logging.getLogger("DUKASCRIPT").disabled = True


def fetch_chunk(symbol_name, instrument, interval, offer_side, chunk_start, chunk_end, is_first):
    """Runs inside a worker process.

    dukascopy_python.fetch() is inclusive on both start and end, so adjacent
    chunks (chunk N's end == chunk N+1's start) both return the boundary
    row. Drop it from every chunk except the symbol's first, so consecutive
    chunks never write a duplicate row.
    """
    df = dukascopy_python.fetch(
        instrument,
        interval,
        offer_side,
        chunk_start,
        chunk_end,
        max_retries=MAX_RETRIES_PER_CHUNK,
    )
    if df is not None and not df.empty:
        if not is_first:
            # chunk_start is naive; dukascopy_python converts it via
            # datetime.timestamp(), which treats naive datetimes as local
            # time. Match that exact conversion so the boundary timestamp
            # lines up with the UTC-aware df index regardless of the
            # machine's timezone.
            boundary = pd.Timestamp(chunk_start.timestamp(), unit="s", tz="UTC")
            df = df[df.index != boundary]
        df["symbol"] = symbol_name
    return symbol_name, chunk_start, chunk_end, df


def date_chunks(start: datetime, end: datetime, chunk_days: int):
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


# =================================================================
# Resume/progress checkpoint — records which (symbol, chunk_start) pairs
# have already been written to CSV, so a killed/restarted run doesn't
# redo completed work.
# =================================================================
def _chunk_key(symbol_name, chunk_start):
    return f"{symbol_name}|{chunk_start.isoformat()}"


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(done_keys):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done_keys), f)
    os.replace(tmp, PROGRESS_FILE)


# =================================================================
# Main (parent process — collects results and writes files)
# =================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = {}          # symbol -> list[(start, end)]
    for symbol_name in SYMBOLS:
        all_chunks[symbol_name] = date_chunks(START, END, CHUNK_DAYS)

    done_keys = load_progress()

    pending = [
        (symbol_name, SYMBOLS[symbol_name], chunk_start, chunk_end, i == 0)
        for symbol_name, chunks in all_chunks.items()
        for i, (chunk_start, chunk_end) in enumerate(chunks)
        if _chunk_key(symbol_name, chunk_start) not in done_keys
    ]

    total_jobs = sum(len(c) for c in all_chunks.values())
    already_done = total_jobs - len(pending)

    print(f"{total_jobs} total chunks across {len(SYMBOLS)} symbols "
          f"({already_done} already done, {len(pending)} remaining), "
          f"{MAX_WORKERS} worker processes, {len(PROXIES)} proxies configured.\n")

    total_rows = {s: 0 for s in SYMBOLS}

    # one tqdm bar per symbol, stacked, each pre-filled with resumed progress
    bars = {}
    for i, symbol_name in enumerate(SYMBOLS):
        n_total = len(all_chunks[symbol_name])
        n_done = sum(1 for cs, _ in all_chunks[symbol_name]
                     if _chunk_key(symbol_name, cs) in done_keys)
        bars[symbol_name] = tqdm(
            total=n_total, initial=n_done, position=i, desc=f"{symbol_name:8s}",
            unit="chunk", leave=True,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {unit} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )

    round_num = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                              initializer=_init_worker,
                              initargs=(PROXIES,)) as pool:
        while pending and round_num < MAX_RETRY_ROUNDS:
            round_num += 1
            if round_num > 1:
                for b in bars.values():
                    b.write(f"--- retry round {round_num}: "
                            f"{len(pending)} chunk(s) still failing, "
                            f"backing off {RETRY_BACKOFF_SECONDS}s ---")
                time.sleep(RETRY_BACKOFF_SECONDS)

            futures = [
                pool.submit(fetch_chunk, symbol_name, instrument, INTERVAL, OFFER_SIDE,
                            chunk_start, chunk_end, is_first)
                for symbol_name, instrument, chunk_start, chunk_end, is_first in pending
            ]
            failed = []

            for future in as_completed(futures):
                try:
                    symbol_name, chunk_start, chunk_end, df_chunk = future.result()
                    n_rows = 0
                    if df_chunk is not None and not df_chunk.empty:
                        out_path = os.path.join(OUTPUT_DIR, f"{symbol_name}.csv")
                        header = not os.path.exists(out_path)
                        df_chunk.to_csv(out_path, mode="a", header=header)
                        n_rows = len(df_chunk)
                        total_rows[symbol_name] += n_rows

                    done_keys.add(_chunk_key(symbol_name, chunk_start))
                    save_progress(done_keys)
                    bars[symbol_name].update(1)
                    bars[symbol_name].set_postfix_str(f"+{n_rows}rows", refresh=False)
                except Exception as e:
                    # future.result() doesn't tell us which job failed on
                    # exception, so match it back via the futures list.
                    idx = futures.index(future)
                    job = pending[idx]
                    failed.append(job)
                    bars[job[0]].write(f"FAILED {job[0]} {job[2].date()}->{job[3].date()}: {e}")

            pending = failed

    for b in bars.values():
        b.close()

    if pending:
        print(f"\nGave up after {MAX_RETRY_ROUNDS} rounds — {len(pending)} chunk(s) still failing:")
        for symbol_name, _, chunk_start, chunk_end, _ in pending:
            print(f"  {symbol_name} {chunk_start.date()} -> {chunk_end.date()}")
    else:
        print("\nAll chunks downloaded successfully.")

    print("Row counts this run:")
    for symbol_name, n in total_rows.items():
        print(f"  {symbol_name}: {n} rows -> {OUTPUT_DIR}/{symbol_name}.csv")


if __name__ == "__main__":
    main()
