"""Verify a strict v2 agent test receipt and its physical log."""
from __future__ import annotations
import json, sys
from pathlib import Path
from receipt_contract import ReceiptError, verify

def main(raw: str) -> int:
    try: payload=verify(Path(raw))
    except (ReceiptError, OSError) as exc: raise SystemExit(f"invalid receipt: {exc}") from exc
    print(json.dumps({"verified":True,"commit_sha":payload["commit_sha"],"command_argv":payload["command_argv"]}))
    return 0
if __name__=="__main__": raise SystemExit(main(sys.argv[1]))
