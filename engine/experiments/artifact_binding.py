"""Physical artifact bindings for independent Tribunal verification."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from engine.experiments.canonical import normalize_logical_path, sha256_payload
from engine.experiments.errors import EvidenceValidationError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_physical_files(bindings: Iterable[dict[str, Any]], *, allowed_root: Path,
                        error_type: type[Exception] = EvidenceValidationError) -> list[dict[str, Any]]:
    """Hash regular, non-symlink files under ``allowed_root``.

    Caller hashes and sizes are assertions that must match, never identities
    accepted without inspecting bytes.
    """
    root = Path(allowed_root).resolve(strict=True)
    result: list[dict[str, Any]] = []
    logical_seen: set[str] = set()
    for raw in bindings:
        try:
            logical = normalize_logical_path(raw.get("logical_path", ""))
            physical = Path(raw["physical_path"])
            if physical.is_symlink():
                raise ValueError("symlinks are forbidden")
            resolved = physical.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise ValueError("physical path is not a regular file")
            if logical in logical_seen:
                raise ValueError(f"duplicate logical path {logical!r}")
            logical_seen.add(logical)
            size = resolved.stat().st_size
            digest = sha256_file(resolved)
            if raw.get("expected_sha256") is not None and raw["expected_sha256"] != digest:
                raise ValueError(f"hash mismatch for {logical}")
            if raw.get("expected_size_bytes") is not None and raw["expected_size_bytes"] != size:
                raise ValueError(f"size mismatch for {logical}")
        except (KeyError, OSError, ValueError) as exc:
            raise error_type(f"invalid physical binding: {exc}") from exc
        result.append({"logical_path": logical, "sha256": digest, "size_bytes": size})
    result.sort(key=lambda item: item["logical_path"])
    return result


def binding_set_sha256(bindings: list[dict[str, Any]]) -> str:
    return sha256_payload(bindings)
