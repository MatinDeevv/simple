"""Deterministic canonical JSON for hash commitments.

Every Tribunal artifact hash is computed over ``canonical_json_bytes`` of the
payload: UTF-8, sorted keys, compact separators, no NaN/Infinity, UUIDs and
timestamps converted deterministically, forward-slash relative logical paths.
Dictionary insertion order never influences a hash.

This module is deliberately self-contained (no imports from ``engine.core``)
so the governance subsystem cannot drift when shared utilities change.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import uuid as uuid_module
from datetime import date, datetime, timezone
from typing import Any

from engine.experiments.errors import CanonicalJsonError, PathSecurityError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 50 * 1024 * 1024  # documented sanity ceiling for artifact payloads
_MAX_JSON_DEPTH = 64
_MAX_INTEGER_BITS = 16_384  # safely below Python's default 4,300-decimal-digit ceiling


def _normalize_value(value: Any, *, depth: int = 0) -> Any:
    """Convert a payload to plain JSON types, rejecting anything non-deterministic."""
    if depth > _MAX_JSON_DEPTH:
        raise CanonicalJsonError(f"payload nesting exceeds maximum depth {_MAX_JSON_DEPTH}")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise CanonicalJsonError(
                f"integer exceeds maximum bit length {_MAX_INTEGER_BITS}")
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalJsonError("NaN and infinity are not permitted in canonical JSON")
        return value
    if isinstance(value, uuid_module.UUID):
        return str(value)
    if isinstance(value, datetime):
        return normalize_utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"canonical JSON object keys must be strings, got {type(key).__name__}")
            if key in normalized:
                raise CanonicalJsonError(f"duplicate key after normalization: {key!r}")
            normalized[key] = _normalize_value(item, depth=depth + 1)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item, depth=depth + 1) for item in value]
    raise CanonicalJsonError(f"unsupported type in canonical JSON payload: {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    normalized = _normalize_value(payload)
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise CanonicalJsonError(f"canonical payload exceeds size limit ({len(encoded)} bytes)")
    return encoded


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json_text(payload: Any) -> str:
    """Human-readable serialization sharing the canonical normalization rules."""
    normalized = _normalize_value(payload)
    return json.dumps(normalized, sort_keys=True, indent=2,
                      ensure_ascii=True, allow_nan=False) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_strict_json_text(text: str) -> Any:
    """Parse JSON rejecting duplicate keys and NaN/Infinity literals."""
    if len(text.encode("utf-8", errors="replace")) > _MAX_JSON_BYTES:
        raise CanonicalJsonError("JSON document exceeds size limit")

    def _reject_nonfinite(token: str) -> float:
        raise CanonicalJsonError(f"non-finite JSON constant not permitted: {token}")

    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=_reject_nonfinite)
    except CanonicalJsonError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CanonicalJsonError(f"invalid JSON document: {exc}") from exc


def normalize_logical_path(path: str) -> str:
    """Normalize an artifact logical path to relative, forward-slash, NFC form.

    Rejects absolute paths (POSIX and Windows), drive letters, ``..`` segments,
    empty segments, and null bytes so a manifest key can never escape its root.
    """
    if not isinstance(path, str) or not path:
        raise PathSecurityError("logical path must be a non-empty string")
    if "\x00" in path:
        raise PathSecurityError("logical path contains a null byte")
    normalized = unicodedata.normalize("NFC", path).replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        raise PathSecurityError(f"absolute logical path not permitted: {path!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise PathSecurityError(f"drive-letter logical path not permitted: {path!r}")
    segments = normalized.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise PathSecurityError(f"unsafe logical path segment in {path!r}")
    return "/".join(segments)


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def is_git_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_GIT_SHA_RE.match(value))


def normalize_uuid(value: Any) -> str:
    """Return the canonical lowercase-hyphenated form of a UUID string."""
    if isinstance(value, uuid_module.UUID):
        return str(value)
    if not isinstance(value, str):
        raise CanonicalJsonError(f"UUID must be a string, got {type(value).__name__}")
    try:
        return str(uuid_module.UUID(value.strip()))
    except ValueError as exc:
        raise CanonicalJsonError(f"malformed UUID: {value!r}") from exc


def normalize_utc_timestamp(value: Any) -> str:
    """Return ``YYYY-MM-DDTHH:MM:SS(.ffffff)?+00:00`` for a tz-aware UTC moment.

    Naive timestamps are rejected: an audit chain must never depend on the
    local clock's implied timezone.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise CanonicalJsonError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise CanonicalJsonError(f"timestamp must be str or datetime, got {type(value).__name__}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CanonicalJsonError(f"timestamp lacks a timezone: {value!r}")
    return parsed.astimezone(timezone.utc).isoformat()


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def validate_against_schema(schema: dict[str, Any], instance: Any, *, path: str = "$") -> list[str]:
    """Minimal JSON-schema-subset validation (type, required, properties, enum,
    items, additionalProperties). Self-contained twin of the repo's generic
    validator so the Tribunal has no runtime dependency outside this package."""
    errors: list[str] = []
    declared = schema.get("type")
    if declared is not None:
        allowed = declared if isinstance(declared, list) else [declared]
        actual = _json_type_name(instance)
        matches = any(actual == option or (option == "number" and actual == "integer")
                      for option in allowed)
        if not matches:
            errors.append(f"{path}: expected type {allowed}, got {actual}")
            return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, dict) and schema.get("type") == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required field {name!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            if extra:
                errors.append(f"{path}: unexpected fields not permitted by schema: {extra}")
        for name, value in instance.items():
            subschema = properties.get(name)
            if subschema is not None:
                errors.extend(validate_against_schema(subschema, value, path=f"{path}.{name}"))
    if isinstance(instance, list) and schema.get("type") == "array":
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                errors.extend(validate_against_schema(item_schema, item, path=f"{path}[{index}]"))
    return errors
