"""Small, explicit JSON Schema subset used by repository contracts.

Supported keywords are ``type``, ``required``, ``properties``, ``items``,
``enum`` and ``additionalProperties``.  This is deliberately not a partial
claim of full JSON Schema support: malformed documents and unsupported
keywords are rejected before they can silently weaken a contract.
"""
from __future__ import annotations

import math
from typing import Any

_TYPES = {"object", "array", "string", "boolean", "integer", "number", "null"}
_KEYWORDS = {"schema_version", "title", "type", "required", "properties", "items", "enum", "additionalProperties", "description"}


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    # numpy scalar support without making numpy a validator dependency.
    module = type(value).__module__
    if module.startswith("numpy"):
        if hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iu":
            return "integer"
        if hasattr(value, "dtype") and getattr(value.dtype, "kind", "") == "f":
            return "number"
    if isinstance(value, int): return "integer"
    if isinstance(value, float): return "number"
    if isinstance(value, dict): return "object"
    if isinstance(value, list): return "array"
    if isinstance(value, str): return "string"
    if value is None: return "null"
    return type(value).__name__


def _finite_number(value: Any) -> bool:
    return _json_type_name(value) in {"integer", "number"} and math.isfinite(float(value))


def _type_matches(value: Any, declared: str) -> bool:
    actual = _json_type_name(value)
    if declared == "number":
        return actual in {"integer", "number"} and _finite_number(value)
    if declared == "integer":
        return actual == "integer"
    return actual == declared


def _validate_subschema(schema: Any, path: str) -> str | None:
    if not isinstance(schema, dict): return f"{path} must be an object"
    unknown = sorted(set(schema) - _KEYWORDS)
    if unknown: return f"{path} uses unsupported keywords: {unknown}"
    declared = schema.get("type")
    if declared is None: return f"{path} must declare type"
    allowed = declared if isinstance(declared, list) else [declared]
    if not allowed or not all(isinstance(item, str) and item in _TYPES for item in allowed):
        return f"{path}.type must contain supported JSON types"
    if "required" in schema and (not isinstance(schema["required"], list) or not all(isinstance(x, str) for x in schema["required"])):
        return f"{path}.required must be a list of strings"
    if "enum" in schema and not isinstance(schema["enum"], list): return f"{path}.enum must be an array"
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool): return f"{path}.additionalProperties must be boolean"
    if "properties" in schema:
        if not isinstance(schema["properties"], dict): return f"{path}.properties must be an object"
        for key, child in schema["properties"].items():
            err = _validate_subschema(child, f"{path}.properties.{key}")
            if err: return err
        missing = set(schema.get("required", [])) - set(schema["properties"])
        if missing: return f"{path}.required references undeclared properties: {sorted(missing)}"
    if "items" in schema:
        err = _validate_subschema(schema["items"], f"{path}.items")
        if err: return err
    return None


def validate_schema_document(document: Any) -> str | None:
    if not isinstance(document, dict): return "schema document must be a JSON object"
    for key in ("schema_version", "title", "type", "properties"):
        if key not in document: return f"schema document missing required key {key!r}"
    if not isinstance(document["schema_version"], str) or not document["schema_version"]: return "schema_version must be a non-empty string"
    if not isinstance(document["title"], str) or not document["title"]: return "title must be a non-empty string"
    if document["type"] != "object": return "top-level schema type must be 'object'"
    return _validate_subschema(document, "$")


def validate_instance(schema: dict[str, Any], instance: Any, *, path: str = "$") -> list[str]:
    """Return precise errors for the supported schema subset."""
    schema_error = _validate_subschema(schema, path)
    if schema_error: return [schema_error]
    errors: list[str] = []
    allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    if not any(_type_matches(instance, item) for item in allowed):
        errors.append(f"{path}: expected type {allowed}, got {_json_type_name(instance)}")
        return errors
    if _json_type_name(instance) == "number" and not _finite_number(instance):
        errors.append(f"{path}: number must be finite")
        return errors
    if "enum" in schema and instance not in schema["enum"]: errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance: errors.append(f"{path}: missing required field {name!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            if extra: errors.append(f"{path}: unexpected fields not permitted by schema: {extra}")
        for name, value in instance.items():
            if name in properties: errors.extend(validate_instance(properties[name], value, path=f"{path}.{name}"))
    if isinstance(instance, list) and "items" in schema:
        for i, value in enumerate(instance): errors.extend(validate_instance(schema["items"], value, path=f"{path}[{i}]"))
    return errors
