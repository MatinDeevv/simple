"""Minimal, dependency-free JSON-schema-subset validator.

This intentionally implements a small fixed subset of JSON Schema (type,
required, properties, items, enum, additionalProperties) rather than pulling
in a full JSON Schema library. It is enough to catch the failure modes this
repository cares about: missing required fields, type drift, disallowed NaN
(modeled as JSON ``null`` plus an explicit ``"nullable": false`` contract),
invalid enum values, and unexpected extra fields when a schema is strict.

Schema documents in ``config/schemas/*.schema.json`` use this convention:

    {
      "schema_version": "fxsim-<name>-v1",
      "title": "human title",
      "type": "object",
      "required": ["field", ...],
      "properties": {"field": {"type": "string", ...}, ...},
      "additionalProperties": true | false
    }
"""

from __future__ import annotations

from typing import Any

_PY_TYPE_NAMES = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    for python_type, name in _PY_TYPE_NAMES.items():
        if python_type is bool or python_type is int:
            continue
        if isinstance(value, python_type):
            return name
    return type(value).__name__


def _type_matches(value: Any, declared: str) -> bool:
    actual = _json_type_name(value)
    if declared == "number":
        return actual in ("number", "integer")
    return actual == declared


def validate_schema_document(document: Any) -> str | None:
    """Return an error string if ``document`` is not a well-formed schema, else None."""
    if not isinstance(document, dict):
        return "schema document must be a JSON object"
    for key in ("schema_version", "title", "type", "properties"):
        if key not in document:
            return f"schema document missing required key {key!r}"
    if not isinstance(document["schema_version"], str) or not document["schema_version"]:
        return "schema_version must be a non-empty string"
    if not isinstance(document["title"], str) or not document["title"]:
        return "title must be a non-empty string"
    if document["type"] != "object":
        return "top-level schema type must be 'object'"
    properties = document["properties"]
    if not isinstance(properties, dict):
        return "properties must be an object"
    for name, subschema in properties.items():
        if not isinstance(subschema, dict) or "type" not in subschema:
            return f"properties.{name} must declare a type"
    required = document.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        return "required must be a list of strings"
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        return f"required references undeclared properties: {unknown_required}"
    return None


def validate_instance(schema: dict[str, Any], instance: Any, *, path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``; return a list of error strings (empty if valid)."""
    errors: list[str] = []
    declared_type = schema.get("type")
    if declared_type is not None:
        allowed = declared_type if isinstance(declared_type, list) else [declared_type]
        if not any(_type_matches(instance, option) for option in allowed):
            errors.append(f"{path}: expected type {allowed}, got {_json_type_name(instance)}")
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
                errors.extend(validate_instance(subschema, value, path=f"{path}.{name}"))
    if isinstance(instance, list) and schema.get("type") == "array":
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(instance):
                errors.extend(validate_instance(item_schema, item, path=f"{path}[{index}]"))
    return errors
