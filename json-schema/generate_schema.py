#!/usr/bin/env python3
"""Generate JSON Schema from JSON data using streaming parsing.

Iteratively refines a schema by finding non-matching objects and updating
the schema to accommodate them. Uses ijson for memory-efficient streaming
of large files.

Usage:
    python generate_schema.py data.json
    python generate_schema.py data.json -o schema.json
    python generate_schema.py data.json -s existing_schema.json -o refined.json
    python generate_schema.py data.ndjson -f ndjson -n 500
    python generate_schema.py wrapped.json -p objects     # {"objects": [...]}
"""

import argparse
import json
import sys
from decimal import Decimal

import ijson
import jsonschema


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def minimal_schema():
    """Return a minimal schema that matches only an empty object."""
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def infer_schema(value):
    """Infer a JSON Schema from a single Python value."""
    if value is None:
        return {"type": "null"}
    # bool before int — bool is a subclass of int in Python
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, (float, Decimal)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if not value:
            return {"type": "array"}
        items_schema = infer_schema(value[0])
        for item in value[1:]:
            items_schema = merge_schemas(items_schema, infer_schema(item))
        return {"type": "array", "items": items_schema}
    if isinstance(value, dict):
        properties = {k: infer_schema(v) for k, v in value.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": sorted(properties.keys()),
            "additionalProperties": False,
        }
    return {}


# ---------------------------------------------------------------------------
# Schema merging
# ---------------------------------------------------------------------------

def merge_schemas(s1, s2):
    """Merge two schemas so the result accepts values valid under either."""
    if not s1:
        return s2
    if not s2:
        return s1

    # Handle anyOf on either side
    if "anyOf" in s1 or "anyOf" in s2:
        variants1 = s1.get("anyOf", [s1])
        variants2 = s2.get("anyOf", [s2])
        merged = list(variants1)
        for v2 in variants2:
            matched = False
            for i, v1 in enumerate(merged):
                if _same_type(v1, v2):
                    merged[i] = merge_schemas(v1, v2)
                    matched = True
                    break
            if not matched:
                merged.append(v2)
        return {"anyOf": merged} if len(merged) > 1 else merged[0]

    t1 = s1.get("type")
    t2 = s2.get("type")

    if t1 == t2:
        if t1 == "object":
            return _merge_objects(s1, s2)
        if t1 == "array":
            return _merge_arrays(s1, s2)
        return s1  # identical primitive type

    # integer + number → number
    if {t1, t2} == {"integer", "number"}:
        return {"type": "number"}

    return {"anyOf": [s1, s2]}


def _same_type(s1, s2):
    t1 = s1.get("type")
    t2 = s2.get("type")
    return t1 is not None and t1 == t2


def _merge_objects(s1, s2):
    p1 = s1.get("properties", {})
    p2 = s2.get("properties", {})
    r1 = set(s1.get("required", []))
    r2 = set(s2.get("required", []))

    merged_props = {}
    for key in set(p1) | set(p2):
        if key in p1 and key in p2:
            merged_props[key] = merge_schemas(p1[key], p2[key])
        else:
            merged_props[key] = p1.get(key) or p2.get(key)

    # When one side has no properties (e.g. the minimal seed schema),
    # adopt the other side's required set — the empty side has no opinion.
    # Otherwise, a field is required only if required in both schemas.
    if not p1:
        required = sorted(r2 & set(merged_props))
    elif not p2:
        required = sorted(r1 & set(merged_props))
    else:
        required = sorted(r1 & r2)

    result = {
        "type": "object",
        "properties": merged_props,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _merge_arrays(s1, s2):
    i1 = s1.get("items")
    i2 = s2.get("items")
    if i1 and i2:
        return {"type": "array", "items": merge_schemas(i1, i2)}
    if i1 or i2:
        return {"type": "array", "items": i1 or i2}
    return {"type": "array"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validates(instance, schema):
    """Return True if *instance* conforms to *schema*."""
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------

def stream_objects(filepath, fmt=None, path=None):
    """Yield objects from a JSON file (JSON array, NDJSON, or nested array).

    *fmt* can be ``"array"``, ``"ndjson"``, or ``None`` (auto-detect).
    *path* is a dotted key path to the array inside a JSON object
    (e.g. ``"objects"`` for ``{"objects": [...]}``, or ``"data.items"``
    for ``{"data": {"items": [...]}}``.  When given, the file is always
    read as JSON (not NDJSON).
    """
    if path is not None:
        yield from _read_json_array(filepath, path)
        return
    if fmt == "ndjson":
        yield from _read_ndjson(filepath)
        return
    if fmt == "array":
        yield from _read_json_array(filepath)
        return
    # auto-detect by first non-whitespace byte
    with open(filepath, "rb") as fh:
        ch = fh.read(1)
        while ch and ch in b" \t\n\r":
            ch = fh.read(1)
    if ch == b"[":
        yield from _read_json_array(filepath)
    elif ch == b"{":
        # Object wrapper — find the first array key automatically
        detected = _detect_array_path(filepath)
        if detected is not None:
            yield from _read_json_array(filepath, detected)
        else:
            # Single object, no array found — yield it as one item
            yield from _read_ndjson(filepath)
    else:
        yield from _read_ndjson(filepath)


def _read_json_array(filepath, path=None):
    """Stream items from a JSON array.

    *path* is an optional dotted key path (e.g. ``"objects"`` or
    ``"data.items"``).  The ijson prefix is built as ``"<path>.item"``
    when a path is given, or just ``"item"`` for a top-level array.
    """
    prefix = f"{path}.item" if path else "item"
    with open(filepath, "rb") as fh:
        for obj in ijson.items(fh, prefix, use_float=True):
            yield obj


def _detect_array_path(filepath):
    """Return the dotted key path to the first array inside a JSON object.

    Uses ijson's event-based parser so only the beginning of the file is
    read — no need to load the whole thing.  Returns ``None`` if no array
    is found (or if the file isn't valid single-document JSON, e.g. NDJSON).
    """
    try:
        with open(filepath, "rb") as fh:
            parser = ijson.parse(fh, use_float=True)
            for prefix, event, _value in parser:
                if event == "start_array" and prefix:
                    return prefix
                # Stop early if we hit a deeply nested value without an array
                if prefix.count(".") > 10:
                    break
    except ijson.common.IncompleteJSONError:
        # Not a single JSON document (e.g. NDJSON) — no array path to detect
        pass
    return None


def _read_ndjson(filepath):
    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def generate_schema(filepath, schema=None, max_iterations=1000, fmt=None,
                    path=None):
    """Generate or refine a JSON schema by streaming through *filepath*.

    Returns ``(schema, iterations, objects_seen, complete)``.

    *complete* is ``True`` when every object in the file matched the final
    schema; ``False`` when the iteration limit was hit first.
    *path* is an optional dotted key to the array inside a wrapper object.
    """
    if schema is None:
        schema = minimal_schema()

    iterations = 0
    objects_seen = 0

    for obj in stream_objects(filepath, fmt, path=path):
        objects_seen += 1
        if not validates(obj, schema):
            schema = merge_schemas(schema, infer_schema(obj))
            iterations += 1
            if iterations >= max_iterations:
                return schema, iterations, objects_seen, False

    return schema, iterations, objects_seen, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate JSON Schema from a JSON file using streaming.",
    )
    ap.add_argument("input", help="Input JSON file (array or NDJSON)")
    ap.add_argument("-o", "--output", help="Output schema file (default: stdout)")
    ap.add_argument(
        "-s", "--schema",
        help="Existing schema file to continue refining",
    )
    ap.add_argument(
        "-n", "--max-iterations",
        type=int,
        default=1000,
        help="Max number of schema updates (default: 1000)",
    )
    ap.add_argument(
        "-p", "--path",
        help='Dotted key path to the array inside a JSON object '
             '(e.g. "objects" for {"objects": [...]}, '
             '"data.items" for {"data": {"items": [...]}}). '
             'Auto-detected when omitted.',
    )
    ap.add_argument(
        "-f", "--format",
        choices=["array", "ndjson", "auto"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON output indentation (default: 2)",
    )
    args = ap.parse_args()

    # Load existing schema if provided
    existing = None
    if args.schema:
        with open(args.schema) as fh:
            existing = json.load(fh)

    fmt = None if args.format == "auto" else args.format

    schema, iterations, seen, complete = generate_schema(
        args.input,
        schema=existing,
        max_iterations=args.max_iterations,
        fmt=fmt,
        path=args.path,
    )

    out = json.dumps(schema, indent=args.indent, sort_keys=False)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out + "\n")
    else:
        print(out)

    # Stats to stderr so they don't pollute piped output
    print(f"\nObjects scanned: {seen}", file=sys.stderr)
    print(f"Schema updates:  {iterations}", file=sys.stderr)
    if complete:
        print("Status: complete - schema matches all objects", file=sys.stderr)
    else:
        print(
            f"Status: incomplete - stopped after {iterations} updates "
            f"(limit: {args.max_iterations}). "
            f"Re-run with -s <schema-file> to continue.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
