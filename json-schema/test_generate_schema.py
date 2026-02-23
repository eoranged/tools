"""Tests for generate_schema — JSON Schema generator with streaming parser.

Each test writes one or more JSON fixture files, runs generate_schema(),
and then validates that **every** original object conforms to the produced
schema.  This guarantees the generated schema is both structurally correct
and complete.
"""

import json
import random

import jsonschema
import pytest

from generate_schema import (
    generate_schema,
    infer_schema,
    merge_schemas,
    minimal_schema,
    validates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_json_array(path, objects):
    """Write *objects* as a JSON array file and return the path."""
    fp = path / "data.json"
    fp.write_text(json.dumps(objects))
    return str(fp)


def write_ndjson(path, objects, filename="data.ndjson"):
    """Write *objects* as newline-delimited JSON and return the path."""
    fp = path / filename
    fp.write_text("\n".join(json.dumps(o) for o in objects) + "\n")
    return str(fp)


def assert_schema_validates_all(schema, objects):
    """Assert that every object in *objects* passes validation."""
    for i, obj in enumerate(objects):
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as exc:
            pytest.fail(
                f"Object at index {i} failed validation: {exc.message}\n"
                f"Object: {json.dumps(obj)}\n"
                f"Schema: {json.dumps(schema, indent=2)}"
            )


# ===================================================================
# Unit tests — infer / merge helpers
# ===================================================================

class TestInferSchema:
    def test_null(self):
        assert infer_schema(None) == {"type": "null"}

    def test_bool(self):
        assert infer_schema(True) == {"type": "boolean"}
        assert infer_schema(False) == {"type": "boolean"}

    def test_bool_not_int(self):
        """bool must not be confused with int."""
        assert infer_schema(True)["type"] == "boolean"

    def test_integer(self):
        assert infer_schema(42) == {"type": "integer"}

    def test_float(self):
        assert infer_schema(3.14) == {"type": "number"}

    def test_string(self):
        assert infer_schema("hello") == {"type": "string"}

    def test_empty_list(self):
        assert infer_schema([]) == {"type": "array"}

    def test_list_of_ints(self):
        s = infer_schema([1, 2, 3])
        assert s == {"type": "array", "items": {"type": "integer"}}

    def test_list_mixed(self):
        s = infer_schema([1, "two"])
        assert s["type"] == "array"
        assert "anyOf" in s["items"]

    def test_object(self):
        s = infer_schema({"a": 1, "b": "x"})
        assert s["type"] == "object"
        assert s["properties"]["a"] == {"type": "integer"}
        assert s["properties"]["b"] == {"type": "string"}
        assert sorted(s["required"]) == ["a", "b"]
        assert s["additionalProperties"] is False


class TestMergeSchemas:
    def test_same_primitive(self):
        assert merge_schemas({"type": "string"}, {"type": "string"}) == {"type": "string"}

    def test_int_and_number(self):
        assert merge_schemas({"type": "integer"}, {"type": "number"}) == {"type": "number"}
        assert merge_schemas({"type": "number"}, {"type": "integer"}) == {"type": "number"}

    def test_different_types_become_anyof(self):
        m = merge_schemas({"type": "string"}, {"type": "integer"})
        assert "anyOf" in m
        types = {v["type"] for v in m["anyOf"]}
        assert types == {"string", "integer"}

    def test_objects_union_properties(self):
        s1 = infer_schema({"a": 1})
        s2 = infer_schema({"b": "x"})
        m = _merge_and_check(s1, s2)
        assert "a" in m["properties"]
        assert "b" in m["properties"]
        # Neither field is required (only present in one side)
        assert "required" not in m or m.get("required") == []

    def test_objects_required_intersection(self):
        s1 = infer_schema({"a": 1, "b": 2})
        s2 = infer_schema({"a": 3, "c": 4})
        m = _merge_and_check(s1, s2)
        assert m.get("required") == ["a"]

    def test_empty_with_value(self):
        assert merge_schemas({}, {"type": "string"}) == {"type": "string"}
        assert merge_schemas({"type": "string"}, {}) == {"type": "string"}

    def test_arrays_merge_items(self):
        s1 = {"type": "array", "items": {"type": "integer"}}
        s2 = {"type": "array", "items": {"type": "string"}}
        m = merge_schemas(s1, s2)
        assert m["type"] == "array"
        assert "anyOf" in m["items"]


def _merge_and_check(s1, s2):
    m = merge_schemas(s1, s2)
    assert m["type"] == "object"
    return m


# ===================================================================
# Integration tests — full generate_schema on JSON files
# ===================================================================

class TestHomogeneousObjects:
    """All objects have exactly the same shape."""

    def test_flat_strings(self, tmp_path):
        objects = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        fp = write_json_array(tmp_path, objects)
        schema, iters, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 3
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema.get("required", [])
        assert_schema_validates_all(schema, objects)

    def test_multiple_fields(self, tmp_path):
        objects = [
            {"id": 1, "name": "a", "active": True},
            {"id": 2, "name": "b", "active": False},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["id"]["type"] == "integer"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["active"]["type"] == "boolean"
        assert_schema_validates_all(schema, objects)


class TestOptionalFields:
    """Some objects have fields others lack."""

    def test_one_optional_field(self, tmp_path):
        objects = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b", "email": "b@test.com"},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        # email is optional — not in required
        required = set(schema.get("required", []))
        assert "email" not in required
        assert "id" in required
        assert "name" in required
        # but it IS in properties
        assert "email" in schema["properties"]
        assert_schema_validates_all(schema, objects)

    def test_many_optional_fields(self, tmp_path):
        objects = [
            {"a": 1},
            {"a": 2, "b": "x"},
            {"a": 3, "c": True},
            {"a": 4, "b": "y", "c": False, "d": 0.5},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert set(schema["properties"].keys()) == {"a", "b", "c", "d"}
        assert schema.get("required") == ["a"]
        assert_schema_validates_all(schema, objects)


class TestMixedTypes:
    """A single field has different types across objects."""

    def test_string_or_null(self, tmp_path):
        objects = [
            {"value": "hello"},
            {"value": None},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        val_schema = schema["properties"]["value"]
        assert "anyOf" in val_schema
        types = {v["type"] for v in val_schema["anyOf"]}
        assert types == {"string", "null"}
        assert_schema_validates_all(schema, objects)

    def test_integer_and_float(self, tmp_path):
        objects = [{"v": 1}, {"v": 2.5}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["v"]["type"] == "number"
        assert_schema_validates_all(schema, objects)

    def test_string_int_bool(self, tmp_path):
        objects = [{"x": "a"}, {"x": 1}, {"x": True}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        val = schema["properties"]["x"]
        assert "anyOf" in val
        assert_schema_validates_all(schema, objects)


class TestNestedObjects:
    """Objects containing sub-objects."""

    def test_simple_nesting(self, tmp_path):
        objects = [
            {"user": {"name": "Alice", "age": 30}},
            {"user": {"name": "Bob", "age": 25}},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        user_props = schema["properties"]["user"]["properties"]
        assert user_props["name"]["type"] == "string"
        assert user_props["age"]["type"] == "integer"
        assert_schema_validates_all(schema, objects)

    def test_nested_with_optional_subfield(self, tmp_path):
        objects = [
            {"address": {"city": "NYC"}},
            {"address": {"city": "LA", "zip": "90001"}},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        addr = schema["properties"]["address"]
        assert "city" in addr["properties"]
        assert "zip" in addr["properties"]
        assert "zip" not in addr.get("required", [])
        assert "city" in addr.get("required", [])
        assert_schema_validates_all(schema, objects)

    def test_deeply_nested(self, tmp_path):
        objects = [
            {"a": {"b": {"c": {"d": 1}}}},
            {"a": {"b": {"c": {"d": 2, "e": "x"}}}},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        d_level = schema["properties"]["a"]["properties"]["b"]["properties"]["c"]
        assert "d" in d_level["properties"]
        assert "e" in d_level["properties"]
        assert_schema_validates_all(schema, objects)


class TestArrayFields:
    """Objects containing array fields."""

    def test_array_of_strings(self, tmp_path):
        objects = [
            {"tags": ["a", "b"]},
            {"tags": ["c"]},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"]["type"] == "string"
        assert_schema_validates_all(schema, objects)

    def test_empty_and_nonempty_arrays(self, tmp_path):
        objects = [
            {"items": []},
            {"items": [1, 2, 3]},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["items"]["type"] == "array"
        assert_schema_validates_all(schema, objects)

    def test_array_of_objects(self, tmp_path):
        objects = [
            {"records": [{"id": 1}, {"id": 2}]},
            {"records": [{"id": 3, "label": "x"}]},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        items_schema = schema["properties"]["records"]["items"]
        assert items_schema["type"] == "object"
        assert "id" in items_schema["properties"]
        assert "label" in items_schema["properties"]
        assert_schema_validates_all(schema, objects)

    def test_mixed_type_array(self, tmp_path):
        objects = [{"vals": [1, "two", None]}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["vals"]["type"] == "array"
        assert "anyOf" in schema["properties"]["vals"]["items"]
        assert_schema_validates_all(schema, objects)


class TestNDJSON:
    """NDJSON (newline-delimited JSON) input."""

    def test_basic_ndjson(self, tmp_path):
        objects = [
            {"id": 1, "msg": "hello"},
            {"id": 2, "msg": "world"},
        ]
        fp = write_ndjson(tmp_path, objects)
        schema, _, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 2
        assert_schema_validates_all(schema, objects)

    def test_ndjson_with_optional(self, tmp_path):
        objects = [
            {"ts": 100, "level": "info", "msg": "ok"},
            {"ts": 200, "level": "error", "msg": "fail", "trace": "..."},
        ]
        fp = write_ndjson(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert "trace" in schema["properties"]
        assert "trace" not in schema.get("required", [])
        assert_schema_validates_all(schema, objects)

    def test_explicit_ndjson_format(self, tmp_path):
        """Force ndjson format with fmt parameter."""
        objects = [{"x": 1}, {"x": 2}]
        fp = write_ndjson(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp, fmt="ndjson")
        assert complete
        assert_schema_validates_all(schema, objects)


class TestAutoDetectFormat:
    """Format auto-detection from file content."""

    def test_detects_array(self, tmp_path):
        objects = [{"a": 1}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert_schema_validates_all(schema, objects)

    def test_detects_ndjson(self, tmp_path):
        objects = [{"a": 1}]
        fp = write_ndjson(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert_schema_validates_all(schema, objects)

    def test_leading_whitespace_array(self, tmp_path):
        """Array file with leading whitespace still detected correctly."""
        fp = tmp_path / "data.json"
        fp.write_text("  \n  [" + json.dumps({"x": 1}) + "]")
        schema, _, _, complete = generate_schema(str(fp))
        assert complete
        assert_schema_validates_all(schema, [{"x": 1}])


class TestIterationLimit:
    """Max-iterations cap and continuation."""

    def test_stops_at_limit(self, tmp_path):
        # Each object introduces a new property → each triggers a schema update
        objects = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
        fp = write_ndjson(tmp_path, objects)
        schema, iters, seen, complete = generate_schema(fp, max_iterations=2)
        assert not complete
        assert iters == 2
        # seen ≤ total because we stop mid-stream
        assert seen <= 5

    def test_continue_with_schema(self, tmp_path):
        objects = [
            {"a": 1, "b": "x"},
            {"a": 2, "c": True},
            {"a": 3, "d": 0.5},
        ]
        fp = write_ndjson(tmp_path, objects)

        # First pass — limited to 1 update
        schema1, _, _, complete1 = generate_schema(fp, max_iterations=1)
        assert not complete1

        # Second pass — continue with schema from first pass
        schema2, _, _, complete2 = generate_schema(fp, schema=schema1)
        assert complete2
        assert_schema_validates_all(schema2, objects)

    def test_full_continuation_cycle(self, tmp_path):
        """Simulate a real continuation workflow across multiple runs."""
        objects = [{"k" + str(i): i} for i in range(20)]
        fp = write_ndjson(tmp_path, objects)

        schema = None
        total_iters = 0
        for _ in range(50):  # safety limit
            schema, iters, _, complete = generate_schema(
                fp, schema=schema, max_iterations=5,
            )
            total_iters += iters
            if complete:
                break
        assert complete
        assert_schema_validates_all(schema, objects)


class TestLargeFile:
    """Larger generated datasets to exercise streaming."""

    def test_1000_uniform_objects(self, tmp_path):
        objects = [{"id": i, "name": f"user_{i}"} for i in range(1000)]
        fp = write_json_array(tmp_path, objects)
        schema, iters, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 1000
        # Only 1 update needed (first object defines the shape)
        assert iters == 1
        assert_schema_validates_all(schema, objects[:10])  # spot-check
        assert_schema_validates_all(schema, objects[-10:])

    def test_varied_shapes(self, tmp_path):
        rng = random.Random(42)
        objects = []
        for i in range(500):
            obj = {"id": i}
            if rng.random() < 0.5:
                obj["name"] = f"u{i}"
            if rng.random() < 0.3:
                obj["score"] = rng.uniform(0, 100)
            if rng.random() < 0.2:
                obj["tags"] = [f"t{j}" for j in range(rng.randint(0, 3))]
            if rng.random() < 0.1:
                obj["meta"] = {"src": "api"}
            objects.append(obj)
        fp = write_json_array(tmp_path, objects)
        schema, _, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 500
        assert_schema_validates_all(schema, objects)


class TestNullableFields:
    """Fields that alternate between a value and null."""

    def test_nullable_string(self, tmp_path):
        objects = [
            {"name": "Alice"},
            {"name": None},
            {"name": "Bob"},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        val = schema["properties"]["name"]
        assert "anyOf" in val
        types = {v["type"] for v in val["anyOf"]}
        assert types == {"string", "null"}
        assert_schema_validates_all(schema, objects)

    def test_nullable_nested_object(self, tmp_path):
        objects = [
            {"data": {"x": 1}},
            {"data": None},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert_schema_validates_all(schema, objects)


class TestBooleanNotInt:
    """Booleans must remain booleans and not collapse to integer."""

    def test_bool_and_int_separate(self, tmp_path):
        objects = [{"flag": True}, {"flag": 1}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert_schema_validates_all(schema, objects)

    def test_pure_bool_field(self, tmp_path):
        objects = [
            {"active": True, "id": 1},
            {"active": False, "id": 2},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["active"]["type"] == "boolean"
        assert_schema_validates_all(schema, objects)


class TestEdgeCases:
    """Edge cases and corner scenarios."""

    def test_empty_array_file(self, tmp_path):
        fp = write_json_array(tmp_path, [])
        schema, iters, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 0
        assert iters == 0
        # Schema is the minimal empty-object schema
        assert schema == minimal_schema()

    def test_single_object(self, tmp_path):
        objects = [{"only": "one"}]
        fp = write_json_array(tmp_path, objects)
        schema, iters, seen, complete = generate_schema(fp)
        assert complete
        assert seen == 1
        assert iters == 1
        assert_schema_validates_all(schema, objects)

    def test_all_null_values(self, tmp_path):
        objects = [{"a": None, "b": None}] * 3
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["a"]["type"] == "null"
        assert_schema_validates_all(schema, objects)

    def test_empty_nested_object(self, tmp_path):
        objects = [{"meta": {}}]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["meta"]["type"] == "object"
        assert_schema_validates_all(schema, objects)

    def test_unicode_strings(self, tmp_path):
        objects = [
            {"text": "hello"},
            {"text": "\u00e9\u00e8\u00ea"},
            {"text": "\u4f60\u597d"},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert_schema_validates_all(schema, objects)


class TestComplexRealWorld:
    """Realistic multi-shape datasets."""

    def test_user_records(self, tmp_path):
        objects = [
            {"id": 1, "name": "Alice", "email": "alice@test.com",
             "age": 30, "roles": ["admin", "user"]},
            {"id": 2, "name": "Bob", "age": 25,
             "roles": ["user"]},
            {"id": 3, "name": "Carol", "email": "carol@test.com",
             "age": None, "roles": [],
             "address": {"street": "123 Main", "city": "NYC"}},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        props = schema["properties"]
        assert "id" in props
        assert "name" in props
        assert "email" in props
        assert "roles" in props
        assert "address" in props
        required = set(schema.get("required", []))
        assert {"id", "name", "roles"} <= required
        assert "email" not in required
        assert "address" not in required
        assert_schema_validates_all(schema, objects)

    def test_api_responses(self, tmp_path):
        objects = [
            {"status": 200, "data": {"items": [{"id": 1}], "total": 1}},
            {"status": 200, "data": {"items": [], "total": 0}},
            {"status": 404, "error": "not found"},
            {"status": 500, "error": "internal", "trace": "stack..."},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert "status" in schema["properties"]
        assert "data" in schema["properties"]
        assert "error" in schema["properties"]
        assert_schema_validates_all(schema, objects)

    def test_event_log(self, tmp_path):
        objects = [
            {"ts": 1000, "event": "click", "x": 100, "y": 200},
            {"ts": 1001, "event": "scroll", "dx": 0, "dy": -50},
            {"ts": 1002, "event": "resize", "w": 1920, "h": 1080},
            {"ts": 1003, "event": "click", "x": 150, "y": 300,
             "target": {"id": "btn1", "class": "primary"}},
        ]
        fp = write_ndjson(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert set(schema.get("required", [])) == {"ts", "event"}
        assert_schema_validates_all(schema, objects)

    def test_mixed_numeric_precision(self, tmp_path):
        """Integers and floats for the same field merge to number."""
        objects = [
            {"value": 1},
            {"value": 1.5},
            {"value": 2},
            {"value": 2.7},
        ]
        fp = write_json_array(tmp_path, objects)
        schema, _, _, complete = generate_schema(fp)
        assert complete
        assert schema["properties"]["value"]["type"] == "number"
        assert_schema_validates_all(schema, objects)


class TestCLI:
    """Test the CLI via subprocess to verify end-to-end behavior."""

    def test_stdout_output(self, tmp_path):
        import subprocess
        objects = [{"a": 1}, {"a": 2}]
        fp = write_json_array(tmp_path, objects)
        result = subprocess.run(
            ["python3", "generate_schema.py", fp],
            capture_output=True, text=True,
            cwd="/home/user/tools/json-schema",
        )
        assert result.returncode == 0
        schema = json.loads(result.stdout)
        assert schema["properties"]["a"]["type"] == "integer"

    def test_output_file(self, tmp_path):
        import subprocess
        objects = [{"x": "y"}]
        fp = write_json_array(tmp_path, objects)
        out = str(tmp_path / "schema.json")
        result = subprocess.run(
            ["python3", "generate_schema.py", fp, "-o", out],
            capture_output=True, text=True,
            cwd="/home/user/tools/json-schema",
        )
        assert result.returncode == 0
        schema = json.loads((tmp_path / "schema.json").read_text())
        assert_schema_validates_all(schema, objects)

    def test_schema_continuation_flag(self, tmp_path):
        import subprocess
        objects = [{"a": 1}, {"b": 2}, {"c": 3}]
        fp = write_ndjson(tmp_path, objects)
        partial = str(tmp_path / "partial.json")

        # Pass 1 — limit to 1 update
        subprocess.run(
            ["python3", "generate_schema.py", fp, "-n", "1", "-o", partial],
            capture_output=True, text=True,
            cwd="/home/user/tools/json-schema",
        )
        assert (tmp_path / "partial.json").exists()

        # Pass 2 — continue
        final = str(tmp_path / "final.json")
        subprocess.run(
            ["python3", "generate_schema.py", fp, "-s", partial, "-o", final],
            capture_output=True, text=True,
            cwd="/home/user/tools/json-schema",
        )
        schema = json.loads((tmp_path / "final.json").read_text())
        assert_schema_validates_all(schema, objects)

    def test_stderr_stats(self, tmp_path):
        import subprocess
        objects = [{"a": 1}]
        fp = write_json_array(tmp_path, objects)
        result = subprocess.run(
            ["python3", "generate_schema.py", fp],
            capture_output=True, text=True,
            cwd="/home/user/tools/json-schema",
        )
        assert "Objects scanned: 1" in result.stderr
        assert "Schema updates:" in result.stderr
        assert "complete" in result.stderr
