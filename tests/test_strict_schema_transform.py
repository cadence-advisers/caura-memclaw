"""Unit tests for the strict-schema transform (BP-7 — Anthropic compat).

Anthropic's OpenAI-compatible endpoint requires json_schema.strict=true AND a
strict-compliant schema (every object: additionalProperties=false + all props
required). ``_to_strict_schema`` makes a Pydantic schema compliant. Verified live
against api.anthropic.com on the ExtractedGraph schema; these tests pin the
transform's shape behaviour.
"""
import copy

from common.llm.providers.openai import _to_strict_schema


def test_object_gets_additional_properties_false_and_all_required():
    s = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    out = _to_strict_schema(s)
    assert out["additionalProperties"] is False
    assert sorted(out["required"]) == ["a", "b"]


def test_recurses_into_items_defs_and_combiners():
    s = {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}},
            "opt": {"anyOf": [{"type": "object", "properties": {"y": {"type": "string"}}}, {"type": "null"}]},
        },
        "$defs": {"D": {"type": "object", "properties": {"z": {"type": "string"}}}},
    }
    out = _to_strict_schema(s)
    assert out["properties"]["rows"]["items"]["additionalProperties"] is False
    assert out["properties"]["rows"]["items"]["required"] == ["x"]
    assert out["properties"]["opt"]["anyOf"][0]["additionalProperties"] is False
    assert out["$defs"]["D"]["additionalProperties"] is False
    assert out["$defs"]["D"]["required"] == ["z"]


def test_is_pure_does_not_mutate_input():
    s = {"type": "object", "properties": {"a": {"type": "string"}}}
    before = copy.deepcopy(s)
    _to_strict_schema(s)
    assert s == before


def test_non_dict_passthrough():
    assert _to_strict_schema("x") == "x"
    assert _to_strict_schema([1, 2]) == [1, 2]
