"""BP-10 — Anthropic schemaless-JSON path + lenient parsing.

Anthropic's OpenAI-compatible endpoint rejects ``response_format={"type":
"json_object"}`` (400: "response_format.type: Input should be 'json_schema'").
The schemaless callers (enrichment, contradiction detection) call
``complete_json`` with no schema, so for Anthropic we omit ``response_format``
and parse the reply leniently. These tests pin both behaviours.
"""

import json
import types

import pytest

from common.llm.providers.openai import OpenAILLMProvider, _loads_json_lenient
from common.provider_names import ProviderName


# --- lenient parser (pure) ------------------------------------------------


def test_plain_json_object():
    assert _loads_json_lenient('{"a": 1}') == {"a": 1}


def test_fenced_json_block():
    assert _loads_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_bare_fence_no_lang():
    assert _loads_json_lenient('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_then_object():
    assert _loads_json_lenient('Here is the result:\n{"a": 1, "b": [2,3]}') == {
        "a": 1,
        "b": [2, 3],
    }


def test_array_payload():
    assert _loads_json_lenient("[1, 2, 3]") == [1, 2, 3]


def test_unparseable_raises():
    with pytest.raises(json.JSONDecodeError):
        _loads_json_lenient("not json at all")


# --- response_format selection (light async monkeypatch) ------------------


def _provider(provider_name: str) -> OpenAILLMProvider:
    # Construction opens no network connection; we replace the create() call.
    return OpenAILLMProvider(api_key="k", model="m", provider_name=provider_name)


def _patch_capture(
    provider: OpenAILLMProvider, captured: dict, content: str = '{"ok": true}'
):
    async def fake_create(**kwargs):
        captured.update(kwargs)
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    provider._client.chat.completions.create = fake_create  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_anthropic_no_schema_omits_response_format():
    p = _provider(ProviderName.ANTHROPIC)
    cap: dict = {}
    _patch_capture(p, cap)
    await p.complete_json("give me json")
    assert "response_format" not in cap  # Anthropic 400s on json_object


@pytest.mark.asyncio
async def test_anthropic_no_schema_parses_fenced_reply():
    p = _provider(ProviderName.ANTHROPIC)
    cap: dict = {}
    _patch_capture(p, cap, content='```json\n{"title": "Hi"}\n```')
    assert await p.complete_json("give me json") == {"title": "Hi"}


@pytest.mark.asyncio
async def test_openai_no_schema_keeps_json_object():
    p = _provider(ProviderName.OPENAI)
    cap: dict = {}
    _patch_capture(p, cap)
    await p.complete_json("give me json")
    assert cap["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_anthropic_with_schema_uses_strict_json_schema():
    p = _provider(ProviderName.ANTHROPIC)
    cap: dict = {}
    _patch_capture(p, cap)
    await p.complete_json(
        "x", response_schema={"type": "object", "properties": {"a": {"type": "string"}}}
    )
    rf = cap["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
