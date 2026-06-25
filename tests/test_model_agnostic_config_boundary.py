"""Config-boundary guard (BP-7): the brain must be model-agnostic.

Mandate: changing the model/provider/embedding-dimension is a CONFIG change,
never a code change — and that boundary is enforced here so it cannot regress.

This guard scans all RUNTIME source and fails if a hardcoded model identifier,
an LLM base URL, or a provider string literal *used for selection* appears
OUTSIDE the explicit allowlist of config-boundary files. Model/provider
specifics belong only in the narrow config boundary (provider constants,
credential resolution, env-config, provider adapters, the ProviderName enum,
and the now config-driven per-tenant defaults). Everything else routes through
``common.llm.registry`` / ``common.embedding`` by name.

The scan is AST-based:
  * Comments are not in the AST, so they are ignored for free.
  * Docstrings are detected and skipped.
  * Prompt-template constants (UPPERCASE names containing PROMPT / TEMPLATE)
    are skipped for the model-identifier check, because few-shot prompts
    legitimately contain illustrative model names (e.g. ``EXTRACTION_PROMPT``).

See ``docs/adr/0001-model-agnostic-config-boundary.md`` for the full rule and
for how to add a new model/provider (config only).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level runtime source roots scanned by the guard.
RUNTIME_ROOTS = (
    "core-api",
    "core-worker",
    "core-storage-api",
    "core-operations",
    "common",
    "clients",
    "plugin",
)

# ── Config-boundary allowlist ────────────────────────────────────────
# Files where model identifiers / provider names / base URLs legitimately
# live. Keep this list MINIMAL and documented — every entry is a place the
# guard intentionally cannot protect, so each needs a reason.
ALLOWLISTED_FILES: dict[str, str] = {
    "common/provider_names.py": "ProviderName enum — source of truth for provider strings",
    "common/llm/constants.py": "LLM model defaults + provider base URLs (config boundary)",
    "common/llm/_credentials.py": "LLM credential + base-url resolution (config boundary)",
    "common/embedding/constants.py": "embedding model default (config boundary)",
    "common/embedding/_platform.py": "platform embedding wiring (config boundary)",
    "common/embedding/_registry.py": "embedding provider construction (config boundary)",
    "common/embedding/_service.py": "embedding service wiring (config boundary)",
    "core-api/src/core_api/config.py": "global env-config defaults + legacy vertex remap (config boundary)",
    "core-api/src/core_api/services/organization_settings.py": (
        "config-driven per-tenant defaults, informational model suggestions, legacy vertex remap"
    ),
}

# Directory prefixes whose files are all config boundary.
ALLOWLISTED_DIRS: dict[str, str] = {
    "common/llm/providers": "LLM provider adapters legitimately name their provider + a default model",
    "common/embedding/providers": "embedding provider adapters legitimately name their provider + a default model",
}

# ── Per-file token exceptions ────────────────────────────────────────
# A token that matches a guard pattern but is provably not a model/provider
# selector. Keyed by repo-relative path → {token: reason}. Before flagging a
# string, the exempted tokens are stripped from it and the pattern re-tested,
# so an exempted token embedded in a larger literal (e.g. an API param
# description) is also covered. Keep this list minimal and documented.
LITERAL_EXCEPTIONS: dict[str, dict[str, str]] = {
    "core-api/src/core_api/routes/plugin.py": {
        "claude-code": "Claude Code skill-agent identifier (agent name / param doc), not an LLM model",
    },
}

# Hardcoded model-family identifiers. Matching tokens per BP-7.
MODEL_ID_RE = re.compile(
    r"(claude-|gpt-|gemini-|text-embedding|bge-|o1-|o3-|llama|mistral|qwen|deepseek)",
    re.IGNORECASE,
)
# LLM provider base URLs / host fragments.
BASE_URL_RE = re.compile(
    r"(googleapis\.com|api\.anthropic\.com|api\.openai\.com|openrouter\.ai|:11434)"
)
# Provider names that, as a bare string literal in a comparison / match,
# indicate provider-selection logic that should route through ProviderName.
SELECTION_PROVIDER_NAMES = frozenset(
    {"openai", "anthropic", "gemini", "openrouter", "vertex", "ollama"}
)

_PROMPT_NAME_RE = re.compile(r"PROMPT|TEMPLATE")


class Violation(NamedTuple):
    rel_path: str
    lineno: int
    kind: str
    text: str


def _is_allowlisted(rel_path: str) -> bool:
    if rel_path in ALLOWLISTED_FILES:
        return True
    return any(rel_path == d or rel_path.startswith(d + "/") for d in ALLOWLISTED_DIRS)


def _skip_constant_ids(tree: ast.AST) -> set[int]:
    """ids() of string Constant nodes to skip: docstrings + prompt constants."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        # Docstrings: first statement of a module/class/function body.
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                skip.add(id(body[0].value))
        # Prompt/template constants legitimately embed illustrative model names.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(_PROMPT_NAME_RE.search(n.upper()) for n in names):
                skip.add(id(node.value))
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _PROMPT_NAME_RE.search(node.target.id.upper())
        ):
            skip.add(id(node.value))
    return skip


def _scan_source(
    rel_path: str,
    source: str,
    *,
    literal_exceptions: dict[str, dict[str, str]] | None = None,
) -> list[Violation]:
    """Return guard violations for one source string. AST-based."""
    exc = set((literal_exceptions or {}).get(rel_path, {}))
    tree = ast.parse(source)
    skip_ids = _skip_constant_ids(tree)
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip_ids:
                continue
            s = node.value
            # Strip exempted tokens, then re-test — covers exempted tokens
            # embedded in a larger literal (e.g. an API-param description).
            probe = s
            for token in exc:
                probe = probe.replace(token, "")
            lineno = getattr(node, "lineno", 0)
            if MODEL_ID_RE.search(probe):
                out.append(Violation(rel_path, lineno, "model-identifier", s[:80]))
            elif BASE_URL_RE.search(probe):
                out.append(Violation(rel_path, lineno, "llm-base-url", s[:80]))
        elif isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if (
                    isinstance(operand, ast.Constant)
                    and isinstance(operand.value, str)
                    and operand.value in SELECTION_PROVIDER_NAMES
                    and operand.value not in exc
                ):
                    out.append(
                        Violation(
                            rel_path,
                            getattr(operand, "lineno", 0),
                            "provider-selection-literal",
                            operand.value,
                        )
                    )
        elif isinstance(node, ast.MatchValue):
            v = node.value
            if (
                isinstance(v, ast.Constant)
                and isinstance(v.value, str)
                and v.value in SELECTION_PROVIDER_NAMES
                and v.value not in exc
            ):
                out.append(
                    Violation(rel_path, getattr(v, "lineno", 0), "provider-selection-literal", v.value)
                )
    return out


def _iter_runtime_files() -> list[Path]:
    files: list[Path] = []
    skip_segments = {"tests", "test", "__pycache__", ".venv", "node_modules", "build", "dist"}
    for root in RUNTIME_ROOTS:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        for py in root_path.rglob("*.py"):
            parts = set(py.relative_to(REPO_ROOT).parts)
            if parts & skip_segments:
                continue
            files.append(py)
    return files


def scan_tree() -> list[Violation]:
    out: list[Violation] = []
    for py in _iter_runtime_files():
        rel = py.relative_to(REPO_ROOT).as_posix()
        if _is_allowlisted(rel):
            continue
        try:
            source = py.read_text(encoding="utf-8")
            out.extend(_scan_source(rel, source, literal_exceptions=LITERAL_EXCEPTIONS))
        except SyntaxError:
            # A file that doesn't parse under the test interpreter is a lint
            # concern, not a config-boundary concern. Don't fail the guard on it.
            continue
    return out


# ── The enforcement ──────────────────────────────────────────────────


@pytest.mark.unit
def test_runtime_source_has_no_hardcoded_model_or_provider():
    """No model id / base URL / provider-selection literal outside the boundary."""
    violations = scan_tree()
    if violations:
        lines = "\n".join(
            f"  {v.rel_path}:{v.lineno}  [{v.kind}]  {v.text!r}" for v in violations
        )
        pytest.fail(
            "Model-agnostic config boundary violated — model/provider details must "
            "live only in the config boundary and be selected by name via the "
            "registry. Move the value behind config, or (if it is a config-boundary "
            "file) add it to ALLOWLISTED_FILES/DIRS, or add a documented exact-string "
            "exception. See docs/adr/0001-model-agnostic-config-boundary.md.\n"
            f"{lines}"
        )


# ── Self-tests: prove the guard CATCHES violations and ignores noise ──

_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "model_boundary_violation_sample.txt"


@pytest.mark.unit
def test_guard_catches_injected_violations():
    found = _scan_source("fixtures/_planted.py", _FIXTURE.read_text(encoding="utf-8"))
    kinds = {v.kind for v in found}
    assert "model-identifier" in kinds, found
    assert "llm-base-url" in kinds, found
    assert "provider-selection-literal" in kinds, found
    # Exactly the three planted violations — the comment, docstring, and
    # *_PROMPT constant model names must NOT be flagged.
    assert len(found) == 3, found


@pytest.mark.unit
def test_literal_exception_suppresses_match():
    src = 'AGENT_NAME = "claude-code"\n'
    assert _scan_source("p.py", src), "claude-code should match the model-id pattern"
    assert not _scan_source(
        "p.py", src, literal_exceptions={"p.py": {"claude-code"}}
    ), "documented exact-string exception should suppress the match"


@pytest.mark.unit
def test_allowlist_path_matching():
    assert _is_allowlisted("common/llm/constants.py")
    assert _is_allowlisted("common/embedding/providers/openai.py")
    assert _is_allowlisted("common/provider_names.py")
    assert not _is_allowlisted("core-api/src/core_api/services/recall.py")
