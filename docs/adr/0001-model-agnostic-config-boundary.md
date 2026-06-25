# ADR 0001 — Model-agnostic config boundary

- **Status:** Accepted
- **Date:** 2026-06-24
- **Tags:** architecture, llm, embedding, config, ci-guard

## Context

The brain (MemClaw) calls an LLM for enrichment, recall summarisation, entity
extraction, the dedup/contradiction judge, and a fallback path, and an embedder
for vector search. We must be able to run *any* provider and *any* model for
each of those jobs — OpenAI, Anthropic, Gemini, OpenRouter, a local embedder,
or whatever comes next — and switching one must be a **configuration** change,
never a code change.

Two failure modes had crept in and would keep recurring without a hard rule:

1. **Hardcoded model identifiers leaking into business logic.** A default like
   `"gpt-4o-mini"` or a base URL like `https://api.openai.com/v1` written inline
   in a service file silently couples that file to one vendor. Each one is an
   edit-and-redeploy the next time the model changes.
2. **Selection allowlists masquerading as validation.** The settings service
   carried a static `(task × provider × model)` table (`PROVIDER_OPTIONS`) that
   *looked* authoritative. Adding Anthropic or an OpenRouter model to a task
   meant editing that table — i.e. a code change to do what should be config.

## Decision

### 1. There is exactly one config boundary

Model identifiers, provider names used for *selection*, and LLM base URLs live
**only** in a small, enumerated set of config-boundary modules:

- `common/provider_names.py` — the `ProviderName` enum (the source of truth for
  provider strings).
- `common/llm/constants.py`, `common/llm/_credentials.py`,
  `common/llm/providers/*` — LLM model defaults, base URLs, credential
  resolution, and the per-provider adapters.
- `common/embedding/constants.py`, `common/embedding/_platform.py`,
  `common/embedding/_registry.py`, `common/embedding/_service.py`,
  `common/embedding/providers/*` — embedding defaults and wiring.
- `core-api/src/core_api/config.py` — global env-config defaults.
- `core-api/src/core_api/services/organization_settings.py` — config-driven
  per-tenant defaults and the informational model *suggestions*.

Everything else selects a provider/model **by name** at runtime through
`common.llm.registry` (`get_llm_provider`) and `common.embedding`. No other
module names a model, a vendor base URL, or a provider string used to branch.

### 2. Validation is pass-through; suggestions are cosmetic

`PROVIDER_OPTIONS` in `organization_settings.py` is now an **informational
suggestion list** surfaced at `GET /settings/providers` so the settings UI can
show known-good models per task. It is **not** a validation allowlist:

- `update_settings` accepts any tenant `provider`/`model` as a bare string (the
  leaf-type validator pins them to `str` and nothing more).
- The LLM and embedding registries route every provider through their adapter
  and degrade gracefully on an unknown provider/model.

So a tenant can select **any** `ProviderName` value (`openai`, `anthropic`,
`openrouter`, `gemini`, `local`, `fake`, `none`) and **any** model string with
no code edit. Anthropic and OpenRouter are first-class selectable providers for
every LLM task; the embedding suggestions include the deployed `local`
(`BAAI/bge-m3`) embedder, not just OpenAI.

The suggestion list itself is extensible without code: set
`MEMCLAW_SUGGESTED_MODELS_JSON` (a JSON object of `{task: {provider: [models…]}}`)
in the environment and it is merged over the defaults at import time. Malformed
JSON is logged and ignored — suggestions are cosmetic and must never crash boot.

### 3. The boundary is CI-enforced

`tests/test_model_agnostic_config_boundary.py` is an AST-based guard that scans
all runtime source (`core-api`, `core-worker`, `core-storage-api`,
`core-operations`, `common`, `clients`, `plugin`) and **fails** if, outside the
boundary allowlist, it finds:

- a hardcoded model-family identifier
  (`claude-`, `gpt-`, `gemini-`, `text-embedding`, `bge-`, `o1-`/`o3-`,
  `llama`, `mistral`, `qwen`, `deepseek`),
- an LLM base URL / host fragment
  (`googleapis.com`, `api.anthropic.com`, `api.openai.com`, `openrouter.ai`,
  `:11434`), or
- a provider string literal used in a comparison or `match` arm
  (`openai`, `anthropic`, `gemini`, `openrouter`, `vertex`, `ollama`) — i.e.
  selection logic that should route through `ProviderName`.

Because the scan is AST-based it does not false-positive on:

- **comments** (absent from the AST entirely),
- **docstrings** (detected and skipped), or
- **prompt/template constants** — `UPPERCASE` names containing `PROMPT` or
  `TEMPLATE` legitimately embed illustrative model names (e.g. a few-shot
  `EXTRACTION_PROMPT`) and are skipped for the model-identifier check.

The guard ships with a planted-violation fixture
(`tests/fixtures/model_boundary_violation_sample.txt`) and a self-test proving
it **catches** an introduced model id, base URL, and provider-selection literal
while ignoring the comment/docstring/prompt noise around them — so the guard
cannot silently rot into a no-op.

### 4. Exceptions are explicit and documented

The guard maintains two minimal, documented allowlists:

- `ALLOWLISTED_FILES` / `ALLOWLISTED_DIRS` — the config-boundary modules above.
  Every entry carries a one-line reason.
- `LITERAL_EXCEPTIONS` — per-file exact tokens that match a pattern but are
  provably not a model/provider selector (e.g. `"claude-code"`, the Claude Code
  skill-agent identifier in a route param doc). Exempted tokens are stripped
  from the literal and the pattern re-tested, so an exempted token embedded in a
  larger string is also covered.

Keep both lists minimal: each entry is a place the guard intentionally cannot
protect.

## How to add a new model or provider (config only)

1. **Same provider, new model** — set the relevant env var (e.g.
   `ENRICHMENT_MODEL`, `OPENAI_EMBEDDING_MODEL`) or the per-tenant setting via
   `PATCH /settings`. No code change. Optionally add it to the UI suggestion
   list via `MEMCLAW_SUGGESTED_MODELS_JSON`.
2. **New provider already in `ProviderName`** (`anthropic`, `openrouter`,
   `gemini`, `local`, …) — set the provider + model + credentials env/settings.
   No code change.
3. **A genuinely new vendor not yet in `ProviderName`** — this *is* a code
   change, but a contained one: add the enum value in `common/provider_names.py`
   and an adapter under `common/llm/providers/` (or `common/embedding/providers/`).
   Both are inside the boundary, so the guard allows the model/provider strings
   there. Nothing outside the boundary changes.

## Embedding dimension is also config (with a caveat)

The embedding vector **dimension** is config too: `VECTOR_DIM` (env, default
`1024`) in `common/constants.py`. A same-dimension model swap is pure config. A
*cross*-dimension swap additionally requires a schema migration plus a re-embed,
because pgvector encodes the dimension in the `vector(N)` column type and cannot
resize it in place. See
[`embedding-dimension-and-reindex.md`](../embedding-dimension-and-reindex.md)
and `scripts/reindex_embeddings.py` for the runbook. The CI guard keeps model
identifiers out of business logic but cannot enforce the `VECTOR_DIM`/column
agreement — that is the operator's responsibility during a reindex.

## Consequences

- **Positive:** changing the model/provider for any job is a config/env change,
  CI-guarded so it cannot regress. New models reach production without a code
  edit; the brain is genuinely model-agnostic.
- **Cost:** a small, explicit allowlist must be maintained. Adding a truly new
  vendor still touches the boundary modules — by design, so that vendor strings
  live in exactly one place.
- **Limitation:** the guard is a string/AST heuristic. An obfuscated model id
  (assembled from fragments at runtime) would slip past it. That is an
  acceptable trade — the rule targets honest mistakes, not adversarial code.
