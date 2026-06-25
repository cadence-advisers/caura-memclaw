# Embedding dimension as config, and how to reindex

The embedding vector dimension is a **configuration value**, not a hardcoded
constant. It is read once at import time from the `VECTOR_DIM` environment
variable (default `1024`) in [`common/constants.py`](../common/constants.py)
and flows to every consumer: the pgvector column types
(`memories.embedding`, `entities.name_embedding`, `documents.embedding`), the
OpenAI Matryoshka truncation knob, the deterministic embedding cache key, and
the fake/local providers.

This is part of the **model-agnostic config boundary** — see
[ADR 0001](adr/0001-model-agnostic-config-boundary.md). Changing the embedding
model (or provider) must be a config change. Whether you also need a schema
change depends entirely on the **dimension** of the new model.

## Two kinds of embedding swap

### 1. Same-dimension swap — pure config, no reindex

Swapping one embedder for another that produces vectors of the **same**
dimension as the current schema (e.g. one 1024-dim model for another 1024-dim
model) is a pure configuration change:

- Set the provider/model env vars (e.g. `EMBEDDING_PROVIDER`,
  `OPENAI_EMBEDDING_MODEL`, or the platform embedding settings).
- `VECTOR_DIM` and the `vector(N)` column type are unchanged.
- **No migration, no reindex.** New writes embed with the new model; existing
  rows keep their old vectors.

Caveat: cosine distances are only comparable **within** one embedding space.
Mixing vectors from two different models in the same column degrades retrieval
even at the same dimension. For a clean cutover, re-embed existing rows with
the new model (see the reindex runbook below, steps 4–5) even though the schema
does not change. For a gradual cutover, accept transient mixed-space recall
until lifecycle re-embeds touch the old rows.

### 2. Cross-dimension swap — config **and** a reindex migration

Swapping to a model with a **different** dimension (e.g. 1024 → 1536) requires
both a config change and a schema change, because pgvector encodes the
dimension in the column type and **cannot widen/narrow a `vector` column in
place** — an existing N-dim value is not coercible to M dims.

Procedure (this is what `scripts/reindex_embeddings.py` documents and helps
verify):

1. **Pick the new model + dimension.** Confirm the model's native output
   dimension (or the Matryoshka dimension you will truncate to).
2. **Set `VECTOR_DIM`** to the new dimension in the environment for *every*
   service (core-api, core-worker, core-storage-api). They all import
   `common.constants`, so they must agree.
3. **Ship a migration** modelled on
   [`012_vector_dim_1024.py`](../core-storage-api/src/core_storage_api/database/migrations/versions/012_vector_dim_1024.py):
   drop the HNSW indexes `CONCURRENTLY`, `NULL` the existing vectors, `ALTER
   TABLE ... ALTER COLUMN ... TYPE vector(<new>)`, and recreate the indexes
   `CONCURRENTLY`. Reuse migration 012's safety gate
   (`MEMCLAW_RUN_DESTRUCTIVE_MIGRATIONS=true`) — the column rewrite destroys
   existing embeddings, which is irreversible.
4. **Re-embed every row.** After the migration, all embeddings are `NULL`.
   - *Lazy:* any new write, or any read path that calls
     `get_or_cache_embedding`, re-embeds the touched row. Steady-state
     deployments converge over hours/days.
   - *Eager (recommended for production cutovers):* run the core-worker
     backfill task that scans `WHERE embedding IS NULL` and embeds in batches.
     **This eager backfill worker is tracked as a follow-on** (see migration
     012's module docstring — "run the backfill task in core-worker (separate
     PR)"). `scripts/reindex_embeddings.py` is the operator-facing preflight /
     runbook companion; it does not itself replace that worker.
5. **Verify.** Re-run `scripts/reindex_embeddings.py --report` to confirm the
   live column dimension matches the configured `VECTOR_DIM` and that the count
   of `NULL`-embedding rows has drained.

## Why dimension stays coupled to the schema

`VECTOR_DIM` being env-driven does **not** make cross-dimension swaps free —
the database column type is the hard constraint. Treat `VECTOR_DIM` and the
`vector(N)` column as a single fact expressed in two places that must agree.
The CI config-boundary guard
([`tests/test_model_agnostic_config_boundary.py`](../tests/test_model_agnostic_config_boundary.py))
keeps model identifiers out of business logic, but it cannot enforce the
DB/`VECTOR_DIM` agreement — that is the operator's responsibility during a
reindex, which is why this runbook and the preflight script exist.
