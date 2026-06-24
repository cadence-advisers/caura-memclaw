#!/usr/bin/env python3
"""Operator preflight + runbook for an embedding-dimension reindex.

The embedding vector dimension is a config value (``VECTOR_DIM`` env, default
1024 — see ``common/constants.py`` and ``docs/embedding-dimension-and-reindex.md``).
Swapping to an embedder of a DIFFERENT dimension requires a schema migration
plus a re-embed of every row, because pgvector cannot resize a ``vector``
column in place.

This script is the operator-facing companion to that procedure. It does two
things, both READ-ONLY:

  * ``--runbook`` (default): print the step-by-step reindex procedure.
  * ``--report``: connect to the database and report, per embedding column,
    the live ``vector(N)`` dimension vs the configured ``VECTOR_DIM``, and how
    many rows still need (re-)embedding (``embedding IS NULL``).

It deliberately does NOT perform the re-embed. The eager backfill that scans
``WHERE embedding IS NULL`` and embeds in batches lives in core-worker and is
tracked as a follow-on (see migration ``012_vector_dim_1024.py``). Lazy
re-embedding also happens automatically on the read/write hot path via
``get_or_cache_embedding``.

Usage:
    python scripts/reindex_embeddings.py --runbook
    python scripts/reindex_embeddings.py --report
    python scripts/reindex_embeddings.py --report \\
        --db-url postgresql://memclaw:changeme@localhost:5432/memclaw
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# (table, embedding_column) tuples touched by a dimension change. Mirrors the
# three columns altered by alembic migration 012.
EMBEDDING_COLUMNS: list[tuple[str, str]] = [
    ("memories", "embedding"),
    ("entities", "name_embedding"),
    ("documents", "embedding"),
]

RUNBOOK = """\
Embedding-dimension reindex runbook
===================================

A same-dimension model swap is PURE CONFIG (set the provider/model env vars;
stop here). The steps below are ONLY for a CROSS-dimension swap.

1. Pick the new model and confirm its output dimension D_new.
2. Set VECTOR_DIM=<D_new> in the env for ALL services (core-api, core-worker,
   core-storage-api) and set the new embedding provider/model env vars.
3. Ship a migration modelled on 012_vector_dim_1024.py:
     - DROP INDEX CONCURRENTLY the HNSW indexes
     - UPDATE ... SET <col> = NULL   (old vectors are not coercible)
     - ALTER TABLE ... ALTER COLUMN ... TYPE vector(<D_new>)
     - CREATE INDEX CONCURRENTLY the HNSW indexes
   Keep the MEMCLAW_RUN_DESTRUCTIVE_MIGRATIONS=true safety gate — the column
   rewrite destroys existing embeddings.
4. Re-embed every row:
     - Lazy: hot-path reads/writes re-embed touched rows over time.
     - Eager (recommended for production): run the core-worker backfill task
       (follow-on PR) over rows WHERE embedding IS NULL.
5. Verify with:  python scripts/reindex_embeddings.py --report
   The live vector(N) dimension must equal VECTOR_DIM, and the NULL-embedding
   counts should drain toward zero as re-embedding proceeds.

See docs/embedding-dimension-and-reindex.md for the full rationale.
"""


def _db_url_from_env() -> str | None:
    """Build an asyncpg DSN from the same POSTGRES_* env vars the app uses."""
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    user = os.environ.get("POSTGRES_USER", "memclaw")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "memclaw")
    port = os.environ.get("POSTGRES_PORT", "5432")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"


def _configured_vector_dim() -> int:
    # Import lazily so --runbook works without the package on PYTHONPATH.
    from common.constants import VECTOR_DIM

    return VECTOR_DIM


async def _report(db_url: str) -> int:
    try:
        import asyncpg
    except ImportError:
        print("ERROR: --report needs asyncpg installed.", file=sys.stderr)
        return 2

    configured = _configured_vector_dim()
    print(f"Configured VECTOR_DIM = {configured}\n")

    conn = await asyncpg.connect(dsn=db_url)
    try:
        ok = True
        for table, column in EMBEDDING_COLUMNS:
            coltype = await conn.fetchval(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = $1 AND a.attname = $2
                  AND a.attnum > 0 AND NOT a.attisdropped
                """,
                table,
                column,
            )
            if coltype is None:
                print(f"  {table}.{column:<16} MISSING (table/column not found)")
                ok = False
                continue
            # coltype looks like 'vector(1024)'; extract the integer dim.
            live_dim = "".join(ch for ch in coltype if ch.isdigit())
            row = await conn.fetchrow(
                f"SELECT count(*) FILTER (WHERE {column} IS NULL) AS null_n, "  # noqa: S608 — identifiers from a fixed in-code allowlist, not user input
                f"count(*) AS total FROM {table}"  # noqa: S608
            )
            match = "OK" if live_dim == str(configured) else "MISMATCH"
            if match != "OK":
                ok = False
            print(
                f"  {table}.{column:<16} {coltype:<14} vs VECTOR_DIM={configured} "
                f"[{match}]  needs_embedding={row['null_n']}/{row['total']}"
            )
        print()
        if not ok:
            print(
                "One or more columns disagree with VECTOR_DIM — a migration is "
                "required before the new dimension is usable.",
                file=sys.stderr,
            )
            return 1
        print("All embedding columns match the configured VECTOR_DIM.")
        return 0
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="connect to the DB and report live vs configured dimension + re-embed backlog",
    )
    parser.add_argument(
        "--runbook",
        action="store_true",
        help="print the reindex runbook (default when no mode is given)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="asyncpg DSN; defaults to POSTGRES_* env vars",
    )
    args = parser.parse_args()

    if not args.report:
        print(RUNBOOK)
        return 0

    db_url = args.db_url or _db_url_from_env()
    if not db_url:
        print(
            "ERROR: --report needs a database. Pass --db-url or set POSTGRES_HOST etc.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_report(db_url))


if __name__ == "__main__":
    raise SystemExit(main())
