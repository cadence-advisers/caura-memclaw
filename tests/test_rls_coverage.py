"""Guard: every model-backed table is covered by the RLS migration.

The tenant-isolation security layer (row-level security + per-table policies)
is defined in alembic migration ``026``. RLS is NOT part of the SQLAlchemy
models and is NOT applied by the test harness's ``Base.metadata.create_all``,
so a new table can silently ship without RLS — exactly the regression that left
``organization_settings`` / ``lifecycle_audit`` / etc. world-writable via
PostgREST until it was caught by the Supabase linter.

This is a pure unit test (no database): it asserts that every table declared by
the shared SQLAlchemy models appears in migration ``026``'s RLS coverage lists.
When someone adds a new model, this test fails until they also classify the new
table as tenant-scoped (``TENANT_TABLES``) or app-role-only (``ENGINE_TABLES``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_migration_026():
    """Load the hyphen/numeric-named migration module by path."""
    repo = Path(__file__).resolve().parent.parent
    mig_path = (
        repo
        / "core-storage-api"
        / "src"
        / "core_storage_api"
        / "database"
        / "migrations"
        / "versions"
        / "026_rls_enable_force_and_tenant_policies.py"
    )
    spec = importlib.util.spec_from_file_location("_test_alembic_026_rls", mig_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_alembic_026_rls"] = module
    spec.loader.exec_module(module)
    return module


def _model_table_names() -> set[str]:
    """All table names declared by the shared SQLAlchemy models.

    ``common.models`` omits organization_settings from its package __init__,
    so import it explicitly to fully populate ``Base.metadata``.
    """
    import common.models  # noqa: F401  (populates most of the metadata)
    import common.models.organization_settings  # noqa: F401
    from common.models.base import Base

    return set(Base.metadata.tables.keys())


def test_tenant_and_engine_lists_are_disjoint():
    mig = _load_migration_026()
    overlap = set(mig.TENANT_TABLES) & set(mig.ENGINE_TABLES)
    assert not overlap, f"tables classified as both tenant and engine: {sorted(overlap)}"


def test_every_model_table_has_rls_coverage():
    mig = _load_migration_026()
    covered = set(mig.TENANT_TABLES) | set(mig.ENGINE_TABLES)
    model_tables = _model_table_names()

    uncovered = model_tables - covered
    assert not uncovered, (
        "Model-backed tables missing from RLS migration 026 — every new table "
        "must be classified as tenant-scoped (TENANT_TABLES) or app-role-only "
        f"(ENGINE_TABLES): {sorted(uncovered)}"
    )


def test_tenant_tables_actually_have_tenant_id():
    """A table in TENANT_TABLES must carry a tenant_id column; otherwise its
    tenant-predicated policies would be invalid. Only checks model-backed
    tables (migration-only tables like session_traces aren't introspectable
    here)."""
    mig = _load_migration_026()
    import common.models  # noqa: F401
    import common.models.organization_settings  # noqa: F401
    from common.models.base import Base

    tables = Base.metadata.tables
    missing = [
        t
        for t in mig.TENANT_TABLES
        if t in tables and "tenant_id" not in tables[t].columns
    ]
    assert not missing, f"TENANT_TABLES entries without a tenant_id column: {missing}"
