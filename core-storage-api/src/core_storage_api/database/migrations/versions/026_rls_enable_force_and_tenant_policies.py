"""Codify the full RLS layer (role + grants + ENABLE/FORCE RLS + policies).

Until now the tenant-isolation security layer for this database lived ONLY as
manual SQL applied out-of-band to the live project: the ``memclaw_engine``
application role, its grants, row-level security, and the per-table policies
were never expressed in a migration. The consequence was drift — tables added
by later migrations (``014_*`` organization_settings, ``015_*`` lifecycle_audit,
and the join/audit tables) silently shipped with **RLS disabled**, leaving them
readable and writable by the public PostgREST roles (``anon`` / ``authenticated``).

This migration makes the RLS layer reproducible from scratch so a fresh database
(``alembic upgrade head``) is secured identically to production, and so the
linter's ``rls_disabled_in_public`` class of error cannot recur for the tables
that exist today. It is intentionally the LAST migration so every table it
governs already exists when it runs.

Two policy shapes, matching production exactly:

* **Tenant tables** (carry a ``tenant_id`` column) get the four ``tenant_*``
  policies (``TO public``) gated on the ``app.tenant_id`` / ``app.readable_tenant_ids``
  GUCs that the application sets per request (see ``core_api.db.session``).
* **Engine tables** (org-global config, audit, migration bookkeeping, and the
  memory<->entity join table — none of which carry a ``tenant_id``) get a single
  ``engine_all`` policy scoped to the application role. They are not tenant-
  filtered, but they must never be reachable by ``anon`` / ``authenticated``.

``service_role`` and ``postgres`` are ``BYPASSRLS``, so Supabase service access
and alembic migrations (which run as ``postgres``) are unaffected by FORCE RLS.

SECRET HANDLING: the ``memclaw_engine`` LOGIN role is created here WITHOUT a
password. Its password is provisioned out-of-band by the deploy from a secret
(``ALTER ROLE memclaw_engine PASSWORD ...``) and must never be committed.

Idempotent by construction (guarded role creation, ``GRANT`` is repeatable,
``ENABLE/FORCE`` are no-ops if already set, ``DROP POLICY IF EXISTS`` before
``CREATE``), so it re-asserts cleanly against the already-patched production DB.

Revision ID: 026
Revises: 025
Create Date: 2026-06-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "026"
down_revision: str | None = "025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "memclaw_engine"

# Tables that carry a tenant_id and are isolated per tenant.
TENANT_TABLES: tuple[str, ...] = (
    "agents",
    "analysis_reports",
    "audit_chain_head",
    "audit_log",
    "background_task_log",
    "capability_usage",
    "dedup_reviews",
    "documents",
    "entities",
    "fleet_commands",
    "fleet_nodes",
    "forge_rejected_fingerprints",
    "idempotency_responses",
    "memories",
    "relations",
    "session_traces",
    "tenant_suppression",
)

# Org-global / internal tables with no tenant_id: app-role access only,
# public PostgREST roles fully denied.
ENGINE_TABLES: tuple[str, ...] = (
    "alembic_version",
    "lifecycle_audit",
    "memory_entity_links",
    "organization_settings",
    "organization_settings_audit",
)

ALL_TABLES = TENANT_TABLES + ENGINE_TABLES

# Read is broader than write: admin context, own tenant, the shared-lessons
# corpus, and any explicitly-readable tenant ids.
_SELECT_USING = (
    "(current_setting('app.tenant_id', true) = '__admin__')"
    " OR (tenant_id = current_setting('app.tenant_id', true))"
    " OR (tenant_id = 'shared-lessons')"
    " OR (tenant_id = ANY (string_to_array(current_setting('app.readable_tenant_ids', true), ',')))"
)
# Writes are limited to admin context or the caller's own tenant.
_WRITE_PRED = (
    "(current_setting('app.tenant_id', true) = '__admin__')"
    " OR (tenant_id = current_setting('app.tenant_id', true))"
)


def _ensure_role() -> None:
    # Create the application login role if absent. No password here — it is set
    # out-of-band from a deploy secret. NOINHERIT/INHERIT left at default.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} LOGIN;
            END IF;
        END
        $$;
        """
    )


def _grant_role() -> None:
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    # Future tables/sequences created by postgres auto-grant to the app role,
    # so a new table can never silently regress to "no app access".
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
    )
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}")


def _enable_force_rls(table: str) -> None:
    op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY")


def _apply_tenant_policies(table: str) -> None:
    for name in ("tenant_select", "tenant_insert", "tenant_modify", "tenant_delete"):
        op.execute(f"DROP POLICY IF EXISTS {name} ON public.{table}")
    op.execute(f"CREATE POLICY tenant_select ON public.{table} FOR SELECT TO public USING ({_SELECT_USING})")
    op.execute(
        f"CREATE POLICY tenant_insert ON public.{table} FOR INSERT TO public WITH CHECK ({_WRITE_PRED})"
    )
    op.execute(
        f"CREATE POLICY tenant_modify ON public.{table} "
        f"FOR UPDATE TO public USING ({_WRITE_PRED}) WITH CHECK ({_WRITE_PRED})"
    )
    op.execute(f"CREATE POLICY tenant_delete ON public.{table} FOR DELETE TO public USING ({_WRITE_PRED})")


def _apply_engine_policy(table: str) -> None:
    op.execute(f"DROP POLICY IF EXISTS engine_all ON public.{table}")
    op.execute(
        f"CREATE POLICY engine_all ON public.{table} FOR ALL TO {APP_ROLE} USING (true) WITH CHECK (true)"
    )


# Trigger functions whose body only calls pg_catalog builtins (to_tsvector);
# pinning an empty search_path closes the "mutable search_path" advisor without
# touching the body (pg_catalog is always implicitly resolvable).
_SEARCH_PATH_FUNCS = (
    "memories_search_vector_update()",
    "entities_search_vector_update()",
)


def _harden_platform_advisors() -> None:
    """Close the Supabase security-advisor WARNs, safely on any deployment.

    Three fixes, all idempotent and guarded so a non-Supabase (vanilla Postgres,
    no ``extensions`` schema) deployment is unaffected:

    * pin ``search_path`` on the two tsvector trigger functions;
    * add ``extensions`` to the app role's ``search_path`` (a non-existent
      schema in the path is simply ignored, so this is harmless everywhere);
    * relocate the ``vector`` extension out of ``public`` into ``extensions``
      ONLY when that schema exists and vector is currently in ``public`` —
      granting the app role ``USAGE`` on ``extensions`` first so its bare
      ``<=>`` / ``::vector`` operators keep resolving.
    """
    for fn in _SEARCH_PATH_FUNCS:
        op.execute(f"ALTER FUNCTION public.{fn} SET search_path = ''")

    # Harmless when `extensions` doesn't exist (ignored entries in search_path).
    op.execute(f'ALTER ROLE {APP_ROLE} SET search_path = "$user", public, extensions')

    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'extensions')
               AND EXISTS (
                   SELECT 1 FROM pg_extension e
                   JOIN pg_namespace n ON n.oid = e.extnamespace
                   WHERE e.extname = 'vector' AND n.nspname = 'public'
               ) THEN
                GRANT USAGE ON SCHEMA extensions TO {APP_ROLE};
                ALTER EXTENSION vector SET SCHEMA extensions;
            END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    _ensure_role()
    _grant_role()
    for table in TENANT_TABLES:
        _enable_force_rls(table)
        _apply_tenant_policies(table)
    for table in ENGINE_TABLES:
        _enable_force_rls(table)
        _apply_engine_policy(table)
    _harden_platform_advisors()


def downgrade() -> None:
    # Best-effort teardown: drop policies and disable RLS. The role and its
    # grants are intentionally left in place (other objects may depend on it,
    # and dropping a LOGIN role mid-fleet is more dangerous than leaving it).
    for table in TENANT_TABLES:
        for name in ("tenant_select", "tenant_insert", "tenant_modify", "tenant_delete"):
            op.execute(f"DROP POLICY IF EXISTS {name} ON public.{table}")
        op.execute(f"ALTER TABLE public.{table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY")
    for table in ENGINE_TABLES:
        op.execute(f"DROP POLICY IF EXISTS engine_all ON public.{table}")
        op.execute(f"ALTER TABLE public.{table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY")
