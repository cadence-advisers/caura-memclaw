"""Codify the brain ops layer (self-maintaining graph + isolation/health monitors).

Until now the brain's operational layer lived ONLY as manual SQL applied
out-of-band to the live project (the same anti-pattern migration 026 closed for
RLS). Three pg_cron jobs, their functions, and the ``cora_ops`` observability
schema were never expressed in a migration — so a fresh database (and, crucially,
a **forked** instance: the licensing/NewCo path) would spin up *without* the
self-maintaining knowledge graph or any of the monitors.

This migration makes that layer reproducible from scratch:

* ``public.infer_relations_cooccurrence()`` — idempotent co-occurrence relation
  builder (the self-maintaining graph). Scheduled hourly @:23.
* ``cora_ops`` schema + ``check_tenant_isolation()`` — a tripwire over the
  hash-chained ``audit_log`` that records any ``cross_tenant_read`` into
  ``cora_ops.tenant_isolation_alerts`` (high-water deduped). Scheduled @:37.
* ``cora_ops.check_enrichment_health()`` — anti-silent enrichment monitor
  (the tripwire the Jun-17 enrichment outage lacked): rolling-window signals for
  enrichment backlog / stuck-pending / embed backlog into
  ``cora_ops.enrichment_health_alerts`` (6h per-signal dedup). Scheduled */15.

Environment-safe: the schema, tables, functions and views are created
everywhere. The pg_cron extension + ``cron.schedule`` calls are guarded behind
``pg_available_extensions`` AND wrapped in an exception handler, so a plain-
Postgres deployment (local / CI, no pg_cron) gets the objects without the
scheduling and without failing the migration. ``cron.schedule`` upserts by job
name, so re-running re-asserts cleanly against the already-patched production DB.

Migrations run as ``postgres`` (BYPASSRLS); the monitor functions are
SECURITY DEFINER so the scheduled jobs can scan across tenants regardless of the
job-owner role on a fork.

Revision ID: 027
Revises: 026
Create Date: 2026-06-25
"""

from collections.abc import Sequence

from alembic import op

revision: str = "027"
down_revision: str | None = "026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ── Self-maintaining graph: co-occurrence relation inference ──────────────────
_INFER_RELATIONS_FN = """
CREATE OR REPLACE FUNCTION public.infer_relations_cooccurrence()
 RETURNS TABLE(created bigint, updated bigint)
 LANGUAGE sql
 SET search_path TO 'public', 'pg_temp'
AS $function$
  with pairs as (
    select mem.tenant_id, a.entity_id as from_id, b.entity_id as to_id, count(*) as cooccur
    from memory_entity_links a
    join memory_entity_links b on a.memory_id=b.memory_id and a.entity_id<b.entity_id
    join memories mem on mem.id=a.memory_id and mem.deleted_at is null
    join entities ea on ea.id=a.entity_id and ea.tenant_id=mem.tenant_id
    join entities eb on eb.id=b.entity_id and eb.tenant_id=mem.tenant_id
    group by mem.tenant_id, a.entity_id, b.entity_id
    having count(*) >= 2
  ),
  upd as (
    update relations r set weight = least(p.cooccur*0.1, 1.0)
    from pairs p
    where r.tenant_id=p.tenant_id and r.relation_type='related_to'
      and r.from_entity_id=p.from_id and r.to_entity_id=p.to_id
      and r.weight is distinct from least(p.cooccur*0.1, 1.0)
    returning 1
  ),
  ins as (
    insert into relations (tenant_id, fleet_id, from_entity_id, relation_type, to_entity_id, weight)
    select p.tenant_id, null, p.from_id, 'related_to', p.to_id, least(p.cooccur*0.1, 1.0)
    from pairs p
    on conflict on constraint uq_relations_natural_key do nothing
    returning 1
  )
  select (select count(*) from ins), (select count(*) from upd);
$function$;
"""

# ── cora_ops observability schema ─────────────────────────────────────────────
_CORA_OPS_SCHEMA = "CREATE SCHEMA IF NOT EXISTS cora_ops;"

_TENANT_ISOLATION_TABLES = """
CREATE TABLE IF NOT EXISTS cora_ops.tenant_isolation_alerts (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  detected_at   timestamptz NOT NULL DEFAULT now(),
  source_tenant text,
  home_tenant   text,
  agent_id      text,
  action        text NOT NULL,
  event_count   int  NOT NULL DEFAULT 1,
  source_seq    bigint,
  sample        json,
  acknowledged  boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_tia_unack ON cora_ops.tenant_isolation_alerts (detected_at) WHERE NOT acknowledged;

CREATE TABLE IF NOT EXISTS cora_ops.tenant_isolation_monitor_state (
  only_row boolean PRIMARY KEY DEFAULT true CHECK (only_row),
  last_seq bigint NOT NULL DEFAULT 0
);
INSERT INTO cora_ops.tenant_isolation_monitor_state (only_row, last_seq)
VALUES (true, COALESCE((SELECT max(seq) FROM public.audit_log), 0))
ON CONFLICT (only_row) DO NOTHING;
"""

_TENANT_ISOLATION_FN = """
CREATE OR REPLACE FUNCTION cora_ops.check_tenant_isolation()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = cora_ops, public
AS $$
DECLARE
  v_last bigint; v_max bigint; v_new int := 0;
BEGIN
  SELECT last_seq INTO v_last FROM cora_ops.tenant_isolation_monitor_state;
  IF v_last IS NULL THEN v_last := 0; END IF;

  WITH ins AS (
    INSERT INTO cora_ops.tenant_isolation_alerts
      (source_tenant, home_tenant, agent_id, action, source_seq, sample)
    SELECT a.tenant_id, a.detail->>'home_tenant_id', a.agent_id, a.action, a.seq, a.detail
    FROM public.audit_log a
    WHERE a.action = 'cross_tenant_read' AND a.seq > v_last
    RETURNING 1
  )
  SELECT count(*) INTO v_new FROM ins;

  SELECT max(seq) INTO v_max FROM public.audit_log;
  UPDATE cora_ops.tenant_isolation_monitor_state
     SET last_seq = GREATEST(COALESCE(v_max, 0), v_last);

  IF v_new > 0 THEN
    RAISE WARNING '[tenant-isolation-monitor] % new cross-tenant-read event(s) detected', v_new;
  END IF;
  RETURN v_new;
END;
$$;

CREATE OR REPLACE VIEW cora_ops.v_tenant_isolation_recent AS
SELECT a.created_at, a.tenant_id AS source_tenant, a.detail->>'home_tenant_id' AS home_tenant,
  a.agent_id, a.detail->>'query_summary' AS query_summary,
  a.detail->>'result_count_from_this_tenant' AS result_count
FROM public.audit_log a
WHERE a.action = 'cross_tenant_read' AND a.created_at >= now() - interval '30 days'
ORDER BY a.created_at DESC;
"""

_ENRICHMENT_HEALTH = """
CREATE TABLE IF NOT EXISTS cora_ops.enrichment_health_alerts (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  detected_at  timestamptz NOT NULL DEFAULT now(),
  signal       text NOT NULL,
  observed     int  NOT NULL,
  threshold    int  NOT NULL,
  detail       text,
  acknowledged boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_eha_unack ON cora_ops.enrichment_health_alerts (signal, detected_at) WHERE NOT acknowledged;

CREATE OR REPLACE FUNCTION cora_ops._eh_raise(p_signal text, p_observed int, p_threshold int, p_detail text)
RETURNS int
LANGUAGE plpgsql
AS $$
BEGIN
  IF p_observed > p_threshold
     AND NOT EXISTS (
       SELECT 1 FROM cora_ops.enrichment_health_alerts
       WHERE signal = p_signal AND NOT acknowledged
         AND detected_at >= now() - interval '6 hours'
     ) THEN
    INSERT INTO cora_ops.enrichment_health_alerts (signal, observed, threshold, detail)
    VALUES (p_signal, p_observed, p_threshold, p_detail);
    RAISE WARNING '[enrichment-health] % breached: % > % (%)', p_signal, p_observed, p_threshold, p_detail;
    RETURN 1;
  END IF;
  RETURN 0;
END;
$$;

CREATE OR REPLACE FUNCTION cora_ops.check_enrichment_health()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = cora_ops, public
AS $$
DECLARE
  v_backlog int; v_pending int; v_embed int; v_new int := 0;
BEGIN
  SELECT count(*) INTO v_backlog FROM public.memories
    WHERE title IS NULL
      AND created_at >= now() - interval '2 hours'
      AND created_at <  now() - interval '15 minutes';
  SELECT count(*) INTO v_pending FROM public.memories
    WHERE (metadata::jsonb->>'enrichment_pending') = 'true'
      AND created_at < now() - interval '30 minutes';
  SELECT count(*) INTO v_embed FROM public.memories
    WHERE embedding IS NULL
      AND created_at < now() - interval '30 minutes';

  v_new := v_new + cora_ops._eh_raise('enrichment_backlog', v_backlog, 8, 'title-null memories aged 15m-2h (LLM enrichment not completing)');
  v_new := v_new + cora_ops._eh_raise('pending_stuck',      v_pending, 5, 'enrichment_pending=true older than 30m (worker/LLM stuck)');
  v_new := v_new + cora_ops._eh_raise('embed_backlog',      v_embed,   5, 'embedding-null older than 30m (embed path degraded)');
  RETURN v_new;
END;
$$;

CREATE OR REPLACE VIEW cora_ops.v_enrichment_health AS
SELECT
  (SELECT count(*) FROM public.memories
     WHERE title IS NULL AND created_at >= now()-interval '2 hours'
       AND created_at < now()-interval '15 minutes')               AS enrichment_backlog,
  (SELECT count(*) FROM public.memories
     WHERE (metadata::jsonb->>'enrichment_pending')='true'
       AND created_at < now()-interval '30 minutes')               AS pending_stuck,
  (SELECT count(*) FROM public.memories
     WHERE embedding IS NULL AND created_at < now()-interval '30 minutes') AS embed_backlog,
  (SELECT count(*) FROM public.memories WHERE title IS NULL)        AS title_null_all,
  (SELECT max(created_at) FROM public.memories)                    AS latest_write;
"""

# pg_cron scheduling — guarded (extension may be absent on plain Postgres) and
# exception-wrapped (a scheduling failure must not abort the whole migration; the
# objects above are already committed and useful without the schedule).
_SCHEDULE_CRON = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_cron') THEN
    BEGIN
      CREATE EXTENSION IF NOT EXISTS pg_cron;
      PERFORM cron.schedule('brain-relation-inference', '23 * * * *',  'select public.infer_relations_cooccurrence()');
      PERFORM cron.schedule('tenant-isolation-monitor', '37 * * * *',  'SELECT cora_ops.check_tenant_isolation();');
      PERFORM cron.schedule('enrichment-health-monitor', '*/15 * * * *', 'SELECT cora_ops.check_enrichment_health();');
    EXCEPTION WHEN OTHERS THEN
      RAISE WARNING 'ops-layer: pg_cron present but scheduling failed (%); schedule the 3 jobs manually', SQLERRM;
    END;
  ELSE
    RAISE NOTICE 'ops-layer: pg_cron unavailable; created functions/tables only (no scheduled jobs)';
  END IF;
END $$;
"""

_UNSCHEDULE_CRON = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_cron')
     AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='cron' AND table_name='job') THEN
    PERFORM cron.unschedule(jobid) FROM cron.job
     WHERE jobname IN ('brain-relation-inference','tenant-isolation-monitor','enrichment-health-monitor');
  END IF;
END $$;
"""


def upgrade() -> None:
    op.execute(_INFER_RELATIONS_FN)
    op.execute(_CORA_OPS_SCHEMA)
    op.execute(_TENANT_ISOLATION_TABLES)
    op.execute(_TENANT_ISOLATION_FN)
    op.execute(_ENRICHMENT_HEALTH)
    op.execute(_SCHEDULE_CRON)


def downgrade() -> None:
    # Unschedule first (guarded), then drop the objects. Best-effort; the
    # self-maintaining relation function is left in place (harmless, idempotent).
    op.execute(_UNSCHEDULE_CRON)
    op.execute("DROP VIEW IF EXISTS cora_ops.v_enrichment_health;")
    op.execute("DROP VIEW IF EXISTS cora_ops.v_tenant_isolation_recent;")
    op.execute("DROP FUNCTION IF EXISTS cora_ops.check_enrichment_health();")
    op.execute("DROP FUNCTION IF EXISTS cora_ops._eh_raise(text, int, int, text);")
    op.execute("DROP FUNCTION IF EXISTS cora_ops.check_tenant_isolation();")
    op.execute("DROP TABLE IF EXISTS cora_ops.enrichment_health_alerts;")
    op.execute("DROP TABLE IF EXISTS cora_ops.tenant_isolation_alerts;")
    op.execute("DROP TABLE IF EXISTS cora_ops.tenant_isolation_monitor_state;")
    op.execute("DROP SCHEMA IF EXISTS cora_ops;")
    op.execute("DROP FUNCTION IF EXISTS public.infer_relations_cooccurrence();")
