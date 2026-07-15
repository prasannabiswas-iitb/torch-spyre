# ClickHouse Schema Migrations

Idempotent SQL scripts for provisioning the `spyre` ClickHouse database schema.

## Quick Start

```bash
export CLICKHOUSE_HOST="<host>"
export CLICKHOUSE_PASS="<password>"
export CLICKHOUSE_PORT=443          # optional, default 443
export CLICKHOUSE_USER=default      # optional, default "default"
export CLICKHOUSE_DB=spyre          # optional, default "spyre"

python3 run_migrations.py
```

## How It Works

1. `run_migrations.py` connects to ClickHouse using the same env vars as the ingest scripts.
2. Creates a `spyre.schema_migrations` tracking table (if not present).
3. Reads all `NNN_*.sql` files in this directory, sorted by numeric prefix.
4. Skips files already recorded in `schema_migrations`.
5. Executes each pending file and records it as applied.

All DDL uses `CREATE TABLE IF NOT EXISTS`, so migrations are safe to re-run
even without the tracking table.

## Commands

| Command | Description |
|---------|-------------|
| `python3 run_migrations.py` | Apply all pending migrations |
| `python3 run_migrations.py --dry-run` | Show what would be applied |

## Running Manually

Each `.sql` file can also be executed directly via the ClickHouse client:

```bash
clickhouse-client --host <host> --port 9440 --secure \
    --user default --password <pass> \
    --database spyre \
    --multiquery < 002_create_benchmark_tables.sql
```

## Adding a New Migration

1. Create a new file: `NNN_short_description.sql` (next sequential number).
2. Use `CREATE TABLE IF NOT EXISTS` or `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
   for idempotency.
3. Each file may contain multiple statements separated by `;`.
4. Add a comment header with the migration name and purpose.
5. Update `tests/docs/clickhouse.md` to reflect any schema changes.

## Conventions

- **Numbering:** Three-digit zero-padded prefix (`001`, `002`, ...).
- **Naming:** `NNN_verb_noun.sql` — e.g., `004_add_platform_column.sql`.
- **Idempotent:** Every statement must be safe to run multiple times.
- **Forward-only:** No rollback scripts. Write a new corrective migration if needed.
- **One concern per file:** Group related tables, but don't mix unrelated changes.
