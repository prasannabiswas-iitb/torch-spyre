#!/usr/bin/env python3
"""
ClickHouse schema migration runner for the spyre database.

Executes numbered .sql migration files in order, tracking which have been
applied in a `spyre.schema_migrations` table to prevent double-execution.

Environment variables (same as ingest scripts):
    CLICKHOUSE_HOST     (required)
    CLICKHOUSE_PASS     (required)
    CLICKHOUSE_PORT     (default: 443)
    CLICKHOUSE_USER     (default: "default")
    CLICKHOUSE_DB       (default: "spyre")

Usage:
    python3 run_migrations.py              # Apply all pending migrations
    python3 run_migrations.py --dry-run    # Show what would be applied
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect


MIGRATIONS_DIR = Path(__file__).parent

TRACKING_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations
(
    migration_id String NOT NULL,
    applied_at   DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (migration_id)
SETTINGS index_granularity = 8192
"""


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def ensure_tracking_table(client):
    client.command(TRACKING_TABLE_DDL)


def get_applied_migrations(client) -> set[str]:
    result = client.query("SELECT migration_id FROM schema_migrations")
    return {row[0] for row in result.result_rows}


def discover_migrations() -> list[Path]:
    sql_files = sorted(MIGRATIONS_DIR.glob("[0-9]*.sql"))
    return sql_files


def apply_migration(client, migration_path: Path) -> None:
    sql = migration_path.read_text(encoding="utf-8")
    for statement in sql.split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            client.command(statement)


def record_migration(client, migration_id: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    client.insert(
        "schema_migrations",
        [[migration_id, now]],
        column_names=["migration_id", "applied_at"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply pending ClickHouse schema migrations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which migrations would be applied without executing them",
    )
    args = parser.parse_args()

    migrations = discover_migrations()
    if not migrations:
        print("No migration files found.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("Connected.\n")

    ensure_tracking_table(client)
    applied = get_applied_migrations(client)

    pending = [m for m in migrations if m.name not in applied]

    if not pending:
        print(f"All {len(migrations)} migrations already applied. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(pending)} pending migration(s):\n")
    for m in pending:
        print(f"  {'[DRY RUN] ' if args.dry_run else ''}  {m.name}")

    if args.dry_run:
        print(f"\nDry run complete. {len(pending)} migration(s) would be applied.")
        sys.exit(0)

    print()
    applied_count = 0
    for m in pending:
        print(f"Applying: {m.name} ...", end=" ")
        try:
            apply_migration(client, m)
            record_migration(client, m.name)
            print("OK")
            applied_count += 1
        except Exception as exc:
            print(f"FAILED\n  Error: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. {applied_count} migration(s) applied successfully.")


if __name__ == "__main__":
    main()
