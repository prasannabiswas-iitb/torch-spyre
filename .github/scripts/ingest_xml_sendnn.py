#!/usr/bin/env python3
"""
Parses sendnn benchmark XML files and batch-inserts into ClickHouse.

Designed to run independently on the sendnn pod (no torch-spyre dependency).

Usage:
    python3 ingest_xml_sendnn.py \
        --xml-file sendnn_report.xml \
        --workflow "sendnn-benchmark" \
        --branch "main" \
        --run-id "12345678"
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree
import clickhouse_connect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag_props(tc_el) -> dict:
    """Return a flat dict of tag__ → value parsed from <properties>."""
    result = {}
    props_el = tc_el.find("properties")
    if props_el is None:
        return result
    for p in props_el.findall("property"):
        name = p.get("name", "").strip()
        value = p.get("value", "").strip()
        if name == "tag" and "__" in value:
            key, _, val = value.partition("__")
            result[key] = val
    return result


def _opt_float(d: dict, key: str):
    try:
        return float(d[key])
    except (KeyError, ValueError, TypeError):
        return None


def _opt_int(d: dict, key: str):
    try:
        v = d[key]
        if v == "null" or v is None:
            return None
        return int(v)
    except (KeyError, ValueError, TypeError):
        return None


def _tag_val(tags: dict, key: str, default=None):
    v = tags.get(key, default)
    if v == "null" or v is None:
        return None
    return v


# ---------------------------------------------------------------------------
# SendNN XML detection & parsing
# ---------------------------------------------------------------------------


def is_sendnn_xml(root) -> bool:
    """Return True if every testcase has classname containing 'sendnn'."""
    cases = root.findall(".//testcase")
    if not cases:
        return False
    return all("sendnn" in (tc.get("classname", "")) for tc in cases)


def parse_sendnn_xml(xml_path: Path):
    """
    Parse a sendnn benchmark XML into (run_meta, list[benchmark_row]).

    Returns:
        run_meta  : dict  – data for sendnn_runs
        benchmarks: list[dict] – data for sendnn_benchmarks
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    suite = root.find(".//testsuite")
    if suite is None:
        print(f"  [warn] No <testsuite> in {xml_path.name}", file=sys.stderr)
        return None, []

    ts_str = suite.get("timestamp", "")
    try:
        created_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        created_at = datetime.now(timezone.utc)

    benchmarks = []
    for tc in suite.findall(".//testcase"):
        tags = _tag_props(tc)

        benchmarks.append(
            {
                "benchmark_id": uuid.uuid4().int >> 64,
                "record_type": _tag_val(tags, "record_type", "op"),
                "operation_name": _tag_val(tags, "op"),
                "config_name": _tag_val(tags, "config_name"),
                "input_shapes": _tag_val(tags, "input_shapes"),
                "batch_size": _opt_int(tags, "batch_size"),
                "prompt_length": _opt_int(tags, "prompt_length"),
                "run_mode": _tag_val(tags, "run_mode", "sendnn_benchmark"),
                "total_duration_ms": _opt_float(tags, "total_duration_ms"),
                "cpu_ms": _opt_float(tags, "cpu_ms"),
                "spyre_ms": _opt_float(tags, "spyre_ms"),
                "kernel_mean_ms": _opt_float(tags, "kernel_mean_ms"),
                "memory_transfer_mean_ms": _opt_float(tags, "memory_transfer_mean_ms"),
                "pt_util_percent": _opt_float(tags, "pt_util_percent"),
                "custom_op_file": None,
                "created_at": created_at,
            }
        )

    run_meta = {
        "source_file": xml_path.name,
        "created_at": created_at,
    }
    return run_meta, benchmarks


# ---------------------------------------------------------------------------
# ClickHouse insertion
# ---------------------------------------------------------------------------


def get_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 443)),
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASS"],
        database=os.environ.get("CLICKHOUSE_DB", "spyre"),
        secure=True,
    )


def insert_sendnn_run(client, run_id: int, run_meta: dict) -> None:
    client.insert(
        "sendnn_runs",
        [
            [
                run_id,
                run_meta["source_file"],
                run_meta["created_at"].replace(tzinfo=None),
            ]
        ],
        column_names=["run_id", "source_file", "created_at"],
    )


def insert_sendnn_benchmarks(client, run_id: int, benchmarks: list[dict]) -> None:
    if not benchmarks:
        return
    client.insert(
        "sendnn_benchmarks",
        [
            [
                b["benchmark_id"],
                run_id,
                b["record_type"],
                b["operation_name"],
                b["config_name"],
                b["input_shapes"],
                b["batch_size"],
                b["prompt_length"],
                b["run_mode"],
                b["total_duration_ms"],
                b["cpu_ms"],
                b["spyre_ms"],
                b["kernel_mean_ms"],
                b["memory_transfer_mean_ms"],
                b["pt_util_percent"],
                b["custom_op_file"],
                b["created_at"].replace(tzinfo=None),
            ]
            for b in benchmarks
        ],
        column_names=[
            "benchmark_id",
            "run_id",
            "record_type",
            "operation_name",
            "config_name",
            "input_shapes",
            "batch_size",
            "prompt_length",
            "run_mode",
            "total_duration_ms",
            "cpu_ms",
            "spyre_ms",
            "kernel_mean_ms",
            "memory_transfer_mean_ms",
            "pt_util_percent",
            "custom_op_file",
            "created_at",
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ingest sendnn benchmark XML into ClickHouse"
    )
    parser.add_argument("--xml-dir", default=None)
    parser.add_argument("--xml-file", default=None)
    parser.add_argument("--workflow", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.xml_file:
        xml_files = [Path(args.xml_file)]
    elif args.xml_dir:
        xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    else:
        print("Error: provide --xml-dir or --xml-file")
        sys.exit(1)

    if not xml_files:
        print("No XML files found — nothing to ingest.")
        sys.exit(0)

    print(
        f"Connecting to ClickHouse at "
        f"{os.environ['CLICKHOUSE_HOST']}:{os.environ.get('CLICKHOUSE_PORT', 443)} ..."
    )
    client = get_client()
    client.command("SELECT 1")
    print("Connected.\n")

    total_benchmarks = 0

    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")

        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        if not is_sendnn_xml(root):
            print(
                f"  [skip] Not a sendnn XML (classname missing 'sendnn'): {xml_path.name}"
            )
            continue

        run_meta, benchmarks = parse_sendnn_xml(xml_path)
        if run_meta is None:
            continue

        existing = client.query(
            "SELECT count() FROM sendnn_runs WHERE source_file = {sf:String}",
            parameters={"sf": run_meta["source_file"]},
        )
        if existing.result_rows[0][0] > 0:
            print(f"  Already ingested — skipping {run_meta['source_file']}")
            continue

        run_id = uuid.uuid4().int >> 64
        print(f"  run_id={run_id}  benchmarks={len(benchmarks)}")

        insert_sendnn_run(client, run_id, run_meta)
        insert_sendnn_benchmarks(client, run_id, benchmarks)

        total_benchmarks += len(benchmarks)
        print(f"  Inserted {len(benchmarks)} sendnn benchmark rows")

    print(f"\nDone. {len(xml_files)} file(s) processed.")
    print(f"  Benchmarks ingested: {total_benchmarks}")


if __name__ == "__main__":
    main()
