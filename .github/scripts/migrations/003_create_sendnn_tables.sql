-- Migration: 003_create_sendnn_tables
-- Creates sendnn_runs and sendnn_benchmarks tables for the SendNN
-- benchmarking pipeline (ingest_xml_sendnn.py).

CREATE TABLE IF NOT EXISTS spyre.sendnn_runs
(
    run_id      UInt64 DEFAULT rand64(),
    source_file String NOT NULL,
    created_at  DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (run_id)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS spyre.sendnn_benchmarks
(
    benchmark_id            UInt64 DEFAULT rand64(),
    run_id                  UInt64 NOT NULL,
    record_type             String NOT NULL,
    operation_name          Nullable(String),
    config_name             Nullable(String),
    input_shapes            Nullable(String),
    batch_size              Nullable(Int32),
    prompt_length           Nullable(Int32),
    run_mode                Nullable(String),
    total_duration_ms       Nullable(Float64),
    cpu_ms                  Nullable(Float64),
    spyre_ms                Nullable(Float64),
    kernel_mean_ms          Nullable(Float64),
    memory_transfer_mean_ms Nullable(Float64),
    pt_util_percent         Nullable(Float64),
    custom_op_file          Nullable(String),
    created_at              DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (run_id, benchmark_id)
SETTINGS index_granularity = 8192;
