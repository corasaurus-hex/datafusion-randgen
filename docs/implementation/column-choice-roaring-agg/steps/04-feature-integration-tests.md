# Column Choice Roaring Aggregate Step 04 — Feature Integration Tests

## Goal

Rewrite Parquet and Arrow IPC column-choice integration tests around explicit SQL
aggregation and sampling.

## Crates/files touched

- `tests/column_choice_parquet.rs`
- `tests/column_choice_arrow_ipc.rs`
- `src/randgen/column_choice.rs` if integration exposes behavior gaps
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review:

- How each test registers table data with DataFusion.
- How to register `column_choice_udfs()` and `column_choice_udafs()`.
- SQL shape for aggregate once and sample many rows.

## Red

Rewrite tests before implementation fixes:

- Parquet `UInt32` source: aggregate source column, sample generated rows, assert
  all outputs are from the distinct non-null set.
- Parquet `UInt64` source: same.
- Arrow IPC file and stream variants: same for the supported existing coverage.
- `all_udfs()` plus `all_udafs()` registration path works for column choice.
- Old cache-control tests are removed or replaced with aggregate registration
  assertions.

Expected red commands:

```bash
cargo test --features column-choice-parquet --test column_choice_parquet
cargo test --features column-choice-arrow-ipc --test column_choice_arrow_ipc
```

The expected failure is stale tests or missing integration behavior.

## Green

Implement integration fixes:

- Register source files as DataFusion tables where possible.
- Use SQL CTEs or subqueries to compute `randgen_roaring_agg(column)` once and
  feed it to `randgen_column_choice`.
- Register aggregate functions through `register_udaf`.
- Remove dependence on file-path scalar loading.

## Refactor

- Deduplicate SQL registration helpers in tests where it improves clarity.
- Keep feature tests focused on feature behavior, not generic aggregate internals
  already covered by Step 02 and Step 03.

## Verification Criteria

```bash
cargo fmt --check
cargo test --features column-choice-parquet --test column_choice_parquet
cargo test --features column-choice-arrow-ipc --test column_choice_arrow_ipc
cargo test --all-features --test column_choice_parquet --test column_choice_arrow_ipc
```

## Docs Requirements

Update `CONTEXT.md` with any final integration SQL patterns.

## Done When

- Parquet and Arrow IPC feature tests use aggregate-to-sampler SQL.
- No feature test depends on shared cache state or file-path sampler behavior.

## Carry-Forward Notes

Step 05 should remove stale implementation and docs now that integration proves
the new path.

