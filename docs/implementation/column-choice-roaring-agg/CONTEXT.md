# Column Choice Roaring Aggregate Context

## Purpose

Pivot column-choice sampling away from a shared cache design. The new model is
explicit SQL dataflow: aggregate source values into a serialized roaring set
with `randgen_roaring_agg`, then pass that value into `randgen_column_choice` to
sample generated rows.

## Current Status

Planning complete. Implementation has not started.

Current step: Step 02, roaring aggregate accumulator.

## Locked Constraints

- Remove the old file-path `randgen_column_choice(source_path, column_name[,
  null_probability])` API.
- Remove the shared cache changes, SQL cache controls, and normal `tokio`
  dependency from the outstanding diff.
- Scope `randgen_roaring_agg` to the column-choice feature area.
- Keep aggregate registration separate from scalar registration because
  DataFusion uses `register_udaf` separately from `register_udf`.
- Do not add a dependency on `datafusion::SessionContext` to the library crate.
- Do not create commits unless explicitly requested by the user.

## Target API

- `column_choice_udf() -> ScalarUDF`
- `column_choice_udfs() -> Vec<ScalarUDF>`
- `roaring_agg_udaf() -> AggregateUDF`
- `column_choice_udafs() -> Vec<AggregateUDF>`
- `all_udafs() -> Vec<AggregateUDF>`

Feature-gated public types:

- `ColumnChoice`
- `RoaringAgg`

Target SQL:

```sql
WITH choices AS (
  SELECT randgen_roaring_agg(user_id) AS user_ids
  FROM users
)
SELECT randgen_column_choice(user_ids)
FROM choices, generate_series(1, 100);
```

## Behavior Rules

- `randgen_roaring_agg(UInt32)` returns `Binary`.
- `randgen_roaring_agg(UInt64)` returns `LargeBinary`.
- The aggregate ignores null input rows.
- Empty and all-null aggregate groups return valid serialized empty roaring
  values.
- `randgen_column_choice(Binary[, Float64])` returns `UInt32`.
- `randgen_column_choice(LargeBinary[, Float64])` returns `UInt64`.
- `randgen_column_choice(NULL)` produces null output rows.
- `randgen_column_choice` errors when the non-null roaring input has cardinality
  zero.
- The optional null probability remains finite `Float64` in `0.0..=1.0`.

## Important Repository Context

- Existing optional features:
  - `column-choice-parquet = ["dep:parquet", "dep:roaring"]`
  - `column-choice-arrow-ipc = ["dep:arrow-ipc", "dep:roaring"]`
- The current outstanding diff contains shared cache work that should be removed
  during implementation.
- Integration tests already exist for Parquet and Arrow IPC column-choice paths;
  they need to be rewritten around aggregate-to-sampler SQL.
- `README.md` currently includes stale cache-control documentation from the
  outstanding diff.

## Completed Steps

- Step 01, API and registration contracts: completed without a commit. Added
  feature-gated registration tests, exposed `RoaringAgg`, added
  `roaring_agg_udaf()`, `column_choice_udafs()`, and `all_udafs()`, narrowed
  `column_choice_udfs()` to the scalar sampler only, and changed
  `randgen_column_choice` planning to `Binary`/`LargeBinary` input contracts.
  Verification passed:
  - `cargo fmt --check`
  - `cargo check --features column-choice-parquet --all-targets`
  - `cargo check --features column-choice-arrow-ipc --all-targets`
  - `cargo test --features column-choice-parquet --test registration`

## Carry-Forward Risks

- DataFusion aggregate APIs require `AggregateUDF` registration via
  `SessionContext::register_udaf`; examples must show both scalar and aggregate
  registration.
- `Binary` vs `LargeBinary` is the type signal for sampler output type; tests
  must guard this contract.
- Empty roaring values are valid aggregate output but invalid sampler input for
  random selection.
- The old file-path loader/cache internals still exist after Step 01 and produce
  dead-code warnings because the planning contract no longer uses them. Later
  steps should remove them rather than preserving compatibility.
- `RoaringAggAccumulator` is only a Step 01 skeleton. It currently returns empty
  serialized placeholder values and must be replaced with real update, state,
  merge, and evaluate behavior in Step 02.
