# Column Choice Roaring Aggregate Design

## Problem

The current outstanding column-choice changes moved toward a shared in-process
cache with SQL cache-control functions. The intended direction is different:
build a distinct roaring value set explicitly in SQL with an aggregate, then pass
that value into the random sampler. The roaring aggregate should be scoped to
the column-choice feature area, not exposed as a broad general-purpose aggregate
outside this feature.

## Goals

- Replace the shared cache direction with an explicit aggregate-to-sampler flow.
- Add a feature-gated `randgen_roaring_agg` aggregate for `UInt32` and `UInt64`
  values.
- Make `randgen_column_choice` sample from the aggregate result rather than
  requiring a shared cache.
- Remove the old file-path sampler API; making that path fast requires cache
  machinery that is out of scope for this feature.
- Keep generated outputs typed as `UInt32` for `UInt32` inputs and `UInt64` for
  `UInt64` inputs.
- Preserve optional `null_probability Float64` support for the sampler.
- Keep all normal tests offline and deterministic except for the intended random
  membership assertions.
- Remove the new shared cache API, SQL cache controls, and Tokio runtime
  dependency from the outstanding work.

## Non-Goals

- Do not expose a general roaring manipulation package.
- Do not add SQL cache inspection, clearing, eviction, or configuration UDFs.
- Do not introduce a background cleanup task.
- Do not support arbitrary numeric types in the first implementation.
- Do not support string, signed integer, or floating-point column-choice sets in
  this feature.

## Proposed Architecture

### Feature Scope

The roaring aggregate and aggregate-driven sampler are compiled only when a
column-choice feature is enabled:

- `column-choice-parquet`
- `column-choice-arrow-ipc`

Both features already depend on `roaring`. The aggregate should live in the
existing `randgen::column_choice` module or a child module under it, so the
scope remains tied to column choice.

### Aggregate UDAF

Add a DataFusion aggregate UDF, tentatively named `randgen_roaring_agg`.

Input and output mapping:

- `UInt32` input -> `Binary` output containing a serialized `RoaringBitmap`.
- `UInt64` input -> `LargeBinary` output containing a serialized
  `RoaringTreemap`.

The accumulator:

- Ignores null input rows.
- Inserts distinct non-null values into the corresponding roaring set.
- Serializes the roaring set in `state()` and `evaluate()`.
- Merges partial states by deserializing and unioning roaring sets.
- Returns a valid serialized empty roaring set for an all-null or empty input
  group.

### Sampler Scalar UDF

Update `randgen_column_choice` to accept:

```sql
randgen_column_choice(values[, null_probability])
```

Where `values` is the scalar result of `randgen_roaring_agg`.

Type behavior:

- `Binary` input -> returns `UInt32`.
- `LargeBinary` input -> returns `UInt64`.
- Null aggregate input produces null output rows.
- Empty non-null roaring sets error with a clear message when sampled.

Runtime behavior:

- Deserialize the scalar roaring value once per invocation.
- Sample by random rank using roaring `select`.
- Apply optional native null probability row-by-row.

### Registration API

Because DataFusion registers scalar UDFs and aggregate UDFs separately, add
aggregate constructors alongside the scalar constructors.

Likely additions:

- `column_choice_udf() -> ScalarUDF`
- `column_choice_udfs() -> Vec<ScalarUDF>`
- `column_choice_udafs() -> Vec<AggregateUDF>`
- `roaring_agg_udaf() -> AggregateUDF`
- `all_udafs() -> Vec<AggregateUDF>`

`all_udfs()` should remain scalar-only because the crate does not own a
`SessionContext`, and `SessionContext` has separate `register_udf` and
`register_udaf` methods. Documentation should show registering both sets:

```rust
for udf in datafusion_randgen::column_choice_udfs() {
    ctx.register_udf(udf);
}
for udaf in datafusion_randgen::column_choice_udafs() {
    ctx.register_udaf(udaf);
}
```

## SQL Shape

Expected usage:

```sql
WITH choices AS (
  SELECT randgen_roaring_agg(user_id) AS user_ids
  FROM users
)
SELECT randgen_column_choice(user_ids)
FROM choices, generate_series(1, 100);
```

With null probability:

```sql
WITH choices AS (
  SELECT randgen_roaring_agg(user_id) AS user_ids
  FROM users
)
SELECT randgen_column_choice(user_ids, 0.05)
FROM choices, generate_series(1, 100);
```

## Migration From Current Outstanding Changes

- Remove `ColumnChoiceCache`, cache stats/config/control types, and cache-control
  UDFs.
- Remove the normal `tokio` dependency added for cache cleanup.
- Replace cache-control README sections with aggregate-driven usage.
- Remove the old file-path `randgen_column_choice(source_path, column_name[,
  null_probability])` API.
- Rewrite Parquet and Arrow IPC tests to register source tables and aggregate
  source columns through SQL rather than checking cache behavior.

## Testing Strategy

Use tests-first red/green/refactor for each implementation slice.

Coverage layers:

- Unit tests for roaring serialization, deserialization, union, and empty/null
  behavior.
- Aggregate SQL integration tests for `UInt32`, `UInt64`, duplicates, nulls,
  partial/final merge behavior where practical, and grouped aggregates.
- Scalar sampler tests for `Binary` and `LargeBinary` aggregate outputs,
  optional null probability, bad argument types, empty sets, and null aggregate
  inputs.
- Feature-gated Parquet and Arrow IPC integration tests proving the aggregate can
  be used against registered source data under each feature.
- Registration tests proving aggregate constructors are documented and registered
  separately from scalar UDFs.
- Source scans or compile checks proving cache-control functions and normal
  Tokio dependency are gone.
- README/doc examples that match actual registration requirements.

## Formal TDD/RGR Step Plan

Do not create the detailed step files until the open questions are answered.
The expected steps are:

1. Red: lock public API and registration contracts.
   Green: add aggregate constructor stubs and scalar signature changes behind
   the column-choice feature.
   Refactor: keep scalar and aggregate registration paths clearly separated.

2. Red: add accumulator tests for `UInt32` and `UInt64` aggregate behavior.
   Green: implement `randgen_roaring_agg` accumulator update, evaluate, state,
   and merge.
   Refactor: centralize serialization/deserialization helpers.

3. Red: add scalar sampler tests using serialized roaring aggregate values.
   Green: update `randgen_column_choice` to accept aggregate values and sample
   from `Binary`/`LargeBinary`.
   Refactor: simplify or remove file-path cache-specific helpers.

4. Red: rewrite Parquet and Arrow IPC integration tests around SQL aggregation
   and sampling.
   Green: register source tables and prove aggregate-to-sampler works under each
   feature.
   Refactor: deduplicate test helpers across the feature tests where useful.

5. Red: add regression tests proving the cache API is gone and registration docs
   compile conceptually.
   Green: remove cache-control exports, docs, and the normal `tokio` dependency.
   Refactor: clean stale names and feature-gated code.

6. Red: add README/example checks or focused docs review expectations.
   Green: update README and public rustdoc for the aggregate-driven workflow.
   Refactor: tighten examples and keep feature-scope wording explicit.

7. Final hardening.
   Run formatting, targeted feature tests, all-feature tests, and the normal
   project check command where practical. Fix any integration drift found by the
   broader suite.

## Open Questions

1. Should the old file-path API
   `randgen_column_choice(source_path, column_name[, null_probability])` be
   removed in this pivot, or kept as a convenience path separate from the new
   aggregate flow? Answer: remove it.
2. Is the proposed encoding acceptable: `UInt32 -> Binary` and
   `UInt64 -> LargeBinary`, letting `randgen_column_choice` infer output type
   from the aggregate value type? Answer: yes.
3. For an all-null source group, should `randgen_roaring_agg` return null and
   `randgen_column_choice(null)` produce null rows, or should the sampler error
   with "requires at least one non-null source value"? Answer: an empty roaring
   bitmap/treemap is valid aggregate output, but `randgen_column_choice` should
   error when asked to pull random values from it.
4. Should `all_udafs()` be added as the aggregate counterpart to `all_udfs()`,
   or do you prefer only feature-specific aggregate constructors? Answer:
   provide a way to register both column-choice scalar and aggregate functions,
   without adding a `SessionContext` dependency to this crate.
5. Should the aggregate name be exactly `randgen_roaring_agg`, or should it be
   more feature-specific, such as `randgen_column_choice_agg`? Answer:
   `randgen_roaring_agg`.

## Answered Assumptions

- The shared cache design is not desired.
- The roaring aggregate is scoped to the column-choice feature.
- The implementation should proceed through tests-first red/green/refactor.
- `randgen_column_choice` will no longer read source files directly.
- `randgen_roaring_agg(UInt32)` returns `Binary`.
- `randgen_roaring_agg(UInt64)` returns `LargeBinary`.
- `randgen_roaring_agg` returns valid serialized empty roaring values for empty
  or all-null groups.
- `randgen_column_choice` errors when sampling from an empty non-null roaring
  value.
