# Column Choice Roaring Aggregate Step 03 — Aggregate-Value Sampler

## Goal

Update `randgen_column_choice` so it samples from the serialized roaring value
produced by `randgen_roaring_agg`.

## Crates/files touched

- `src/randgen/column_choice.rs`
- `tests/column_choice_roaring_agg.rs` or `tests/column_choice_sampler.rs`
- Existing column-choice integration tests as needed
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review and define:

- `ColumnValues` as the deserialized sampler representation.
- Scalar argument parsing for `Binary`, `LargeBinary`, and null values.
- Return field inference:
  - `Binary` -> `UInt32`
  - `LargeBinary` -> `UInt64`
- Empty non-null roaring set error path.

## Red

Write failing sampler tests first:

- `randgen_column_choice(randgen_roaring_agg(u32_col))` returns only values from
  the source set and returns `UInt32`.
- `randgen_column_choice(randgen_roaring_agg(u64_col))` returns only values from
  the source set and returns `UInt64`.
- Optional null probability `1.0` produces null rows.
- Bad null probability values error.
- A valid empty serialized `Binary` or `LargeBinary` input errors when sampled.
- A null scalar roaring input produces null rows.
- Old file-path call shape no longer plans.

Expected red commands:

```bash
cargo test --features column-choice-parquet --test column_choice_roaring_agg
```

The expected failure is that the scalar UDF still expects path/column arguments
or cannot deserialize aggregate values.

## Green

Implement:

- `randgen_column_choice(values[, null_probability])`.
- Scalar binary extraction for `Binary`, `LargeBinary`, and compatible binary
  scalar variants if locally appropriate.
- Deserialization into `ColumnValues`.
- Sampling by random rank.
- Null propagation for null roaring input.
- Empty-set error on non-null roaring input.

## Refactor

- Delete or isolate file-reader-specific sampler paths that are no longer used.
- Keep `ColumnValues::sample` small and shared between bitmap and treemap.
- Ensure error messages name `randgen_column_choice`.

## Verification Criteria

```bash
cargo fmt --check
cargo test --features column-choice-parquet --test column_choice_roaring_agg
cargo test --features column-choice-arrow-ipc --test column_choice_roaring_agg
cargo check --all-features --all-targets
```

## Docs Requirements

Update `CONTEXT.md` with final null and empty-set behavior.

## Done When

- Sampler tests pass for `Binary` and `LargeBinary`.
- Old path/column API is no longer accepted.
- Null probability behavior matches other generator UDFs.

## Carry-Forward Notes

Step 04 should test the complete aggregate-to-sampler flow against feature-backed
source data.

