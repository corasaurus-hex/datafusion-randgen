# Column Choice Roaring Aggregate Step 02 — Roaring Aggregate Accumulator

## Goal

Implement `randgen_roaring_agg` for `UInt32` and `UInt64`, including update,
state, merge, and evaluate behavior.

## Crates/files touched

- `src/randgen/column_choice.rs`
- `tests/column_choice_roaring_agg.rs` or an equivalent focused integration test
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review and define:

- Internal aggregate kind: `UInt32` vs `UInt64`.
- Serialization helpers:
  - `serialize_bitmap(&RoaringBitmap) -> Result<Vec<u8>>`
  - `deserialize_bitmap(&[u8], name) -> Result<RoaringBitmap>`
  - `serialize_treemap(&RoaringTreemap) -> Result<Vec<u8>>`
  - `deserialize_treemap(&[u8], name) -> Result<RoaringTreemap>`
- Accumulator state as serialized roaring bytes, matching the aggregate return
  type.

## Red

Write failing aggregate tests first:

- `randgen_roaring_agg(UInt32)` returns `Binary` that deserializes to distinct
  non-null `UInt32` values.
- `randgen_roaring_agg(UInt64)` returns `LargeBinary` that deserializes to
  distinct non-null `UInt64` values.
- Duplicates collapse.
- Nulls are ignored.
- All-null or empty input produces a valid serialized empty roaring value, not a
  cache lookup and not a null.
- Partial/final merge behavior unions serialized states. Prefer a query shape or
  direct accumulator test that exercises `state()` and `merge_batch()`.

Expected red commands:

```bash
cargo test --features column-choice-parquet --test column_choice_roaring_agg
```

The expected failure is missing or incomplete aggregate behavior.

## Green

Implement the accumulator:

- Validate one input array.
- Support only `UInt32` and `UInt64`.
- Ignore null rows.
- Serialize in both `state()` and `evaluate()`.
- Merge by deserializing each non-null state row and unioning.
- Report allocated size using roaring serialized size plus struct overhead.

## Refactor

- Share serialization/deserialization helpers with Step 03.
- Keep errors clear and function-name specific.
- Keep feature gates tied to column-choice features.

## Verification Criteria

```bash
cargo fmt --check
cargo test --features column-choice-parquet --test column_choice_roaring_agg
cargo test --features column-choice-arrow-ipc --test column_choice_roaring_agg
cargo check --all-features --all-targets
```

## Docs Requirements

Update `CONTEXT.md` with any final encoding details or accumulator behavior
changes.

## Done When

- Aggregate tests pass for both `UInt32` and `UInt64`.
- Empty aggregate output is valid serialized roaring data.
- Partial state merge is covered.

## Carry-Forward Notes

Step 03 uses the same deserialization helpers and empty-set behavior.

