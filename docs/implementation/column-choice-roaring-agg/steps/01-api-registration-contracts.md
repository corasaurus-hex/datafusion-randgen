# Column Choice Roaring Aggregate Step 01 — API and Registration Contracts

## Goal

Lock the public API and registration contract for aggregate-driven column choice
before implementing accumulator internals.

## Crates/files touched

- `src/lib.rs`
- `src/randgen/column_choice.rs`
- `tests/registration.rs`
- Potential new focused test file if registration coverage fits better there.
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review and define:

- `RoaringAgg` as the `AggregateUDFImpl` type.
- `ColumnChoice` as a scalar sampler over serialized roaring values.
- Public constructors:
  - `column_choice_udf() -> ScalarUDF`
  - `column_choice_udfs() -> Vec<ScalarUDF>`
  - `roaring_agg_udaf() -> AggregateUDF`
  - `column_choice_udafs() -> Vec<AggregateUDF>`
  - `all_udafs() -> Vec<AggregateUDF>`

## Red

Write tests/checks that fail before implementation:

- A compile/integration test imports and calls the new UDAF constructors behind a
  column-choice feature.
- A registration test registers `column_choice_udfs()` with `register_udf` and
  `column_choice_udafs()` with `register_udaf`.
- A source or compile check proves cache-control constructors are not part of the
  intended API.

Expected red commands:

```bash
cargo test --features column-choice-parquet --test registration
cargo check --features column-choice-parquet --all-targets
```

The expected failure is missing aggregate constructors/types or stale cache API.

## Green

Implement only enough public API scaffolding to satisfy registration and compile
contracts:

- Add `AggregateUDF` import and constructor exports.
- Add `RoaringAgg` skeleton with type signatures.
- Keep accumulator methods stubbed only as far as compile/tests allow.
- Change `ColumnChoice` signature planning to accept `Binary`/`LargeBinary`, but
  deeper sampling can remain for Step 03.

## Refactor

- Keep feature gates minimal and consistent.
- Avoid pulling `datafusion` or `SessionContext` into the library.
- Remove any public cache exports introduced by the outstanding diff if they
  conflict with the new contract.

## Verification Criteria

```bash
cargo fmt --check
cargo check --features column-choice-parquet --all-targets
cargo check --features column-choice-arrow-ipc --all-targets
cargo test --features column-choice-parquet --test registration
```

## Docs Requirements

Update `CONTEXT.md` with the final constructor names and any API deviations.

## Done When

- New constructors compile.
- Registration tests prove callers can register scalar and aggregate functions
  separately.
- No implementation step has introduced cache-control public API.

## Carry-Forward Notes

Step 02 depends on the final `RoaringAgg` type and constructor names.

