# Column Choice Roaring Aggregate Step 06 — Docs and Examples

## Goal

Update user-facing docs, rustdoc, and examples for aggregate-driven column choice
registration and SQL usage.

## Crates/files touched

- `README.md`
- `src/lib.rs`
- `src/randgen/column_choice.rs`
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review the final public API from previous steps:

- Scalar constructors.
- Aggregate constructors.
- Feature flags.
- SQL examples.

## Red

Add or run checks that expose stale docs:

- Source scan for cache-control names in docs.
- Source scan for old `randgen_column_choice('/path', 'column')` examples.
- Rustdoc/compile check for public docs.

Expected red commands:

```bash
rg "cache|randgen_column_choice\\('/|source_path|column_name" README.md src/lib.rs src/randgen/column_choice.rs
cargo test --doc --all-features
```

The expected failure is stale docs or examples before edits.

## Green

Update docs:

- Explain registering `column_choice_udfs()` and `column_choice_udafs()`.
- Show aggregate-to-sampler SQL.
- Document `randgen_roaring_agg` input and return types.
- Document empty-set sampler error.
- Remove cache-control documentation.
- Remove file-path sampler examples.

## Refactor

- Keep README concise.
- Avoid duplicating a long function list in multiple places unless useful.
- Ensure rustdoc links compile.

## Verification Criteria

```bash
cargo fmt --check
cargo test --doc --all-features
! rg "randgen_column_choice_cache_|ColumnChoiceCache" README.md src
! rg "randgen_column_choice\\('/|source_path|column_name" README.md src/lib.rs src/randgen/column_choice.rs
```

If `source_path` or `column_name` remains in historical context, confirm it is
not documenting supported behavior.

## Docs Requirements

This is the docs step. Update `CONTEXT.md` with final docs status and examples.

## Done When

- README and rustdoc describe the aggregate-driven API accurately.
- No cache-control docs remain.
- Doc tests pass or any lack of doc tests is explicitly recorded.

## Carry-Forward Notes

Step 07 should run the broad project gate and catch consistency drift.

