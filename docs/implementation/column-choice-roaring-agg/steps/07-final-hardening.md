# Column Choice Roaring Aggregate Step 07 — Final Hardening

## Goal

Run broad verification, fix drift, and leave the working tree in a coherent
state ready for review.

## Crates/files touched

- Any files needed to fix verification failures
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review the final architecture:

- Explicit `randgen_roaring_agg` aggregate.
- `randgen_column_choice` samples only from serialized roaring values.
- No shared cache or file-path sampler.
- Separate scalar and aggregate registration.

## Red

Run the broad checks before making final hardening edits:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --all-features --test column_choice_parquet --test column_choice_arrow_ipc
cargo test --doc --all-features
```

Any failure is red for this hardening step.

## Green

Fix only issues exposed by final verification:

- Formatting.
- Clippy warnings.
- Feature-gate compile failures.
- Test flakiness caused by overly narrow random sample sizes.
- Stale docs/source-scan issues.

## Refactor

- Keep cleanup tightly scoped.
- Do not introduce new behavior in final hardening unless required by a failing
  check.

## Verification Criteria

Required where practical:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --features column-choice-parquet --test column_choice_parquet
cargo test --features column-choice-arrow-ipc --test column_choice_arrow_ipc
cargo test --all-features --test column_choice_parquet --test column_choice_arrow_ipc
cargo test --doc --all-features
```

If `just check` is available and not materially slower than the above, run it
too or instead of the overlapping base commands:

```bash
just check
```

## Docs Requirements

Update `CONTEXT.md` with final verification results and remaining risks, if any.

## Done When

- Broad checks pass or any skipped checks are explicitly justified.
- No cache or file-path sampler surface remains.
- The plan context records completion.

## Carry-Forward Notes

Final handoff should summarize changed API, removed cache surface, and commands
run.

