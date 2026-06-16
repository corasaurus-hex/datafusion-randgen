# Column Choice Roaring Aggregate Step 05 — Remove Cache Surface

## Goal

Remove the shared cache implementation, cache-control SQL API, stale exports,
and normal Tokio dependency from the outstanding changes.

## Crates/files touched

- `Cargo.toml`
- `Cargo.lock` if dependency resolution changes
- `src/lib.rs`
- `src/randgen/column_choice.rs`
- `README.md` only if needed before the dedicated docs step
- `tests/column_choice_parquet.rs`
- `tests/column_choice_arrow_ipc.rs`
- `docs/implementation/column-choice-roaring-agg/CONTEXT.md`

## Start with abstractions/types

Review what should remain:

- `ColumnChoice`
- `RoaringAgg`
- Serialization helpers
- Optional source-file test writers only where tests still need them

Review what must go:

- `ColumnChoiceCache`
- `ColumnChoiceCacheConfig`
- `ColumnChoiceCacheStats`
- `ColumnChoiceCacheControl`
- `randgen_column_choice_cache_*`
- Normal `tokio` dependency added for cleanup
- File-path sampler planning and invocation

## Red

Add checks that fail while cache artifacts remain:

- Source scan test or scripted check for `randgen_column_choice_cache_`.
- Source scan check for `ColumnChoiceCache`.
- Cargo manifest check that `tokio` is not a normal dependency.
- Compile check that stale cache exports are gone.

Expected red commands:

```bash
rg "randgen_column_choice_cache_|ColumnChoiceCache" src README.md tests
rg '^tokio = .*rt.*time' Cargo.toml
cargo check --all-features --all-targets
```

The expected failure is finding stale cache symbols or manifest entries.

## Green

Remove stale cache code and exports. Keep dev-dependency `tokio` only if tests
still require it.

## Refactor

- Simplify imports.
- Remove file metadata/cache key helpers that are no longer needed.
- Keep the column-choice module cohesive and feature-gated.

## Verification Criteria

```bash
cargo fmt --check
! rg "randgen_column_choice_cache_|ColumnChoiceCache" src README.md tests
! rg '^tokio = .*rt.*time' Cargo.toml
cargo check --all-features --all-targets
cargo test --all-features --test column_choice_parquet --test column_choice_arrow_ipc
```

If the shell does not support `!` in the execution context, run `rg` and confirm
it exits with no matches.

## Docs Requirements

Update `CONTEXT.md` with removed API notes. Carry README cleanup into Step 06 if
not completed here.

## Done When

- Cache code and docs are gone.
- Normal `tokio` dependency is gone.
- Feature tests still pass.

## Carry-Forward Notes

Step 06 must ensure README and rustdoc explain aggregate registration instead of
cache configuration.

