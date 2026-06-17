@default:
    echo
    echo 'Usage:'
    echo
    echo '    Run `just` to list all tasks.'
    echo '    Run `just <task>` to run a task.'
    echo
    echo 'Tasks:'
    echo
    just --list --unsorted --list-heading '' --list-submodules
    echo

alias ls := default

test *args:
    RUST_BACKTRACE=1 cargo nextest run --all-targets {{args}}

test-all-features *args:
    RUST_BACKTRACE=1 cargo nextest run --all-targets --all-features {{args}}

test-cargo *args:
    RUST_BACKTRACE=1 cargo test --all-targets {{args}}

type-check:
    cargo check --all-targets

type-check-all-features:
    cargo check --all-targets --all-features

alias type := type-check

fmt-check:
    cargo fmt --check

fmt:
    cargo fmt

lint:
    cargo clippy --all-targets --all-features -- -D warnings

doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

deny-check:
    cargo deny check

package-check:
    cargo publish --dry-run --locked --allow-dirty

fuzz-build:
    cargo +nightly fuzz build

fuzz-smoke:
    cargo +nightly fuzz run direct_api -- -runs=256
    cargo +nightly fuzz run sql_api -- -runs=64

stress:
    RUST_BACKTRACE=1 cargo test --test public_udf_invariants stress_public_udfs_over_large_batches

stress-sql:
    RUST_BACKTRACE=1 cargo test --test sql_udf_invariants stress_sql_public_udfs_over_large_batches

stress-column-choice:
    RUST_BACKTRACE=1 cargo test --all-features --test sql_udf_invariants stress_sql_column_choice_over_large_source_and_sample_batches

stress-all: stress stress-sql stress-column-choice

soak:
    RUST_BACKTRACE=1 cargo test --test public_udf_invariants -- --ignored --nocapture

soak-sql:
    RUST_BACKTRACE=1 cargo test --test sql_udf_invariants soak_sql_public_udfs_over_repeated_large_batches -- --ignored --nocapture

soak-all: soak soak-sql

coverage:
    cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

check: fmt-check type-check type-check-all-features lint test test-all-features

release-check: check doc-check package-check deny-check
