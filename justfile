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

test-cargo *args:
    RUST_BACKTRACE=1 cargo test --all-targets {{args}}

type-check:
    cargo check --all-targets

alias type := type-check

fmt-check:
    cargo fmt --check

fmt:
    cargo fmt

lint:
    cargo clippy --all-targets -- -D warnings

check: fmt-check type-check lint test
