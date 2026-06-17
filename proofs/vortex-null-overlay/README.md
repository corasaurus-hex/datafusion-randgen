# Vortex Null Overlay Proof

This proof checks whether a Vortex file round trip preserves physical values
under Arrow null slots.

Run it with:

```sh
cargo run --manifest-path proofs/vortex-null-overlay/Cargo.toml
```

This proof is not part of normal CI because Vortex is outside this crate's
dependency surface, has a newer MSRV than this crate, and compiles a large
dependency graph.
