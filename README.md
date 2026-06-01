# datafusion-randgen

Random data generator UDFs for Apache DataFusion.

This crate exports `ScalarUDF` constructors and leaves registration to the
application that owns the DataFusion `SessionContext`.

```rust
use datafusion::prelude::SessionContext;

let ctx = SessionContext::new();
for udf in datafusion_randgen::all_udfs() {
    ctx.register_udf(udf);
}
```

## UDFs

| Function | Arguments | Return type | Behavior |
| --- | --- | --- | --- |
| `randgen_int64_uniform` | `min Int64, max Int64` | `Int64` | Uniform integer in the inclusive range `min..=max`. |
| `randgen_float64_uniform` | `min Float64, max Float64` | `Float64` | Uniform float in the inclusive range `min..=max`. |
| `randgen_float64_normal` | `mean Float64, stddev Float64` | `Float64` | Random value from a normal distribution. |
| `randgen_bool` | `probability Float64` | `Boolean` | `true` with probability `0.0..=1.0`. |
| `randgen_utf8` | `characters Utf8, min_length Int64, max_length Int64` | `Utf8` | String using the distinct characters from `characters`. |
| `randgen_choice` | `choices List<T>` | `T` | Random element from a non-empty list. |
| `randgen_date32` | `min Date32, max Date32` | `Date32` | Date in the inclusive range `min..=max`. |
| `randgen_timestamp_millisecond` | `min Timestamp(Millisecond), max Timestamp(Millisecond)` | `Timestamp(Millisecond)` | Timestamp in the inclusive range `min..=max`. |

Bounds are inclusive. Null arguments produce null outputs for that row. Invalid
ranges or parameters return DataFusion execution errors.

## Examples

```sql
SELECT randgen_int64_uniform(1, 10)
FROM generate_series(1, 100);

SELECT randgen_utf8('ABC123', 8, 16)
FROM generate_series(1, 100);

SELECT randgen_choice(['UTC', 'America/New_York', 'Europe/London'])
FROM generate_series(1, 100);
```

## Development

The repository uses `just` for local checks:

```bash
just check
```

That runs formatting, type checking, clippy with warnings denied, and the test
suite through `cargo-nextest`.

Useful release checks:

```bash
cargo package --locked
cargo publish --dry-run --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```
