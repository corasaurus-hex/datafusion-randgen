# datafusion-randgen

Random data generator UDFs for Apache DataFusion.

This crate exports `ScalarUDF` constructors. It does not modify a
`SessionContext` or register functions as a side effect. The application that
owns the DataFusion session chooses which UDFs to register.

## Install

```toml
[dependencies]
datafusion-randgen = "0.1.0"
```

## Register

```rust
use datafusion::prelude::SessionContext;

let ctx = SessionContext::new();
for udf in datafusion_randgen::all_udfs() {
    ctx.register_udf(udf);
}
```

If you only want part of the set, register individual UDFs:

```rust
ctx.register_udf(datafusion_randgen::int64_uniform_udf());
ctx.register_udf(datafusion_randgen::utf8_udf());
```

## UDFs

Every generator is a volatile DataFusion scalar function. Bounds are inclusive.
Null required arguments produce null output for that row. Invalid parameters
return DataFusion errors; generators do not clamp or swap bad ranges.

| Function                        | Arguments                                                | Returns                  | Rules                                                                                     |
| ------------------------------- | -------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| `randgen_int64_uniform`         | `min Int64, max Int64`                                   | `Int64`                  | Requires `min <= max`; samples from `min..=max`.                                          |
| `randgen_uint64_uniform`        | `min UInt64, max UInt64`                                 | `UInt64`                 | Requires `min <= max`; accepts nonnegative signed integer inputs.                         |
| `randgen_float64_uniform`       | `min Float64, max Float64`                               | `Float64`                | Requires finite bounds, finite span, and `min <= max`.                                    |
| `randgen_float64_normal`        | `mean Float64, stddev Float64`                           | `Float64`                | Requires finite arguments and `stddev > 0`.                                               |
| `randgen_int64_normal`          | `min Int64, max Int64, mean Int64, stddev Float64`       | `Int64`                  | Requires `min <= max`; samples a rounded f64 normal and rejects values outside the range. |
| `randgen_uint64_normal`         | `min UInt64, max UInt64, mean UInt64, stddev Float64`    | `UInt64`                 | Requires `min <= max`; accepts nonnegative signed integer inputs.                         |
| `randgen_bool`                  | `probability Float64`                                    | `Boolean`                | Requires a finite probability in `0.0..=1.0`.                                             |
| `randgen_utf8`                  | `characters Utf8, min_length Int64, max_length Int64`    | `Utf8`                   | Uses the distinct characters from `characters`; requires `0 <= min_length <= max_length`. |
| `randgen_choice`                | `choices List<T>`                                        | `T`                      | Samples one element from a non-empty list for each row.                                   |
| `randgen_date32`                | `min Date32, max Date32`                                 | `Date32`                 | Requires `min <= max`; samples from the inclusive day range.                              |
| `randgen_timestamp_millisecond` | `min Timestamp(Millisecond), max Timestamp(Millisecond)` | `Timestamp(Millisecond)` | Requires matching timestamp timezones and `min <= max`.                                   |

The `UInt64` generators accept `UInt64` values and nonnegative signed integer
inputs. Plain SQL calls can mix literals such as `0` with values larger than
`Int64::MAX`; DataFusion parses `18446744073709551615` as `UInt64`.

For the integer normal generators, `min..=max` is a truncation bound, not an
input used to calculate `stddev`. The implementation samples an f64 normal
centered on `mean`, rounds to the nearest integer, and retries values outside
`min..=max`. A single-value range returns that value. Very low-probability tail
ranges can error after a bounded number of retries; widen the range or move
`mean` closer to the requested output range in that case.

Use `randgen_int64_uniform` or `randgen_uint64_uniform` when every integer in a
large range must be directly representable. Integer normal sampling is
f64-backed and inherits f64 spacing limits for very large magnitudes.

## Examples

```sql
SELECT randgen_int64_uniform(1, 10)
FROM generate_series(1, 100);

SELECT randgen_uint64_uniform(0, 18446744073709551615)
FROM generate_series(1, 100);

SELECT randgen_int64_normal(0, 200, 100, 15.0)
FROM generate_series(1, 100);

SELECT randgen_utf8('ABC123', 8, 16)
FROM generate_series(1, 100);

SELECT randgen_choice(['UTC', 'America/New_York', 'Europe/London'])
FROM generate_series(1, 100);
```

Column arguments work the same way as constants:

```sql
SELECT randgen_int64_uniform(min_value, max_value)
FROM bounds;

SELECT randgen_utf8(alphabet, min_len, max_len)
FROM string_specs;
```

## Development

The local gate is:

```bash
just check
```

That runs:

- `cargo fmt --check`
- `cargo check --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `cargo nextest run --all-targets`

The integration suite includes property tests for every public UDF and a bounded
stress test over larger batches. A longer soak pass is opt-in:

```bash
just soak
RANDGEN_SOAK_ITERATIONS=100 RANDGEN_SOAK_ROWS=16384 just soak
```

Release checks:

```bash
cargo publish --dry-run --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check
just coverage
```

Use `cargo package --list --locked` to check package contents. Benchmarks and
integration tests are not included in the published crate.

`deny.toml` contains one advisory ignore for `paste`, which is pulled in by
`datafusion 53.1.0`. The advisory marks `paste` unmaintained and lists no safe
upgrade. Drop the ignore once DataFusion stops depending on it. Duplicate
dependency versions remain warnings unless they point to a security or size
problem.

## License

Licensed under MIT or Apache-2.0, at your option.
