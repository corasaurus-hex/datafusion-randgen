# datafusion-randgen

Random data generator UDFs for Apache DataFusion.

The crate gives you `ScalarUDF` values. It does not modify your `SessionContext` or
register anything as a side effect. Register the UDFs yourself, in the
DataFusion application that owns the session.

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

All generators are volatile DataFusion scalar functions. Bounds are inclusive.
A null in any required argument means the output for that row is null. Invalid parameters
return DataFusion errors. They do not silently clamp or swap values.

| Function                        | Arguments                                                | Returns                  | Rules                                                                                     |
| ------------------------------- | -------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| `randgen_int64_uniform`         | `min Int64, max Int64`                                   | `Int64`                  | Requires `min <= max`; samples from `min..=max`.                                          |
| `randgen_uint64_uniform`        | `min UInt64, max UInt64`                                 | `UInt64`                 | Requires `min <= max`; supports the full `UInt64` range.                                  |
| `randgen_float64_uniform`       | `min Float64, max Float64`                               | `Float64`                | Requires finite bounds, finite span, and `min <= max`.                                    |
| `randgen_float64_normal`        | `mean Float64, stddev Float64`                           | `Float64`                | Requires finite arguments and `stddev > 0`.                                               |
| `randgen_int64_normal`          | `min Int64, max Int64, mean Int64, stddev Float64`       | `Int64`                  | Requires `min <= max`; `stddev` is caller-supplied; large `stddev` uses low-bit dither.   |
| `randgen_uint64_normal`         | `min UInt64, max UInt64, mean UInt64, stddev Float64`    | `UInt64`                 | Requires `min <= max`; `stddev` is caller-supplied; large `stddev` uses low-bit dither.   |
| `randgen_bool`                  | `probability Float64`                                    | `Boolean`                | Requires a finite probability in `0.0..=1.0`.                                             |
| `randgen_utf8`                  | `characters Utf8, min_length Int64, max_length Int64`    | `Utf8`                   | Uses the distinct characters from `characters`; requires `0 <= min_length <= max_length`. |
| `randgen_choice`                | `choices List<T>`                                        | `T`                      | Samples one element from a non-empty list for each row.                                   |
| `randgen_date32`                | `min Date32, max Date32`                                 | `Date32`                 | Requires `min <= max`; samples from the inclusive day range.                              |
| `randgen_timestamp_millisecond` | `min Timestamp(Millisecond), max Timestamp(Millisecond)` | `Timestamp(Millisecond)` | Requires matching timestamp timezones and `min <= max`.                                   |

For the integer normal generators, `min..=max` is a truncation bound, not an input used to calculate `stddev`. `mean` centers the unbounded
distribution and is allowed to sit outside the output range. Small and moderate
`stddev` values use rounded f64 normal offsets. Large `stddev` values use
centered low-bit integer dither so f64 spacing doesn't leave
regular gaps between reachable integers. Every integer in the requested range
remains reachable.

The hybrid router estimates the probability that an unbounded draw lands inside
`min..=max`. High-acceptance calls stay on the f64 or dithered fast path. When
the estimated acceptance probability drops below 5%, the sampler switches to
exact integer-domain proposals: bounded uniform rejection for small ranges near
the center, and a discrete exponential proposal for one-sided tails. If f64
rounding can't represent the required integer domain exactly, full-range and
high-mass calls fall back to Karney-style discrete normal sampling. A
single-value range returns that value, which is the correct truncated
distribution.

## Examples

```sql
SELECT randgen_int64_uniform(1, 10)
FROM generate_series(1, 100);

SELECT randgen_uint64_uniform(arrow_cast(0, 'UInt64'), arrow_cast(18446744073709551615, 'UInt64'))
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

The integration suite covers property tests for every public UDF, plus a
bounded stress test over larger batches. A longer soak pass is opt-in:

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
integration tests are excluded from the published crate.

`deny.toml` contains one narrow advisory ignore for `paste`, which is currently
pulled in by `datafusion 53.1.0`. The advisory marks `paste` unmaintained
and lists no safe upgrade. Drop the ignore once DataFusion stops depending on
it. Duplicate dependency versions remain warnings unless they point to a real
security or size problem.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
