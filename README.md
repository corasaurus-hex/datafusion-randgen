# datafusion-randgen

Random data generator UDFs for Apache DataFusion.

The crate gives you `ScalarUDF` values. It doesn't take ownership of your
`SessionContext`, and it doesn't register anything by side effect. Register the
UDFs in the DataFusion application that owns the session.

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

Register individual UDFs when you only want part of the set:

```rust
ctx.register_udf(datafusion_randgen::int64_uniform_udf());
ctx.register_udf(datafusion_randgen::utf8_udf());
```

## UDFs

All generators are volatile DataFusion scalar functions. Bounds are inclusive.
When any required argument is null, the output for that row is null. Invalid
parameters return DataFusion errors instead of silently clamping or swapping
values.

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

For integer normal generators, `min..=max` is a truncation bound, not an input
used to calculate `stddev`. `mean` is the center of the unbounded distribution
and may sit outside the output range. Small and moderate `stddev` values use
rounded f64 normal offsets. Large `stddev` values add centered low-bit integer
dither so f64 spacing doesn't leave regular integer gaps. Every integer in the
requested range remains reachable.

The hybrid router estimates the probability that an unbounded draw will land in
`min..=max`. High-acceptance calls stay on the f64 or dithered fast path. If the
estimated acceptance probability drops below 5%, the sampler uses exact
integer-domain proposals instead: bounded uniform rejection for small ranges
near the center and a discrete exponential proposal for one-sided tails. When
f64 rounding can't represent the required integer domain exactly, full-range and
high-mass calls use Karney-style discrete normal sampling. A single-value range
returns that value, which is the correct truncated distribution.

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

Release checks:

```bash
cargo package --locked
cargo publish --dry-run --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny check advisories
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
```

`deny.toml` contains a narrow advisory ignore for `paste`, which is currently
pulled in by `datafusion 53.1.0`. Remove that ignore when DataFusion no longer
depends on it.
