use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

mod integer_normal {
    #![allow(dead_code, unused_imports)]

    include!("../src/randgen/integer_normal.rs");
}

use integer_normal::IntegerNormalSampler;

const SAMPLES: usize = 64;

fn bench_integer_normal_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_normal_algorithms");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(SAMPLES as u64));

    let samplers = [
        (
            "f64_round",
            IntegerNormalSampler::for_offset_range(-(1_i128 << 40), 1_i128 << 40, 2.0, "bench")
                .unwrap(),
        ),
        (
            "f64_dither",
            IntegerNormalSampler::for_offset_range(
                i64::MIN as i128,
                i64::MAX as i128,
                9_007_199_254_740_992.0,
                "bench",
            )
            .unwrap(),
        ),
        (
            "karney_unbounded",
            IntegerNormalSampler::for_offset_range(-(1_i128 << 53) - 1, 1_i128, 2.0, "bench")
                .unwrap(),
        ),
        (
            "exact_uniform_range",
            IntegerNormalSampler::for_offset_range(-8, 8, 1_000_000.0, "bench").unwrap(),
        ),
        (
            "exact_tail",
            IntegerNormalSampler::for_offset_range(1_000, 10_000, 10.0, "bench").unwrap(),
        ),
    ];

    for (name, sampler) in samplers {
        group.bench_function(BenchmarkId::new("sample_offset", name), |b| {
            let mut rng = rand::rng();
            b.iter(|| {
                let mut sum = 0_i128;
                for _ in 0..SAMPLES {
                    sum = sum
                        .wrapping_add(sampler.sample_offset(&mut rng, black_box("bench")).unwrap());
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_integer_normal_algorithms);
criterion_main!(benches);
