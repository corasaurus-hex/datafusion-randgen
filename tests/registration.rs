use datafusion::prelude::SessionContext;
use datafusion_randgen::all_udfs;

async fn query_succeeds(sql: &str) {
    let ctx = SessionContext::new();
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    assert!(!batches.is_empty());
}

#[tokio::test]
async fn all_udfs_registers_all_generators() {
    query_succeeds(
        "SELECT \
            randgen_int64_uniform(1, 1), \
            randgen_uint64_uniform(arrow_cast(1, 'UInt64'), arrow_cast(1, 'UInt64')), \
            randgen_float64_uniform(1.0, 1.0), \
            randgen_float64_normal(0.0, 1.0), \
            randgen_int64_normal(-10, 10, 0, 1.0), \
            randgen_uint64_normal(arrow_cast(0, 'UInt64'), arrow_cast(10, 'UInt64'), arrow_cast(0, 'UInt64'), 1.0), \
            randgen_bool(1.0), \
            randgen_utf8('A', 1, 1), \
            randgen_choice(['UTC']), \
            randgen_date32(to_date('2024-01-01'), to_date('2024-01-01')), \
            randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z')) \
         FROM generate_series(1, 1)",
    )
    .await;
}
