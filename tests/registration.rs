use arrow_array::Array;
use arrow_array::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_randgen::all_udfs;
#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
use datafusion_randgen::{all_udafs, column_choice_udafs, column_choice_udfs, roaring_agg_udaf};

async fn query_batches(sql: &str) -> Vec<RecordBatch> {
    let ctx = SessionContext::new();
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    assert!(!batches.is_empty());
    batches
}

async fn query_succeeds(sql: &str) {
    query_batches(sql).await;
}

async fn query_fails(sql: &str) {
    let ctx = SessionContext::new();
    for udf in all_udfs() {
        ctx.register_udf(udf);
    }
    if let Ok(df) = ctx.sql(sql).await {
        assert!(df.collect().await.is_err());
    }
}

#[tokio::test]
async fn all_udfs_registers_all_generators() {
    query_succeeds(
        "SELECT \
            randgen_int64_uniform(1, 1), \
            randgen_uint64_uniform(1, 1), \
            randgen_float64_uniform(1.0, 1.0), \
            randgen_float64_normal(0.0, 1.0), \
            randgen_int64_normal(-10, 10, 0, 1.0), \
            randgen_uint64_normal(0, 10, 0, 1.0), \
            randgen_bool(1.0), \
            randgen_utf8('A', 1, 1), \
            randgen_choice(['UTC']), \
            randgen_nullable(randgen_int64_uniform(1, 1), 0.0), \
            randgen_date32(to_date('2024-01-01'), to_date('2024-01-01')), \
            randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z')) \
         FROM generate_series(1, 1)",
    )
    .await;
}

#[tokio::test]
async fn native_nullability_arguments_work_in_sql() {
    let batches = query_batches(
        "SELECT \
            randgen_int64_uniform(1, 1, 1.0), \
            randgen_uint64_uniform(1, 1, 1.0), \
            randgen_float64_uniform(1.0, 1.0, 1.0), \
            randgen_float64_normal(0.0, 1.0, 1.0), \
            randgen_int64_normal(-10, 10, 0, 1.0, 1.0), \
            randgen_uint64_normal(0, 10, 0, 1.0, 1.0), \
            randgen_bool(1.0, 1.0), \
            randgen_utf8('A', 1, 1, 1.0), \
            randgen_choice(['UTC'], 1.0), \
            randgen_date32(to_date('2024-01-01'), to_date('2024-01-01'), 1.0), \
            randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z'), 1.0) \
         FROM generate_series(1, 3)",
    )
    .await;

    for batch in batches {
        for column in batch.columns() {
            assert_eq!(column.null_count(), column.len());
        }
    }
}

#[tokio::test]
async fn native_nullability_rejects_bad_sql_probabilities() {
    query_fails("SELECT randgen_int64_uniform(1, 1, -0.1) FROM generate_series(1, 1)").await;
    query_fails("SELECT randgen_int64_uniform(1, 1, 1.1) FROM generate_series(1, 1)").await;
    query_fails("SELECT randgen_bool(0.5, p) FROM (VALUES (CAST(-0.1 AS DOUBLE))) AS t(p)").await;
}

#[cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
#[tokio::test]
async fn column_choice_registers_scalar_and_aggregate_functions_separately() {
    let ctx = SessionContext::new();

    let scalar_udfs = column_choice_udfs();
    let scalar_names = scalar_udfs.iter().map(|udf| udf.name()).collect::<Vec<_>>();
    assert_eq!(scalar_names, vec!["randgen_column_choice"]);
    assert!(!scalar_names.iter().any(|name| name.contains("_cache_")));
    for udf in scalar_udfs {
        ctx.register_udf(udf);
    }

    let aggregate_udfs = column_choice_udafs();
    let aggregate_names = aggregate_udfs
        .iter()
        .map(|udaf| udaf.name())
        .collect::<Vec<_>>();
    assert_eq!(aggregate_names, vec!["randgen_roaring_agg"]);
    for udaf in aggregate_udfs {
        ctx.register_udaf(udaf);
    }

    assert_eq!(roaring_agg_udaf().name(), "randgen_roaring_agg");
    assert_eq!(
        all_udafs()
            .iter()
            .map(|udaf| udaf.name())
            .collect::<Vec<_>>(),
        vec!["randgen_roaring_agg"]
    );

    let all_scalar_udfs = all_udfs();
    let all_scalar_names = all_scalar_udfs
        .iter()
        .map(|udf| udf.name())
        .collect::<Vec<_>>();
    assert!(all_scalar_names.contains(&"randgen_column_choice"));
    assert!(!all_scalar_names.iter().any(|name| name.contains("_cache_")));
}
