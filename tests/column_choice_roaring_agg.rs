#![cfg(any(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, BinaryArray, LargeBinaryArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion_randgen::{column_choice_udafs, column_choice_udfs};
use datafusion_roaring::decode_bitmap;
use roaring::RoaringTreemap;

fn context_with_column_choice() -> SessionContext {
    let ctx = SessionContext::new();
    for udf in column_choice_udfs() {
        ctx.register_udf(udf);
    }
    for udaf in column_choice_udafs() {
        ctx.register_udaf(udaf);
    }
    ctx
}

fn register_u32_table(ctx: &SessionContext, name: &str, values: UInt32Array) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

fn register_u64_table(ctx: &SessionContext, name: &str, values: UInt64Array) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

async fn collect(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

async fn collect_error(ctx: &SessionContext, sql: &str) -> String {
    match ctx.sql(sql).await {
        Ok(df) => match df.collect().await {
            Ok(_) => String::new(),
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    }
}

fn first_binary_value(batches: &[RecordBatch]) -> Vec<u8> {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
        .value(0)
        .to_vec()
}

fn first_large_binary_value(batches: &[RecordBatch]) -> Vec<u8> {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap()
        .value(0)
        .to_vec()
}

fn u32_values(batches: &[RecordBatch]) -> Vec<Option<u32>> {
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            (0..column.len()).map(|row| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row))
                }
            })
        })
        .collect()
}

fn u64_values(batches: &[RecordBatch]) -> Vec<Option<u64>> {
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..column.len()).map(|row| {
                if column.is_null(row) {
                    None
                } else {
                    Some(column.value(row))
                }
            })
        })
        .collect()
}

#[tokio::test]
async fn roaring_agg_uint32_serializes_distinct_non_null_values() {
    let ctx = context_with_column_choice();
    register_u32_table(
        &ctx,
        "source",
        UInt32Array::from(vec![Some(7), Some(7), None, Some(9)]),
    );

    let batches = collect(&ctx, "SELECT randgen_roaring_agg(id) FROM source").await;
    assert_eq!(batches[0].column(0).data_type(), &DataType::Binary);
    let bitmap = decode_bitmap(first_binary_value(&batches).as_slice()).unwrap();

    assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![7, 9]);
}

#[tokio::test]
async fn roaring_agg_uint64_serializes_distinct_non_null_values() {
    let ctx = context_with_column_choice();
    register_u64_table(
        &ctx,
        "source",
        UInt64Array::from(vec![
            Some(1),
            Some(u64::from(u32::MAX) + 5),
            Some(u64::from(u32::MAX) + 5),
            None,
        ]),
    );

    let batches = collect(&ctx, "SELECT randgen_roaring_agg(id) FROM source").await;
    assert_eq!(batches[0].column(0).data_type(), &DataType::LargeBinary);
    let treemap =
        RoaringTreemap::deserialize_from(first_large_binary_value(&batches).as_slice()).unwrap();

    assert_eq!(
        treemap.iter().collect::<Vec<_>>(),
        vec![1, u64::from(u32::MAX) + 5]
    );
}

#[tokio::test]
async fn roaring_agg_all_null_group_serializes_empty_set() {
    let ctx = context_with_column_choice();
    register_u32_table(
        &ctx,
        "source",
        UInt32Array::from(vec![None::<u32>, None::<u32>]),
    );

    let batches = collect(&ctx, "SELECT randgen_roaring_agg(id) FROM source").await;
    let bitmap = decode_bitmap(first_binary_value(&batches).as_slice()).unwrap();

    assert!(bitmap.is_empty());
}

#[tokio::test]
async fn column_choice_samples_uint32_aggregate_values() {
    let ctx = context_with_column_choice();
    register_u32_table(
        &ctx,
        "source",
        UInt32Array::from(vec![Some(3), Some(5), Some(5), None]),
    );

    let batches = collect(
        &ctx,
        "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
         SELECT randgen_column_choice(ids) FROM choices, generate_series(1, 128)",
    )
    .await;

    let allowed = HashSet::from([3, 5]);
    assert!(
        u32_values(&batches)
            .into_iter()
            .all(|value| value.is_some_and(|value| allowed.contains(&value)))
    );
}

#[tokio::test]
async fn column_choice_samples_datafusion_roaring_values() {
    let ctx = SessionContext::new();
    ctx.register_udf(datafusion_randgen::column_choice_udf());
    ctx.register_udaf(datafusion_roaring::roaring_agg_udaf());
    register_u32_table(
        &ctx,
        "source",
        UInt32Array::from(vec![Some(11), Some(13), None]),
    );

    let batches = collect(
        &ctx,
        "WITH choices AS (SELECT roaring_agg(id) AS ids FROM source) \
         SELECT randgen_column_choice(ids) FROM choices, generate_series(1, 128)",
    )
    .await;

    let allowed = HashSet::from([11, 13]);
    assert!(
        u32_values(&batches)
            .into_iter()
            .all(|value| value.is_some_and(|value| allowed.contains(&value)))
    );
}

#[tokio::test]
async fn column_choice_samples_uint64_aggregate_values() {
    let ctx = context_with_column_choice();
    let high = u64::from(u32::MAX) + 11;
    register_u64_table(
        &ctx,
        "source",
        UInt64Array::from(vec![Some(2), Some(high), Some(high), None]),
    );

    let batches = collect(
        &ctx,
        "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
         SELECT randgen_column_choice(ids) FROM choices, generate_series(1, 128)",
    )
    .await;

    let allowed = HashSet::from([2, high]);
    assert!(
        u64_values(&batches)
            .into_iter()
            .all(|value| value.is_some_and(|value| allowed.contains(&value)))
    );
}

#[tokio::test]
async fn column_choice_null_probability_can_null_aggregate_values() {
    let ctx = context_with_column_choice();
    register_u32_table(&ctx, "source", UInt32Array::from(vec![Some(3), Some(5)]));

    let batches = collect(
        &ctx,
        "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
         SELECT randgen_column_choice(ids, 1.0) FROM choices, generate_series(1, 8)",
    )
    .await;

    assert!(
        u32_values(&batches)
            .into_iter()
            .all(|value| value.is_none())
    );
}

#[tokio::test]
async fn column_choice_errors_for_empty_roaring_set() {
    let ctx = context_with_column_choice();
    register_u32_table(
        &ctx,
        "source",
        UInt32Array::from(vec![None::<u32>, None::<u32>]),
    );

    let error = collect_error(
        &ctx,
        "WITH choices AS (SELECT randgen_roaring_agg(id) AS ids FROM source) \
         SELECT randgen_column_choice(ids) FROM choices, generate_series(1, 1)",
    )
    .await;

    assert!(error.contains("requires at least one non-null source value"));
}

#[tokio::test]
async fn column_choice_rejects_old_file_path_call_shape() {
    let ctx = context_with_column_choice();

    let error = collect_error(
        &ctx,
        "SELECT randgen_column_choice('/tmp/source.parquet', 'id') FROM generate_series(1, 1)",
    )
    .await;

    assert!(error.contains("expects Binary or LargeBinary"));
}
