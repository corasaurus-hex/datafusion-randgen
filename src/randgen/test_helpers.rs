#[cfg(test)]
pub(crate) mod querying {
    use arrow_array::types::ArrowPrimitiveType;
    use arrow_array::{PrimitiveArray, RecordBatch};
    use arrow_schema::DataType;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::ScalarUDF;

    pub(crate) async fn query_result(
        udf: ScalarUDF,
        query: &str,
    ) -> datafusion_common::Result<Vec<RecordBatch>> {
        let ctx = SessionContext::new();
        ctx.register_udf(udf);
        let df = ctx.sql(query).await?;
        df.collect().await
    }

    pub(crate) async fn query_to_string_values(udf: ScalarUDF, query: &str) -> Vec<Option<String>> {
        let batches = query_result(udf, query).await.unwrap();
        let values = batches
            .into_iter()
            .flat_map(|batch| {
                let col = batch.column(0);
                assert_eq!(col.data_type(), &DataType::Utf8);
                col.as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap()
                    .iter()
                    .map(|value| value.map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(!values.is_empty());
        values
    }

    pub(crate) async fn query_to_bool_values(udf: ScalarUDF, query: &str) -> Vec<Option<bool>> {
        let batches = query_result(udf, query).await.unwrap();
        let values = batches
            .into_iter()
            .flat_map(|batch| {
                let col = batch.column(0);
                assert_eq!(col.data_type(), &DataType::Boolean);
                col.as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(!values.is_empty());
        values
    }

    pub(crate) async fn query_to_values<T>(
        udf: ScalarUDF,
        query: &str,
        data_type: DataType,
    ) -> Vec<Option<T::Native>>
    where
        T: ArrowPrimitiveType,
    {
        let ctx = SessionContext::new();
        ctx.register_udf(udf);
        let df = ctx.sql(query).await.unwrap();
        let batches = df.collect().await.unwrap();
        let values = batches
            .into_iter()
            .flat_map(|batch| {
                let col = batch.column(0);
                assert_eq!(col.data_type(), &data_type);
                col.as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(!values.is_empty());
        values
    }
}
