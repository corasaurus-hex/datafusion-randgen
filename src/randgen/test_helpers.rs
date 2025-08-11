#[cfg(test)]
pub(crate) mod querying {
    use datafusion::arrow::array::{ArrowPrimitiveType, PrimitiveArray};
    use datafusion::arrow::datatypes::DataType;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::SessionContext;
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
        assert!(values.len() > 0);
        values
    }
}
