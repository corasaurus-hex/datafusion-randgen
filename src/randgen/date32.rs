//! Date32 generator UDF.
//!
//! `randgen_date32(min, max[, null_probability])` samples a date from the
//! inclusive day range `min..=max`. Null bounds produce null output for that
//! row. Non-null bounds must satisfy `min <= max`. The optional null
//! probability must be finite and within `0.0..=1.0`.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Date32Type;
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

use crate::randgen::utils::{
    NullProbability, optional_args, primitive_range_array, primitive_range_scalar_array,
    two_array_args,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_date32(min, max[, null_probability])`.
pub struct Date32 {
    signature: &'static Signature,
}

static DATE32_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![DataType::Date32, DataType::Date32]),
            TypeSignature::Exact(vec![DataType::Date32, DataType::Date32, DataType::Float64]),
        ],
        Volatility::Volatile,
    )
});

impl Date32 {
    /// Creates the `randgen_date32` implementation.
    pub fn new() -> Self {
        Self {
            signature: &DATE32_SIGNATURE,
        }
    }
}

impl Default for Date32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Date32 {
    fn invoke_scalar_args(
        &self,
        min: Option<i32>,
        max: Option<i32>,
        number_rows: usize,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(Arc::new(
            primitive_range_scalar_array::<Date32Type>(
                min,
                max,
                number_rows,
                self.name(),
                null_probability,
            )?,
        )))
    }
}

impl ScalarUDFImpl for Date32 {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_date32"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Date32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let ([min, max], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Date32(min)),
            ColumnarValue::Scalar(ScalarValue::Date32(max)),
        ) = (&min, &max)
        {
            return self.invoke_scalar_args(*min, *max, number_rows, &null_probability);
        }

        let (min_array, max_array) = two_array_args(
            vec![min, max],
            (DataType::Date32, "Date32 arguments"),
            (DataType::Date32, "Date32 arguments"),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<Date32Type>();
        let max_values = max_array.as_primitive::<Date32Type>();

        Ok(ColumnarValue::Array(Arc::new(primitive_range_array(
            min_values,
            max_values,
            number_rows,
            self.name(),
            &null_probability,
        )?)))
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_values};

    use super::*;

    #[tokio::test]
    async fn date32_values_stay_in_range() {
        let values = query_to_values::<Date32Type>(
            ScalarUDF::from(Date32::new()),
            "SELECT randgen_date32(to_date('2024-01-01'), to_date('2024-01-31')) FROM generate_series(1, 1000)",
            DataType::Date32,
        )
        .await;
        assert!(
            values
                .iter()
                .all(|value| value.is_some_and(|value| (19723..=19753).contains(&value)))
        );
    }

    #[tokio::test]
    async fn date32_equal_bounds_are_deterministic() {
        let values = query_to_values::<Date32Type>(
            ScalarUDF::from(Date32::new()),
            "SELECT randgen_date32(to_date('2024-01-15'), to_date('2024-01-15')) FROM generate_series(1, 100)",
            DataType::Date32,
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(19737)));
    }

    #[tokio::test]
    async fn date32_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(Date32::new()),
            "SELECT randgen_date32(to_date('2024-02-01'), to_date('2024-01-01')) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn date32_array_bounds_propagate_nulls() {
        let values = query_to_values::<Date32Type>(
            ScalarUDF::from(Date32::new()),
            "SELECT randgen_date32(min_date, max_date) FROM (VALUES (to_date('2024-01-15'), to_date('2024-01-15')), (CAST(NULL AS DATE), to_date('2024-01-31')), (to_date('2024-01-01'), CAST(NULL AS DATE))) AS t(min_date, max_date)",
            DataType::Date32,
        )
        .await;

        assert_eq!(values, vec![Some(19737), None, None]);
    }

    #[tokio::test]
    async fn date32_array_bounds_without_nulls_use_column_values() {
        let values = query_to_values::<Date32Type>(
            ScalarUDF::from(Date32::new()),
            "SELECT randgen_date32(min_date, max_date) FROM (VALUES (to_date('2024-01-15'), to_date('2024-01-15')), (to_date('2024-01-16'), to_date('2024-01-16'))) AS t(min_date, max_date)",
            DataType::Date32,
        )
        .await;

        assert_eq!(values, vec![Some(19737), Some(19738)]);
    }
}
