use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Date32Type;
use arrow_array::{Array, Date32Array};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;

use crate::randgen::utils::two_array_args;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Date32 {
    signature: &'static Signature,
}

static DATE32_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![DataType::Date32, DataType::Date32],
        Volatility::Volatile,
    )
});

impl Date32 {
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
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            let (Some(min), Some(max)) = (min, max) else {
                values.push(None);
                continue;
            };

            if min > max {
                return exec_err!(
                    "{} requires min <= max, got min {min} and max {max}",
                    self.name()
                );
            }

            values.push(Some(rng.random_range(min..=max)));
        }

        Ok(ColumnarValue::Array(Arc::new(Date32Array::from(values))))
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
        let [min, max] = crate::randgen::utils::exact_args(args, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Date32(min)),
            ColumnarValue::Scalar(ScalarValue::Date32(max)),
        ) = (&min, &max)
        {
            return self.invoke_scalar_args(*min, *max, number_rows);
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

        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if min_values.is_null(row) || max_values.is_null(row) {
                values.push(None);
                continue;
            }

            let min = min_values.value(row);
            let max = max_values.value(row);
            if min > max {
                return exec_err!(
                    "{} requires min <= max, got min {min} and max {max}",
                    self.name()
                );
            }

            values.push(Some(rng.random_range(min..=max)));
        }

        Ok(ColumnarValue::Array(Arc::new(Date32Array::from(values))))
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
}
