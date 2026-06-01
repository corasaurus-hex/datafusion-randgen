//! Int64 uniform random generator.
//!
//! `randgen_int64_uniform(min, max)` samples from the inclusive integer range
//! `min..=max`. Null bounds produce null output for that row. Non-null bounds
//! must satisfy `min <= max`.

use std::any::Any;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;
use std::sync::{Arc, LazyLock};

use crate::randgen::utils::two_array_args;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_int64_uniform(min, max)`.
pub struct Int64Uniform {
    signature: &'static Signature,
}

static INT64_UNIFORM_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(vec![DataType::Int64, DataType::Int64], Volatility::Volatile)
});

impl Int64Uniform {
    /// Creates the `randgen_int64_uniform` implementation.
    pub fn new() -> Self {
        Self {
            signature: &INT64_UNIFORM_SIGNATURE,
        }
    }
}

impl Default for Int64Uniform {
    fn default() -> Self {
        Self::new()
    }
}

impl Int64Uniform {
    fn invoke_scalar_args(
        &self,
        min: Option<i64>,
        max: Option<i64>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(min), Some(max)) = (min, max) {
            let mut values = Vec::with_capacity(number_rows);
            for _ in 0..number_rows {
                if min > max {
                    return exec_err!(
                        "{} requires min <= max, got min {min} and max {max}",
                        self.name()
                    );
                }

                values.push(rng.random_range(min..=max));
            }

            return Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            values.push(None);
        }

        Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))))
    }
}

impl ScalarUDFImpl for Int64Uniform {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_int64_uniform"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [min, max] = crate::randgen::utils::exact_args(args, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Int64(min)),
            ColumnarValue::Scalar(ScalarValue::Int64(max)),
        ) = (&min, &max)
        {
            return self.invoke_scalar_args(*min, *max, number_rows);
        }

        let (min_array, max_array) = two_array_args(
            vec![min, max],
            (DataType::Int64, "Int64 arguments"),
            (DataType::Int64, "Int64 arguments"),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<Int64Type>();
        let max_values = max_array.as_primitive::<Int64Type>();

        let mut rng = rand::rng();
        if min_values.null_count() == 0 && max_values.null_count() == 0 {
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                let min = min_values.value(row);
                let max = max_values.value(row);
                if min > max {
                    return exec_err!(
                        "{} requires min <= max, got min {min} and max {max}",
                        self.name()
                    );
                }

                values.push(rng.random_range(min..=max));
            }

            return Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))));
        }

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

        Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Int64Type;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::query_to_values;

    use super::*;

    #[tokio::test]
    async fn int64_uniform_min_const_max_const() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, 10) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await
        {
            assert!(value.unwrap() >= 1);
            assert!(value.unwrap() <= 10);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_const_max_const_varies_by_row() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, 2) as x from generate_series(1, 1000)",
            DataType::Int64,
        )
        .await;
        assert!(values.contains(&Some(1)));
        assert!(values.contains(&Some(2)));
    }

    #[tokio::test]
    async fn int64_uniform_min_equals_max_is_deterministic() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(7, 7) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(7)));
    }

    #[tokio::test]
    async fn int64_uniform_invalid_range_errors() {
        let result = crate::randgen::test_helpers::querying::query_result(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(10, 1) as x from generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn int64_uniform_min_array_max_const() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(y, 20) as x from (select randgen_int64_uniform(1, 10) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value.unwrap() >= 1);
            assert!(value.unwrap() <= 20);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_const_max_array() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, y) as x from (select randgen_int64_uniform(11, 20) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value.unwrap() >= 1);
            assert!(value.unwrap() <= 20);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_array_max_array() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(x, y) as x from (select randgen_int64_uniform(1, 10) as x, randgen_int64_uniform(11, 20) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value.unwrap() >= 1);
            assert!(value.unwrap() <= 20);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_const_max_null() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, null) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|v| v.is_none()));
    }

    #[tokio::test]
    async fn int64_uniform_min_array_max_null() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(x, null) as x from (select randgen_int64_uniform(1, 10) as x from generate_series(1, 100))",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|v| v.is_none()));
    }

    #[tokio::test]
    async fn int64_uniform_min_null_max_null() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(null, null) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|v| v.is_none()));
    }

    #[tokio::test]
    async fn int64_uniform_min_null_max_const() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(null, 10) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|v| v.is_none()));
    }

    #[tokio::test]
    async fn int64_uniform_min_null_max_array() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(null, y) as x from (select randgen_int64_uniform(1, 10) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await;
        assert!(values.iter().all(|v| v.is_none()));
    }
}
