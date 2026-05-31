use std::any::Any;
use std::sync::LazyLock;

use datafusion::arrow::array::{Array, AsArray, Float64Array};
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::common::{exec_err, internal_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::RngExt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Float64Uniform {
    signature: &'static Signature,
}

static FLOAT64_UNIFORM_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![DataType::Float64, DataType::Float64],
        Volatility::Volatile,
    )
});

impl Float64Uniform {
    pub fn new() -> Self {
        Self {
            signature: &FLOAT64_UNIFORM_SIGNATURE,
        }
    }
}

impl Default for Float64Uniform {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for Float64Uniform {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "randgen_float64_uniform"
    }
    fn signature(&self) -> &Signature {
        self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [min, max]: [ColumnarValue; 2] = match args.try_into() {
            Ok(args) => args,
            Err(_) => return internal_err!("{} expects exactly two arguments", self.name()),
        };

        if min.data_type() != DataType::Float64 || max.data_type() != DataType::Float64 {
            return internal_err!("{} expects Float64 arguments", self.name());
        }

        let min_array = min.into_array_of_size(number_rows)?;
        let max_array = max.into_array_of_size(number_rows)?;
        let min_values = min_array.as_primitive::<Float64Type>();
        let max_values = max_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if min_values.is_null(row) || max_values.is_null(row) {
                values.push(None);
                continue;
            }

            let min = min_values.value(row);
            let max = max_values.value(row);
            if !min.is_finite() || !max.is_finite() {
                return exec_err!("{} requires finite bounds", self.name());
            }
            if min > max {
                return exec_err!(
                    "{} requires min <= max, got min {min} and max {max}",
                    self.name()
                );
            }
            if !(max - min).is_finite() {
                return exec_err!(
                    "{} requires finite distance between bounds, got min {min} and max {max}",
                    self.name()
                );
            }

            let value = if min == max {
                min
            } else {
                rng.random_range(min..=max)
            };
            values.push(Some(value));
        }

        Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::datatypes::{DataType, Float64Type},
        logical_expr::ScalarUDF,
    };

    use crate::randgen::test_helpers::querying::{query_result, query_to_values};

    use super::*;

    #[tokio::test]
    async fn float64_uniform_values_stay_in_range() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(1.5, 2.5) FROM generate_series(1, 1000)",
            DataType::Float64,
        )
        .await;
        assert!(
            values
                .iter()
                .all(|value| value.is_some_and(|value| (1.5..=2.5).contains(&value)))
        );
    }

    #[tokio::test]
    async fn float64_uniform_equal_bounds_are_deterministic() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(3.5, 3.5) FROM generate_series(1, 100)",
            DataType::Float64,
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(3.5)));
    }

    #[tokio::test]
    async fn float64_uniform_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(10.0, 1.0) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn float64_uniform_overflowing_span_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(-1.7976931348623157e308, 1.7976931348623157e308) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }
}
