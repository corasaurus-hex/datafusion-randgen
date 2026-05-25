use std::any::Any;
use std::sync::LazyLock;

use datafusion::arrow::array::{Array, AsArray, Float64Array};
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::common::{exec_err, internal_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::Rng;
use rand_distr::Normal;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Float64Normal {
    signature: &'static Signature,
}

static FLOAT64_NORMAL_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![DataType::Float64, DataType::Float64],
        Volatility::Volatile,
    )
});

impl Float64Normal {
    pub fn new() -> Self {
        Self {
            signature: &FLOAT64_NORMAL_SIGNATURE,
        }
    }
}

impl Default for Float64Normal {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for Float64Normal {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "randgen_float64_normal"
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
        let [mean, stddev]: [ColumnarValue; 2] = match args.try_into() {
            Ok(args) => args,
            Err(_) => return internal_err!("{} expects exactly two arguments", self.name()),
        };

        if mean.data_type() != DataType::Float64 || stddev.data_type() != DataType::Float64 {
            return internal_err!("{} expects Float64 arguments", self.name());
        }

        let mean_array = mean.into_array_of_size(number_rows)?;
        let stddev_array = stddev.into_array_of_size(number_rows)?;
        let mean_values = mean_array.as_primitive::<Float64Type>();
        let stddev_values = stddev_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if mean_values.is_null(row) || stddev_values.is_null(row) {
                values.push(None);
                continue;
            }

            let mean = mean_values.value(row);
            let stddev = stddev_values.value(row);
            if !mean.is_finite() || !stddev.is_finite() {
                return exec_err!("{} requires finite mean and stddev", self.name());
            }
            if stddev <= 0.0 {
                return exec_err!("{} requires stddev > 0", self.name());
            }
            let normal = Normal::new(mean, stddev).map_err(|error| {
                datafusion::error::DataFusionError::Execution(format!(
                    "{} invalid normal distribution parameters: {error}",
                    self.name()
                ))
            })?;
            values.push(Some(rng.sample(normal)));
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
    async fn float64_normal_outputs_finite_values() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(10.0, 2.0) FROM generate_series(1, 1000)",
            DataType::Float64,
        )
        .await;
        assert!(values.iter().all(|value| value.is_some_and(f64::is_finite)));
    }

    #[tokio::test]
    async fn float64_normal_invalid_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(10.0, 0.0) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }
}
