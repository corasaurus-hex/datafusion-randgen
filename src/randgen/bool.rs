//! Boolean random generator UDF implementation.

use std::any::Any;
use std::sync::LazyLock;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, BooleanArray};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;
use std::sync::Arc;

use crate::randgen::utils::one_array_arg;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Implements `randgen_bool(probability)`.
pub struct Bool {
    signature: &'static Signature,
}

static BOOL_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::exact(vec![DataType::Float64], Volatility::Volatile));

impl Bool {
    /// Creates a `randgen_bool` UDF implementation.
    pub fn new() -> Self {
        Self {
            signature: &BOOL_SIGNATURE,
        }
    }
}

impl Default for Bool {
    fn default() -> Self {
        Self::new()
    }
}

impl Bool {
    fn invoke_scalar_args(
        &self,
        probability: Option<f64>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);

        for _ in 0..number_rows {
            let Some(probability) = probability else {
                values.push(None);
                continue;
            };

            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return exec_err!(
                    "{} requires probability between 0.0 and 1.0 inclusive",
                    self.name()
                );
            }

            values.push(Some(rng.random_bool(probability)));
        }

        Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(values))))
    }
}

impl ScalarUDFImpl for Bool {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_bool"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [probability] = crate::randgen::utils::exact_args(args, self.name())?;
        if let ColumnarValue::Scalar(ScalarValue::Float64(probability)) = &probability {
            return self.invoke_scalar_args(*probability, number_rows);
        }

        let probability_array = one_array_arg(
            vec![probability],
            (DataType::Float64, "a Float64 probability"),
            number_rows,
            self.name(),
        )?;
        let probabilities = probability_array.as_primitive::<Float64Type>();
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);

        for row in 0..number_rows {
            if probabilities.is_null(row) {
                values.push(None);
                continue;
            }

            let probability = probabilities.value(row);
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return exec_err!(
                    "{} requires probability between 0.0 and 1.0 inclusive",
                    self.name()
                );
            }

            values.push(Some(rng.random_bool(probability)));
        }

        Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_bool_values};

    use super::*;

    #[tokio::test]
    async fn bool_zero_probability_is_false() {
        let values = query_to_bool_values(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(0.0) FROM generate_series(1, 100)",
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(false)));
    }

    #[tokio::test]
    async fn bool_one_probability_is_true() {
        let values = query_to_bool_values(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(1.0) FROM generate_series(1, 100)",
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(true)));
    }

    #[tokio::test]
    async fn bool_half_probability_smoke() {
        let values = query_to_bool_values(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(0.5) FROM generate_series(1, 1000)",
        )
        .await;
        assert!(values.contains(&Some(false)));
        assert!(values.contains(&Some(true)));
    }

    #[tokio::test]
    async fn bool_invalid_probability_errors() {
        let result = query_result(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(1.1) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }
}
