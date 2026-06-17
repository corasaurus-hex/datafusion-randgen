//! Boolean generator UDF.
//!
//! `randgen_bool(probability[, null_probability])` returns `true` with the
//! supplied probability. The optional second probability controls null output.
//! Both probabilities must be finite and within `0.0..=1.0`; a null probability
//! value produces null output for that row.

use std::any::Any;
use std::sync::LazyLock;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, BooleanArray};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use rand::Rng;
use std::sync::Arc;

use crate::randgen::utils::{NullProbability, one_array_arg, optional_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_bool(probability[, null_probability])`.
pub struct Bool {
    signature: &'static Signature,
}

static BOOL_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![DataType::Float64]),
            TypeSignature::Exact(vec![DataType::Float64, DataType::Float64]),
        ],
        Volatility::Volatile,
    )
});

impl Bool {
    /// Creates the `randgen_bool` implementation.
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
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);

        for row in 0..number_rows {
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

            if null_probability.is_null(row, &mut rng, self.name())? {
                values.push(None);
            } else {
                values.push(Some(rng.random_bool(probability)));
            }
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
        let ([probability], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let ColumnarValue::Scalar(ScalarValue::Float64(probability)) = &probability {
            return self.invoke_scalar_args(*probability, number_rows, &null_probability);
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

            if null_probability.is_null(row, &mut rng, self.name())? {
                values.push(None);
            } else {
                values.push(Some(rng.random_bool(probability)));
            }
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

    #[tokio::test]
    async fn bool_negative_probability_errors() {
        let result = query_result(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(-0.1) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn bool_array_probability_propagates_nulls() {
        let values = query_to_bool_values(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(p) FROM (VALUES (0.0), (1.0), (CAST(NULL AS DOUBLE))) AS t(p)",
        )
        .await;

        assert_eq!(values, vec![Some(false), Some(true), None]);
    }

    #[tokio::test]
    async fn bool_array_invalid_probability_errors() {
        let result = query_result(
            ScalarUDF::from(Bool::new()),
            "SELECT randgen_bool(p) FROM (VALUES (0.5), (1.1)) AS t(p)",
        )
        .await;

        assert!(result.is_err());
    }
}
