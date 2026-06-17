//! Float64 uniform generator UDF.
//!
//! `randgen_float64_uniform(min, max[, null_probability])` samples from the
//! inclusive range `min..=max`. Bounds must be finite, the span must be finite,
//! and `min` must not exceed `max`. Null bounds produce null output for that
//! row. The optional null probability must be finite and within `0.0..=1.0`.

use std::any::Any;
use std::sync::LazyLock;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, Float64Array};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use rand::Rng;
use std::sync::Arc;

use crate::randgen::utils::{NullProbability, optional_args, two_array_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_float64_uniform(min, max[, null_probability])`.
pub struct Float64Uniform {
    signature: &'static Signature,
}

static FLOAT64_UNIFORM_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![DataType::Float64, DataType::Float64]),
            TypeSignature::Exact(vec![
                DataType::Float64,
                DataType::Float64,
                DataType::Float64,
            ]),
        ],
        Volatility::Volatile,
    )
});

impl Float64Uniform {
    /// Creates the `randgen_float64_uniform` implementation.
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

impl Float64Uniform {
    fn invoke_scalar_args(
        &self,
        min: Option<f64>,
        max: Option<f64>,
        number_rows: usize,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(min), Some(max)) = (min, max) {
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

            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                let value = if min == max {
                    min
                } else {
                    rng.random_range(min..=max)
                };
                if null_probability.is_null(row, &mut rng, self.name())? {
                    values.push(None);
                } else {
                    values.push(Some(value));
                }
            }

            return Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            values.push(None);
        }

        Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))))
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
        let ([min, max], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Float64(min)),
            ColumnarValue::Scalar(ScalarValue::Float64(max)),
        ) = (&min, &max)
        {
            return self.invoke_scalar_args(*min, *max, number_rows, &null_probability);
        }

        let (min_array, max_array) = two_array_args(
            vec![min, max],
            (DataType::Float64, "Float64 arguments"),
            (DataType::Float64, "Float64 arguments"),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<Float64Type>();
        let max_values = max_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        if min_values.null_count() == 0
            && max_values.null_count() == 0
            && matches!(null_probability, NullProbability::None)
        {
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
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
                values.push(value);
            }

            return Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))));
        }

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
            if null_probability.is_null(row, &mut rng, self.name())? {
                values.push(None);
            } else {
                values.push(Some(value));
            }
        }

        Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Float64Type;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;

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
    async fn float64_uniform_accepts_integer_literals() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(1, 10) FROM generate_series(1, 100)",
            DataType::Float64,
        )
        .await;

        assert!(
            values
                .iter()
                .all(|value| value.is_some_and(|value| (1.0..=10.0).contains(&value)))
        );
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
    async fn float64_uniform_nonfinite_min_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(CAST('NaN' AS DOUBLE), 1.0) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn float64_uniform_nonfinite_max_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(0.0, CAST('Infinity' AS DOUBLE)) FROM generate_series(1, 10)",
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

    #[tokio::test]
    async fn float64_uniform_array_bounds_propagate_nulls() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(min_value, max_value) FROM (VALUES (3.5, 3.5), (CAST(NULL AS DOUBLE), 2.5), (1.0, CAST(NULL AS DOUBLE))) AS t(min_value, max_value)",
            DataType::Float64,
        )
        .await;

        assert_eq!(values, vec![Some(3.5), None, None]);
    }

    #[tokio::test]
    async fn float64_uniform_array_bounds_without_nulls_use_column_values() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Uniform::new()),
            "SELECT randgen_float64_uniform(min_value, max_value) FROM (VALUES (3.5, 3.5), (4.5, 4.5)) AS t(min_value, max_value)",
            DataType::Float64,
        )
        .await;

        assert_eq!(values, vec![Some(3.5), Some(4.5)]);
    }
}
