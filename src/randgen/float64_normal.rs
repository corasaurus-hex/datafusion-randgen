//! Float64 normal-distribution generator UDF.
//!
//! `randgen_float64_normal(mean, stddev[, null_probability])` samples from a
//! normal distribution. Both required arguments must be finite, and `stddev`
//! must be greater than zero. A null required argument produces null output for
//! that row. The optional null probability must be finite and within
//! `0.0..=1.0`.

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
use rand_distr::{Distribution, Normal};
use std::sync::Arc;

use crate::randgen::utils::{NullProbability, optional_args, two_array_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_float64_normal(mean, stddev[, null_probability])`.
pub struct Float64Normal {
    signature: &'static Signature,
}

static FLOAT64_NORMAL_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
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

struct FloatNormalSampler {
    normal: Normal<f64>,
}

impl FloatNormalSampler {
    fn new(mean: f64, stddev: f64, name: &str) -> Result<Self> {
        if !mean.is_finite() || !stddev.is_finite() {
            return exec_err!("{name} requires finite mean and stddev");
        }
        if stddev <= 0.0 {
            return exec_err!("{name} requires stddev > 0");
        }

        let normal = Normal::new(mean, stddev).map_err(|error| {
            datafusion_common::DataFusionError::Execution(format!(
                "{name} invalid normal distribution parameters: {error}"
            ))
        })?;
        Ok(Self { normal })
    }

    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: Rng + ?Sized,
    {
        self.normal.sample(rng)
    }
}

impl Float64Normal {
    /// Creates the `randgen_float64_normal` implementation.
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

impl Float64Normal {
    fn invoke_scalar_args(
        &self,
        mean: Option<f64>,
        stddev: Option<f64>,
        number_rows: usize,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(mean), Some(stddev)) = (mean, stddev) {
            if number_rows == 0 {
                return Ok(ColumnarValue::Array(Arc::new(Float64Array::from(
                    Vec::<f64>::new(),
                ))));
            }

            let sampler = FloatNormalSampler::new(mean, stddev, self.name())?;
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                if null_probability.is_null(row, &mut rng, self.name())? {
                    values.push(None);
                } else {
                    values.push(Some(sampler.sample(&mut rng)));
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

impl ScalarUDFImpl for Float64Normal {
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
        let ([mean, stddev], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Float64(mean)),
            ColumnarValue::Scalar(ScalarValue::Float64(stddev)),
        ) = (&mean, &stddev)
        {
            return self.invoke_scalar_args(*mean, *stddev, number_rows, &null_probability);
        }

        let (mean_array, stddev_array) = two_array_args(
            vec![mean, stddev],
            (DataType::Float64, "Float64 arguments"),
            (DataType::Float64, "Float64 arguments"),
            number_rows,
            self.name(),
        )?;
        let mean_values = mean_array.as_primitive::<Float64Type>();
        let stddev_values = stddev_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        if mean_values.null_count() == 0
            && stddev_values.null_count() == 0
            && matches!(null_probability, NullProbability::None)
        {
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                let mean = mean_values.value(row);
                let stddev = stddev_values.value(row);
                let sampler = FloatNormalSampler::new(mean, stddev, self.name())?;
                values.push(sampler.sample(&mut rng));
            }

            return Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if mean_values.is_null(row) || stddev_values.is_null(row) {
                values.push(None);
                continue;
            }

            let mean = mean_values.value(row);
            let stddev = stddev_values.value(row);
            let sampler = FloatNormalSampler::new(mean, stddev, self.name())?;
            if null_probability.is_null(row, &mut rng, self.name())? {
                values.push(None);
            } else {
                values.push(Some(sampler.sample(&mut rng)));
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
    async fn float64_normal_accepts_integer_literals() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(10, 2) FROM generate_series(1, 100)",
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

    #[tokio::test]
    async fn float64_normal_negative_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(10.0, -1.0) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn float64_normal_nonfinite_mean_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(CAST('NaN' AS DOUBLE), 1.0) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn float64_normal_nonfinite_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(10.0, CAST('Infinity' AS DOUBLE)) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn float64_normal_array_args_propagate_nulls() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(mean, stddev) FROM (VALUES (10.0, 1.0), (CAST(NULL AS DOUBLE), 1.0), (10.0, CAST(NULL AS DOUBLE))) AS t(mean, stddev)",
            DataType::Float64,
        )
        .await;

        assert!(values[0].is_some_and(f64::is_finite));
        assert_eq!(values[1..], [None, None]);
    }

    #[tokio::test]
    async fn float64_normal_array_args_without_nulls_outputs_finite_values() {
        let values = query_to_values::<Float64Type>(
            ScalarUDF::from(Float64Normal::new()),
            "SELECT randgen_float64_normal(mean, stddev) FROM (VALUES (10.0, 1.0), (20.0, 2.0)) AS t(mean, stddev)",
            DataType::Float64,
        )
        .await;

        assert!(values.iter().all(|value| value.is_some_and(f64::is_finite)));
    }
}
