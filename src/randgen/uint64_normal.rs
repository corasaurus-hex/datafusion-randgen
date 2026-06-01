//! UInt64 normal-distribution random generator.
//!
//! `randgen_uint64_normal(mean, stddev)` samples in floating-point normal
//! space, rounds to the nearest integer, and clamps to the `UInt64` range. The
//! mean is `UInt64`; `stddev` is `Float64` and must be finite and greater than
//! zero. Null input yields null output for that row.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt64Type};
use arrow_array::{Array, UInt64Array};
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;
use rand_distr::Normal;

use crate::randgen::utils::two_array_args;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_uint64_normal(mean, stddev)`.
pub struct UInt64Normal {
    signature: &'static Signature,
}

static UINT64_NORMAL_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![DataType::UInt64, DataType::Float64],
        Volatility::Volatile,
    )
});

fn normal_distribution(mean: u64, stddev: f64, name: &str) -> Result<Normal<f64>> {
    if !stddev.is_finite() {
        return exec_err!("{name} requires finite stddev");
    }
    if stddev <= 0.0 {
        return exec_err!("{name} requires stddev > 0");
    }

    Normal::new(mean as f64, stddev).map_err(|error| {
        DataFusionError::Execution(format!(
            "{name} invalid normal distribution parameters: {error}"
        ))
    })
}

fn sample_uint64<R: Rng + ?Sized>(rng: &mut R, normal: Normal<f64>) -> u64 {
    let value: f64 = rng.sample(normal);
    let value = value.round();
    if value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

impl UInt64Normal {
    /// Creates the `randgen_uint64_normal` implementation.
    pub fn new() -> Self {
        Self {
            signature: &UINT64_NORMAL_SIGNATURE,
        }
    }
}

impl Default for UInt64Normal {
    fn default() -> Self {
        Self::new()
    }
}

impl UInt64Normal {
    fn invoke_scalar_args(
        &self,
        mean: Option<u64>,
        stddev: Option<f64>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(mean), Some(stddev)) = (mean, stddev) {
            let normal = normal_distribution(mean, stddev, self.name())?;
            let mut values = Vec::with_capacity(number_rows);
            for _ in 0..number_rows {
                values.push(sample_uint64(&mut rng, normal));
            }

            return Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for _ in 0..number_rows {
            values.push(None);
        }

        Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(values))))
    }
}

impl ScalarUDFImpl for UInt64Normal {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "randgen_uint64_normal"
    }
    fn signature(&self) -> &Signature {
        self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [mean, stddev] = crate::randgen::utils::exact_args(args, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::UInt64(mean)),
            ColumnarValue::Scalar(ScalarValue::Float64(stddev)),
        ) = (&mean, &stddev)
        {
            return self.invoke_scalar_args(*mean, *stddev, number_rows);
        }

        let (mean_array, stddev_array) = two_array_args(
            vec![mean, stddev],
            (DataType::UInt64, "UInt64, Float64 arguments"),
            (DataType::Float64, "UInt64, Float64 arguments"),
            number_rows,
            self.name(),
        )?;
        let mean_values = mean_array.as_primitive::<UInt64Type>();
        let stddev_values = stddev_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        if mean_values.null_count() == 0 && stddev_values.null_count() == 0 {
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                let normal = normal_distribution(
                    mean_values.value(row),
                    stddev_values.value(row),
                    self.name(),
                )?;
                values.push(sample_uint64(&mut rng, normal));
            }

            return Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if mean_values.is_null(row) || stddev_values.is_null(row) {
                values.push(None);
                continue;
            }

            let normal = normal_distribution(
                mean_values.value(row),
                stddev_values.value(row),
                self.name(),
            )?;
            values.push(Some(sample_uint64(&mut rng, normal)));
        }

        Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::UInt64Type;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_values};

    use super::*;

    #[tokio::test]
    async fn uint64_normal_outputs_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(arrow_cast(10, 'UInt64'), 2.0) FROM generate_series(1, 100)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 100);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn uint64_normal_invalid_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(arrow_cast(10, 'UInt64'), 0.0) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_array_args_propagate_nulls() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(mean_value, stddev) FROM (VALUES (arrow_cast(10, 'UInt64'), 1.0), (arrow_cast(NULL, 'UInt64'), 1.0), (arrow_cast(10, 'UInt64'), CAST(NULL AS DOUBLE))) AS t(mean_value, stddev)",
            DataType::UInt64,
        )
        .await;

        assert!(values[0].is_some());
        assert_eq!(values[1..], [None, None]);
    }

    #[tokio::test]
    async fn uint64_normal_array_args_without_nulls_output_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(mean_value, stddev) FROM (VALUES (arrow_cast(10, 'UInt64'), 1.0), (arrow_cast(20, 'UInt64'), 2.0)) AS t(mean_value, stddev)",
            DataType::UInt64,
        )
        .await;

        assert!(values.iter().all(Option::is_some));
    }
}
