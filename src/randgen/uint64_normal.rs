//! UInt64 normal-distribution random generator.
//!
//! `randgen_uint64_normal(min, max, mean, stddev)` samples an integer normal
//! distribution centered on `mean` and truncated to the inclusive `UInt64`
//! range `min..=max`. The mean may be outside the output range. A null argument
//! produces null output for that row. Integer arguments may be `UInt64` values
//! or nonnegative signed integer values.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt64Type};
use arrow_array::{Array, UInt64Array};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;

use crate::randgen::integer_normal::RoundedIntegerNormalSampler;
use crate::randgen::utils::{coerce_float64_argument, coerce_uint64_argument, four_array_args};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_uint64_normal(min, max, mean, stddev)`.
pub struct UInt64Normal {
    signature: &'static Signature,
}

static UINT64_NORMAL_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

fn sampler_for_range(
    min: u64,
    max: u64,
    mean: u64,
    stddev: f64,
    name: &str,
) -> Result<RoundedIntegerNormalSampler> {
    RoundedIntegerNormalSampler::new(min as i128, max as i128, mean as i128, stddev, name)
}

fn sample_uint64<R: Rng + ?Sized>(
    rng: &mut R,
    sampler: &RoundedIntegerNormalSampler,
    name: &str,
) -> Result<u64> {
    Ok(sampler.sample(rng, name)? as u64)
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
        min: Option<u64>,
        max: Option<u64>,
        mean: Option<u64>,
        stddev: Option<f64>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(min), Some(max), Some(mean), Some(stddev)) = (min, max, mean, stddev) {
            let sampler = sampler_for_range(min, max, mean, stddev, self.name())?;
            let mut values = Vec::with_capacity(number_rows);
            for _ in 0..number_rows {
                values.push(sample_uint64(&mut rng, &sampler, self.name())?);
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

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 4 {
            let argument = if arg_types.len() == 1 {
                "argument"
            } else {
                "arguments"
            };
            return exec_err!(
                "{} expects exactly 4 arguments, got {} {argument}",
                self.name(),
                arg_types.len()
            );
        }

        Ok(vec![
            coerce_uint64_argument(&arg_types[0], self.name())?,
            coerce_uint64_argument(&arg_types[1], self.name())?,
            coerce_uint64_argument(&arg_types[2], self.name())?,
            coerce_float64_argument(&arg_types[3], self.name())?,
        ])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [min, max, mean, stddev] = crate::randgen::utils::exact_args(args, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::UInt64(min)),
            ColumnarValue::Scalar(ScalarValue::UInt64(max)),
            ColumnarValue::Scalar(ScalarValue::UInt64(mean)),
            ColumnarValue::Scalar(ScalarValue::Float64(stddev)),
        ) = (&min, &max, &mean, &stddev)
        {
            return self.invoke_scalar_args(*min, *max, *mean, *stddev, number_rows);
        }

        let expected = "UInt64, UInt64, UInt64, Float64 arguments";
        let (min_array, max_array, mean_array, stddev_array) = four_array_args(
            vec![min, max, mean, stddev],
            (DataType::UInt64, expected),
            (DataType::UInt64, expected),
            (DataType::UInt64, expected),
            (DataType::Float64, expected),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<UInt64Type>();
        let max_values = max_array.as_primitive::<UInt64Type>();
        let mean_values = mean_array.as_primitive::<UInt64Type>();
        let stddev_values = stddev_array.as_primitive::<Float64Type>();

        let mut rng = rand::rng();
        if min_values.null_count() == 0
            && max_values.null_count() == 0
            && mean_values.null_count() == 0
            && stddev_values.null_count() == 0
        {
            let mut values = Vec::with_capacity(number_rows);
            for row in 0..number_rows {
                let min = min_values.value(row);
                let max = max_values.value(row);
                let mean = mean_values.value(row);
                let stddev = stddev_values.value(row);
                let sampler = sampler_for_range(min, max, mean, stddev, self.name())?;
                values.push(sample_uint64(&mut rng, &sampler, self.name())?);
            }

            return Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(values))));
        }

        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if min_values.is_null(row)
                || max_values.is_null(row)
                || mean_values.is_null(row)
                || stddev_values.is_null(row)
            {
                values.push(None);
                continue;
            }

            let min = min_values.value(row);
            let max = max_values.value(row);
            let mean = mean_values.value(row);
            let stddev = stddev_values.value(row);
            let sampler = sampler_for_range(min, max, mean, stddev, self.name())?;
            values.push(Some(sample_uint64(&mut rng, &sampler, self.name())?));
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
            "SELECT randgen_uint64_normal(0, 20, 10, 2.0) FROM generate_series(1, 100)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 100);
        assert!(values.iter().all(Option::is_some));
        assert!(
            values
                .iter()
                .flatten()
                .all(|value| (0..=20).contains(value))
        );
    }

    #[tokio::test]
    async fn uint64_normal_full_range_outputs_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 18446744073709551615, 9223372036854775807, 1.0) FROM generate_series(1, 10)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn uint64_normal_large_stddev_full_range_outputs_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 18446744073709551615, 9223372036854775807, 9007199254740992.0) FROM generate_series(1, 10)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn uint64_normal_huge_stddev_single_value_range_outputs_value() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 0, 0, 9007199254740992.0) FROM generate_series(1, 1)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values, [Some(0)]);
    }

    #[tokio::test]
    async fn uint64_normal_invalid_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 20, 10, 0.0) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(20, 0, 10, 1.0) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_mean_just_outside_range_outputs_tail_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 20, 21, 5.0) FROM generate_series(1, 10)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(
            values
                .iter()
                .flatten()
                .all(|value| (0..=20).contains(value))
        );
    }

    #[tokio::test]
    async fn uint64_normal_far_tail_errors_after_retry_cap() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 1, 1000, 1.0) FROM generate_series(1, 1)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_negative_min_literal_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(-1, 10, 0, 1.0) FROM generate_series(1, 1)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_negative_max_literal_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, -1, 0, 1.0) FROM generate_series(1, 1)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_negative_mean_literal_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(0, 10, -1, 1.0) FROM generate_series(1, 1)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_negative_column_value_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(min_value, max_value, mean_value, stddev) FROM (VALUES (0, 10, 5, 1.0), (0, 10, -1, 1.0)) AS t(min_value, max_value, mean_value, stddev)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_normal_array_args_propagate_nulls() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(min_value, max_value, mean_value, stddev) FROM (VALUES (0, 20, 10, 1.0), (NULL, 20, 10, 1.0), (0, 20, NULL, 1.0), (0, 20, 10, CAST(NULL AS DOUBLE))) AS t(min_value, max_value, mean_value, stddev)",
            DataType::UInt64,
        )
        .await;

        assert!(values[0].is_some());
        assert_eq!(values[1..], [None, None, None]);
    }

    #[tokio::test]
    async fn uint64_normal_array_args_without_nulls_output_values() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Normal::new()),
            "SELECT randgen_uint64_normal(min_value, max_value, mean_value, stddev) FROM (VALUES (0, 20, 10, 1.0), (10, 30, 20, 2.0)) AS t(min_value, max_value, mean_value, stddev)",
            DataType::UInt64,
        )
        .await;

        assert!(values.iter().all(Option::is_some));
    }
}
