//! Int64 normal-distribution random generator.
//!
//! `randgen_int64_normal(min, max, mean, stddev)` samples an integer normal
//! distribution centered on `mean` and truncated to the inclusive `Int64`
//! range `min..=max`. The mean may be outside the output range. Null input
//! yields null output for that row.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, Int64Array};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use rand::Rng;

use crate::randgen::integer_normal::IntegerNormalSampler;
use crate::randgen::utils::four_array_args;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_int64_normal(min, max, mean, stddev)`.
pub struct Int64Normal {
    signature: &'static Signature,
}

static INT64_NORMAL_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
            DataType::Float64,
        ],
        Volatility::Volatile,
    )
});

fn sampler_for_range(
    min: i64,
    max: i64,
    mean: i64,
    stddev: f64,
    name: &str,
) -> Result<IntegerNormalSampler> {
    if min > max {
        return exec_err!("{name} requires min <= max");
    }

    IntegerNormalSampler::for_offset_range(
        min as i128 - mean as i128,
        max as i128 - mean as i128,
        stddev,
        name,
    )
}

fn sample_int64<R: Rng + ?Sized>(
    rng: &mut R,
    mean: i64,
    sampler: &IntegerNormalSampler,
    name: &str,
) -> Result<i64> {
    let offset = sampler.sample_offset(rng, name)?;
    Ok((mean as i128 + offset) as i64)
}

impl Int64Normal {
    /// Creates the `randgen_int64_normal` implementation.
    pub fn new() -> Self {
        Self {
            signature: &INT64_NORMAL_SIGNATURE,
        }
    }
}

impl Default for Int64Normal {
    fn default() -> Self {
        Self::new()
    }
}

impl Int64Normal {
    fn invoke_scalar_args(
        &self,
        min: Option<i64>,
        max: Option<i64>,
        mean: Option<i64>,
        stddev: Option<f64>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let mut rng = rand::rng();
        if let (Some(min), Some(max), Some(mean), Some(stddev)) = (min, max, mean, stddev) {
            let sampler = sampler_for_range(min, max, mean, stddev, self.name())?;
            let mut values = Vec::with_capacity(number_rows);
            for _ in 0..number_rows {
                values.push(sample_int64(&mut rng, mean, &sampler, self.name())?);
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

impl ScalarUDFImpl for Int64Normal {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "randgen_int64_normal"
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
        let [min, max, mean, stddev] = crate::randgen::utils::exact_args(args, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::Int64(min)),
            ColumnarValue::Scalar(ScalarValue::Int64(max)),
            ColumnarValue::Scalar(ScalarValue::Int64(mean)),
            ColumnarValue::Scalar(ScalarValue::Float64(stddev)),
        ) = (&min, &max, &mean, &stddev)
        {
            return self.invoke_scalar_args(*min, *max, *mean, *stddev, number_rows);
        }

        let expected = "Int64, Int64, Int64, Float64 arguments";
        let (min_array, max_array, mean_array, stddev_array) = four_array_args(
            vec![min, max, mean, stddev],
            (DataType::Int64, expected),
            (DataType::Int64, expected),
            (DataType::Int64, expected),
            (DataType::Float64, expected),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<Int64Type>();
        let max_values = max_array.as_primitive::<Int64Type>();
        let mean_values = mean_array.as_primitive::<Int64Type>();
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
                values.push(sample_int64(&mut rng, mean, &sampler, self.name())?);
            }

            return Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))));
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
            values.push(Some(sample_int64(&mut rng, mean, &sampler, self.name())?));
        }

        Ok(ColumnarValue::Array(Arc::new(Int64Array::from(values))))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Int64Type;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;

    use crate::randgen::test_helpers::querying::{query_result, query_to_values};

    use super::*;

    #[test]
    fn int64_normal_uses_f64_for_exact_offset_ranges() {
        let sampler =
            sampler_for_range(i64::MAX - 10, i64::MAX, i64::MAX - 5, 1.0, "test").unwrap();

        assert!(sampler.uses_f64());
    }

    #[test]
    fn int64_normal_uses_integer_domain_for_full_type_range() {
        let sampler = sampler_for_range(i64::MIN, i64::MAX, 0, 1.0, "test").unwrap();

        assert!(sampler.uses_integer_domain());
    }

    #[test]
    fn int64_normal_uses_dither_for_large_stddev() {
        let sampler =
            sampler_for_range(i64::MIN, i64::MAX, 0, (1_u64 << 53) as f64, "test").unwrap();

        assert!(sampler.uses_f64());
        assert!(sampler.uses_dither());
    }

    #[tokio::test]
    async fn int64_normal_outputs_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(0, 20, 10, 2.0) FROM generate_series(1, 100)",
            DataType::Int64,
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
    async fn int64_normal_full_range_outputs_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(-9223372036854775807, 9223372036854775807, 0, 1.0) FROM generate_series(1, 10)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn int64_normal_large_stddev_full_range_outputs_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(-9223372036854775807, 9223372036854775807, 0, 9007199254740992.0) FROM generate_series(1, 10)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn int64_normal_huge_stddev_single_value_range_outputs_value() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(0, 0, 0, 9007199254740992.0) FROM generate_series(1, 1)",
            DataType::Int64,
        )
        .await;

        assert_eq!(values, [Some(0)]);
    }

    #[tokio::test]
    async fn int64_normal_invalid_stddev_errors() {
        let result = query_result(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(0, 20, 10, 0.0) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn int64_normal_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(20, 0, 10, 1.0) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn int64_normal_mean_outside_range_outputs_tail_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(0, 20, 30, 1.0) FROM generate_series(1, 10)",
            DataType::Int64,
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
    async fn int64_normal_array_args_propagate_nulls() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(min_value, max_value, mean_value, stddev) FROM (VALUES (0, 20, 10, 1.0), (CAST(NULL AS BIGINT), 20, 10, 1.0), (0, 20, CAST(NULL AS BIGINT), 1.0), (0, 20, 10, CAST(NULL AS DOUBLE))) AS t(min_value, max_value, mean_value, stddev)",
            DataType::Int64,
        )
        .await;

        assert!(values[0].is_some());
        assert_eq!(values[1..], [None, None, None]);
    }

    #[tokio::test]
    async fn int64_normal_array_args_without_nulls_output_values() {
        let values = query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Normal::new()),
            "SELECT randgen_int64_normal(min_value, max_value, mean_value, stddev) FROM (VALUES (0, 20, 10, 1.0), (10, 30, 20, 2.0)) AS t(min_value, max_value, mean_value, stddev)",
            DataType::Int64,
        )
        .await;

        assert!(values.iter().all(Option::is_some));
    }
}
