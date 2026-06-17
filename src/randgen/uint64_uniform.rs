//! UInt64 uniform generator UDF.
//!
//! `randgen_uint64_uniform(min, max[, null_probability])` samples from the
//! inclusive integer range `min..=max`. Null bounds produce null output for
//! that row. Non-null bounds must satisfy `min <= max`. Arguments may be
//! `UInt64` values or nonnegative signed integer values. The optional null
//! probability must be finite and within `0.0..=1.0`.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use crate::randgen::utils::{
    NullProbability, coerce_optional_null_probability, coerce_uint64_argument, optional_args,
    primitive_range_array, primitive_range_scalar_array, two_array_args,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_uint64_uniform(min, max[, null_probability])`.
pub struct UInt64Uniform {
    signature: &'static Signature,
}

static UINT64_UNIFORM_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

impl UInt64Uniform {
    /// Creates the `randgen_uint64_uniform` implementation.
    pub fn new() -> Self {
        Self {
            signature: &UINT64_UNIFORM_SIGNATURE,
        }
    }
}

impl Default for UInt64Uniform {
    fn default() -> Self {
        Self::new()
    }
}

impl UInt64Uniform {
    fn invoke_scalar_args(
        &self,
        min: Option<u64>,
        max: Option<u64>,
        number_rows: usize,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(Arc::new(
            primitive_range_scalar_array::<UInt64Type>(
                min,
                max,
                number_rows,
                self.name(),
                null_probability,
            )?,
        )))
    }
}

impl ScalarUDFImpl for UInt64Uniform {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_uint64_uniform"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_optional_null_probability(arg_types, 2, self.name(), |arg_types| {
            Ok(vec![
                coerce_uint64_argument(&arg_types[0], self.name())?,
                coerce_uint64_argument(&arg_types[1], self.name())?,
            ])
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let ([min, max], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        if let (
            ColumnarValue::Scalar(ScalarValue::UInt64(min)),
            ColumnarValue::Scalar(ScalarValue::UInt64(max)),
        ) = (&min, &max)
        {
            return self.invoke_scalar_args(*min, *max, number_rows, &null_probability);
        }

        let (min_array, max_array) = two_array_args(
            vec![min, max],
            (DataType::UInt64, "UInt64 arguments"),
            (DataType::UInt64, "UInt64 arguments"),
            number_rows,
            self.name(),
        )?;
        let min_values = min_array.as_primitive::<UInt64Type>();
        let max_values = max_array.as_primitive::<UInt64Type>();

        Ok(ColumnarValue::Array(Arc::new(primitive_range_array(
            min_values,
            max_values,
            number_rows,
            self.name(),
            &null_probability,
        )?)))
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
    async fn uint64_uniform_values_stay_in_range() {
        for value in query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(1, 10) FROM generate_series(1, 100)",
            DataType::UInt64,
        )
        .await
        {
            assert!(value.unwrap() >= 1);
            assert!(value.unwrap() <= 10);
        }
    }

    #[tokio::test]
    async fn uint64_uniform_equal_bounds_are_deterministic() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(18446744073709551615, 18446744073709551615) FROM generate_series(1, 10)",
            DataType::UInt64,
        )
        .await;

        assert!(values.iter().all(|value| *value == Some(u64::MAX)));
    }

    #[tokio::test]
    async fn uint64_uniform_full_range_smoke() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(0, 18446744073709551615) FROM generate_series(1, 10)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values.len(), 10);
        assert!(values.iter().all(Option::is_some));
    }

    #[tokio::test]
    async fn uint64_uniform_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(10, 1) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_uniform_negative_min_literal_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(-1, 10) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_uniform_negative_max_literal_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(0, -1) FROM generate_series(1, 10)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_uniform_negative_column_value_errors() {
        let result = query_result(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(min_value, max_value) FROM (VALUES (0, 10), (0, -1)) AS t(min_value, max_value)",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn uint64_uniform_array_bounds_propagate_nulls() {
        let values = query_to_values::<UInt64Type>(
            ScalarUDF::from(UInt64Uniform::new()),
            "SELECT randgen_uint64_uniform(min_value, max_value) FROM (VALUES (7, 7), (NULL, 9), (1, NULL)) AS t(min_value, max_value)",
            DataType::UInt64,
        )
        .await;

        assert_eq!(values, vec![Some(7), None, None]);
    }
}
