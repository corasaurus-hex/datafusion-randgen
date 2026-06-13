//! Millisecond timestamp random generator.
//!
//! `randgen_timestamp_millisecond(min, max)` samples from the inclusive
//! timestamp range `min..=max`. Arguments must use millisecond precision and
//! matching timezones. Null bounds produce null output for that row.

use std::any::Any;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMillisecondType;
use arrow_schema::{DataType, TimeUnit};
use datafusion_common::plan_err;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TIMEZONE_WILDCARD, TypeSignature,
    Volatility,
};

use crate::randgen::utils::{exact_args, primitive_range_array, primitive_range_scalar_array};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// `ScalarUDFImpl` for `randgen_timestamp_millisecond(min, max)`.
pub struct TimestampMillisecond {
    signature: &'static Signature,
}

static TIMESTAMP_MILLISECOND_TYPE: LazyLock<DataType> =
    LazyLock::new(|| DataType::Timestamp(TimeUnit::Millisecond, None));

static TIMESTAMP_MILLISECOND_TIMEZONE_TYPE: LazyLock<DataType> =
    LazyLock::new(|| DataType::Timestamp(TimeUnit::Millisecond, Some(TIMEZONE_WILDCARD.into())));

static TIMESTAMP_MILLISECOND_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::one_of(
        vec![
            TypeSignature::Exact(vec![
                TIMESTAMP_MILLISECOND_TYPE.clone(),
                TIMESTAMP_MILLISECOND_TYPE.clone(),
            ]),
            TypeSignature::Exact(vec![
                TIMESTAMP_MILLISECOND_TIMEZONE_TYPE.clone(),
                TIMESTAMP_MILLISECOND_TIMEZONE_TYPE.clone(),
            ]),
        ],
        Volatility::Volatile,
    )
});

fn timestamp_millisecond_type(
    min_type: &DataType,
    max_type: &DataType,
    name: &str,
) -> Result<DataType> {
    match (min_type, max_type) {
        (
            DataType::Timestamp(TimeUnit::Millisecond, min_timezone),
            DataType::Timestamp(TimeUnit::Millisecond, max_timezone),
        ) if min_timezone == max_timezone => Ok(DataType::Timestamp(
            TimeUnit::Millisecond,
            min_timezone.clone(),
        )),
        (
            DataType::Timestamp(TimeUnit::Millisecond, _),
            DataType::Timestamp(TimeUnit::Millisecond, _),
        ) => plan_err!("{name} requires matching timestamp timezones"),
        _ => plan_err!("{name} expects Timestamp(Millisecond, timezone) arguments"),
    }
}

impl TimestampMillisecond {
    /// Creates the `randgen_timestamp_millisecond` implementation.
    pub fn new() -> Self {
        Self {
            signature: &TIMESTAMP_MILLISECOND_SIGNATURE,
        }
    }
}

impl Default for TimestampMillisecond {
    fn default() -> Self {
        Self::new()
    }
}

impl TimestampMillisecond {
    fn invoke_scalar_args(
        &self,
        min: Option<i64>,
        max: Option<i64>,
        timezone: Option<Arc<str>>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let values = primitive_range_scalar_array::<TimestampMillisecondType>(
            min,
            max,
            number_rows,
            self.name(),
        )?;
        Ok(ColumnarValue::Array(Arc::new(
            values.with_timezone_opt(timezone),
        )))
    }
}

impl ScalarUDFImpl for TimestampMillisecond {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_timestamp_millisecond"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [min_type, max_type] = arg_types else {
            return plan_err!("{} expects exactly two arguments", self.name());
        };
        timestamp_millisecond_type(min_type, max_type, self.name())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [min, max] = exact_args(args, self.name())?;

        let output_type =
            timestamp_millisecond_type(&min.data_type(), &max.data_type(), self.name())?;

        if let (
            ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(min_value, _)),
            ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(max_value, _)),
        ) = (&min, &max)
        {
            let DataType::Timestamp(TimeUnit::Millisecond, timezone) = output_type else {
                unreachable!("timestamp_millisecond_type only returns millisecond timestamps");
            };
            return self.invoke_scalar_args(*min_value, *max_value, timezone, number_rows);
        }

        let min_array = min.into_array_of_size(number_rows)?;
        let max_array = max.into_array_of_size(number_rows)?;
        let min_values = min_array.as_primitive::<TimestampMillisecondType>();
        let max_values = max_array.as_primitive::<TimestampMillisecondType>();

        let values = primitive_range_array(min_values, max_values, number_rows, self.name())?;
        let DataType::Timestamp(TimeUnit::Millisecond, timezone) = output_type else {
            unreachable!("timestamp_millisecond_type only returns millisecond timestamps");
        };
        Ok(ColumnarValue::Array(Arc::new(
            values.with_timezone_opt(timezone),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, TimestampMillisecondArray};
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::ScalarUDF;
    use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};

    use crate::randgen::test_helpers::querying::{query_result, query_to_values};

    use super::*;

    #[tokio::test]
    async fn timestamp_millisecond_values_stay_in_range() {
        let values = query_to_values::<TimestampMillisecondType>(
            ScalarUDF::from(TimestampMillisecond::new()),
            "SELECT randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-02T00:00:00Z')) FROM generate_series(1, 1000)",
            TIMESTAMP_MILLISECOND_TYPE.clone(),
        )
        .await;
        assert!(values.iter().all(|value| {
            value.is_some_and(|value| (1_704_067_200_000..=1_704_153_600_000).contains(&value))
        }));
    }

    #[tokio::test]
    async fn timestamp_millisecond_equal_bounds_are_deterministic() {
        let values = query_to_values::<TimestampMillisecondType>(
            ScalarUDF::from(TimestampMillisecond::new()),
            "SELECT randgen_timestamp_millisecond(to_timestamp_millis('2024-01-01T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z')) FROM generate_series(1, 100)",
            TIMESTAMP_MILLISECOND_TYPE.clone(),
        )
        .await;
        assert!(values.iter().all(|value| *value == Some(1_704_067_200_000)));
    }

    #[tokio::test]
    async fn timestamp_millisecond_invalid_range_errors() {
        let result = query_result(
            ScalarUDF::from(TimestampMillisecond::new()),
            "SELECT randgen_timestamp_millisecond(to_timestamp_millis('2024-01-02T00:00:00Z'), to_timestamp_millis('2024-01-01T00:00:00Z')) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }

    #[test]
    fn timestamp_millisecond_preserves_matching_timezone() {
        let data_type = DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into()));
        let min =
            TimestampMillisecondArray::from(vec![Some(1_704_067_200_000)]).with_timezone("+00:00");
        let max =
            TimestampMillisecondArray::from(vec![Some(1_704_067_200_000)]).with_timezone("+00:00");
        let field = Arc::new(Field::new("value", data_type.clone(), true));

        let result = TimestampMillisecond::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(min)),
                    ColumnarValue::Array(Arc::new(max)),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("min", data_type.clone(), true)),
                    Arc::new(Field::new("max", data_type.clone(), true)),
                ],
                number_rows: 1,
                return_field: field,
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        assert_eq!(array.data_type(), &data_type);
    }

    #[test]
    fn timestamp_millisecond_array_args_propagate_nulls() {
        let data_type = TIMESTAMP_MILLISECOND_TYPE.clone();
        let min = TimestampMillisecondArray::from(vec![Some(1_704_067_200_000), None]);
        let max =
            TimestampMillisecondArray::from(vec![Some(1_704_067_200_000), Some(1_704_153_600_000)]);

        let result = TimestampMillisecond::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(min)),
                    ColumnarValue::Array(Arc::new(max)),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("min", data_type.clone(), true)),
                    Arc::new(Field::new("max", data_type.clone(), true)),
                ],
                number_rows: 2,
                return_field: Arc::new(Field::new("value", data_type, true)),
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let values = array.as_primitive::<TimestampMillisecondType>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![Some(1_704_067_200_000), None]
        );
    }

    #[test]
    fn timestamp_millisecond_rejects_mismatched_timezones() {
        let result = timestamp_millisecond_type(
            &DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
            &DataType::Timestamp(TimeUnit::Millisecond, Some("+08:00".into())),
            "randgen_timestamp_millisecond",
        );

        assert!(result.is_err());
    }
}
