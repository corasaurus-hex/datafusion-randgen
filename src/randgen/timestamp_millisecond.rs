use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::{Array, AsArray, TimestampMillisecondArray};
use datafusion::arrow::datatypes::{DataType, TimeUnit, TimestampMillisecondType};
use datafusion::common::{exec_err, internal_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use rand::Rng;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimestampMillisecond {
    signature: &'static Signature,
}

static TIMESTAMP_MILLISECOND_TYPE: LazyLock<DataType> =
    LazyLock::new(|| DataType::Timestamp(TimeUnit::Millisecond, None));

static TIMESTAMP_MILLISECOND_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(
        vec![
            TIMESTAMP_MILLISECOND_TYPE.clone(),
            TIMESTAMP_MILLISECOND_TYPE.clone(),
        ],
        Volatility::Volatile,
    )
});

impl TimestampMillisecond {
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

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(TIMESTAMP_MILLISECOND_TYPE.clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args, number_rows, ..
        } = args;
        let [min, max]: [ColumnarValue; 2] = match args.try_into() {
            Ok(args) => args,
            Err(_) => return internal_err!("{} expects exactly two arguments", self.name()),
        };

        if min.data_type() != *TIMESTAMP_MILLISECOND_TYPE
            || max.data_type() != *TIMESTAMP_MILLISECOND_TYPE
        {
            return internal_err!(
                "{} expects Timestamp(Millisecond, None) arguments",
                self.name()
            );
        }

        let min_array = min.into_array_of_size(number_rows)?;
        let max_array = max.into_array_of_size(number_rows)?;
        let min_values = min_array.as_primitive::<TimestampMillisecondType>();
        let max_values = max_array.as_primitive::<TimestampMillisecondType>();

        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(number_rows);
        for row in 0..number_rows {
            if min_values.is_null(row) || max_values.is_null(row) {
                values.push(None);
                continue;
            }

            let min = min_values.value(row);
            let max = max_values.value(row);
            if min > max {
                return exec_err!(
                    "{} requires min <= max, got min {min} and max {max}",
                    self.name()
                );
            }

            values.push(Some(rng.random_range(min..=max)));
        }

        Ok(ColumnarValue::Array(Arc::new(
            TimestampMillisecondArray::from(values),
        )))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::ScalarUDF;

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

    #[tokio::test]
    async fn timestamp_millisecond_rejects_other_timestamp_units() {
        let result = query_result(
            ScalarUDF::from(TimestampMillisecond::new()),
            "SELECT randgen_timestamp_millisecond(to_timestamp('2024-01-01T00:00:00Z'), to_timestamp('2024-01-02T00:00:00Z')) FROM generate_series(1, 10)",
        )
        .await;
        assert!(result.is_err());
    }
}
