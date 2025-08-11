use std::any::Any;
use std::cell::RefCell;

use datafusion::arrow::array::{Array, AsArray, Int64Array};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::{DataType, Int64Type};
use datafusion::common::internal_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;
use rand::Rng;
use rand_distr::Uniform;
use std::sync::{Arc, LazyLock};

#[derive(Debug, Clone)]
pub struct Int64Uniform {
    signature: &'static Signature,
}

static INT64_UNIFORM_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::exact(vec![DataType::Int64, DataType::Int64], Volatility::Volatile)
});

impl Int64Uniform {
    pub fn new() -> Self {
        Self {
            signature: &INT64_UNIFORM_SIGNATURE,
        }
    }
}

impl Default for Int64Uniform {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for Int64Uniform {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_int64_uniform"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        let ScalarFunctionArgs { mut args, .. } = args;
        let max = args.pop().unwrap();
        let min = args.pop().unwrap();

        assert_eq!(max.data_type(), DataType::Int64);
        assert_eq!(min.data_type(), DataType::Int64);

        match (max, min) {
            (
                ColumnarValue::Scalar(ScalarValue::Int64(max)),
                ColumnarValue::Scalar(ScalarValue::Int64(min)),
            ) => {
                let results = sample_uniform_min_const_max_const(min.unwrap(), max.unwrap());
                Ok(ColumnarValue::Scalar(ScalarValue::from(results)))
            }
            (ColumnarValue::Array(base_array), ColumnarValue::Scalar(ScalarValue::Int64(max))) => {
                let results = sample_uniform_min_array_max_const(base_array, max.unwrap());
                Ok(ColumnarValue::Array(results))
            }

            (ColumnarValue::Scalar(ScalarValue::Int64(min)), ColumnarValue::Array(max_array)) => {
                let results = sample_uniform_min_const_max_array(min.unwrap(), max_array);
                Ok(ColumnarValue::Array(results))
            }

            (ColumnarValue::Array(min_array), ColumnarValue::Array(max_array)) => {
                let results = sample_uniform_min_array_max_array(min_array, max_array);
                Ok(ColumnarValue::Array(results))
            }
            _ => internal_err!("Unsupported argument types"),
        }
    }
}

fn sample_uniform_min_const_max_const(min: i64, max: i64) -> i64 {
    let mut rng = rand::rng();
    let uniform = Uniform::new_inclusive(min, max).unwrap();
    rng.sample(uniform)
}

fn sample_uniform_min_array_max_const(min_array: Arc<dyn Array>, max: i64) -> Arc<Int64Array> {
    let rng = RefCell::new(rand::rng());
    let min_values = min_array.as_primitive::<Int64Type>();
    let results: Int64Array = compute::unary(min_values, |min| {
        let uniform = Uniform::new_inclusive(min, max).unwrap();
        rng.borrow_mut().sample(uniform)
    });
    Arc::new(results)
}

fn sample_uniform_min_const_max_array(min: i64, max_array: Arc<dyn Array>) -> Arc<Int64Array> {
    let rng = RefCell::new(rand::rng());
    let max_values = max_array.as_primitive::<Int64Type>();
    let results: Int64Array = compute::unary(max_values, |max| {
        let uniform = Uniform::new_inclusive(min, max).unwrap();
        rng.borrow_mut().sample(uniform)
    });
    Arc::new(results)
}

fn sample_uniform_min_array_max_array(
    min_array: Arc<dyn Array>,
    max_array: Arc<dyn Array>,
) -> Arc<Int64Array> {
    let rng = RefCell::new(rand::rng());
    let min_values = min_array.as_primitive::<Int64Type>();
    let max_values = max_array.as_primitive::<Int64Type>();
    let results: Int64Array = compute::try_binary(min_values, max_values, |min, max| {
        let uniform = Uniform::new_inclusive(min, max).unwrap();
        Ok(rng.borrow_mut().sample(uniform))
    })
    .unwrap();
    Arc::new(results)
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::datatypes::{DataType, Int64Type},
        logical_expr::ScalarUDF,
    };

    use crate::randgen::test_helpers::querying::query_to_values;

    use super::*;

    #[tokio::test]
    async fn int64_uniform_min_const_max_const() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, 10) as x from generate_series(1, 100)",
            DataType::Int64,
        )
        .await
        {
            assert!(value >= 1);
            assert!(value <= 10);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_array_max_const() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(y, 20) as x from (select randgen_int64_uniform(1, 10) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value >= 1);
            assert!(value <= 20);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_const_max_array() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(1, y) as x from (select randgen_int64_uniform(11, 20) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value >= 1);
            assert!(value <= 20);
        }
    }

    #[tokio::test]
    async fn int64_uniform_min_array_max_array() {
        for value in query_to_values::<Int64Type>(
            ScalarUDF::from(Int64Uniform::new()),
            "SELECT randgen_int64_uniform(x, y) as x from (select randgen_int64_uniform(1, 10) as x, randgen_int64_uniform(11, 20) as y from generate_series(1, 100))",
            DataType::Int64,
        )
        .await
        {
            assert!(value >= 1);
            assert!(value <= 20);
        }
    }
}
