//! Aggregate-backed column-choice generator.
//!
//! `randgen_roaring_agg(source_column)` accepts `UInt32` or `UInt64`, ignores
//! null source rows, and serializes the distinct source values as a roaring set.
//! `randgen_column_choice(values[, null_probability])` samples with replacement
//! from that set. `Binary` input returns `UInt32`; `LargeBinary` input returns
//! `UInt64`. Null input values produce null output rows, and empty non-null sets
//! return an error because there is no value to sample.

use std::any::Any;
use std::io;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BinaryArray, LargeBinaryArray, UInt32Array, UInt64Array, new_empty_array,
    new_null_array,
};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err, internal_err, plan_err};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
    function::{AccumulatorArgs, StateFieldsArgs},
};
use rand::Rng;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::randgen::utils::{NullProbability, coerce_float64_argument, optional_args};

/// `ScalarUDFImpl` for the `randgen_column_choice` sampler.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnChoice {
    signature: &'static Signature,
}

/// `AggregateUDFImpl` for `randgen_roaring_agg(source_column)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RoaringAgg {
    signature: &'static Signature,
}

static COLUMN_CHOICE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

static ROARING_AGG_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::uniform(
        1,
        vec![DataType::UInt32, DataType::UInt64],
        Volatility::Immutable,
    )
});

#[derive(Debug)]
enum ColumnValues {
    UInt32(RoaringBitmap),
    UInt64(RoaringTreemap),
}

#[derive(Debug)]
enum RoaringAggAccumulator {
    UInt32(RoaringBitmap),
    UInt64(RoaringTreemap),
}

impl ColumnChoice {
    /// Creates the `randgen_column_choice` implementation.
    pub fn new() -> Self {
        Self {
            signature: &COLUMN_CHOICE_SIGNATURE,
        }
    }
}

impl Default for ColumnChoice {
    fn default() -> Self {
        Self::new()
    }
}

impl RoaringAgg {
    /// Creates the `randgen_roaring_agg` implementation.
    pub fn new() -> Self {
        Self {
            signature: &ROARING_AGG_SIGNATURE,
        }
    }
}

impl Default for RoaringAgg {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the aggregate-backed column-choice scalar UDFs.
pub fn column_choice_udfs() -> Vec<ScalarUDF> {
    vec![ScalarUDF::from(ColumnChoice::new())]
}

fn io_error(name: &str, action: &str, error: io::Error) -> DataFusionError {
    DataFusionError::Execution(format!("{name} could not {action}: {error}"))
}

fn serialize_bitmap(values: &RoaringBitmap, name: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.serialized_size());
    values
        .serialize_into(&mut bytes)
        .map_err(|error| io_error(name, "serialize UInt32 roaring set", error))?;
    Ok(bytes)
}

fn deserialize_bitmap(bytes: &[u8], name: &str) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes)
        .map_err(|error| io_error(name, "deserialize UInt32 roaring set", error))
}

fn serialize_treemap(values: &RoaringTreemap, name: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.serialized_size());
    values
        .serialize_into(&mut bytes)
        .map_err(|error| io_error(name, "serialize UInt64 roaring set", error))?;
    Ok(bytes)
}

fn deserialize_treemap(bytes: &[u8], name: &str) -> Result<RoaringTreemap> {
    RoaringTreemap::deserialize_from(bytes)
        .map_err(|error| io_error(name, "deserialize UInt64 roaring set", error))
}

impl ColumnValues {
    fn data_type(&self) -> DataType {
        match self {
            Self::UInt32(_) => DataType::UInt32,
            Self::UInt64(_) => DataType::UInt64,
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::UInt32(values) => values.len(),
            Self::UInt64(values) => values.len(),
        }
    }

    fn sample_one<R>(&self, name: &str, rng: &mut R) -> Result<ScalarValue>
    where
        R: Rng + ?Sized,
    {
        if self.len() == 0 {
            return exec_err!("{name} requires at least one non-null source value");
        }

        match self {
            Self::UInt32(values) => {
                let rank = rng.random_range(0..values.len());
                let value = values.select(rank as u32).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "{name} failed to select source value at rank {rank}"
                    ))
                })?;
                Ok(ScalarValue::UInt32(Some(value)))
            }
            Self::UInt64(values) => {
                let rank = rng.random_range(0..values.len());
                let value = values.select(rank).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "{name} failed to select source value at rank {rank}"
                    ))
                })?;
                Ok(ScalarValue::UInt64(Some(value)))
            }
        }
    }

    fn sample(
        &self,
        number_rows: usize,
        name: &str,
        null_probability: &NullProbability,
    ) -> Result<ColumnarValue> {
        if self.len() == 0 {
            return exec_err!("{name} requires at least one non-null source value");
        }

        let mut rng = rand::rng();
        match self {
            Self::UInt32(values) => {
                let len = values.len();
                let mut output = Vec::with_capacity(number_rows);
                for row in 0..number_rows {
                    if null_probability.is_null(row, &mut rng, name)? {
                        output.push(None);
                        continue;
                    }
                    let rank = rng.random_range(0..len);
                    let value = values.select(rank as u32).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "{name} failed to select source value at rank {rank}"
                        ))
                    })?;
                    output.push(Some(value));
                }
                Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(output))))
            }
            Self::UInt64(values) => {
                let len = values.len();
                let mut output = Vec::with_capacity(number_rows);
                for row in 0..number_rows {
                    if null_probability.is_null(row, &mut rng, name)? {
                        output.push(None);
                        continue;
                    }
                    let rank = rng.random_range(0..len);
                    let value = values.select(rank).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "{name} failed to select source value at rank {rank}"
                        ))
                    })?;
                    output.push(Some(value));
                }
                Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(output))))
            }
        }
    }
}

fn scalar_binary_values(value: &ScalarValue, name: &str) -> Result<Option<ColumnValues>> {
    match value {
        ScalarValue::Binary(Some(bytes)) => {
            Ok(Some(ColumnValues::UInt32(deserialize_bitmap(bytes, name)?)))
        }
        ScalarValue::Binary(None) => Ok(None),
        ScalarValue::LargeBinary(Some(bytes)) => Ok(Some(ColumnValues::UInt64(
            deserialize_treemap(bytes, name)?,
        ))),
        ScalarValue::LargeBinary(None) => Ok(None),
        value => exec_err!(
            "{name} expects Binary or LargeBinary, got {}",
            value.data_type()
        ),
    }
}

fn sample_binary_array(
    values: &BinaryArray,
    number_rows: usize,
    name: &str,
    null_probability: &NullProbability,
) -> Result<ColumnarValue> {
    let mut rng = rand::rng();
    let mut output = Vec::with_capacity(number_rows);
    for row in 0..number_rows {
        if values.is_null(row) || null_probability.is_null(row, &mut rng, name)? {
            output.push(None);
            continue;
        }
        let choices = ColumnValues::UInt32(deserialize_bitmap(values.value(row), name)?);
        let ScalarValue::UInt32(value) = choices.sample_one(name, &mut rng)? else {
            return internal_err!("{name} sampled non-UInt32 value from Binary input");
        };
        output.push(value);
    }
    Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(output))))
}

fn sample_large_binary_array(
    values: &LargeBinaryArray,
    number_rows: usize,
    name: &str,
    null_probability: &NullProbability,
) -> Result<ColumnarValue> {
    let mut rng = rand::rng();
    let mut output = Vec::with_capacity(number_rows);
    for row in 0..number_rows {
        if values.is_null(row) || null_probability.is_null(row, &mut rng, name)? {
            output.push(None);
            continue;
        }
        let choices = ColumnValues::UInt64(deserialize_treemap(values.value(row), name)?);
        let ScalarValue::UInt64(value) = choices.sample_one(name, &mut rng)? else {
            return internal_err!("{name} sampled non-UInt64 value from LargeBinary input");
        };
        output.push(value);
    }
    Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(output))))
}

impl Accumulator for RoaringAggAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 1 {
            return exec_err!("randgen_roaring_agg expects one argument");
        }

        match self {
            Self::UInt32(output) => {
                if values[0].data_type() != &DataType::UInt32 {
                    return internal_err!(
                        "randgen_roaring_agg expected UInt32 input, got {}",
                        values[0].data_type()
                    );
                }
                let input = values[0].as_primitive::<UInt32Type>();
                for row in 0..input.len() {
                    if !input.is_null(row) {
                        output.insert(input.value(row));
                    }
                }
            }
            Self::UInt64(output) => {
                if values[0].data_type() != &DataType::UInt64 {
                    return internal_err!(
                        "randgen_roaring_agg expected UInt64 input, got {}",
                        values[0].data_type()
                    );
                }
                let input = values[0].as_primitive::<UInt64Type>();
                for row in 0..input.len() {
                    if !input.is_null(row) {
                        output.insert(input.value(row));
                    }
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        match self {
            Self::UInt32(values) => Ok(ScalarValue::Binary(Some(serialize_bitmap(
                values,
                "randgen_roaring_agg",
            )?))),
            Self::UInt64(values) => Ok(ScalarValue::LargeBinary(Some(serialize_treemap(
                values,
                "randgen_roaring_agg",
            )?))),
        }
    }

    fn size(&self) -> usize {
        let serialized_size = match self {
            Self::UInt32(values) => values.serialized_size(),
            Self::UInt64(values) => values.serialized_size(),
        };
        std::mem::size_of_val(self) + serialized_size
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 1 {
            return exec_err!("randgen_roaring_agg expects one state field");
        }

        match self {
            Self::UInt32(output) => {
                if states[0].data_type() != &DataType::Binary {
                    return internal_err!(
                        "randgen_roaring_agg expected Binary state, got {}",
                        states[0].data_type()
                    );
                }
                let states = states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
                for row in 0..states.len() {
                    if !states.is_null(row) {
                        *output |= &deserialize_bitmap(states.value(row), "randgen_roaring_agg")?;
                    }
                }
            }
            Self::UInt64(output) => {
                if states[0].data_type() != &DataType::LargeBinary {
                    return internal_err!(
                        "randgen_roaring_agg expected LargeBinary state, got {}",
                        states[0].data_type()
                    );
                }
                let states = states[0]
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .unwrap();
                for row in 0..states.len() {
                    if !states.is_null(row) {
                        *output |= &deserialize_treemap(states.value(row), "randgen_roaring_agg")?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl AggregateUDFImpl for RoaringAgg {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_roaring_agg"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::UInt32] => Ok(DataType::Binary),
            [DataType::UInt64] => Ok(DataType::LargeBinary),
            [data_type] => plan_err!("{} expects UInt32 or UInt64, got {data_type}", self.name()),
            _ => plan_err!("{} expects one argument", self.name()),
        }
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(
            self.name(),
            self.return_type(
                &arg_fields
                    .iter()
                    .map(|field| field.data_type().clone())
                    .collect::<Vec<_>>(),
            )?,
            false,
        )))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        match acc_args.return_type() {
            DataType::Binary => Ok(Box::new(
                RoaringAggAccumulator::UInt32(RoaringBitmap::new()),
            )),
            DataType::LargeBinary => Ok(Box::new(RoaringAggAccumulator::UInt64(
                RoaringTreemap::new(),
            ))),
            data_type => internal_err!("{} cannot accumulate {data_type}", self.name()),
        }
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            format!("{}_state", args.name),
            args.return_type().clone(),
            false,
        ))])
    }
}

impl ScalarUDFImpl for ColumnChoice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "randgen_column_choice"
    }

    fn signature(&self) -> &Signature {
        self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::Binary] | [DataType::Binary, DataType::Float64] => Ok(DataType::UInt32),
            [DataType::LargeBinary] | [DataType::LargeBinary, DataType::Float64] => {
                Ok(DataType::UInt64)
            }
            [data_type] | [data_type, DataType::Float64] => {
                plan_err!(
                    "{} expects Binary or LargeBinary, got {data_type}",
                    self.name()
                )
            }
            [_, data_type] => plan_err!(
                "{} expects a Float64 null probability, got {data_type}",
                self.name()
            ),
            _ => plan_err!("{} expects one or two arguments", self.name()),
        }
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        if args.arg_fields.len() != 1 && args.arg_fields.len() != 2 {
            return plan_err!("{} expects one or two arguments", self.name());
        }

        Ok(Arc::new(Field::new(
            self.name(),
            self.return_type(
                &args
                    .arg_fields
                    .iter()
                    .map(|field| field.data_type().clone())
                    .collect::<Vec<_>>(),
            )?,
            true,
        )))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 1 && arg_types.len() != 2 {
            return exec_err!("{} expects one or two arguments", self.name());
        }

        match arg_types[0] {
            DataType::Binary | DataType::LargeBinary => {}
            ref data_type => {
                return exec_err!(
                    "{} expects Binary or LargeBinary, got {data_type}",
                    self.name()
                );
            }
        }

        let mut coerced = vec![arg_types[0].clone()];
        if let Some(null_probability_type) = arg_types.get(1) {
            coerced.push(coerce_float64_argument(null_probability_type, self.name())?);
        }
        Ok(coerced)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            number_rows,
            return_field,
            ..
        } = args;
        let ([values], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;

        if number_rows == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                return_field.data_type(),
            )));
        }

        match values {
            ColumnarValue::Scalar(value) => {
                let Some(values) = scalar_binary_values(&value, self.name())? else {
                    return Ok(ColumnarValue::Array(new_null_array(
                        return_field.data_type(),
                        number_rows,
                    )));
                };
                if values.data_type() != *return_field.data_type() {
                    return internal_err!(
                        "{} planned return type {} but decoded {}",
                        self.name(),
                        return_field.data_type(),
                        values.data_type()
                    );
                }
                values.sample(number_rows, self.name(), &null_probability)
            }
            ColumnarValue::Array(values) => match values.data_type() {
                DataType::Binary => {
                    if return_field.data_type() != &DataType::UInt32 {
                        return internal_err!("{} planned non-UInt32 Binary output", self.name());
                    }
                    sample_binary_array(
                        values.as_any().downcast_ref::<BinaryArray>().unwrap(),
                        number_rows,
                        self.name(),
                        &null_probability,
                    )
                }
                DataType::LargeBinary => {
                    if return_field.data_type() != &DataType::UInt64 {
                        return internal_err!(
                            "{} planned non-UInt64 LargeBinary output",
                            self.name()
                        );
                    }
                    sample_large_binary_array(
                        values.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
                        number_rows,
                        self.name(),
                        &null_probability,
                    )
                }
                data_type => internal_err!(
                    "{} expected Binary or LargeBinary, got {data_type}",
                    self.name()
                ),
            },
        }
    }
}
