//! Parquet-backed column choice random generator.
//!
//! `randgen_column_choice(source_path, column_name)` samples with replacement
//! from the distinct non-null values in an unsigned integer Parquet column.
//! Both arguments must be scalar strings known at planning time. The UDF reads
//! the source as Parquet regardless of extension.

use std::any::Any;
use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::UNIX_EPOCH;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{Array, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::ParquetError;
use rand::Rng;
use roaring::{RoaringBitmap, RoaringTreemap};

/// `ScalarUDFImpl` for `randgen_column_choice(source_path, column_name)`.
#[derive(Debug)]
pub struct ColumnChoice {
    signature: &'static Signature,
    cache: Mutex<HashMap<CacheKey, Arc<ColumnValues>>>,
}

static COLUMN_CHOICE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::exact(vec![DataType::Utf8, DataType::Utf8], Volatility::Volatile));

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    path: String,
    column: String,
    file_len: u64,
    modified: Option<(u64, u32)>,
}

#[derive(Debug)]
enum ColumnValues {
    UInt32(RoaringBitmap),
    UInt64(RoaringTreemap),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    UInt32,
    UInt64,
}

impl PartialEq for ColumnChoice {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ColumnChoice {}

impl Hash for ColumnChoice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ColumnChoice {
    /// Creates the `randgen_column_choice` implementation.
    pub fn new() -> Self {
        Self {
            signature: &COLUMN_CHOICE_SIGNATURE,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for ColumnChoice {
    fn default() -> Self {
        Self::new()
    }
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

    fn sample(&self, number_rows: usize, name: &str) -> Result<ColumnarValue> {
        if self.len() == 0 {
            return exec_err!("{name} requires at least one non-null source value");
        }

        let mut rng = rand::rng();
        match self {
            Self::UInt32(values) => {
                let len = values.len();
                let mut output = Vec::with_capacity(number_rows);
                for _ in 0..number_rows {
                    let rank = rng.random_range(0..len);
                    let value = values.select(rank as u32).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "{name} failed to select source value at rank {rank}"
                        ))
                    })?;
                    output.push(value);
                }
                Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(output))))
            }
            Self::UInt64(values) => {
                let len = values.len();
                let mut output = Vec::with_capacity(number_rows);
                for _ in 0..number_rows {
                    let rank = rng.random_range(0..len);
                    let value = values.select(rank).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "{name} failed to select source value at rank {rank}"
                        ))
                    })?;
                    output.push(value);
                }
                Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(output))))
            }
        }
    }
}

fn parquet_error(name: &str, path: &str, error: ParquetError) -> DataFusionError {
    DataFusionError::Execution(format!(
        "{name} could not read Parquet source {path}: {error}"
    ))
}

fn scalar_string_arg<'a>(
    values: &'a [Option<&'a ScalarValue>],
    index: usize,
    arg_name: &str,
    name: &str,
) -> Result<&'a str> {
    let Some(Some(value)) = values.get(index) else {
        return plan_err!("{name} requires scalar {arg_name}");
    };

    match value {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => Ok(value),
        ScalarValue::Utf8(None) | ScalarValue::LargeUtf8(None) | ScalarValue::Utf8View(None) => {
            plan_err!("{name} requires non-null scalar {arg_name}")
        }
        _ => plan_err!("{name} requires scalar string {arg_name}"),
    }
}

fn columnar_scalar_string_arg(value: &ColumnarValue, arg_name: &str, name: &str) -> Result<String> {
    match value {
        ColumnarValue::Scalar(
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)),
        ) => Ok(value.clone()),
        ColumnarValue::Scalar(
            ScalarValue::Utf8(None) | ScalarValue::LargeUtf8(None) | ScalarValue::Utf8View(None),
        ) => exec_err!("{name} requires non-null scalar {arg_name}"),
        ColumnarValue::Scalar(_) => exec_err!("{name} requires scalar string {arg_name}"),
        ColumnarValue::Array(_) => exec_err!("{name} requires scalar {arg_name}"),
    }
}

fn file_key(path: String, column: String, name: &str) -> Result<CacheKey> {
    let metadata = fs::metadata(&path).map_err(|error| {
        DataFusionError::Execution(format!("{name} could not stat {path}: {error}"))
    })?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs(), duration.subsec_nanos()));

    Ok(CacheKey {
        path,
        column,
        file_len: metadata.len(),
        modified,
    })
}

fn parquet_column_kind(path: &str, column: &str, name: &str) -> Result<ColumnKind> {
    let file = File::open(path).map_err(|error| {
        DataFusionError::Execution(format!(
            "{name} could not open Parquet source {path}: {error}"
        ))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| parquet_error(name, path, error))?;
    let field = builder
        .schema()
        .field_with_name(column)
        .map_err(|_| DataFusionError::Plan(format!("{name} could not find column {column}")))?;

    match field.data_type() {
        DataType::UInt32 => Ok(ColumnKind::UInt32),
        DataType::UInt64 => Ok(ColumnKind::UInt64),
        data_type => plan_err!("{name} supports UInt32 and UInt64 columns, got {data_type}"),
    }
}

fn projected_reader(
    path: &str,
    column: &str,
    name: &str,
) -> Result<(
    ColumnKind,
    parquet::arrow::arrow_reader::ParquetRecordBatchReader,
)> {
    let file = File::open(path).map_err(|error| {
        DataFusionError::Execution(format!(
            "{name} could not open Parquet source {path}: {error}"
        ))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| parquet_error(name, path, error))?;
    let schema = builder.schema();
    let (column_index, field) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == column)
        .ok_or_else(|| {
            DataFusionError::Execution(format!("{name} could not find column {column}"))
        })?;
    let kind = match field.data_type() {
        DataType::UInt32 => ColumnKind::UInt32,
        DataType::UInt64 => ColumnKind::UInt64,
        data_type => {
            return exec_err!("{name} supports UInt32 and UInt64 columns, got {data_type}");
        }
    };
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column_index]);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|error| parquet_error(name, path, error))?;

    Ok((kind, reader))
}

fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues> {
    let (kind, reader) = projected_reader(&key.path, &key.column, name)?;
    match kind {
        ColumnKind::UInt32 => {
            let mut values = RoaringBitmap::new();
            for batch in reader {
                let batch = batch.map_err(|error| parquet_error(name, &key.path, error.into()))?;
                let column = batch.column(0).as_primitive::<UInt32Type>();
                for row in 0..column.len() {
                    if !column.is_null(row) {
                        values.insert(column.value(row));
                    }
                }
            }
            if values.is_empty() {
                return exec_err!("{name} requires at least one non-null source value");
            }
            Ok(ColumnValues::UInt32(values))
        }
        ColumnKind::UInt64 => {
            let mut values = RoaringTreemap::new();
            for batch in reader {
                let batch = batch.map_err(|error| parquet_error(name, &key.path, error.into()))?;
                let column = batch.column(0).as_primitive::<UInt64Type>();
                for row in 0..column.len() {
                    if !column.is_null(row) {
                        values.insert(column.value(row));
                    }
                }
            }
            if values.is_empty() {
                return exec_err!("{name} requires at least one non-null source value");
            }
            Ok(ColumnValues::UInt64(values))
        }
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

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("{} uses return_field_from_args", self.name())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        if args.arg_fields.len() != 2 {
            return plan_err!("{} expects exactly two arguments", self.name());
        }

        let path = scalar_string_arg(args.scalar_arguments, 0, "source_path", self.name())?;
        let column = scalar_string_arg(args.scalar_arguments, 1, "column_name", self.name())?;
        let data_type = match parquet_column_kind(path, column, self.name())? {
            ColumnKind::UInt32 => DataType::UInt32,
            ColumnKind::UInt64 => DataType::UInt64,
        };

        Ok(Arc::new(Field::new(self.name(), data_type, false)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            number_rows,
            return_field,
            ..
        } = args;
        let [path, column] = crate::randgen::utils::exact_args(args, self.name())?;
        let path = columnar_scalar_string_arg(&path, "source_path", self.name())?;
        let column = columnar_scalar_string_arg(&column, "column_name", self.name())?;
        let key = file_key(path, column, self.name())?;

        let values = if let Some(values) = self
            .cache
            .lock()
            .map_err(|_| {
                DataFusionError::Execution(format!("{} cache lock poisoned", self.name()))
            })?
            .get(&key)
            .cloned()
        {
            values
        } else {
            let loaded_values = Arc::new(load_values(&key, self.name())?);
            let mut cache = self.cache.lock().map_err(|_| {
                DataFusionError::Execution(format!("{} cache lock poisoned", self.name()))
            })?;
            Arc::clone(cache.entry(key).or_insert(loaded_values))
        };

        if values.data_type() != *return_field.data_type() {
            return internal_err!(
                "{} planned return type {} but loaded {}",
                self.name(),
                return_field.data_type(),
                values.data_type()
            );
        }

        values.sample(number_rows, self.name())
    }
}
