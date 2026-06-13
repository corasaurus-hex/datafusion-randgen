//! Column choice random generator.
//!
//! `randgen_column_choice(source_path, column_name[, null_probability])`
//! samples with replacement from the distinct non-null values in an unsigned
//! integer columnar source column. The path and column arguments must be scalar
//! strings known at planning time. The UDF detects supported file formats from
//! file contents instead of relying on file extensions.

use std::any::Any;
use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
#[cfg(feature = "column-choice-parquet")]
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::UNIX_EPOCH;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
#[cfg(feature = "column-choice-arrow-ipc")]
use arrow_ipc::reader::{FileReader as ArrowIpcFileReader, StreamReader as ArrowIpcStreamReader};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
#[cfg(feature = "column-choice-parquet")]
use parquet::arrow::ProjectionMask;
#[cfg(feature = "column-choice-parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "column-choice-parquet")]
use parquet::errors::ParquetError;
use rand::Rng;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::randgen::utils::{NullProbability, coerce_float64_argument, optional_args};

/// `ScalarUDFImpl` for `randgen_column_choice(source_path, column_name[, null_probability])`.
#[derive(Debug)]
pub struct ColumnChoice {
    signature: &'static Signature,
    cache: Mutex<HashMap<CacheKey, Arc<ColumnValues>>>,
}

static COLUMN_CHOICE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::user_defined(Volatility::Volatile));

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    source_format: ColumnChoiceFormat,
    path: String,
    column: String,
    file_len: u64,
    modified: Option<(u64, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ColumnChoiceFormat {
    #[cfg(feature = "column-choice-parquet")]
    Parquet,
    #[cfg(feature = "column-choice-arrow-ipc")]
    ArrowIpcFile,
    #[cfg(feature = "column-choice-arrow-ipc")]
    ArrowIpcStream,
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

trait ColumnChoiceSource {
    const FORMAT: ColumnChoiceFormat;

    fn is_format(path: &str, name: &str) -> Result<bool>;

    fn column_kind(path: &str, column: &str, name: &str) -> Result<ColumnKind>;

    fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues>;
}

enum ColumnValuesBuilder {
    UInt32(RoaringBitmap),
    UInt64(RoaringTreemap),
}

#[cfg(feature = "column-choice-parquet")]
struct ParquetSource;

#[cfg(feature = "column-choice-arrow-ipc")]
struct ArrowIpcFileSource;

#[cfg(feature = "column-choice-arrow-ipc")]
struct ArrowIpcStreamSource;

impl std::fmt::Display for ColumnChoiceFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "column-choice-parquet")]
            Self::Parquet => formatter.write_str("Parquet"),
            #[cfg(feature = "column-choice-arrow-ipc")]
            Self::ArrowIpcFile => formatter.write_str("Arrow IPC file"),
            #[cfg(feature = "column-choice-arrow-ipc")]
            Self::ArrowIpcStream => formatter.write_str("Arrow IPC stream"),
        }
    }
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

impl ColumnValuesBuilder {
    fn new(kind: ColumnKind) -> Self {
        match kind {
            ColumnKind::UInt32 => Self::UInt32(RoaringBitmap::new()),
            ColumnKind::UInt64 => Self::UInt64(RoaringTreemap::new()),
        }
    }

    fn append_projected_batch(&mut self, batch: &RecordBatch, name: &str) -> Result<()> {
        let Some(column) = batch.columns().first() else {
            return internal_err!("{name} projected source reader returned no columns");
        };

        match self {
            Self::UInt32(values) => {
                if column.data_type() != &DataType::UInt32 {
                    return internal_err!(
                        "{name} projected source column changed type from UInt32 to {}",
                        column.data_type()
                    );
                }
                let column = column.as_primitive::<UInt32Type>();
                for row in 0..column.len() {
                    if !column.is_null(row) {
                        values.insert(column.value(row));
                    }
                }
            }
            Self::UInt64(values) => {
                if column.data_type() != &DataType::UInt64 {
                    return internal_err!(
                        "{name} projected source column changed type from UInt64 to {}",
                        column.data_type()
                    );
                }
                let column = column.as_primitive::<UInt64Type>();
                for row in 0..column.len() {
                    if !column.is_null(row) {
                        values.insert(column.value(row));
                    }
                }
            }
        }

        Ok(())
    }

    fn finish(self, name: &str) -> Result<ColumnValues> {
        match self {
            Self::UInt32(values) => {
                if values.is_empty() {
                    return exec_err!("{name} requires at least one non-null source value");
                }
                Ok(ColumnValues::UInt32(values))
            }
            Self::UInt64(values) => {
                if values.is_empty() {
                    return exec_err!("{name} requires at least one non-null source value");
                }
                Ok(ColumnValues::UInt64(values))
            }
        }
    }
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

fn open_source_file(path: &str, name: &str) -> Result<File> {
    File::open(path).map_err(|error| {
        DataFusionError::Execution(format!(
            "{name} could not open column choice source {path}: {error}"
        ))
    })
}

fn file_key(
    path: String,
    column: String,
    source_format: ColumnChoiceFormat,
    name: &str,
) -> Result<CacheKey> {
    let metadata = fs::metadata(&path).map_err(|error| {
        DataFusionError::Execution(format!("{name} could not stat {path}: {error}"))
    })?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| (duration.as_secs(), duration.subsec_nanos()));

    Ok(CacheKey {
        source_format,
        path,
        column,
        file_len: metadata.len(),
        modified,
    })
}

fn column_kind_for_field(data_type: &DataType, name: &str) -> Result<ColumnKind> {
    match data_type {
        DataType::UInt32 => Ok(ColumnKind::UInt32),
        DataType::UInt64 => Ok(ColumnKind::UInt64),
        data_type => exec_err!("{name} supports UInt32 and UInt64 columns, got {data_type}"),
    }
}

fn projected_column_kind(
    schema: &arrow_schema::Schema,
    column: &str,
    name: &str,
) -> Result<(usize, ColumnKind)> {
    let (column_index, field) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == column)
        .ok_or_else(|| {
            DataFusionError::Execution(format!("{name} could not find column {column}"))
        })?;

    Ok((
        column_index,
        column_kind_for_field(field.data_type(), name)?,
    ))
}

fn detect_source_format(path: &str, name: &str) -> Result<ColumnChoiceFormat> {
    #[cfg(feature = "column-choice-parquet")]
    if ParquetSource::is_format(path, name)? {
        return Ok(ParquetSource::FORMAT);
    }

    #[cfg(feature = "column-choice-arrow-ipc")]
    if ArrowIpcFileSource::is_format(path, name)? {
        return Ok(ArrowIpcFileSource::FORMAT);
    }

    #[cfg(feature = "column-choice-arrow-ipc")]
    if ArrowIpcStreamSource::is_format(path, name)? {
        return Ok(ArrowIpcStreamSource::FORMAT);
    }

    exec_err!(
        "{name} could not identify source format for {path}; enabled formats: {}",
        enabled_source_formats()
    )
}

fn enabled_source_formats() -> &'static str {
    #[cfg(all(feature = "column-choice-parquet", feature = "column-choice-arrow-ipc"))]
    {
        "Parquet, Arrow IPC file, Arrow IPC stream"
    }

    #[cfg(all(
        feature = "column-choice-parquet",
        not(feature = "column-choice-arrow-ipc")
    ))]
    {
        "Parquet"
    }

    #[cfg(all(
        not(feature = "column-choice-parquet"),
        feature = "column-choice-arrow-ipc"
    ))]
    {
        "Arrow IPC file, Arrow IPC stream"
    }
}

fn detect_column_kind(
    path: &str,
    column: &str,
    name: &str,
) -> Result<(ColumnChoiceFormat, ColumnKind)> {
    let source_format = detect_source_format(path, name)?;
    Ok((
        source_format,
        column_kind_for_source(source_format, path, column, name)?,
    ))
}

fn column_kind_for_source(
    source_format: ColumnChoiceFormat,
    path: &str,
    column: &str,
    name: &str,
) -> Result<ColumnKind> {
    match source_format {
        #[cfg(feature = "column-choice-parquet")]
        ColumnChoiceFormat::Parquet => ParquetSource::column_kind(path, column, name),
        #[cfg(feature = "column-choice-arrow-ipc")]
        ColumnChoiceFormat::ArrowIpcFile => ArrowIpcFileSource::column_kind(path, column, name),
        #[cfg(feature = "column-choice-arrow-ipc")]
        ColumnChoiceFormat::ArrowIpcStream => ArrowIpcStreamSource::column_kind(path, column, name),
    }
}

fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues> {
    match key.source_format {
        #[cfg(feature = "column-choice-parquet")]
        ColumnChoiceFormat::Parquet => ParquetSource::load_values(key, name),
        #[cfg(feature = "column-choice-arrow-ipc")]
        ColumnChoiceFormat::ArrowIpcFile => ArrowIpcFileSource::load_values(key, name),
        #[cfg(feature = "column-choice-arrow-ipc")]
        ColumnChoiceFormat::ArrowIpcStream => ArrowIpcStreamSource::load_values(key, name),
    }
}

#[cfg(feature = "column-choice-parquet")]
fn parquet_error(name: &str, path: &str, error: ParquetError) -> DataFusionError {
    DataFusionError::Execution(format!(
        "{name} could not read Parquet source {path}: {error}"
    ))
}

#[cfg(feature = "column-choice-parquet")]
fn parquet_magic_matches(path: &str, name: &str) -> Result<bool> {
    const PARQUET_MAGIC: &[u8] = b"PAR1";

    let mut file = open_source_file(path, name)?;
    if file.metadata()?.len() < (PARQUET_MAGIC.len() * 2) as u64 {
        return Ok(false);
    }

    let mut start = [0_u8; 4];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut start)?;
    if start != PARQUET_MAGIC {
        return Ok(false);
    }

    let mut end = [0_u8; 4];
    file.seek(SeekFrom::End(-(PARQUET_MAGIC.len() as i64)))?;
    file.read_exact(&mut end)?;

    Ok(end == PARQUET_MAGIC)
}

#[cfg(feature = "column-choice-parquet")]
fn parquet_projected_reader(
    path: &str,
    column: &str,
    name: &str,
) -> Result<(
    ColumnKind,
    parquet::arrow::arrow_reader::ParquetRecordBatchReader,
)> {
    let file = open_source_file(path, name)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| parquet_error(name, path, error))?;
    let schema = builder.schema();
    let (column_index, kind) = projected_column_kind(schema.as_ref(), column, name)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [column_index]);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|error| parquet_error(name, path, error))?;

    Ok((kind, reader))
}

#[cfg(feature = "column-choice-parquet")]
impl ColumnChoiceSource for ParquetSource {
    const FORMAT: ColumnChoiceFormat = ColumnChoiceFormat::Parquet;

    fn is_format(path: &str, name: &str) -> Result<bool> {
        parquet_magic_matches(path, name)
    }

    fn column_kind(path: &str, column: &str, name: &str) -> Result<ColumnKind> {
        let file = open_source_file(path, name)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|error| parquet_error(name, path, error))?;
        let (_, kind) = projected_column_kind(builder.schema().as_ref(), column, name)?;
        Ok(kind)
    }

    fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues> {
        let (kind, reader) = parquet_projected_reader(&key.path, &key.column, name)?;
        let mut values = ColumnValuesBuilder::new(kind);
        for batch in reader {
            let batch = batch.map_err(|error| parquet_error(name, &key.path, error.into()))?;
            values.append_projected_batch(&batch, name)?;
        }
        values.finish(name)
    }
}

#[cfg(feature = "column-choice-arrow-ipc")]
fn arrow_ipc_error(
    name: &str,
    path: &str,
    format: ColumnChoiceFormat,
    error: arrow_schema::ArrowError,
) -> DataFusionError {
    DataFusionError::Execution(format!(
        "{name} could not read {format} source {path}: {error}"
    ))
}

#[cfg(feature = "column-choice-arrow-ipc")]
impl ColumnChoiceSource for ArrowIpcFileSource {
    const FORMAT: ColumnChoiceFormat = ColumnChoiceFormat::ArrowIpcFile;

    fn is_format(path: &str, name: &str) -> Result<bool> {
        let file = open_source_file(path, name)?;
        Ok(ArrowIpcFileReader::try_new(file, None).is_ok())
    }

    fn column_kind(path: &str, column: &str, name: &str) -> Result<ColumnKind> {
        let file = open_source_file(path, name)?;
        let reader = ArrowIpcFileReader::try_new(file, None)
            .map_err(|error| arrow_ipc_error(name, path, Self::FORMAT, error))?;
        let (_, kind) = projected_column_kind(reader.schema().as_ref(), column, name)?;
        Ok(kind)
    }

    fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues> {
        let file = open_source_file(&key.path, name)?;
        let reader = ArrowIpcFileReader::try_new(file, None)
            .map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;
        let (column_index, kind) =
            projected_column_kind(reader.schema().as_ref(), &key.column, name)?;
        let file = open_source_file(&key.path, name)?;
        let reader = ArrowIpcFileReader::try_new(file, Some(vec![column_index]))
            .map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;

        let mut values = ColumnValuesBuilder::new(kind);
        for batch in reader {
            let batch =
                batch.map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;
            values.append_projected_batch(&batch, name)?;
        }
        values.finish(name)
    }
}

#[cfg(feature = "column-choice-arrow-ipc")]
impl ColumnChoiceSource for ArrowIpcStreamSource {
    const FORMAT: ColumnChoiceFormat = ColumnChoiceFormat::ArrowIpcStream;

    fn is_format(path: &str, name: &str) -> Result<bool> {
        let file = open_source_file(path, name)?;
        Ok(ArrowIpcStreamReader::try_new(file, None).is_ok())
    }

    fn column_kind(path: &str, column: &str, name: &str) -> Result<ColumnKind> {
        let file = open_source_file(path, name)?;
        let reader = ArrowIpcStreamReader::try_new(file, None)
            .map_err(|error| arrow_ipc_error(name, path, Self::FORMAT, error))?;
        let (_, kind) = projected_column_kind(reader.schema().as_ref(), column, name)?;
        Ok(kind)
    }

    fn load_values(key: &CacheKey, name: &str) -> Result<ColumnValues> {
        let file = open_source_file(&key.path, name)?;
        let reader = ArrowIpcStreamReader::try_new(file, None)
            .map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;
        let (column_index, kind) =
            projected_column_kind(reader.schema().as_ref(), &key.column, name)?;
        let file = open_source_file(&key.path, name)?;
        let reader = ArrowIpcStreamReader::try_new(file, Some(vec![column_index]))
            .map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;

        let mut values = ColumnValuesBuilder::new(kind);
        for batch in reader {
            let batch =
                batch.map_err(|error| arrow_ipc_error(name, &key.path, Self::FORMAT, error))?;
            values.append_projected_batch(&batch, name)?;
        }
        values.finish(name)
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
        if args.arg_fields.len() != 2 && args.arg_fields.len() != 3 {
            return plan_err!("{} expects two or three arguments", self.name());
        }

        let path = scalar_string_arg(args.scalar_arguments, 0, "source_path", self.name())?;
        let column = scalar_string_arg(args.scalar_arguments, 1, "column_name", self.name())?;
        let (_, kind) = detect_column_kind(path, column, self.name())?;
        let data_type = match kind {
            ColumnKind::UInt32 => DataType::UInt32,
            ColumnKind::UInt64 => DataType::UInt64,
        };

        Ok(Arc::new(Field::new(
            self.name(),
            data_type,
            args.arg_fields.len() == 3,
        )))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 2 && arg_types.len() != 3 {
            return exec_err!("{} expects two or three arguments", self.name());
        }

        let mut coerced = vec![DataType::Utf8, DataType::Utf8];
        if let Some(null_probability_type) = arg_types.get(2) {
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
        let ([path, column], null_probability) = optional_args(args, self.name())?;
        let null_probability =
            NullProbability::from_optional_arg(null_probability, number_rows, self.name())?;
        let path = columnar_scalar_string_arg(&path, "source_path", self.name())?;
        let column = columnar_scalar_string_arg(&column, "column_name", self.name())?;
        let source_format = detect_source_format(&path, self.name())?;
        let key = file_key(path, column, source_format, self.name())?;

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

        values.sample(number_rows, self.name(), &null_probability)
    }
}
