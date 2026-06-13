use std::io::Cursor;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::{ScalarValue, config::ConfigOptions};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_randgen::{Int64Uniform, Nullable};

const HIDDEN_I64: i64 = 0x0123_4567_89ab_cdef;
const HIDDEN_UTF8: &str = "NULL_SLOT_SECRET";

fn hidden_i64_array() -> Int64Array {
    Int64Array::new(
        vec![10, HIDDEN_I64, 30].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    )
}

fn hidden_utf8_array() -> StringArray {
    let values = format!("alpha{HIDDEN_UTF8}omega").into_bytes();
    let hidden_end = 5 + HIDDEN_UTF8.len() as i32;
    StringArray::new(
        OffsetBuffer::new(vec![0_i32, 5, hidden_end, hidden_end + 5].into()),
        Buffer::from(values),
        Some(NullBuffer::from(vec![true, false, true])),
    )
}

fn record_batch(field: Field, array: ArrayRef) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn invoke_array(
    udf: &impl ScalarUDFImpl,
    args: Vec<ColumnarValue>,
    return_type: DataType,
    number_rows: usize,
) -> ArrayRef {
    let arg_fields = args
        .iter()
        .enumerate()
        .map(|(index, arg)| Arc::new(Field::new(format!("arg{index}"), arg.data_type(), true)))
        .collect();
    let result = udf
        .invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new(udf.name(), return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
        .unwrap();
    let ColumnarValue::Array(array) = result else {
        panic!("expected array result");
    };
    array
}

fn write_arrow_stream_and_read_back(batch: &RecordBatch) -> RecordBatch {
    let mut output = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut output, batch.schema_ref()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }

    let mut reader = StreamReader::try_new(Cursor::new(output), None).unwrap();
    reader.next().unwrap().unwrap()
}

#[test]
fn arrow_ipc_stream_preserves_primitive_values_under_nulls() {
    let batch = record_batch(
        Field::new("value", DataType::Int64, true),
        Arc::new(hidden_i64_array()),
    );

    let read_back = write_arrow_stream_and_read_back(&batch);
    let values = read_back.column(0).as_primitive::<Int64Type>();

    assert!(values.is_null(1));
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![Some(10), None, Some(30)]
    );
    assert_eq!(values.value(1), HIDDEN_I64);
}

#[test]
fn arrow_ipc_stream_preserves_utf8_bytes_under_nulls() {
    let batch = record_batch(
        Field::new("value", DataType::Utf8, true),
        Arc::new(hidden_utf8_array()),
    );

    let read_back = write_arrow_stream_and_read_back(&batch);
    let values = read_back.column(0).as_string::<i32>();

    assert!(values.is_null(1));
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![Some("alpha"), None, Some("omega")]
    );
    assert_eq!(values.value(1), HIDDEN_UTF8);
    assert!(
        values
            .value_data()
            .windows(HIDDEN_UTF8.len())
            .any(|window| window == HIDDEN_UTF8.as_bytes())
    );
}

#[test]
fn randgen_nullable_filters_primitive_values_under_nulls() {
    let filtered = invoke_array(
        &Nullable::new(),
        vec![
            ColumnarValue::Array(Arc::new(hidden_i64_array())),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
        ],
        DataType::Int64,
        3,
    );
    let batch = record_batch(Field::new("value", DataType::Int64, true), filtered);

    let read_back = write_arrow_stream_and_read_back(&batch);
    let values = read_back.column(0).as_primitive::<Int64Type>();

    assert!(values.is_null(1));
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![Some(10), None, Some(30)]
    );
    assert_ne!(values.value(1), HIDDEN_I64);
}

#[test]
fn native_generator_nullability_filters_primitive_values_under_nulls() {
    let filtered = invoke_array(
        &Int64Uniform::new(),
        vec![
            ColumnarValue::Scalar(ScalarValue::Int64(Some(HIDDEN_I64))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(HIDDEN_I64))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
        ],
        DataType::Int64,
        3,
    );
    let batch = record_batch(Field::new("value", DataType::Int64, true), filtered);

    let read_back = write_arrow_stream_and_read_back(&batch);
    let values = read_back.column(0).as_primitive::<Int64Type>();

    assert_eq!(values.iter().collect::<Vec<_>>(), vec![None, None, None]);
    for row in 0..values.len() {
        assert_ne!(values.value(row), HIDDEN_I64);
    }
}

#[cfg(feature = "column-choice-parquet")]
mod parquet_tests {
    use std::fs::{File, remove_file};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::column::reader::ColumnReader;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::{HIDDEN_I64, HIDDEN_UTF8, hidden_i64_array, hidden_utf8_array};

    fn unique_path(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "datafusion_randgen_{test_name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn write_parquet(test_name: &str, field: Field, column: ArrayRef) -> PathBuf {
        let path = unique_path(test_name);
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    #[test]
    fn parquet_arrow_writer_excludes_primitive_values_under_nulls() {
        let path = write_parquet(
            "null_overlay_i64",
            Field::new("value", DataType::Int64, true),
            Arc::new(hidden_i64_array()),
        );

        let file_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        let row_group = file_reader.get_row_group(0).unwrap();
        let mut column_reader = row_group.get_column_reader(0).unwrap();
        let ColumnReader::Int64ColumnReader(reader) = &mut column_reader else {
            panic!("expected int64 column reader");
        };
        let mut def_levels = Vec::new();
        let mut values = Vec::new();

        let read = reader
            .read_records(3, Some(&mut def_levels), None, &mut values)
            .unwrap();

        assert_eq!(read, (3, 2, 3));
        assert_eq!(def_levels, vec![1, 0, 1]);
        assert_eq!(values, vec![10, 30]);
        assert!(!values.contains(&HIDDEN_I64));

        remove_file(path).unwrap();
    }

    #[test]
    fn parquet_arrow_writer_excludes_utf8_values_under_nulls() {
        let path = write_parquet(
            "null_overlay_utf8",
            Field::new("value", DataType::Utf8, true),
            Arc::new(hidden_utf8_array()),
        );

        let file_reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        let row_group = file_reader.get_row_group(0).unwrap();
        let mut column_reader = row_group.get_column_reader(0).unwrap();
        let ColumnReader::ByteArrayColumnReader(reader) = &mut column_reader else {
            panic!("expected byte array column reader");
        };
        let mut def_levels = Vec::new();
        let mut values = Vec::new();

        let read = reader
            .read_records(3, Some(&mut def_levels), None, &mut values)
            .unwrap();
        let decoded = values
            .iter()
            .map(|value| std::str::from_utf8(value.data()).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(read, (3, 2, 3));
        assert_eq!(def_levels, vec![1, 0, 1]);
        assert_eq!(decoded, vec!["alpha", "omega"]);
        assert!(!decoded.contains(&HIDDEN_UTF8));

        remove_file(path).unwrap();
    }
}
