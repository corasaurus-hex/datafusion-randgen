use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field};
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef as VortexArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::{ArrowSessionExt, FromArrowArray};
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::ByteBufferMut;
use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

const HIDDEN_I64: i64 = 0x0123_4567_89ab_cdef;

#[tokio::main]
async fn main() -> vortex::error::VortexResult<()> {
    let arrow_input = Int64Array::new(
        vec![10, HIDDEN_I64, 30].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    assert!(arrow_input.is_null(1));
    assert_eq!(arrow_input.value(1), HIDDEN_I64);

    let vortex_input = VortexArrayRef::from_arrow(&arrow_input, true)?;
    let session = VortexSession::default().with_tokio();

    let mut output = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut output, vortex_input.to_array_stream())
        .await?;

    let file = session.open_options().open_buffer(output)?;
    let vortex_read_back = file.scan()?.into_array_stream()?.read_all().await?;

    let mut ctx = session.create_execution_ctx();
    let target = Field::new("value", DataType::Int64, true);
    let arrow_read_back =
        session
            .arrow()
            .execute_arrow(vortex_read_back, Some(&target), &mut ctx)?;
    let values = arrow_read_back.as_primitive::<Int64Type>();

    assert!(values.is_null(1));
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![Some(10), None, Some(30)]
    );
    assert_eq!(values.value(1), HIDDEN_I64);

    println!("logical values: {:?}", values.iter().collect::<Vec<_>>());
    println!("physical null-slot value: {}", values.value(1));
    println!("vortex_preserves_hidden_value=true");

    Ok(())
}
