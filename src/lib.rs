use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;

use crate::randgen::bool::Bool;
use crate::randgen::choice::Choice;
use crate::randgen::date32::Date32;
use crate::randgen::float64_normal::Float64Normal;
use crate::randgen::float64_uniform::Float64Uniform;
use crate::randgen::int64_uniform::Int64Uniform;
use crate::randgen::timestamp_millisecond::TimestampMillisecond;
use crate::randgen::utf8::Utf8;

pub mod randgen;

pub fn add_udfs(ctx: &mut SessionContext) {
    ctx.register_udf(ScalarUDF::from(Int64Uniform::new()));
    ctx.register_udf(ScalarUDF::from(Float64Uniform::new()));
    ctx.register_udf(ScalarUDF::from(Float64Normal::new()));
    ctx.register_udf(ScalarUDF::from(Bool::new()));
    ctx.register_udf(ScalarUDF::from(Utf8::new()));
    ctx.register_udf(ScalarUDF::from(Choice::new()));
    ctx.register_udf(ScalarUDF::from(Date32::new()));
    ctx.register_udf(ScalarUDF::from(TimestampMillisecond::new()));
}
