use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;

use crate::randgen::int64_uniform::Int64Uniform;

pub mod randgen;

pub fn add_udfs(ctx: &mut SessionContext) {
    ctx.register_udf(ScalarUDF::from(Int64Uniform::new()));
}
