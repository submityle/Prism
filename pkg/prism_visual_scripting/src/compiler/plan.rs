//! Plan 阶段：Graph + NodeRegistry → [`ExecutionPlan`]（纯数据）。
//!
//! 与历史 `compile()` 区别：plan() 输出可序列化的 `ExecutionPlan`，
//! `defs` 用 def_id 字符串表表示而非 `Arc<NodeDefinition>`。后续 instantiate
//! 阶段才在具体 registry 下绑定 trait object。

use super::context::LowerCtx;
use super::{lower, validate};
use crate::error::VsResult;
use crate::ir::ExecutionPlan;
use crate::model::Graph;
use crate::registry::NodeRegistry;

pub fn plan(graph: &Graph, registry: &dyn NodeRegistry) -> VsResult<ExecutionPlan> {
    let mut cx = LowerCtx::new(graph, registry)?;
    validate::run(&cx)?;
    cx.allocate_impure_outputs();
    lower::run(&mut cx)?;
    let program = cx.finish();
    Ok(program.to_plan())
}
