//! Graph -> ExecutionPlan -> SlotProgram 编译。
//!
//! 阶段：
//! 1. **resolve**：把 `Node.def.id` 解析到 `NodeDefinition`。
//! 2. **validate**：类型检查、Wildcard 拒绝、循环检测（[`validate`] 模块）。
//! 3. **lower**：每个 impure 节点 → 一个 Block，pure 依赖按反向拓扑展开
//!    为 `CallPure`（[`lower`] 模块）。
//! 4. **plan**：从 LowerCtx 投影出可序列化 [`ExecutionPlan`]。
//! 5. **instantiate**：在某个 [`NodeRegistry`] 下把 Plan 还原为 [`SlotProgram`]。
//!
//! 这样保证 `Plan` 是纯数据（可缓存、可签名、可远程），`Program` 才持有
//! `Arc<NodeDefinition>` 等 trait object 资源。

mod context;
mod instantiate;
mod lower;
mod plan;
mod validate;

use crate::error::VsResult;
use crate::ir::{ExecutionPlan, SlotProgram};
use crate::model::Graph;
use crate::registry::{NodeLibrary, NodeRegistry};

pub use instantiate::instantiate;
pub use plan::plan;

use context::LowerCtx;

/// 单步编译入口（兼容旧接口）。等价于 `instantiate(plan(graph, lib)?, lib)`。
pub fn compile(graph: &Graph, lib: &NodeLibrary) -> VsResult<SlotProgram> {
    let mut cx = LowerCtx::new(graph, lib)?;
    validate::run(&cx)?;
    cx.allocate_impure_outputs();
    lower::run(&mut cx)?;
    Ok(cx.finish())
}

/// 显式两阶段：Graph -> ExecutionPlan。
pub fn compile_to_plan(graph: &Graph, lib: &dyn NodeRegistry) -> VsResult<ExecutionPlan> {
    plan(graph, lib)
}
