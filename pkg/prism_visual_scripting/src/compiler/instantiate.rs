//! Instantiate 阶段：[`ExecutionPlan`] + NodeRegistry → [`SlotProgram`]。
//!
//! 把可序列化的 plan 在具体 registry 下还原成 in-proc 程序。失败原因通常是
//! plan 引用的 def_id 不在当前 registry 中（registry 升级 / 缺失 ability）。

use crate::error::{VsError, VsResult};
use crate::ir::{ExecutionPlan, SlotProgram};
use crate::registry::NodeRegistry;

pub fn instantiate(plan: &ExecutionPlan, registry: &dyn NodeRegistry) -> VsResult<SlotProgram> {
    if plan.registry_digest != registry.registry_digest() {
        return Err(VsError::Runtime("node registry identity mismatch".into()));
    }
    let mut defs = Vec::with_capacity(plan.def_ids.len());
    for id in &plan.def_ids {
        match registry.get(id) {
            Some(d) => defs.push(d),
            None => return Err(VsError::UnknownDef(id.clone())),
        }
    }
    Ok(SlotProgram {
        slots_count: plan.slots_count,
        blocks: plan.blocks.clone(),
        defs,
        vars: plan.vars.clone(),
        entries: plan.entries.clone(),
        registry_digest: plan.registry_digest,
    })
}
