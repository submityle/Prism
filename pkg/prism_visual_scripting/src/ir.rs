//! 扁平指令 IR：`SlotProgram` + 可序列化中间形态 `ExecutionPlan`。
//!
//! Pipeline:
//! ```text
//! Graph (源, serde)  ──►  ExecutionPlan (计划, serde, 跨进程稳定)  ──►  SlotProgram (程序, in-proc, Arc<NodeDefinition>)
//!         compile::plan                       compile::instantiate
//! ```
//!
//! - **ExecutionPlan** 只包含纯数据：把 `def_id` 写成字符串，不携带 runner
//!   trait object。可被缓存、签名、走远程执行器。
//! - **SlotProgram** 是 instantiate 后的最终形态，VM 直接解释。带
//!   `Arc<NodeDefinition>` 因此不可 serde。
//!
//! 这样把"是什么"（Plan）和"怎么跑"（Program）解耦，避免对 trait object 序列化。

use crate::model::{GraphCall, JoinPolicy};
use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crate::error::{VsError, VsResult};
use crate::registry::NodeDefinition;

pub type SlotId = u32;
pub type BlockId = u32;
pub type DefId = u32;
pub type VarId = u32;

#[derive(Clone, Serialize, Deserialize)]
pub enum Instr {
    /// 把字面量装入 slot。
    LoadConst {
        dst: SlotId,
        value: Value,
    },
    /// 复制 slot。
    Move {
        dst: SlotId,
        src: SlotId,
    },
    /// 读 Graph 变量。
    LoadVar {
        dst: SlotId,
        var: VarId,
    },
    /// 写 Graph 变量。
    StoreVar {
        var: VarId,
        src: SlotId,
    },
    /// 调用 Pure 节点：直接计算并写出 outputs。无控制流后果。
    CallPure {
        def: DefId,
        node_id: u32,
        inputs: Range<SlotId>,
        outputs: Range<SlotId>,
        /// 节点实例的静态配置；必须进入可持久化执行计划。
        config: BTreeMap<String, Value>,
    },
    /// 调用 Impure 节点：执行后由 [`super::vm`] 根据返回的 ExecNext 选择下一块。
    CallImpure {
        def: DefId,
        node_id: u32,
        inputs: Range<SlotId>,
        outputs: Range<SlotId>,
        /// 节点实例的静态配置；必须进入可持久化执行计划。
        config: BTreeMap<String, Value>,
        /// (exec pin name -> 目标 block)，按顺序匹配。
        targets: Vec<(String, BlockId)>,
        /// 当 ExecNext::End 时跳转的 block；可能是 sentinel（程序结束）。
        on_end: BlockId,
        /// `ExecNext::Error` 的错误处理分支。
        error_target: Option<BlockId>,
    },
    /// 无条件跳转。
    Jump {
        target: BlockId,
    },
    FrameEntry {
        node_id: u32,
        outputs: Range<SlotId>,
        target: BlockId,
    },
    Return {
        node_id: u32,
        inputs: Range<SlotId>,
    },
    CallGraph {
        node_id: u32,
        target_graph: GraphCall,
        inputs: Range<SlotId>,
        outputs: Range<SlotId>,
        target: BlockId,
        error_target: Option<BlockId>,
    },
    ForkGraphs {
        node_id: u32,
        branches: Vec<GraphCall>,
        policy: JoinPolicy,
        max_concurrency: u16,
        inputs: Range<SlotId>,
        outputs: Range<SlotId>,
        target: BlockId,
        error_target: Option<BlockId>,
    },
    Transition {
        node_id: u32,
        state: String,
    },
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Block {
    pub instrs: Vec<Instr>,
}

/// 可序列化的执行计划。
///
/// 内容与 [`SlotProgram`] 一一对应，但用 `def_ids: Vec<String>` 取代
/// `defs: Vec<Arc<NodeDefinition>>`，因此可以走 CBOR / JSON / 数据库存储 /
/// 远程 RPC，避免 trait-object 序列化问题。
///
/// 通过 [`crate::compiler::instantiate`] 在某个具体 `NodeLibrary` 下还原成
/// `SlotProgram`。Plan 的内容哈希 + Library 的 `RegistryVersion` 共同决定一份
/// 编译产物的身份。
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub slots_count: u32,
    pub blocks: Vec<Block>,
    /// def_id 字符串表（按 idx 索引）。运行期根据 NodeLibrary 解析为
    /// `Arc<NodeDefinition>`。
    pub def_ids: Vec<String>,
    /// var_id -> 变量名（运行期通过 host.var_get/var_set 寻址）。
    pub vars: Vec<String>,
    /// 入口 block：event 节点 id -> block。
    pub entries: BTreeMap<u32, BlockId>,
    /// 编译时 NodeRegistry 的稳定身份。
    pub registry_digest: [u8; 32],
}

impl ExecutionPlan {
    pub const HALT: BlockId = u32::MAX;
}

#[derive(Clone, Default)]
pub struct SlotProgram {
    pub slots_count: u32,
    pub blocks: Vec<Block>,
    /// def_id -> NodeDefinition（按 idx 索引）。
    pub defs: Vec<Arc<NodeDefinition>>,
    /// var_id -> 变量名（运行期通过 host.var_get/var_set 寻址）。
    pub vars: Vec<String>,
    /// 入口 block：event 节点 id -> block。
    pub entries: BTreeMap<u32, BlockId>,
    pub registry_digest: [u8; 32],
}

impl SlotProgram {
    pub const HALT: BlockId = u32::MAX;

    /// Validate the in-process program before handing it to the interpreter.
    pub fn validate(&self) -> VsResult<()> {
        let valid_target =
            |target: BlockId| target == Self::HALT || (target as usize) < self.blocks.len();
        let valid_range = |start: SlotId, end: SlotId| start <= end && end <= self.slots_count;
        for (block_id, block) in self.blocks.iter().enumerate() {
            for instr in &block.instrs {
                let valid_slot = |slot: SlotId| slot < self.slots_count;
                match instr {
                    Instr::LoadConst { dst, .. }
                    | Instr::LoadVar { dst, .. }
                    | Instr::StoreVar { src: dst, .. } => {
                        if !valid_slot(*dst) {
                            return Err(VsError::Validate("IR slot index out of bounds".into()));
                        }
                    }
                    Instr::Move { dst, src } => {
                        if !valid_slot(*dst) || !valid_slot(*src) {
                            return Err(VsError::Validate("IR slot index out of bounds".into()));
                        }
                    }
                    _ => {}
                }
                let (ranges, targets): (Vec<(SlotId, SlotId)>, Vec<BlockId>) = match instr {
                    Instr::LoadConst { dst, .. } => {
                        (vec![(*dst, dst.saturating_add(1))], Vec::new())
                    }
                    Instr::Move { dst, src } => (
                        vec![(*dst, dst.saturating_add(1)), (*src, src.saturating_add(1))],
                        Vec::new(),
                    ),
                    Instr::LoadVar { dst, var } => {
                        if (*var as usize) >= self.vars.len() {
                            return Err(VsError::Validate("IR variable index out of range".into()));
                        }
                        (vec![(*dst, dst.saturating_add(1))], Vec::new())
                    }
                    Instr::StoreVar { var, src } => {
                        if (*var as usize) >= self.vars.len() {
                            return Err(VsError::Validate("IR variable index out of range".into()));
                        }
                        (vec![(*src, src.saturating_add(1))], Vec::new())
                    }
                    Instr::CallPure {
                        def,
                        inputs,
                        outputs,
                        ..
                    } => {
                        if (*def as usize) >= self.defs.len() {
                            return Err(VsError::Validate(
                                "IR definition index out of range".into(),
                            ));
                        }
                        (
                            vec![(inputs.start, inputs.end), (outputs.start, outputs.end)],
                            Vec::new(),
                        )
                    }
                    Instr::CallImpure {
                        def,
                        inputs,
                        outputs,
                        targets,
                        on_end,
                        error_target,
                        ..
                    } => {
                        if (*def as usize) >= self.defs.len() {
                            return Err(VsError::Validate(
                                "IR definition index out of range".into(),
                            ));
                        }
                        let mut targets = targets
                            .iter()
                            .map(|(_, target)| *target)
                            .collect::<Vec<_>>();
                        targets.push(*on_end);
                        if let Some(target) = error_target {
                            targets.push(*target);
                        }
                        (
                            vec![(inputs.start, inputs.end), (outputs.start, outputs.end)],
                            targets,
                        )
                    }
                    Instr::Jump { target } => (Vec::new(), vec![*target]),
                    Instr::FrameEntry {
                        outputs, target, ..
                    } => (vec![(outputs.start, outputs.end)], vec![*target]),
                    Instr::Return { inputs, .. } => (vec![(inputs.start, inputs.end)], Vec::new()),
                    Instr::CallGraph {
                        inputs,
                        outputs,
                        target,
                        error_target,
                        ..
                    } => {
                        let mut targets = vec![*target];
                        if let Some(target) = error_target {
                            targets.push(*target);
                        }
                        (
                            vec![(inputs.start, inputs.end), (outputs.start, outputs.end)],
                            targets,
                        )
                    }
                    Instr::ForkGraphs {
                        inputs,
                        outputs,
                        target,
                        error_target,
                        ..
                    } => {
                        let mut targets = vec![*target];
                        if let Some(target) = error_target {
                            targets.push(*target);
                        }
                        (
                            vec![(inputs.start, inputs.end), (outputs.start, outputs.end)],
                            targets,
                        )
                    }
                    Instr::Transition { .. } => (Vec::new(), Vec::new()),
                };
                if ranges.iter().any(|(start, end)| !valid_range(*start, *end)) {
                    return Err(VsError::Validate(format!(
                        "IR slot range out of bounds in block {block_id}"
                    )));
                }
                if targets.iter().any(|target| !valid_target(*target)) {
                    return Err(VsError::Validate(format!(
                        "IR block target out of bounds in block {block_id}"
                    )));
                }
            }
        }
        for target in self.entries.values() {
            if !valid_target(*target) {
                return Err(VsError::Validate("IR entry target out of bounds".into()));
            }
        }
        Ok(())
    }

    /// 把 in-proc 程序投影回可序列化 [`ExecutionPlan`]。无损：blocks / vars /
    /// entries 直接克隆，`defs` 投影为 `def_ids`。
    pub fn to_plan(&self) -> ExecutionPlan {
        ExecutionPlan {
            slots_count: self.slots_count,
            blocks: self.blocks.clone(),
            def_ids: self.defs.iter().map(|d| d.id.clone()).collect(),
            vars: self.vars.clone(),
            entries: self.entries.clone(),
            registry_digest: self.registry_digest,
        }
    }
}
