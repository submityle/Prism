//! SlotProgram 解释器 —— 同步 + 异步双路径。
//!
//! 主循环：当前 block -> 顺序执行 instr -> 遇到 CallImpure 时根据 ExecNext
//! 跳到下一 block；其余 instr 都在 block 内顺序计算完毕。
//!
//! 同一个 [`SlotProgram`] 可以混合同步 / 异步 runner：
//! - [`Vm::run_entry`] / [`Vm::run_all`] 只能跑全部为 [`crate::RunnerKind::Sync`]
//!   的程序；遇到 [`crate::RunnerKind::Async`] 节点会返回 [`VsError::AsyncRunnerInSyncVm`]。
//! - [`Vm::run_entry_async`] / [`Vm::run_all_async`] 两种 runner 都能跑，
//!   并且可以接收一个可选的 `ChunkTx` 用于流式输出。

use crate::awaiting::AwaitCondition;
use crate::effect::{EffectRequest, ExecutionClock, TokioClock};
use crate::error::{VsError, VsResult};
use crate::execution::{ExecutionEvent, ExecutionEventKind};
use crate::ir::{BlockId, Instr, SlotId, SlotProgram};
use crate::model::{GraphCall, JoinPolicy};
use crate::registry::{
    ChunkTx, ExecNext, ExecutionEnv, HostServices, InvokeCtx, NodeCallInfo, NodePhase, RunnerKind,
};
use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// VM 默认 step budget。可通过 [`Vm::with_step_budget`] 覆盖。
pub const DEFAULT_STEP_BUDGET: u64 = 1_000_000;

/// 可跨进程保存的 VM continuation。状态只在 block 边界产生。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VmState {
    pub entry_node: u32,
    pub block: BlockId,
    pub slots: Vec<Value>,
    pub budget_remaining: Option<u64>,
    /// Execution identity is part of the continuation boundary. A resumed
    /// checkpoint must never be used by another execution.
    #[serde(default)]
    pub execution_id: Option<String>,
    #[serde(default)]
    pub(crate) next_effect_sequence: u64,
    #[serde(default)]
    pub(crate) pending_suspension: Option<PendingSuspension>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VmRunStatus {
    Completed,
    Cancelled(VmState),
    Suspended(VmState),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Suspension {
    pub id: String,
    #[serde(default)]
    pub condition: AwaitCondition,
    pub payload: Value,
}

impl Suspension {
    pub fn new(id: impl Into<String>, payload: Value) -> Self {
        Self {
            id: id.into(),
            condition: AwaitCondition::External,
            payload,
        }
    }

    pub fn with_condition(mut self, condition: AwaitCondition) -> Self {
        self.condition = condition;
        self
    }

    pub fn timer(id: impl Into<String>, deadline_ms: u64, payload: Value) -> Self {
        Self::new(id, payload).with_condition(AwaitCondition::Timer { deadline_ms })
    }

    pub fn signal(
        id: impl Into<String>,
        topic: impl Into<String>,
        correlation: impl Into<String>,
        payload: Value,
    ) -> Self {
        Self::new(id, payload).with_condition(AwaitCondition::Signal {
            topic: topic.into(),
            correlation: correlation.into(),
        })
    }

    pub fn approval(id: impl Into<String>, subject: impl Into<String>, payload: Value) -> Self {
        Self::new(id, payload).with_condition(AwaitCondition::Approval {
            subject: subject.into(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuspensionResolution {
    pub outputs: Vec<Value>,
    pub next: crate::ExecNext,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PendingSuspension {
    suspension: Suspension,
    outputs: std::ops::Range<SlotId>,
    targets: Vec<(String, BlockId)>,
    on_end: BlockId,
    error_target: Option<BlockId>,
}

impl VmState {
    pub fn suspension(&self) -> Option<&Suspension> {
        self.pending_suspension
            .as_ref()
            .map(|pending| &pending.suspension)
    }

    pub fn resume_suspension(
        &mut self,
        suspension_id: &str,
        resolution: SuspensionResolution,
    ) -> VsResult<()> {
        let pending = self
            .pending_suspension
            .take()
            .ok_or_else(|| VsError::Runtime("VM has no pending suspension".into()))?;
        if pending.suspension.id != suspension_id {
            self.pending_suspension = Some(pending);
            return Err(VsError::Runtime("suspension identity mismatch".into()));
        }
        let expected = match pending.outputs.end.checked_sub(pending.outputs.start) {
            Some(width) => width as usize,
            None => {
                self.pending_suspension = Some(pending);
                return Err(VsError::Runtime(
                    "suspension output slot range is reversed".into(),
                ));
            }
        };
        let end = pending.outputs.end as usize;
        if end > self.slots.len() {
            self.pending_suspension = Some(pending);
            return Err(VsError::Runtime(
                "suspension output slot range exceeds VM slots".into(),
            ));
        }
        if resolution.outputs.len() != expected {
            self.pending_suspension = Some(pending);
            return Err(VsError::Runtime(format!(
                "suspension output arity mismatch: expected {expected}, got {}",
                resolution.outputs.len()
            )));
        }
        let next_block = match resolve_next(
            resolution.next,
            &pending.targets,
            pending.on_end,
            pending.error_target,
        ) {
            Ok(block) => block,
            Err(error) => {
                self.pending_suspension = Some(pending);
                return Err(error);
            }
        };
        let target = self.slots.get_mut(pending.outputs.start as usize..end);
        let Some(target) = target else {
            self.pending_suspension = Some(pending);
            return Err(VsError::Runtime(
                "suspension output slot range is invalid".into(),
            ));
        };
        target.clone_from_slice(&resolution.outputs);
        self.block = next_block;
        Ok(())
    }

    pub(crate) fn set_pending_suspension(&mut self, pending: PendingSuspension) -> VsResult<()> {
        if self.pending_suspension.is_some() {
            return Err(VsError::Runtime(
                "VM already has a pending suspension".into(),
            ));
        }
        self.pending_suspension = Some(pending);
        Ok(())
    }

    pub(crate) fn validate_suspension(&self, program: &SlotProgram) -> VsResult<()> {
        let Some(pending) = &self.pending_suspension else {
            return Ok(());
        };
        let valid_block =
            |block: BlockId| block == SlotProgram::HALT || (block as usize) < program.blocks.len();
        if pending.suspension.id.is_empty()
            || pending.outputs.start > pending.outputs.end
            || pending.outputs.end > program.slots_count
            || !valid_block(pending.on_end)
            || pending
                .targets
                .iter()
                .any(|(_, block)| !valid_block(*block))
            || pending
                .error_target
                .is_some_and(|block| !valid_block(block))
        {
            return Err(VsError::Runtime(
                "resumed VM has an invalid pending suspension".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) enum VmStepOutcome {
    Next(BlockId),
    FrameEntry {
        outputs: std::ops::Range<SlotId>,
        target: BlockId,
    },
    Return(Vec<Value>),
    Call {
        target_graph: GraphCall,
        args: Vec<Value>,
        outputs: std::ops::Range<SlotId>,
        target: BlockId,
        error_target: Option<BlockId>,
    },
    Fork {
        branches: Vec<GraphCall>,
        policy: JoinPolicy,
        max_concurrency: u16,
        args: Vec<Value>,
        outputs: std::ops::Range<SlotId>,
        target: BlockId,
        error_target: Option<BlockId>,
    },
    Transition(String),
    Suspend(PendingSuspension),
}

pub struct Vm {
    program: Arc<SlotProgram>,
    step_budget: u64,
}

#[derive(Copy, Clone)]
pub(crate) struct InvocationScope<'a> {
    pub execution_id: Option<&'a str>,
    pub task_id: u64,
    pub frame_depth: usize,
}

impl Vm {
    pub fn new(program: SlotProgram) -> Self {
        Self {
            program: Arc::new(program),
            step_budget: DEFAULT_STEP_BUDGET,
        }
    }

    pub(crate) fn from_shared(program: Arc<SlotProgram>) -> Self {
        Self {
            program,
            step_budget: DEFAULT_STEP_BUDGET,
        }
    }

    /// 设置 step budget；0 表示不限制（仅推荐受信场景）。
    pub fn with_step_budget(mut self, budget: u64) -> Self {
        self.step_budget = budget;
        self
    }

    pub fn program(&self) -> &SlotProgram {
        &self.program
    }

    // ─────────────────────────── 同步入口 ───────────────────────────

    /// 同步执行：从指定 entry 节点出发。仅支持 Sync runner。
    pub fn run_entry(&self, entry_node: u32, host: &mut dyn HostServices) -> VsResult<()> {
        let cancel = CancellationToken::new();
        let env = ExecutionEnv::new().with_cancel(&cancel);
        self.run_entry_with_env(entry_node, host, &env)
    }

    /// 使用外部执行环境同步执行指定 entry。
    pub fn run_entry_with_env(
        &self,
        entry_node: u32,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<()> {
        match self.run_entry_resumable_with_env(entry_node, None, host, env)? {
            VmRunStatus::Completed => Ok(()),
            VmRunStatus::Cancelled(_) => Err(VsError::Cancelled),
            VmRunStatus::Suspended(state) => Err(VsError::Suspended(
                state
                    .suspension()
                    .map(|value| value.id.clone())
                    .unwrap_or_else(|| "unknown".into()),
            )),
        }
    }

    /// 同步执行所有 entry。
    pub fn run_all(&self, host: &mut dyn HostServices) -> VsResult<()> {
        let cancel = CancellationToken::new();
        let env = ExecutionEnv::new().with_cancel(&cancel);
        self.run_all_with_env(host, &env)
    }

    /// 使用外部执行环境同步执行所有 entry。
    pub fn run_all_with_env(
        &self,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<()> {
        for &entry in self.program.entries.keys() {
            self.run_entry_with_env(entry, host, env)?;
        }
        Ok(())
    }

    // ─────────────────────────── 异步入口 ───────────────────────────

    /// 异步执行：从指定 entry 节点出发。`chunk_tx` 不为 `None` 时即为流式模式，
    /// 节点可通过 [`crate::InvokeCtx::chunk_tx`] 推送 [`crate::Chunk`]。
    pub async fn run_entry_async(
        &self,
        entry_node: u32,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<()> {
        let mut env = ExecutionEnv::new().with_cancel(cancel);
        if let Some(chunk_tx) = chunk_tx {
            env = env.with_chunk_tx(chunk_tx);
        }
        self.run_entry_async_with_env(entry_node, host, &env).await
    }

    /// 使用外部执行环境异步执行指定 entry。
    pub async fn run_entry_async_with_env(
        &self,
        entry_node: u32,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<()> {
        match self
            .run_entry_resumable_async_with_env(entry_node, None, host, env)
            .await?
        {
            VmRunStatus::Completed => Ok(()),
            VmRunStatus::Cancelled(_) => Err(VsError::Cancelled),
            VmRunStatus::Suspended(state) => Err(VsError::Suspended(
                state
                    .suspension()
                    .map(|value| value.id.clone())
                    .unwrap_or_else(|| "unknown".into()),
            )),
        }
    }

    /// 异步执行所有 entry。
    pub async fn run_all_async(
        &self,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<()> {
        let mut env = ExecutionEnv::new().with_cancel(cancel);
        if let Some(chunk_tx) = chunk_tx {
            env = env.with_chunk_tx(chunk_tx);
        }
        self.run_all_async_with_env(host, &env).await
    }

    /// 使用外部执行环境异步执行所有 entry。
    pub async fn run_all_async_with_env(
        &self,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<()> {
        let entries: Vec<u32> = self.program.entries.keys().copied().collect();
        for entry in entries {
            self.run_entry_async_with_env(entry, host, env).await?;
        }
        Ok(())
    }

    /// 从 entry 开始执行，或从上一次 block 边界的状态恢复。
    pub fn run_entry_resumable_with_env(
        &self,
        entry_node: u32,
        state: Option<VmState>,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<VmRunStatus> {
        self.program.validate()?;
        let fallback_cancel = CancellationToken::new();
        let state = self.initial_state(entry_node, state, env.execution_id)?;
        self.run_from_sync(state, host, &fallback_cancel, env)
    }

    /// 异步版本的可恢复执行入口。
    pub async fn run_entry_resumable_async_with_env(
        &self,
        entry_node: u32,
        state: Option<VmState>,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<VmRunStatus> {
        self.program.validate()?;
        let fallback_cancel = CancellationToken::new();
        let state = self.initial_state(entry_node, state, env.execution_id)?;
        self.run_from_async(state, host, &fallback_cancel, env)
            .await
    }

    // ─────────────────────────── 内部 ───────────────────────────

    fn entry_block(&self, entry_node: u32) -> VsResult<BlockId> {
        self.program
            .entries
            .get(&entry_node)
            .copied()
            .ok_or_else(|| VsError::Runtime(format!("no entry block for node {}", entry_node)))
    }

    fn initial_state(
        &self,
        entry_node: u32,
        state: Option<VmState>,
        execution_id: Option<&str>,
    ) -> VsResult<VmState> {
        if let Some(state) = state {
            if state.entry_node != entry_node {
                return Err(VsError::Runtime("resumed state entry mismatch".into()));
            }
            if state.slots.len() != self.program.slots_count as usize {
                return Err(VsError::Runtime("resumed state slot count mismatch".into()));
            }
            if state.execution_id.as_deref() != execution_id {
                return Err(VsError::Runtime(
                    "resumed VM execution identity mismatch".into(),
                ));
            }
            state.validate_suspension(&self.program)?;
            return Ok(state);
        }
        Ok(VmState {
            entry_node,
            block: self.entry_block(entry_node)?,
            slots: vec![Value::Unit; self.program.slots_count as usize],
            budget_remaining: (self.step_budget != 0).then_some(self.step_budget),
            execution_id: execution_id.map(str::to_owned),
            next_effect_sequence: 0,
            pending_suspension: None,
        })
    }

    fn check_budget(&self, budget: &mut Option<u64>) -> VsResult<()> {
        let Some(remaining) = budget else {
            return Ok(());
        };
        if *remaining == 0 {
            return Err(VsError::Runtime("execution step budget exceeded".into()));
        }
        *remaining -= 1;
        Ok(())
    }

    fn run_from_sync(
        &self,
        mut state: VmState,
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<VmRunStatus> {
        while state.block != SlotProgram::HALT {
            if env.cancel.unwrap_or(fallback_cancel).is_cancelled() {
                return Ok(VmRunStatus::Cancelled(state));
            }
            self.check_budget(&mut state.budget_remaining)?;
            match self.run_block_sync(state.block, &mut state.slots, host, fallback_cancel, env)? {
                VmStepOutcome::Next(block) => state.block = block,
                VmStepOutcome::Suspend(pending) => {
                    state.pending_suspension = Some(pending);
                    return Ok(VmRunStatus::Suspended(state));
                }
                _ => return Err(structured_control_error()),
            }
        }
        Ok(VmRunStatus::Completed)
    }

    async fn run_from_async(
        &self,
        mut state: VmState,
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<VmRunStatus> {
        while state.block != SlotProgram::HALT {
            if env.cancel.unwrap_or(fallback_cancel).is_cancelled() {
                return Ok(VmRunStatus::Cancelled(state));
            }
            self.check_budget(&mut state.budget_remaining)?;
            let scope = InvocationScope {
                execution_id: state.execution_id.as_deref(),
                task_id: 0,
                frame_depth: 0,
            };
            match self
                .run_block_async(
                    state.block,
                    &mut state.slots,
                    host,
                    fallback_cancel,
                    env,
                    scope,
                    &mut state.next_effect_sequence,
                )
                .await?
            {
                VmStepOutcome::Next(block) => state.block = block,
                VmStepOutcome::Suspend(pending) => {
                    state.pending_suspension = Some(pending);
                    return Ok(VmRunStatus::Suspended(state));
                }
                _ => return Err(structured_control_error()),
            }
        }
        Ok(VmRunStatus::Completed)
    }

    fn run_block_sync(
        &self,
        block: BlockId,
        slots: &mut [Value],
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<VmStepOutcome> {
        // 注意：与异步版几乎一致；为了避免泛型 bloat 这里手写两份。
        let block_ref = self
            .program
            .blocks
            .get(block as usize)
            .ok_or_else(|| VsError::Runtime(format!("invalid block id {}", block)))?;
        // 借用 instrs 通过 index 推进，避免与 self.dispatch 借用冲突。
        let n = block_ref.instrs.len();
        for i in 0..n {
            // 每条 instr 单独取引用，避免长生命周期借用。
            let instr = &self.program.blocks[block as usize].instrs[i];
            match instr.clone() {
                Instr::LoadConst { dst, value } => {
                    slots[dst as usize] = value;
                }
                Instr::Move { dst, src } => {
                    slots[dst as usize] = slots[src as usize].clone();
                }
                Instr::LoadVar { dst, var } => {
                    let name = &self.program.vars[var as usize];
                    slots[dst as usize] = host.var_get(name).unwrap_or(Value::Unit);
                }
                Instr::StoreVar { var, src } => {
                    let name = self.program.vars[var as usize].clone();
                    host.var_set(&name, slots[src as usize].clone());
                }
                Instr::CallPure {
                    def,
                    node_id,
                    inputs,
                    outputs,
                    config,
                } => {
                    self.dispatch_sync(
                        def,
                        node_id,
                        inputs,
                        outputs.clone(),
                        &config,
                        slots,
                        host,
                        fallback_cancel,
                        env,
                    )?;
                }
                Instr::CallImpure {
                    def,
                    node_id,
                    inputs,
                    outputs,
                    config,
                    targets,
                    on_end,
                    error_target,
                } => {
                    let next = self.dispatch_sync(
                        def,
                        node_id,
                        inputs,
                        outputs.clone(),
                        &config,
                        slots,
                        host,
                        fallback_cancel,
                        env,
                    )?;
                    return resolve_or_suspend(next, outputs, targets, on_end, error_target);
                }
                Instr::Jump { target } => return Ok(VmStepOutcome::Next(target)),
                Instr::FrameEntry {
                    outputs, target, ..
                } => return Ok(VmStepOutcome::FrameEntry { outputs, target }),
                Instr::Return { inputs, .. } => {
                    return Ok(VmStepOutcome::Return(
                        slots[inputs.start as usize..inputs.end as usize].to_vec(),
                    ));
                }
                Instr::CallGraph {
                    target_graph,
                    inputs,
                    outputs,
                    target,
                    error_target,
                    ..
                } => {
                    return Ok(VmStepOutcome::Call {
                        target_graph,
                        args: slots[inputs.start as usize..inputs.end as usize].to_vec(),
                        outputs,
                        target,
                        error_target,
                    });
                }
                Instr::ForkGraphs {
                    branches,
                    policy,
                    max_concurrency,
                    inputs,
                    outputs,
                    target,
                    error_target,
                    ..
                } => {
                    return Ok(VmStepOutcome::Fork {
                        branches,
                        policy,
                        max_concurrency,
                        args: slots[inputs.start as usize..inputs.end as usize].to_vec(),
                        outputs,
                        target,
                        error_target,
                    });
                }
                Instr::Transition { state, .. } => {
                    return Ok(VmStepOutcome::Transition(state));
                }
            }
        }
        Ok(VmStepOutcome::Next(SlotProgram::HALT))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn run_block_async(
        &self,
        block: BlockId,
        slots: &mut [Value],
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
        scope: InvocationScope<'_>,
        next_effect_sequence: &mut u64,
    ) -> VsResult<VmStepOutcome> {
        let block_ref = self
            .program
            .blocks
            .get(block as usize)
            .ok_or_else(|| VsError::Runtime(format!("invalid block id {}", block)))?;
        let n = block_ref.instrs.len();
        for i in 0..n {
            let instr = self.program.blocks[block as usize].instrs[i].clone();
            match instr {
                Instr::LoadConst { dst, value } => {
                    slots[dst as usize] = value;
                }
                Instr::Move { dst, src } => {
                    slots[dst as usize] = slots[src as usize].clone();
                }
                Instr::LoadVar { dst, var } => {
                    let name = &self.program.vars[var as usize];
                    slots[dst as usize] = host.var_get(name).unwrap_or(Value::Unit);
                }
                Instr::StoreVar { var, src } => {
                    let name = self.program.vars[var as usize].clone();
                    host.var_set(&name, slots[src as usize].clone());
                }
                Instr::CallPure {
                    def,
                    node_id,
                    inputs,
                    outputs,
                    config,
                } => {
                    self.dispatch_async(
                        def,
                        node_id,
                        inputs,
                        outputs,
                        &config,
                        slots,
                        host,
                        fallback_cancel,
                        env,
                        scope,
                        next_effect_sequence,
                    )
                    .await?;
                }
                Instr::CallImpure {
                    def,
                    node_id,
                    inputs,
                    outputs,
                    config,
                    targets,
                    on_end,
                    error_target,
                } => {
                    let next = self
                        .dispatch_async(
                            def,
                            node_id,
                            inputs,
                            outputs.clone(),
                            &config,
                            slots,
                            host,
                            fallback_cancel,
                            env,
                            scope,
                            next_effect_sequence,
                        )
                        .await?;
                    return resolve_or_suspend(next, outputs, targets, on_end, error_target);
                }
                Instr::Jump { target } => return Ok(VmStepOutcome::Next(target)),
                Instr::FrameEntry {
                    outputs, target, ..
                } => return Ok(VmStepOutcome::FrameEntry { outputs, target }),
                Instr::Return { inputs, .. } => {
                    return Ok(VmStepOutcome::Return(
                        slots[inputs.start as usize..inputs.end as usize].to_vec(),
                    ));
                }
                Instr::CallGraph {
                    target_graph,
                    inputs,
                    outputs,
                    target,
                    error_target,
                    ..
                } => {
                    return Ok(VmStepOutcome::Call {
                        target_graph,
                        args: slots[inputs.start as usize..inputs.end as usize].to_vec(),
                        outputs,
                        target,
                        error_target,
                    });
                }
                Instr::ForkGraphs {
                    branches,
                    policy,
                    max_concurrency,
                    inputs,
                    outputs,
                    target,
                    error_target,
                    ..
                } => {
                    return Ok(VmStepOutcome::Fork {
                        branches,
                        policy,
                        max_concurrency,
                        args: slots[inputs.start as usize..inputs.end as usize].to_vec(),
                        outputs,
                        target,
                        error_target,
                    });
                }
                Instr::Transition { state, .. } => {
                    return Ok(VmStepOutcome::Transition(state));
                }
            }
        }
        Ok(VmStepOutcome::Next(SlotProgram::HALT))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_sync(
        &self,
        def: u32,
        node_id: u32,
        inputs: std::ops::Range<u32>,
        outputs: std::ops::Range<u32>,
        config: &std::collections::BTreeMap<String, Value>,
        slots: &mut [Value],
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<ExecNext> {
        let def = self.program.defs[def as usize].clone();
        let in_vals: Vec<Value> = slots[inputs.start as usize..inputs.end as usize].to_vec();
        let mut out_vals: Vec<Value> = slots[outputs.start as usize..outputs.end as usize].to_vec();
        if let Some(gate) = env.determinism {
            let class = match &def.runner {
                RunnerKind::Sync(_) => crate::registry::RunnerClass::Sync,
                RunnerKind::Async(_) => crate::registry::RunnerClass::Async,
                RunnerKind::Effect(_) => return Err(VsError::EffectRunnerInSyncVm),
            };
            gate.check_runner(&def.id, class)?;
        }
        let mut ctx = InvokeCtx::from_env(config, host, fallback_cancel, env);
        self.run_middleware(node_id, &def.id, NodePhase::Before, env, &mut ctx)?;
        let next = match &def.runner {
            RunnerKind::Sync(r) => r.invoke(&in_vals, &mut out_vals, &mut ctx)?,
            RunnerKind::Async(_) => return Err(VsError::AsyncRunnerInSyncVm),
            RunnerKind::Effect(_) => return Err(VsError::EffectRunnerInSyncVm),
        };
        self.run_middleware(node_id, &def.id, NodePhase::After, env, &mut ctx)?;
        for (i, v) in out_vals.into_iter().enumerate() {
            slots[outputs.start as usize + i] = v;
        }
        Ok(next)
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_async(
        &self,
        def: u32,
        node_id: u32,
        inputs: std::ops::Range<u32>,
        outputs: std::ops::Range<u32>,
        config: &std::collections::BTreeMap<String, Value>,
        slots: &mut [Value],
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
        scope: InvocationScope<'_>,
        next_effect_sequence: &mut u64,
    ) -> VsResult<ExecNext> {
        let def = self.program.defs[def as usize].clone();
        let in_vals: Vec<Value> = slots[inputs.start as usize..inputs.end as usize].to_vec();
        let mut out_vals: Vec<Value> = slots[outputs.start as usize..outputs.end as usize].to_vec();
        let next = if let RunnerKind::Effect(spec) = &def.runner {
            let response = self
                .execute_effect(
                    spec.clone(),
                    node_id,
                    inputs,
                    outputs.clone(),
                    config.clone(),
                    in_vals,
                    host,
                    env,
                    scope,
                    next_effect_sequence,
                )
                .await?;
            out_vals = response.outputs;
            response.next
        } else {
            if let Some(gate) = env.determinism {
                let class = match &def.runner {
                    RunnerKind::Sync(_) => crate::registry::RunnerClass::Sync,
                    RunnerKind::Async(_) => crate::registry::RunnerClass::Async,
                    RunnerKind::Effect(_) => unreachable!(),
                };
                gate.check_runner(&def.id, class)?;
            }
            let mut ctx = InvokeCtx::from_env(config, host, fallback_cancel, env);
            self.append_event(
                env,
                scope,
                node_id,
                env.clock.map_or(0, |clock| clock.now_millis()),
                ExecutionEventKind::NodeStarted {
                    def_id: def.id.clone(),
                },
            )?;
            self.run_middleware(node_id, &def.id, NodePhase::Before, env, &mut ctx)?;
            let next = match &def.runner {
                RunnerKind::Sync(r) => r.invoke(&in_vals, &mut out_vals, &mut ctx)?,
                RunnerKind::Async(r) => r.invoke(&in_vals, &mut out_vals, &mut ctx).await?,
                RunnerKind::Effect(_) => unreachable!(),
            };
            self.run_middleware(node_id, &def.id, NodePhase::After, env, &mut ctx)?;
            self.append_event(
                env,
                scope,
                node_id,
                env.clock.map_or(0, |clock| clock.now_millis()),
                ExecutionEventKind::NodeCompleted {
                    def_id: def.id.clone(),
                },
            )?;
            next
        };
        for (i, v) in out_vals.into_iter().enumerate() {
            slots[outputs.start as usize + i] = v;
        }
        Ok(next)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_effect(
        &self,
        spec: crate::effect::EffectSpec,
        node_id: u32,
        _inputs: std::ops::Range<u32>,
        outputs: std::ops::Range<u32>,
        config: std::collections::BTreeMap<String, Value>,
        input_values: Vec<Value>,
        _host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
        scope: InvocationScope<'_>,
        next_effect_sequence: &mut u64,
    ) -> VsResult<crate::effect::EffectResponse> {
        let execution_id = scope.execution_id.ok_or(VsError::MissingExecutionId)?;
        let executor = env.effects.ok_or(VsError::MissingEffectExecutor)?;
        if let Some(policy) = env.policy {
            let taint = env
                .taints
                .map(|provider| provider.taint(&input_values))
                .unwrap_or_default();
            policy.check_effect(&spec, &taint)?;
        }
        if let Some(permissions) = env.permissions {
            for permission in &spec.required_permissions {
                permissions.check_permission(permission, &spec.capability)?;
            }
        } else if !spec.required_permissions.is_empty() {
            return Err(VsError::Runtime(format!(
                "effect {} requires permissions but no PermissionGate was provided",
                spec.capability
            )));
        }
        let sequence = *next_effect_sequence;
        *next_effect_sequence = next_effect_sequence
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("effect sequence exhausted".into()))?;
        let id = EffectRequest::stable_id(
            execution_id,
            scope.task_id,
            scope.frame_depth,
            sequence,
            node_id,
        );
        let mut attempt = 1_u16;
        let fallback_clock = TokioClock;
        let clock: &dyn ExecutionClock = env.clock.unwrap_or(&fallback_clock);
        loop {
            let request = EffectRequest {
                id: id.clone(),
                execution_id: execution_id.to_string(),
                task_id: scope.task_id,
                frame_depth: scope.frame_depth,
                sequence,
                node_id,
                spec: spec.clone(),
                inputs: input_values.clone(),
                config: config.clone(),
                attempt,
            };
            self.append_event(
                env,
                scope,
                node_id,
                clock.now_millis(),
                ExecutionEventKind::EffectRequested {
                    id: id.clone(),
                    spec: spec.clone(),
                    attempt,
                },
            )?;
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(spec.timeout_ms),
                executor.execute(request),
            )
            .await
            .map_err(|_| {
                crate::effect::EffectError::retryable(format!(
                    "effect {} exceeded timeout of {}ms",
                    id, spec.timeout_ms
                ))
            })
            .and_then(std::convert::identity);
            match result {
                Ok(response) => {
                    let expected = (outputs.end - outputs.start) as usize;
                    if response.outputs.len() != expected {
                        return Err(VsError::Runtime(format!(
                            "effect {} returned {} outputs, expected {}",
                            id,
                            response.outputs.len(),
                            expected
                        )));
                    }
                    self.append_event(
                        env,
                        scope,
                        node_id,
                        clock.now_millis(),
                        ExecutionEventKind::EffectCompleted {
                            id,
                            outputs: response.outputs.clone(),
                            chunks: response.chunks.clone(),
                        },
                    )?;
                    if let Some(chunk_tx) = env.chunk_tx {
                        for chunk in &response.chunks {
                            let _ = chunk_tx
                                .send(crate::registry::Chunk::Delta(chunk.clone()))
                                .await;
                        }
                    }
                    return Ok(response);
                }
                Err(error) => {
                    self.append_event(
                        env,
                        scope,
                        node_id,
                        clock.now_millis(),
                        ExecutionEventKind::EffectFailed {
                            id: id.clone(),
                            error: error.clone(),
                            attempt,
                        },
                    )?;
                    if !error.retryable || attempt >= spec.retry.max_attempts {
                        return Err(VsError::Effect(error.message));
                    }
                    let delay = spec.retry.backoff_ms(attempt);
                    if delay > 0 {
                        clock.sleep_millis(delay).await;
                    }
                    attempt = attempt
                        .checked_add(1)
                        .ok_or_else(|| VsError::Runtime("effect retry attempt exhausted".into()))?;
                }
            }
        }
    }

    fn append_event(
        &self,
        env: &ExecutionEnv<'_>,
        scope: InvocationScope<'_>,
        node_id: u32,
        timestamp_ms: u64,
        event: ExecutionEventKind,
    ) -> VsResult<()> {
        if let Some(sink) = env.events {
            sink.append(ExecutionEvent {
                sequence: 0,
                execution_id: scope.execution_id.unwrap_or_default().to_string(),
                task_id: scope.task_id,
                frame_depth: scope.frame_depth,
                node_id,
                timestamp_ms,
                event,
            })?;
        }
        Ok(())
    }

    fn run_middleware(
        &self,
        node_id: u32,
        def_id: &str,
        phase: NodePhase,
        env: &ExecutionEnv<'_>,
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<()> {
        for middleware in &env.middleware {
            middleware.on_node(
                NodeCallInfo {
                    node_id,
                    def_id,
                    phase,
                },
                ctx,
            )?;
        }
        Ok(())
    }
}

fn structured_control_error() -> VsError {
    VsError::Runtime("structured control instruction requires WorkflowVm".into())
}

fn resolve_next(
    next: ExecNext,
    targets: &[(String, BlockId)],
    on_end: BlockId,
    error_target: Option<BlockId>,
) -> VsResult<BlockId> {
    match next {
        ExecNext::Pin(name) => Ok(targets
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, b)| *b)
            .unwrap_or(on_end)),
        ExecNext::End => Ok(on_end),
        ExecNext::Error(message) => {
            error_target.ok_or_else(|| VsError::Runtime(format!("unhandled node error: {message}")))
        }
        ExecNext::Suspend(_) => Err(VsError::Runtime(
            "nested suspension cannot be used as a resume target".into(),
        )),
    }
}

fn resolve_or_suspend(
    next: ExecNext,
    outputs: std::ops::Range<SlotId>,
    targets: Vec<(String, BlockId)>,
    on_end: BlockId,
    error_target: Option<BlockId>,
) -> VsResult<VmStepOutcome> {
    match next {
        ExecNext::Suspend(suspension) => Ok(VmStepOutcome::Suspend(PendingSuspension {
            suspension,
            outputs,
            targets,
            on_end,
            error_target,
        })),
        next => resolve_next(next, &targets, on_end, error_target).map(VmStepOutcome::Next),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(outputs: std::ops::Range<SlotId>) -> PendingSuspension {
        PendingSuspension {
            suspension: Suspension::new("resume", Value::I64(0)),
            outputs,
            targets: Vec::new(),
            on_end: SlotProgram::HALT,
            error_target: None,
        }
    }

    fn state(pending_suspension: PendingSuspension) -> VmState {
        VmState {
            entry_node: 0,
            block: SlotProgram::HALT,
            slots: vec![Value::I64(0); 4],
            budget_remaining: None,
            execution_id: None,
            next_effect_sequence: 0,
            pending_suspension: Some(pending_suspension),
        }
    }

    #[test]
    fn resume_rejects_reversed_output_range_without_consuming_suspension() {
        let pending = pending(5..3);
        let mut vm = state(pending.clone());
        let error = vm
            .resume_suspension(
                "resume",
                SuspensionResolution {
                    outputs: Vec::new(),
                    next: ExecNext::End,
                },
            )
            .unwrap_err();
        assert!(matches!(error, VsError::Runtime(message) if message.contains("reversed")));
        assert_eq!(vm.pending_suspension, Some(pending));
    }

    #[test]
    fn resume_rejects_output_range_beyond_slots_without_consuming_suspension() {
        let pending = pending(2..5);
        let mut vm = state(pending.clone());
        let error = vm
            .resume_suspension(
                "resume",
                SuspensionResolution {
                    outputs: vec![Value::I64(0); 3],
                    next: ExecNext::End,
                },
            )
            .unwrap_err();
        assert!(matches!(error, VsError::Runtime(message) if message.contains("exceeds VM slots")));
        assert_eq!(vm.pending_suspension, Some(pending));
    }
}
