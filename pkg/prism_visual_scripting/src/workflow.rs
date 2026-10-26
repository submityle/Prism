//! Durable multi-graph workflow runtime with function frames and cooperative fork/join scheduling.

use crate::error::{VsError, VsResult};
use crate::ir::{BlockId, SlotId, SlotProgram};
use crate::model::{GraphCall, JoinPolicy};
use crate::registry::{ExecutionEnv, HostServices};
use crate::value::Value;
use crate::vm::{
    Suspension, SuspensionResolution, Vm, VmState, VmStepOutcome, DEFAULT_STEP_BUDGET,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_MAX_TASKS: usize = 1_024;
pub const DEFAULT_MAX_CALL_DEPTH: usize = 128;
pub const DEFAULT_MAX_PARALLELISM: usize = 64;

#[derive(Clone, Default)]
pub struct WorkflowProgram {
    graphs: BTreeMap<String, Arc<SlotProgram>>,
}

impl WorkflowProgram {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        id: impl Into<String>,
        program: SlotProgram,
    ) -> Option<Arc<SlotProgram>> {
        self.graphs.insert(id.into(), Arc::new(program))
    }

    pub fn graph(&self, id: &str) -> Option<&SlotProgram> {
        self.graphs.get(id).map(AsRef::as_ref)
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    fn shared_graph(&self, id: &str) -> Option<Arc<SlotProgram>> {
        self.graphs.get(id).cloned()
    }

    fn digest(&self) -> VsResult<[u8; 32]> {
        let plans = self
            .graphs
            .iter()
            .map(|(id, graph)| (id, graph.to_plan()))
            .collect::<BTreeMap<_, _>>();
        let encoded = serde_json::to_vec(&plans)
            .map_err(|error| VsError::Runtime(format!("workflow digest failed: {error}")))?;
        Ok(Sha256::digest(encoded).into())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkflowCompletion {
    pub values: Vec<Value>,
    pub transition: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkflowRunStatus {
    Completed(WorkflowCompletion),
    Cancelled(WorkflowVmState),
    Suspended(WorkflowVmState),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowVmState {
    pub module_digest: [u8; 32],
    pub root_graph: String,
    #[serde(default)]
    pub execution_id: Option<String>,
    root_task: u64,
    next_task_id: u64,
    next_join_id: u64,
    ready: VecDeque<u64>,
    tasks: BTreeMap<u64, TaskState>,
    joins: BTreeMap<u64, JoinState>,
    budget_remaining: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskState {
    id: u64,
    frames: Vec<FrameState>,
    status: TaskStatus,
    join_parent: Option<JoinBranch>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum TaskStatus {
    Ready,
    Waiting(u64),
    Suspended,
}

impl WorkflowVmState {
    pub fn suspensions(&self) -> Vec<Suspension> {
        self.tasks
            .values()
            .filter_map(|task| task.frames.last())
            .filter_map(|frame| frame.vm.suspension().cloned())
            .collect()
    }

    pub fn resume_suspension(
        &mut self,
        suspension_id: &str,
        resolution: SuspensionResolution,
    ) -> VsResult<()> {
        for task in self.tasks.values_mut() {
            let Some(frame) = task.frames.last_mut() else {
                continue;
            };
            if frame.vm.suspension().map(|value| value.id.as_str()) != Some(suspension_id) {
                continue;
            }
            if task.status != TaskStatus::Suspended {
                return Err(VsError::Runtime(
                    "suspension belongs to a task that is not suspended".into(),
                ));
            }
            frame.vm.resume_suspension(suspension_id, resolution)?;
            task.status = TaskStatus::Ready;
            self.ready.push_back(task.id);
            return Ok(());
        }
        Err(VsError::Runtime(format!(
            "unknown workflow suspension {suspension_id}"
        )))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameState {
    graph: String,
    vm: VmState,
    args: Vec<Value>,
    return_to: Option<ReturnTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReturnTarget {
    outputs: Range<SlotId>,
    success: BlockId,
    error: Option<BlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinBranch {
    join_id: u64,
    index: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingBranch {
    index: usize,
    call: GraphCall,
    args: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinState {
    parent_task: u64,
    policy: JoinPolicy,
    outputs: Range<SlotId>,
    success: BlockId,
    error: Option<BlockId>,
    max_concurrency: usize,
    total: usize,
    queued: VecDeque<PendingBranch>,
    running: BTreeMap<u64, usize>,
    results: BTreeMap<usize, Vec<Value>>,
    failures: BTreeMap<usize, String>,
}

pub struct WorkflowVm {
    program: WorkflowProgram,
    step_budget: u64,
    max_tasks: usize,
    max_call_depth: usize,
    max_parallelism: usize,
}

impl WorkflowVm {
    pub fn new(program: WorkflowProgram) -> Self {
        Self {
            program,
            step_budget: DEFAULT_STEP_BUDGET,
            max_tasks: DEFAULT_MAX_TASKS,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            max_parallelism: DEFAULT_MAX_PARALLELISM,
        }
    }

    pub fn with_step_budget(mut self, budget: u64) -> Self {
        self.step_budget = budget;
        self
    }

    pub fn with_limits(
        mut self,
        max_tasks: usize,
        max_call_depth: usize,
        max_parallelism: usize,
    ) -> Self {
        self.max_tasks = max_tasks.max(1);
        self.max_call_depth = max_call_depth.max(1);
        self.max_parallelism = max_parallelism.max(1);
        self
    }

    pub async fn run_entry_resumable_async_with_env(
        &self,
        graph: &str,
        entry_node: u32,
        args: Vec<Value>,
        state: Option<WorkflowVmState>,
        host: &mut dyn HostServices,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<WorkflowRunStatus> {
        let module_digest = self.program.digest()?;
        let mut state = match state {
            Some(state) => {
                self.validate_state(&state, module_digest, graph, env.execution_id)?;
                state
            }
            None => self.initial_state(module_digest, graph, entry_node, args, env.execution_id)?,
        };
        let fallback_cancel = CancellationToken::new();

        loop {
            if env.cancel.unwrap_or(&fallback_cancel).is_cancelled() {
                return Ok(WorkflowRunStatus::Cancelled(state));
            }
            consume_step(&mut state.budget_remaining)?;
            let Some(task_id) = state.ready.pop_front() else {
                if state.tasks.is_empty() {
                    return Err(VsError::Runtime(
                        "workflow finished without a root completion".into(),
                    ));
                }
                if state
                    .tasks
                    .values()
                    .any(|task| task.status == TaskStatus::Suspended)
                {
                    return Ok(WorkflowRunStatus::Suspended(state));
                }
                return Err(VsError::Runtime("workflow scheduler deadlock".into()));
            };
            let Some(mut task) = state.tasks.remove(&task_id) else {
                continue;
            };
            if task.status != TaskStatus::Ready {
                state.tasks.insert(task_id, task);
                continue;
            }

            let outcome = self.step_task(&mut task, host, &fallback_cancel, env).await;
            match outcome {
                Ok(TaskStep::Ready) => {
                    state.tasks.insert(task_id, task);
                    state.ready.push_back(task_id);
                }
                Ok(TaskStep::Suspended) => {
                    let suspension_id = task
                        .frames
                        .last()
                        .and_then(|frame| frame.vm.suspension())
                        .map(|value| value.id.as_str())
                        .ok_or_else(|| {
                            VsError::Runtime("suspended task has no suspension identity".into())
                        })?;
                    if state.tasks.values().any(|existing| {
                        existing
                            .frames
                            .last()
                            .and_then(|frame| frame.vm.suspension())
                            .is_some_and(|value| value.id == suspension_id)
                    }) {
                        return Err(VsError::Runtime(format!(
                            "duplicate workflow suspension {suspension_id}"
                        )));
                    }
                    task.status = TaskStatus::Suspended;
                    state.tasks.insert(task_id, task);
                }
                Ok(TaskStep::Call {
                    target,
                    args,
                    return_to,
                }) => {
                    if task.frames.len() >= self.max_call_depth {
                        return Err(VsError::Runtime("workflow call depth exceeded".into()));
                    }
                    task.frames.push(self.new_frame(
                        &target,
                        args,
                        Some(return_to),
                        state.execution_id.as_deref(),
                    )?);
                    state.tasks.insert(task_id, task);
                    state.ready.push_back(task_id);
                }
                Ok(TaskStep::Fork {
                    branches,
                    policy,
                    max_concurrency,
                    args,
                    outputs,
                    success,
                    error,
                }) => {
                    if max_concurrency == 0 || max_concurrency > self.max_parallelism {
                        return Err(VsError::Runtime(format!(
                            "parallelism {max_concurrency} exceeds runtime limit {}",
                            self.max_parallelism
                        )));
                    }
                    let join_id = state.next_join_id;
                    state.next_join_id = checked_next(state.next_join_id, "join id")?;
                    task.status = TaskStatus::Waiting(join_id);
                    state.tasks.insert(task_id, task);
                    let mut join = JoinState {
                        parent_task: task_id,
                        policy,
                        outputs,
                        success,
                        error,
                        max_concurrency,
                        total: branches.len(),
                        queued: branches
                            .into_iter()
                            .enumerate()
                            .map(|(index, call)| PendingBranch {
                                index,
                                call,
                                args: args.clone(),
                            })
                            .collect(),
                        running: BTreeMap::new(),
                        results: BTreeMap::new(),
                        failures: BTreeMap::new(),
                    };
                    self.fill_join(&mut state, join_id, &mut join)?;
                    state.joins.insert(join_id, join);
                }
                Ok(TaskStep::Finished(completion)) => {
                    if let Some(parent) = task.join_parent {
                        self.finish_join_branch(&mut state, parent, Ok(completion.values))?;
                    } else if task_id == state.root_task {
                        return Ok(WorkflowRunStatus::Completed(completion));
                    } else {
                        return Err(VsError::Runtime("orphan workflow task completed".into()));
                    }
                }
                Err(error) => {
                    if self.route_call_error(&mut task)? {
                        state.tasks.insert(task_id, task);
                        state.ready.push_back(task_id);
                    } else if let Some(parent) = task.join_parent {
                        self.finish_join_branch(&mut state, parent, Err(error.to_string()))?;
                    } else {
                        return Err(error);
                    }
                }
            }
        }
    }

    async fn step_task(
        &self,
        task: &mut TaskState,
        host: &mut dyn HostServices,
        fallback_cancel: &CancellationToken,
        env: &ExecutionEnv<'_>,
    ) -> VsResult<TaskStep> {
        let frame_depth = task.frames.len();
        let frame = task
            .frames
            .last_mut()
            .ok_or_else(|| VsError::Runtime("workflow task has no call frame".into()))?;
        if frame.vm.block == SlotProgram::HALT {
            return self.return_from_frame(task, Vec::new());
        }
        let program = self
            .program
            .shared_graph(&frame.graph)
            .ok_or_else(|| VsError::Runtime(format!("unknown workflow graph {}", frame.graph)))?;
        let vm = Vm::from_shared(program);
        let outcome = vm
            .run_block_async(
                frame.vm.block,
                &mut frame.vm.slots,
                host,
                fallback_cancel,
                env,
                crate::vm::InvocationScope {
                    execution_id: env.execution_id,
                    task_id: task.id,
                    frame_depth,
                },
                &mut frame.vm.next_effect_sequence,
            )
            .await?;
        match outcome {
            VmStepOutcome::Next(block) => {
                frame.vm.block = block;
                Ok(TaskStep::Ready)
            }
            VmStepOutcome::Suspend(pending) => {
                frame.vm.set_pending_suspension(pending)?;
                Ok(TaskStep::Suspended)
            }
            VmStepOutcome::FrameEntry { outputs, target } => {
                assign_slots(&mut frame.vm.slots, &outputs, &frame.args)?;
                frame.vm.block = target;
                Ok(TaskStep::Ready)
            }
            VmStepOutcome::Return(values) => self.return_from_frame(task, values),
            VmStepOutcome::Call {
                target_graph,
                args,
                outputs,
                target,
                error_target,
            } => {
                frame.vm.block = target;
                Ok(TaskStep::Call {
                    target: target_graph,
                    args,
                    return_to: ReturnTarget {
                        outputs,
                        success: target,
                        error: error_target,
                    },
                })
            }
            VmStepOutcome::Fork {
                branches,
                policy,
                max_concurrency,
                args,
                outputs,
                target,
                error_target,
            } => {
                frame.vm.block = target;
                Ok(TaskStep::Fork {
                    branches,
                    policy,
                    max_concurrency: usize::from(max_concurrency),
                    args,
                    outputs,
                    success: target,
                    error: error_target,
                })
            }
            VmStepOutcome::Transition(state) => Ok(TaskStep::Finished(WorkflowCompletion {
                values: Vec::new(),
                transition: Some(state),
            })),
        }
    }

    fn return_from_frame(&self, task: &mut TaskState, values: Vec<Value>) -> VsResult<TaskStep> {
        let return_to = task
            .frames
            .last()
            .ok_or_else(|| VsError::Runtime("workflow call stack underflow".into()))?
            .return_to
            .clone();
        match return_to {
            Some(return_to) => {
                let caller_index = task.frames.len().checked_sub(2).ok_or_else(|| {
                    VsError::Runtime("workflow return target has no caller frame".into())
                })?;
                let caller = &mut task.frames[caller_index];
                assign_slots(&mut caller.vm.slots, &return_to.outputs, &values)?;
                caller.vm.block = return_to.success;
                task.frames.pop();
                Ok(TaskStep::Ready)
            }
            None => {
                task.frames.pop();
                Ok(TaskStep::Finished(WorkflowCompletion {
                    values,
                    transition: None,
                }))
            }
        }
    }

    fn route_call_error(&self, task: &mut TaskState) -> VsResult<bool> {
        if task.frames.len() <= 1 {
            return Ok(false);
        }
        let frame = task
            .frames
            .pop()
            .ok_or_else(|| VsError::Runtime("workflow call stack underflow".into()))?;
        let Some(target) = frame.return_to else {
            return Ok(false);
        };
        let Some(error) = target.error else {
            return Ok(false);
        };
        let caller = task
            .frames
            .last_mut()
            .ok_or_else(|| VsError::Runtime("workflow error has no caller frame".into()))?;
        caller.vm.block = error;
        Ok(true)
    }

    fn initial_state(
        &self,
        module_digest: [u8; 32],
        graph: &str,
        entry_node: u32,
        args: Vec<Value>,
        execution_id: Option<&str>,
    ) -> VsResult<WorkflowVmState> {
        let call = GraphCall {
            graph: graph.to_string(),
            entry: entry_node,
        };
        let root_task = 1;
        let task = TaskState {
            id: root_task,
            frames: vec![self.new_frame(&call, args, None, execution_id)?],
            status: TaskStatus::Ready,
            join_parent: None,
        };
        Ok(WorkflowVmState {
            module_digest,
            root_graph: graph.to_string(),
            execution_id: execution_id.map(str::to_owned),
            root_task,
            next_task_id: 2,
            next_join_id: 1,
            ready: VecDeque::from([root_task]),
            tasks: BTreeMap::from([(root_task, task)]),
            joins: BTreeMap::new(),
            budget_remaining: (self.step_budget != 0).then_some(self.step_budget),
        })
    }

    fn new_frame(
        &self,
        call: &GraphCall,
        args: Vec<Value>,
        return_to: Option<ReturnTarget>,
        execution_id: Option<&str>,
    ) -> VsResult<FrameState> {
        let program = self
            .program
            .graph(&call.graph)
            .ok_or_else(|| VsError::Runtime(format!("unknown workflow graph {}", call.graph)))?;
        let block = program.entries.get(&call.entry).copied().ok_or_else(|| {
            VsError::Runtime(format!(
                "graph {} has no entry node {}",
                call.graph, call.entry
            ))
        })?;
        Ok(FrameState {
            graph: call.graph.clone(),
            vm: VmState {
                entry_node: call.entry,
                block,
                slots: vec![Value::Unit; program.slots_count as usize],
                budget_remaining: None,
                execution_id: execution_id.map(str::to_owned),
                next_effect_sequence: 0,
                pending_suspension: None,
            },
            args,
            return_to,
        })
    }

    fn fill_join(
        &self,
        state: &mut WorkflowVmState,
        join_id: u64,
        join: &mut JoinState,
    ) -> VsResult<()> {
        while join.running.len() < join.max_concurrency {
            let Some(branch) = join.queued.pop_front() else {
                break;
            };
            if state.tasks.len() >= self.max_tasks {
                return Err(VsError::Runtime("workflow task limit exceeded".into()));
            }
            let task_id = state.next_task_id;
            state.next_task_id = checked_next(state.next_task_id, "task id")?;
            let task = TaskState {
                id: task_id,
                frames: vec![self.new_frame(
                    &branch.call,
                    branch.args,
                    None,
                    state.execution_id.as_deref(),
                )?],
                status: TaskStatus::Ready,
                join_parent: Some(JoinBranch {
                    join_id,
                    index: branch.index,
                }),
            };
            join.running.insert(task_id, branch.index);
            state.tasks.insert(task_id, task);
            state.ready.push_back(task_id);
        }
        Ok(())
    }

    fn finish_join_branch(
        &self,
        state: &mut WorkflowVmState,
        branch: JoinBranch,
        result: Result<Vec<Value>, String>,
    ) -> VsResult<()> {
        let mut join = state
            .joins
            .remove(&branch.join_id)
            .ok_or_else(|| VsError::Runtime("parallel branch references missing join".into()))?;
        let task_id = join
            .running
            .iter()
            .find_map(|(task_id, index)| (*index == branch.index).then_some(*task_id))
            .ok_or_else(|| VsError::Runtime("parallel branch is not running".into()))?;
        join.running.remove(&task_id);

        match result {
            Ok(values) => {
                join.results.insert(branch.index, values);
                if join.policy == JoinPolicy::Race {
                    cancel_join_tasks(state, &join);
                    self.resume_join_parent(state, &join, race_values(&join)?)?;
                    return Ok(());
                }
            }
            Err(error) => {
                join.failures.insert(branch.index, error);
                if join.policy == JoinPolicy::All {
                    cancel_join_tasks(state, &join);
                    self.fail_join_parent(state, &join)?;
                    return Ok(());
                }
            }
        }

        self.fill_join(state, branch.join_id, &mut join)?;
        let finished = join.results.len() + join.failures.len();
        if finished == join.total {
            if join.policy == JoinPolicy::Race {
                self.fail_join_parent(state, &join)?;
            } else {
                let values = all_values(&join)?;
                self.resume_join_parent(state, &join, values)?;
            }
        } else {
            state.joins.insert(branch.join_id, join);
        }
        Ok(())
    }

    fn resume_join_parent(
        &self,
        state: &mut WorkflowVmState,
        join: &JoinState,
        values: Vec<Value>,
    ) -> VsResult<()> {
        let parent = state
            .tasks
            .get_mut(&join.parent_task)
            .ok_or_else(|| VsError::Runtime("parallel join parent is missing".into()))?;
        let frame = parent
            .frames
            .last_mut()
            .ok_or_else(|| VsError::Runtime("parallel join parent has no frame".into()))?;
        assign_slots(&mut frame.vm.slots, &join.outputs, &values)?;
        frame.vm.block = join.success;
        parent.status = TaskStatus::Ready;
        state.ready.push_back(join.parent_task);
        Ok(())
    }

    fn fail_join_parent(&self, state: &mut WorkflowVmState, join: &JoinState) -> VsResult<()> {
        let Some(error_target) = join.error else {
            let message = join
                .failures
                .values()
                .next()
                .cloned()
                .unwrap_or_else(|| "all race branches failed".into());
            return Err(VsError::Runtime(format!("parallel join failed: {message}")));
        };
        let parent = state
            .tasks
            .get_mut(&join.parent_task)
            .ok_or_else(|| VsError::Runtime("parallel join parent is missing".into()))?;
        let frame = parent
            .frames
            .last_mut()
            .ok_or_else(|| VsError::Runtime("parallel join parent has no frame".into()))?;
        frame.vm.block = error_target;
        parent.status = TaskStatus::Ready;
        state.ready.push_back(join.parent_task);
        Ok(())
    }

    fn validate_state(
        &self,
        state: &WorkflowVmState,
        module_digest: [u8; 32],
        root_graph: &str,
        execution_id: Option<&str>,
    ) -> VsResult<()> {
        if state.module_digest != module_digest
            || state.root_graph != root_graph
            || state.execution_id.as_deref() != execution_id
        {
            return Err(VsError::Runtime("workflow resume identity mismatch".into()));
        }
        if state.tasks.len() > self.max_tasks {
            return Err(VsError::Runtime(
                "resumed workflow exceeds task limit".into(),
            ));
        }
        if !state.tasks.contains_key(&state.root_task) {
            return Err(VsError::Runtime("resumed workflow has no root task".into()));
        }
        let mut ready_ids = std::collections::BTreeSet::new();
        let mut suspension_ids = std::collections::BTreeSet::new();
        for task_id in &state.ready {
            if !ready_ids.insert(*task_id) {
                return Err(VsError::Runtime(
                    "ready queue contains duplicate task".into(),
                ));
            }
        }
        for (task_id, task) in &state.tasks {
            if task.id != *task_id {
                return Err(VsError::Runtime("workflow task identity mismatch".into()));
            }
            if task.frames.is_empty() || task.frames.len() > self.max_call_depth {
                return Err(VsError::Runtime("invalid resumed call stack depth".into()));
            }
            for (frame_index, frame) in task.frames.iter().enumerate() {
                let program = self.program.graph(&frame.graph).ok_or_else(|| {
                    VsError::Runtime(format!(
                        "resumed state references unknown graph {}",
                        frame.graph
                    ))
                })?;
                if frame.vm.slots.len() != program.slots_count as usize
                    || (frame.vm.block != SlotProgram::HALT
                        && frame.vm.block as usize >= program.blocks.len())
                {
                    return Err(VsError::Runtime("invalid resumed workflow frame".into()));
                }
                if frame.vm.execution_id != state.execution_id {
                    return Err(VsError::Runtime(
                        "resumed workflow frame execution identity mismatch".into(),
                    ));
                }
                if !program.entries.contains_key(&frame.vm.entry_node) {
                    return Err(VsError::Runtime(
                        "resumed frame has invalid entry node".into(),
                    ));
                }
                frame.vm.validate_suspension(program)?;
                if let Some(suspension) = frame.vm.suspension()
                    && !suspension_ids.insert(suspension.id.as_str())
                {
                    return Err(VsError::Runtime(
                        "resumed workflow contains duplicate suspension identities".into(),
                    ));
                }
                if let Some(return_to) = &frame.return_to {
                    if frame_index == 0 {
                        return Err(VsError::Runtime("root frame has a return target".into()));
                    }
                    let caller = &task.frames[frame_index - 1];
                    let caller_program = self.program.graph(&caller.graph).ok_or_else(|| {
                        VsError::Runtime("resumed caller references unknown graph".into())
                    })?;
                    if return_to.outputs.end > caller_program.slots_count
                        || return_to.outputs.start > return_to.outputs.end
                        || !valid_block(caller_program, return_to.success)
                        || return_to
                            .error
                            .is_some_and(|block| !valid_block(caller_program, block))
                    {
                        return Err(VsError::Runtime("invalid resumed return target".into()));
                    }
                }
            }
            match task.status {
                TaskStatus::Ready if !ready_ids.contains(task_id) => {
                    return Err(VsError::Runtime(
                        "ready task is absent from ready queue".into(),
                    ));
                }
                TaskStatus::Waiting(join_id) if !state.joins.contains_key(&join_id) => {
                    return Err(VsError::Runtime("task waits for a missing join".into()));
                }
                TaskStatus::Suspended
                    if task
                        .frames
                        .last()
                        .and_then(|frame| frame.vm.suspension())
                        .is_none() =>
                {
                    return Err(VsError::Runtime(
                        "suspended task has no pending suspension".into(),
                    ));
                }
                TaskStatus::Ready | TaskStatus::Waiting(_)
                    if task
                        .frames
                        .last()
                        .and_then(|frame| frame.vm.suspension())
                        .is_some() =>
                {
                    return Err(VsError::Runtime(
                        "non-suspended task contains a pending suspension".into(),
                    ));
                }
                _ => {}
            }
        }
        for task_id in &state.ready {
            if !state.tasks.contains_key(task_id) {
                return Err(VsError::Runtime(
                    "ready queue references missing task".into(),
                ));
            }
        }
        for (join_id, join) in &state.joins {
            if join.max_concurrency == 0 || join.max_concurrency > self.max_parallelism {
                return Err(VsError::Runtime("invalid resumed join parallelism".into()));
            }
            if !state.tasks.contains_key(&join.parent_task) {
                return Err(VsError::Runtime(
                    "join references missing parent task".into(),
                ));
            }
            if state.tasks[&join.parent_task].status != TaskStatus::Waiting(*join_id) {
                return Err(VsError::Runtime(
                    "join parent is not waiting for this join".into(),
                ));
            }
            if join.total == 0
                || join.total
                    != join.queued.len()
                        + join.running.len()
                        + join.results.len()
                        + join.failures.len()
            {
                return Err(VsError::Runtime("invalid resumed join accounting".into()));
            }
            let mut branch_indexes = std::collections::BTreeSet::new();
            for pending in &join.queued {
                validate_branch_call(&self.program, pending.index, join.total, &pending.call)?;
                if !branch_indexes.insert(pending.index) {
                    return Err(VsError::Runtime("duplicate join branch index".into()));
                }
            }
            for (task_id, index) in &join.running {
                let task = state.tasks.get(task_id).ok_or_else(|| {
                    VsError::Runtime("join references missing branch task".into())
                })?;
                if task.join_parent
                    != Some(JoinBranch {
                        join_id: *join_id,
                        index: *index,
                    })
                    || !branch_indexes.insert(*index)
                {
                    return Err(VsError::Runtime("invalid running join branch".into()));
                }
            }
            for index in join.results.keys().chain(join.failures.keys()) {
                if *index >= join.total || !branch_indexes.insert(*index) {
                    return Err(VsError::Runtime("invalid completed join branch".into()));
                }
            }
        }
        if state.next_task_id <= state.tasks.keys().copied().max().unwrap_or(0)
            || state.next_join_id <= state.joins.keys().copied().max().unwrap_or(0)
        {
            return Err(VsError::Runtime(
                "workflow id allocator moved backwards".into(),
            ));
        }
        Ok(())
    }
}

enum TaskStep {
    Ready,
    Suspended,
    Call {
        target: GraphCall,
        args: Vec<Value>,
        return_to: ReturnTarget,
    },
    Fork {
        branches: Vec<GraphCall>,
        policy: JoinPolicy,
        max_concurrency: usize,
        args: Vec<Value>,
        outputs: Range<SlotId>,
        success: BlockId,
        error: Option<BlockId>,
    },
    Finished(WorkflowCompletion),
}

fn assign_slots(slots: &mut [Value], range: &Range<SlotId>, values: &[Value]) -> VsResult<()> {
    let expected = (range.end - range.start) as usize;
    if expected != values.len() {
        return Err(VsError::Runtime(format!(
            "workflow value arity mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    let target = slots
        .get_mut(range.start as usize..range.end as usize)
        .ok_or_else(|| VsError::Runtime("workflow slot range is invalid".into()))?;
    target.clone_from_slice(values);
    Ok(())
}

fn consume_step(budget: &mut Option<u64>) -> VsResult<()> {
    let Some(remaining) = budget else {
        return Ok(());
    };
    if *remaining == 0 {
        return Err(VsError::Runtime("workflow step budget exceeded".into()));
    }
    *remaining -= 1;
    Ok(())
}

fn checked_next(value: u64, name: &str) -> VsResult<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| VsError::Runtime(format!("{name} exhausted")))
}

fn race_values(join: &JoinState) -> VsResult<Vec<Value>> {
    join.results
        .iter()
        .next()
        .map(|(_, values)| values.clone())
        .ok_or_else(|| VsError::Runtime("race completed without a result".into()))
}

fn all_values(join: &JoinState) -> VsResult<Vec<Value>> {
    if join.results.len() != join.total {
        return Err(VsError::Runtime(
            "join completed without all results".into(),
        ));
    }
    let width = join.results.values().next().map_or(0, Vec::len);
    if join.results.values().any(|values| values.len() != width) {
        return Err(VsError::Runtime("parallel result arity mismatch".into()));
    }
    Ok((0..width)
        .map(|column| {
            Value::List(Arc::new(
                join.results
                    .values()
                    .map(|values| values[column].clone())
                    .collect(),
            ))
        })
        .collect())
}

fn valid_block(program: &SlotProgram, block: BlockId) -> bool {
    block == SlotProgram::HALT || (block as usize) < program.blocks.len()
}

fn validate_branch_call(
    program: &WorkflowProgram,
    index: usize,
    total: usize,
    call: &GraphCall,
) -> VsResult<()> {
    if index >= total {
        return Err(VsError::Runtime("join branch index is out of range".into()));
    }
    let graph = program
        .graph(&call.graph)
        .ok_or_else(|| VsError::Runtime("join branch references unknown graph".into()))?;
    if !graph.entries.contains_key(&call.entry) {
        return Err(VsError::Runtime(
            "join branch references invalid entry".into(),
        ));
    }
    Ok(())
}

fn cancel_join_tasks(state: &mut WorkflowVmState, join: &JoinState) {
    for task_id in join.running.keys() {
        state.tasks.remove(task_id);
    }
    state
        .ready
        .retain(|task_id| !join.running.contains_key(task_id));
}
