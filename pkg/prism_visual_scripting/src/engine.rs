//! `prism_visual_scripting` 的有状态全功能门面。
//!
//! [`VisualScriptEngine`] 只编排内核已有能力，不复制编译器、VM、持久化、
//! Actor、Saga 或流处理实现。一个引擎实例统一持有节点库、Workflow 快照、
//! DurableStore、等待代理、执行事件、调度器以及宿主注入的 Effect/策略依赖。

use crate::actor::DurableActorRuntime;
use crate::analysis::{analyze_graph, StaticModelReport};
use crate::awaiting::{AwaitReady, AwaitRequest, InMemoryAwaitBroker};
use crate::compiler::{compile_to_plan, instantiate};
use crate::coordinator::{RunCoordinator, RunRecord, RunStatus};
use crate::durable::{DurableRecord, DurableStore, InMemoryDurableStore};
use crate::effect::{EffectExecutor, ExecutionClock, TokioClock};
use crate::execution::{
    ExecutionEvent, ExecutionEventSink, ExecutionReplay, InMemoryEventLog, ReplayReport,
};
use crate::ir::{ExecutionPlan, SlotProgram};
use crate::migration::MigrationRegistry;
use crate::model::Graph;
use crate::policy::{EffectPolicy, TaintProvider};
use crate::registry::{
    BudgetMeter, ChunkTx, DeterminismGate, ExecutionEnv, HostServices, NodeDefinition, NodeLibrary,
    NodeMiddleware, PermissionGate, TraceSink,
};
use crate::saga::{SagaOrchestrator, SagaState};
use crate::scheduler::{
    AdmitResult, FairScheduler, ScheduledRun, SchedulerLimits, SchedulerSnapshot,
};
use crate::upgrade::UpgradeCoordinator;
use crate::value::Value;
use crate::vm::{SuspensionResolution, Vm, VmRunStatus, VmState};
use crate::workflow::{
    WorkflowCompletion, WorkflowProgram, WorkflowRunStatus, WorkflowVm, WorkflowVmState,
};
use crate::{VsError, VsResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

/// 单次 Workflow 启动所需的稳定身份和入口参数。
#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowStart {
    pub run_id: String,
    pub execution_id: String,
    pub workflow_id: String,
    pub graph: String,
    pub entry_node: u32,
    pub args: Vec<Value>,
}

/// Durable Workflow 一次推进后的持久化结果。
#[derive(Clone, Debug, PartialEq)]
pub struct DurableWorkflowRun {
    pub revision: u64,
    pub status: WorkflowRunStatus,
}

/// 引擎默认执行与调度边界。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineLimits {
    pub step_budget: u64,
    pub max_tasks: usize,
    pub max_call_depth: usize,
    pub max_parallelism: usize,
    pub scheduler: SchedulerLimits,
}

impl Default for EngineLimits {
    fn default() -> Self {
        Self {
            step_budget: crate::vm::DEFAULT_STEP_BUDGET,
            max_tasks: crate::workflow::DEFAULT_MAX_TASKS,
            max_call_depth: crate::workflow::DEFAULT_MAX_CALL_DEPTH,
            max_parallelism: crate::workflow::DEFAULT_MAX_PARALLELISM,
            scheduler: SchedulerLimits {
                max_active: 64,
                max_active_per_tenant: 16,
                max_queued: 4_096,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredWorkflowCheckpoint {
    Reserved {
        workflow_id: String,
    },
    Resumable {
        workflow_id: String,
        state: WorkflowVmState,
    },
    Completed {
        workflow_id: String,
        completion: WorkflowCompletion,
    },
    Failed {
        workflow_id: String,
        message: String,
    },
}

/// 可视化脚本内核的主入口。
///
/// 门面负责依赖所有权和完整调用链；编辑器资产、领域节点与远程资产发布仍属于上层。
pub struct VisualScriptEngine {
    node_library: NodeLibrary,
    workflows: DashMap<String, Arc<WorkflowProgram>>,
    durable_store: Arc<dyn DurableStore>,
    await_broker: Arc<InMemoryAwaitBroker>,
    event_logs: DashMap<String, Arc<InMemoryEventLog>>,
    run_gates: DashMap<String, Arc<tokio::sync::Mutex<()>>>,
    migrations: MigrationRegistry,
    scheduler: Mutex<FairScheduler>,
    limits: EngineLimits,
    effects: Option<Arc<dyn EffectExecutor>>,
    clock: Arc<dyn ExecutionClock>,
    permissions: Option<Arc<dyn PermissionGate>>,
    budget: Option<Arc<dyn BudgetMeter>>,
    trace: Option<Arc<dyn TraceSink>>,
    policy: Option<Arc<dyn EffectPolicy>>,
    taints: Option<Arc<dyn TaintProvider>>,
    determinism: Option<Arc<dyn DeterminismGate>>,
    middleware: Vec<Arc<dyn NodeMiddleware>>,
    external: Option<Arc<dyn Any + Send + Sync>>,
    event_sink: Option<Arc<dyn ExecutionEventSink>>,
}

struct FanoutEventSink<'a> {
    local: &'a dyn ExecutionEventSink,
    external: Option<&'a dyn ExecutionEventSink>,
}

impl ExecutionEventSink for FanoutEventSink<'_> {
    fn append(&self, event: ExecutionEvent) -> VsResult<()> {
        if let Some(external) = self.external {
            external.append(event.clone())?;
        }
        self.local.append(event)
    }
}

impl Default for VisualScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualScriptEngine {
    /// 创建包含全部内置节点、内存 DurableStore 和默认安全边界的引擎。
    pub fn new() -> Self {
        Self::with_node_library(NodeLibrary::with_builtins())
    }

    /// 使用自定义节点库创建引擎，其余依赖采用可立即运行的默认实现。
    pub fn with_node_library(node_library: NodeLibrary) -> Self {
        let limits = EngineLimits::default();
        let scheduler = FairScheduler::new(limits.scheduler.clone())
            .expect("default scheduler limits must be valid");
        Self {
            node_library,
            workflows: DashMap::new(),
            durable_store: Arc::new(InMemoryDurableStore::default()),
            await_broker: Arc::new(InMemoryAwaitBroker::default()),
            event_logs: DashMap::new(),
            run_gates: DashMap::new(),
            migrations: MigrationRegistry::default(),
            scheduler: Mutex::new(scheduler),
            limits,
            effects: None,
            clock: Arc::new(TokioClock),
            permissions: None,
            budget: None,
            trace: None,
            policy: None,
            taints: None,
            determinism: None,
            middleware: Vec::new(),
            external: None,
            event_sink: None,
        }
    }

    /// 替换 DurableStore。Run、Actor、Effect journal 与升级协调器可共享它。
    pub fn set_durable_store(&mut self, store: Arc<dyn DurableStore>) {
        self.durable_store = store;
    }

    pub fn durable_store(&self) -> Arc<dyn DurableStore> {
        self.durable_store.clone()
    }

    pub fn node_library(&self) -> &NodeLibrary {
        &self.node_library
    }

    pub fn node_library_mut(&mut self) -> &mut NodeLibrary {
        &mut self.node_library
    }

    pub fn register_node(&mut self, definition: NodeDefinition) {
        self.node_library.register(definition);
    }

    pub fn limits(&self) -> &EngineLimits {
        &self.limits
    }

    /// 更新 VM 和调度边界。已有 Workflow 快照继续使用新边界。
    /// 调度器必须为空，避免配置更新时丢失排队或活跃任务。
    pub fn set_limits(&mut self, limits: EngineLimits) -> VsResult<()> {
        let scheduler = self
            .scheduler
            .get_mut()
            .map_err(|_| VsError::Runtime("engine scheduler poisoned".into()))?;
        let snapshot = scheduler.snapshot();
        if !snapshot.active.is_empty() || snapshot.queued.values().any(|queue| !queue.is_empty()) {
            return Err(VsError::Runtime(
                "cannot replace limits while scheduler has active or queued runs".into(),
            ));
        }
        *scheduler = FairScheduler::new(limits.scheduler.clone())?;
        self.limits = limits;
        Ok(())
    }

    pub fn set_effect_executor(&mut self, effects: Arc<dyn EffectExecutor>) {
        self.effects = Some(effects);
    }

    pub fn clear_effect_executor(&mut self) {
        self.effects = None;
    }

    pub fn effect_executor(&self) -> Option<Arc<dyn EffectExecutor>> {
        self.effects.clone()
    }

    pub fn set_clock(&mut self, clock: Arc<dyn ExecutionClock>) {
        self.clock = clock;
    }

    pub fn clock(&self) -> Arc<dyn ExecutionClock> {
        self.clock.clone()
    }

    /// 注入宿主事件后端。本地按 execution 隔离的日志仍会保留用于即时重放。
    pub fn set_event_sink(&mut self, sink: Option<Arc<dyn ExecutionEventSink>>) {
        self.event_sink = sink;
    }

    pub fn set_permission_gate(&mut self, gate: Option<Arc<dyn PermissionGate>>) {
        self.permissions = gate;
    }

    pub fn set_budget_meter(&mut self, meter: Option<Arc<dyn BudgetMeter>>) {
        self.budget = meter;
    }

    pub fn set_trace_sink(&mut self, sink: Option<Arc<dyn TraceSink>>) {
        self.trace = sink;
    }

    pub fn set_effect_policy(&mut self, policy: Option<Arc<dyn EffectPolicy>>) {
        self.policy = policy;
    }

    pub fn set_taint_provider(&mut self, taints: Option<Arc<dyn TaintProvider>>) {
        self.taints = taints;
    }

    pub fn set_determinism_gate(&mut self, gate: Option<Arc<dyn DeterminismGate>>) {
        self.determinism = gate;
    }

    pub fn add_middleware(&mut self, middleware: Arc<dyn NodeMiddleware>) {
        self.middleware.push(middleware);
    }

    pub fn clear_middleware(&mut self) {
        self.middleware.clear();
    }

    pub fn set_external_context<T: Any + Send + Sync>(&mut self, external: Arc<T>) {
        self.external = Some(external);
    }

    pub fn clear_external_context(&mut self) {
        self.external = None;
    }

    /// 运行与 checkpoint 迁移共享的迁移注册表。
    pub fn migrations(&self) -> &MigrationRegistry {
        &self.migrations
    }

    pub fn migrations_mut(&mut self) -> &mut MigrationRegistry {
        &mut self.migrations
    }

    /// 先执行静态安全分析，再由调用方决定是否允许 warning。
    pub fn analyze(&self, graph: &Graph) -> VsResult<StaticModelReport> {
        analyze_graph(graph, &self.node_library)
    }

    /// 编译为可序列化、带注册表身份的计划。
    pub fn compile_plan(&self, graph: &Graph) -> VsResult<ExecutionPlan> {
        compile_to_plan(graph, &self.node_library)
    }

    /// 在当前节点快照下实例化计划，注册表不一致时会被拒绝。
    pub fn instantiate(&self, plan: &ExecutionPlan) -> VsResult<SlotProgram> {
        instantiate(plan, &self.node_library)
    }

    /// 完成 Graph -> ExecutionPlan -> SlotProgram 全编译链。
    pub fn compile(&self, graph: &Graph) -> VsResult<SlotProgram> {
        self.instantiate(&self.compile_plan(graph)?)
    }

    /// 编译多图 Workflow，但不安装到引擎。
    pub fn compile_workflow(&self, graphs: &BTreeMap<String, Graph>) -> VsResult<WorkflowProgram> {
        if graphs.is_empty() {
            return Err(VsError::Validate(
                "workflow must contain at least one graph".into(),
            ));
        }
        let mut workflow = WorkflowProgram::new();
        for (id, graph) in graphs {
            validate_identity("workflow graph", id)?;
            workflow.insert(id.clone(), self.compile(graph)?);
        }
        Ok(workflow)
    }

    /// 编译并安装一个可并发读取的 Workflow 快照。
    pub fn register_workflow(
        &self,
        workflow_id: impl Into<String>,
        graphs: &BTreeMap<String, Graph>,
    ) -> VsResult<Arc<WorkflowProgram>> {
        let workflow_id = workflow_id.into();
        validate_identity("workflow", &workflow_id)?;
        let workflow = Arc::new(self.compile_workflow(graphs)?);
        self.workflows.insert(workflow_id, workflow.clone());
        Ok(workflow)
    }

    /// 安装调用方已经编译好的 Workflow 快照。
    pub fn install_workflow(
        &self,
        workflow_id: impl Into<String>,
        workflow: WorkflowProgram,
    ) -> VsResult<Arc<WorkflowProgram>> {
        let workflow_id = workflow_id.into();
        validate_identity("workflow", &workflow_id)?;
        if workflow.is_empty() {
            return Err(VsError::Validate(
                "workflow id and program must not be empty".into(),
            ));
        }
        let workflow = Arc::new(workflow);
        self.workflows.insert(workflow_id, workflow.clone());
        Ok(workflow)
    }

    pub fn workflow(&self, workflow_id: &str) -> Option<Arc<WorkflowProgram>> {
        self.workflows.get(workflow_id).map(|value| value.clone())
    }

    pub fn remove_workflow(&self, workflow_id: &str) -> Option<Arc<WorkflowProgram>> {
        self.workflows.remove(workflow_id).map(|(_, value)| value)
    }

    /// 执行单图入口；同步与异步 Runner、Effect、策略和事件记录共用同一环境。
    pub async fn run_graph(
        &self,
        execution_id: &str,
        graph: &Graph,
        entry_node: u32,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<VmRunStatus> {
        let program = self.compile(graph)?;
        self.run_program(
            execution_id,
            program,
            entry_node,
            None,
            host,
            cancel,
            chunk_tx,
        )
        .await
    }

    /// 依次执行 Graph 的全部 entry。挂起场景应改用可返回状态的 [`Self::run_graph`]。
    pub async fn run_graph_all(
        &self,
        execution_id: &str,
        graph: &Graph,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<()> {
        validate_identity("execution", execution_id)?;
        let program = self.compile(graph)?;
        let events = self.event_log(execution_id);
        let event_sink = self.fanout_event_sink(events.as_ref());
        let env = self.execution_env(execution_id, cancel, chunk_tx, &event_sink);
        Vm::new(program)
            .with_step_budget(self.limits.step_budget)
            .run_all_async_with_env(host, &env)
            .await
    }

    /// 执行或恢复一个已经实例化的单图计划。
    #[allow(clippy::too_many_arguments)]
    pub async fn run_program(
        &self,
        execution_id: &str,
        program: SlotProgram,
        entry_node: u32,
        state: Option<VmState>,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<VmRunStatus> {
        validate_identity("execution", execution_id)?;
        let events = self.event_log(execution_id);
        let event_sink = self.fanout_event_sink(events.as_ref());
        let env = self.execution_env(execution_id, cancel, chunk_tx, &event_sink);
        let status = Vm::new(program)
            .with_step_budget(self.limits.step_budget)
            .run_entry_resumable_async_with_env(entry_node, state, host, &env)
            .await?;
        self.register_vm_suspension(&status)?;
        Ok(status)
    }

    /// 执行已安装的多图 Workflow；返回的状态可由调用方自行持久化。
    #[allow(clippy::too_many_arguments)]
    pub async fn run_workflow(
        &self,
        execution_id: &str,
        workflow_id: &str,
        graph: &str,
        entry_node: u32,
        args: Vec<Value>,
        state: Option<WorkflowVmState>,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<WorkflowRunStatus> {
        validate_identity("execution", execution_id)?;
        let workflow = self.require_workflow(workflow_id)?;
        let events = self.event_log(execution_id);
        let event_sink = self.fanout_event_sink(events.as_ref());
        let env = self.execution_env(execution_id, cancel, chunk_tx, &event_sink);
        let status = self
            .workflow_vm(workflow.as_ref())
            .run_entry_resumable_async_with_env(graph, entry_node, args, state, host, &env)
            .await?;
        self.register_suspensions(&status)?;
        Ok(status)
    }

    /// 占用 run identity 后执行 Workflow，并用 CAS 持久化完成/挂起状态。
    pub async fn start_workflow(
        &self,
        request: WorkflowStart,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<DurableWorkflowRun> {
        validate_identity("run", &request.run_id)?;
        validate_identity("execution", &request.execution_id)?;
        self.require_workflow(&request.workflow_id)?;
        let gate = self.run_gate(&request.run_id);
        let _guard = gate.lock().await;
        let coordinator = self.run_coordinator();
        let reserved = StoredWorkflowCheckpoint::Reserved {
            workflow_id: request.workflow_id.clone(),
        };
        let reserved = coordinator.start(
            &request.run_id,
            &request.execution_id,
            encode_checkpoint(&reserved)?,
        )?;
        let seed_cancel = CancellationToken::new();
        seed_cancel.cancel();
        let initial = self
            .run_workflow(
                &request.execution_id,
                &request.workflow_id,
                &request.graph,
                request.entry_node,
                request.args.clone(),
                None,
                host,
                &seed_cancel,
                None,
            )
            .await;
        let initial = match initial {
            Ok(WorkflowRunStatus::Cancelled(state)) => state,
            Ok(_) => {
                return self.persist_workflow_failure(
                    &request.run_id,
                    &request.execution_id,
                    reserved.revision,
                    &request.workflow_id,
                    VsError::Runtime("failed to capture initial workflow state".into()),
                );
            }
            Err(error) => {
                return self.persist_workflow_failure(
                    &request.run_id,
                    &request.execution_id,
                    reserved.revision,
                    &request.workflow_id,
                    error,
                );
            }
        };
        let running = coordinator.checkpoint(
            &request.run_id,
            reserved.revision,
            &request.execution_id,
            RunStatus::Running,
            encode_checkpoint(&StoredWorkflowCheckpoint::Resumable {
                workflow_id: request.workflow_id.clone(),
                state: initial.clone(),
            })?,
        )?;
        let result = self
            .run_workflow(
                &request.execution_id,
                &request.workflow_id,
                &request.graph,
                request.entry_node,
                request.args,
                Some(initial),
                host,
                cancel,
                chunk_tx,
            )
            .await;
        match result {
            Ok(status) => self.persist_workflow_status(
                &request.run_id,
                &request.execution_id,
                running.revision,
                &request.workflow_id,
                status,
            ),
            Err(error) => self.persist_workflow_failure(
                &request.run_id,
                &request.execution_id,
                running.revision,
                &request.workflow_id,
                error,
            ),
        }
    }

    /// 从 DurableStore 恢复挂起 Workflow，并原子提交下一状态。
    pub async fn resume_workflow(
        &self,
        run_id: &str,
        suspension_id: &str,
        resolution: SuspensionResolution,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<DurableWorkflowRun> {
        let gate = self.run_gate(run_id);
        let _guard = gate.lock().await;
        let coordinator = self.run_coordinator();
        let (record, run) = coordinator
            .load(run_id)?
            .ok_or_else(|| VsError::Runtime(format!("unknown run {run_id}")))?;
        if run.status != RunStatus::Suspended {
            return Err(VsError::Runtime(format!("run {run_id} is not suspended")));
        }
        let checkpoint = decode_checkpoint(&run.checkpoint)?;
        let StoredWorkflowCheckpoint::Resumable {
            workflow_id,
            mut state,
        } = checkpoint
        else {
            return Err(VsError::Runtime(
                "run checkpoint does not contain resumable workflow state".into(),
            ));
        };
        let root_graph = state.root_graph.clone();
        state.resume_suspension(suspension_id, resolution)?;
        let claimed = coordinator.checkpoint(
            run_id,
            record.revision,
            &run.execution_id,
            RunStatus::Running,
            encode_checkpoint(&StoredWorkflowCheckpoint::Resumable {
                workflow_id: workflow_id.clone(),
                state: state.clone(),
            })?,
        )?;
        self.await_broker.cancel(suspension_id)?;
        let result = self
            .run_workflow(
                &run.execution_id,
                &workflow_id,
                &root_graph,
                0,
                Vec::new(),
                Some(state),
                host,
                cancel,
                chunk_tx,
            )
            .await;
        match result {
            Ok(status) => self.persist_workflow_status(
                run_id,
                &run.execution_id,
                claimed.revision,
                &workflow_id,
                status,
            ),
            Err(error) => self.persist_workflow_failure(
                run_id,
                &run.execution_id,
                claimed.revision,
                &workflow_id,
                error,
            ),
        }
    }

    /// 恢复进程退出前已持久化为 `Running` 的 block-boundary 状态。
    pub async fn recover_workflow(
        &self,
        run_id: &str,
        host: &mut dyn HostServices,
        cancel: &CancellationToken,
        chunk_tx: Option<&ChunkTx>,
    ) -> VsResult<DurableWorkflowRun> {
        let gate = self.run_gate(run_id);
        let _guard = gate.lock().await;
        let coordinator = self.run_coordinator();
        let (record, run) = coordinator
            .load(run_id)?
            .ok_or_else(|| VsError::Runtime(format!("unknown run {run_id}")))?;
        if run.status != RunStatus::Running {
            return Err(VsError::Runtime(format!("run {run_id} is not running")));
        }
        let StoredWorkflowCheckpoint::Resumable { workflow_id, state } =
            decode_checkpoint(&run.checkpoint)?
        else {
            return Err(VsError::Runtime(
                "run checkpoint does not contain recoverable workflow state".into(),
            ));
        };
        let root_graph = state.root_graph.clone();
        let result = self
            .run_workflow(
                &run.execution_id,
                &workflow_id,
                &root_graph,
                0,
                Vec::new(),
                Some(state),
                host,
                cancel,
                chunk_tx,
            )
            .await;
        match result {
            Ok(status) => self.persist_workflow_status(
                run_id,
                &run.execution_id,
                record.revision,
                &workflow_id,
                status,
            ),
            Err(error) => self.persist_workflow_failure(
                run_id,
                &run.execution_id,
                record.revision,
                &workflow_id,
                error,
            ),
        }
    }

    pub fn load_run(&self, run_id: &str) -> VsResult<Option<(DurableRecord, RunRecord)>> {
        self.run_coordinator().load(run_id)
    }

    /// 从持久化的挂起 checkpoint 重建内存 AwaitBroker，供进程重启后继续匹配信号。
    pub fn restore_run_awaits(&self, run_id: &str) -> VsResult<usize> {
        let (_, run) = self
            .load_run(run_id)?
            .ok_or_else(|| VsError::Runtime(format!("unknown run {run_id}")))?;
        if run.status != RunStatus::Suspended {
            return Err(VsError::Runtime(format!("run {run_id} is not suspended")));
        }
        let StoredWorkflowCheckpoint::Resumable { state, .. } = decode_checkpoint(&run.checkpoint)?
        else {
            return Err(VsError::Runtime(
                "run checkpoint does not contain resumable workflow state".into(),
            ));
        };
        let mut restored = 0;
        for suspension in state.suspensions() {
            restored += usize::from(self.await_broker.register(AwaitRequest {
                id: suspension.id,
                condition: suspension.condition,
                payload: suspension.payload,
            })?);
        }
        Ok(restored)
    }

    pub fn await_broker(&self) -> Arc<InMemoryAwaitBroker> {
        self.await_broker.clone()
    }

    pub fn set_await_broker(&mut self, broker: Arc<InMemoryAwaitBroker>) {
        self.await_broker = broker;
    }

    pub fn drain_ready_awaits(&self) -> VsResult<Vec<AwaitReady>> {
        self.await_broker.drain_ready()
    }

    pub fn event_log(&self, execution_id: &str) -> Arc<InMemoryEventLog> {
        self.event_logs
            .entry(execution_id.to_string())
            .or_insert_with(|| Arc::new(InMemoryEventLog::default()))
            .clone()
    }

    pub fn events(&self, execution_id: &str) -> Vec<ExecutionEvent> {
        self.event_logs
            .get(execution_id)
            .map_or_else(Vec::new, |log| log.events())
    }

    pub fn validate_replay(&self, execution_id: &str) -> VsResult<ReplayReport> {
        ExecutionReplay::validate(&self.events(execution_id), execution_id)
    }

    pub fn clear_events(&self, execution_id: &str) -> bool {
        self.event_logs.remove(execution_id).is_some()
    }

    pub fn run_coordinator(&self) -> RunCoordinator {
        RunCoordinator::new(self.durable_store.clone())
    }

    pub fn actor_runtime(&self) -> DurableActorRuntime {
        DurableActorRuntime::new(self.durable_store.clone())
    }

    pub fn upgrade_coordinator(&self) -> UpgradeCoordinator {
        UpgradeCoordinator::new(self.durable_store.clone())
    }

    pub async fn compensate_saga(&self, saga: &mut SagaState) -> VsResult<()> {
        let effects = self.effects.as_deref().ok_or_else(|| {
            VsError::Runtime("engine has no effect executor for Saga compensation".into())
        })?;
        SagaOrchestrator::new(effects).compensate(saga).await
    }

    pub fn admit(
        &self,
        run_id: impl Into<String>,
        tenant: impl Into<String>,
        priority: u8,
        deadline_ms: Option<u64>,
    ) -> VsResult<AdmitResult> {
        self.scheduler_lock()?
            .admit(run_id, tenant, priority, deadline_ms)
    }

    pub fn poll_scheduled(&self, now_ms: u64) -> VsResult<Vec<ScheduledRun>> {
        Ok(self.scheduler_lock()?.poll(now_ms))
    }

    pub fn complete_scheduled(&self, run_id: &str) -> VsResult<()> {
        self.scheduler_lock()?.complete(run_id)
    }

    pub fn scheduler_snapshot(&self) -> VsResult<SchedulerSnapshot> {
        Ok(self.scheduler_lock()?.snapshot())
    }

    fn workflow_vm(&self, workflow: &WorkflowProgram) -> WorkflowVm {
        WorkflowVm::new(workflow.clone())
            .with_step_budget(self.limits.step_budget)
            .with_limits(
                self.limits.max_tasks,
                self.limits.max_call_depth,
                self.limits.max_parallelism,
            )
    }

    fn require_workflow(&self, workflow_id: &str) -> VsResult<Arc<WorkflowProgram>> {
        self.workflow(workflow_id)
            .ok_or_else(|| VsError::Runtime(format!("unknown workflow {workflow_id}")))
    }

    fn execution_env<'a>(
        &'a self,
        execution_id: &'a str,
        cancel: &'a CancellationToken,
        chunk_tx: Option<&'a ChunkTx>,
        events: &'a dyn ExecutionEventSink,
    ) -> ExecutionEnv<'a> {
        ExecutionEnv {
            cancel: Some(cancel),
            chunk_tx,
            permissions: self.permissions.as_deref(),
            budget: self.budget.as_deref(),
            trace: self.trace.as_deref(),
            external: self.external.as_deref(),
            execution_id: Some(execution_id),
            effects: self.effects.as_deref(),
            clock: Some(self.clock.as_ref()),
            events: Some(events),
            policy: self.policy.as_deref(),
            taints: self.taints.as_deref(),
            determinism: self.determinism.as_deref(),
            middleware: self
                .middleware
                .iter()
                .map(|middleware| middleware.as_ref())
                .collect(),
        }
    }

    fn fanout_event_sink<'a>(&'a self, local: &'a dyn ExecutionEventSink) -> FanoutEventSink<'a> {
        FanoutEventSink {
            local,
            external: self.event_sink.as_deref(),
        }
    }

    fn register_suspensions(&self, status: &WorkflowRunStatus) -> VsResult<()> {
        let state = match status {
            WorkflowRunStatus::Suspended(state) | WorkflowRunStatus::Cancelled(state) => state,
            WorkflowRunStatus::Completed(_) => return Ok(()),
        };
        for suspension in state.suspensions() {
            self.await_broker.register(AwaitRequest {
                id: suspension.id,
                condition: suspension.condition,
                payload: suspension.payload,
            })?;
        }
        Ok(())
    }

    fn register_vm_suspension(&self, status: &VmRunStatus) -> VsResult<()> {
        let state = match status {
            VmRunStatus::Suspended(state) | VmRunStatus::Cancelled(state) => state,
            VmRunStatus::Completed => return Ok(()),
        };
        let Some(suspension) = state.suspension() else {
            return Ok(());
        };
        self.await_broker.register(AwaitRequest {
            id: suspension.id.clone(),
            condition: suspension.condition.clone(),
            payload: suspension.payload.clone(),
        })?;
        Ok(())
    }

    fn run_gate(&self, run_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.run_gates
            .entry(run_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    fn persist_workflow_status(
        &self,
        run_id: &str,
        execution_id: &str,
        expected_revision: u64,
        workflow_id: &str,
        status: WorkflowRunStatus,
    ) -> VsResult<DurableWorkflowRun> {
        let (run_status, checkpoint) = match &status {
            WorkflowRunStatus::Completed(completion) => (
                RunStatus::Completed,
                StoredWorkflowCheckpoint::Completed {
                    workflow_id: workflow_id.to_string(),
                    completion: completion.clone(),
                },
            ),
            WorkflowRunStatus::Suspended(state) => (
                RunStatus::Suspended,
                StoredWorkflowCheckpoint::Resumable {
                    workflow_id: workflow_id.to_string(),
                    state: state.clone(),
                },
            ),
            WorkflowRunStatus::Cancelled(state) => (
                RunStatus::Running,
                StoredWorkflowCheckpoint::Resumable {
                    workflow_id: workflow_id.to_string(),
                    state: state.clone(),
                },
            ),
        };
        let record = self.run_coordinator().checkpoint(
            run_id,
            expected_revision,
            execution_id,
            run_status,
            encode_checkpoint(&checkpoint)?,
        )?;
        Ok(DurableWorkflowRun {
            revision: record.revision,
            status,
        })
    }

    fn persist_workflow_failure(
        &self,
        run_id: &str,
        execution_id: &str,
        expected_revision: u64,
        workflow_id: &str,
        error: VsError,
    ) -> VsResult<DurableWorkflowRun> {
        let message = error.to_string();
        let checkpoint = StoredWorkflowCheckpoint::Failed {
            workflow_id: workflow_id.to_string(),
            message: message.clone(),
        };
        match self.run_coordinator().checkpoint(
            run_id,
            expected_revision,
            execution_id,
            RunStatus::Failed,
            encode_checkpoint(&checkpoint)?,
        ) {
            Ok(_) => Err(error),
            Err(persist_error) => Err(VsError::Runtime(format!(
                "workflow failed ({message}); persist failure also failed: {persist_error}"
            ))),
        }
    }

    fn scheduler_lock(&self) -> VsResult<std::sync::MutexGuard<'_, FairScheduler>> {
        self.scheduler
            .lock()
            .map_err(|_| VsError::Runtime("engine scheduler poisoned".into()))
    }
}

const MAX_ENGINE_ID_BYTES: usize = 128;

fn validate_identity(kind: &str, value: &str) -> VsResult<()> {
    if value.is_empty() {
        Err(VsError::Validate(format!(
            "{kind} identity must not be empty"
        )))
    } else if value.len() > MAX_ENGINE_ID_BYTES {
        Err(VsError::Validate(format!(
            "{kind} identity exceeds {MAX_ENGINE_ID_BYTES} bytes"
        )))
    } else {
        Ok(())
    }
}

fn encode_checkpoint(checkpoint: &StoredWorkflowCheckpoint) -> VsResult<Vec<u8>> {
    serde_json::to_vec(checkpoint)
        .map_err(|error| VsError::Runtime(format!("encode workflow checkpoint: {error}")))
}

fn decode_checkpoint(bytes: &[u8]) -> VsResult<StoredWorkflowCheckpoint> {
    serde_json::from_slice(bytes)
        .map_err(|error| VsError::Runtime(format!("decode workflow checkpoint: {error}")))
}
