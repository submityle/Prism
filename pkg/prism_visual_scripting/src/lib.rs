//! prism_visual_scripting —— Prism 可视化脚本与持久化执行引擎。
//!
//! The backend-neutral graph compiler, slot VM, durable workflow runtime, and
//! effect contracts live here. Host integrations remain explicit so ECS and
//! non-Send runtime state never leak into the core VM.

extern crate alloc;

pub mod actor;
pub mod analysis;
pub mod awaiting;
pub mod builtin;
pub mod compiler;
pub mod coordinator;
pub mod durable;
pub mod effect;
pub mod engine;
pub mod error;
pub mod execution;
pub mod ir;
pub mod migration;
pub mod model;
pub mod policy;
pub mod prelude;
pub mod prism_adapter;
pub mod registry;
pub mod saga;
pub mod scheduler;
pub mod simulation;
pub mod stream;
pub mod stream_advanced;
pub mod upgrade;
pub mod value;
pub mod vm;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod workflow;

pub use actor::{
    ActorHandler, ActorId, ActorLease, ActorMessage, ActorSignal, ActorState, ActorStore,
    DurableActorRuntime, InMemoryActorStore,
};
pub use analysis::{
    analyze_graph, FindingSeverity, StaticBounds, StaticFinding, StaticModelReport,
};
pub use awaiting::{
    AwaitBrokerState, AwaitCondition, AwaitReady, AwaitRequest, InMemoryAwaitBroker,
};
pub use compiler::{compile, compile_to_plan, instantiate, plan};
pub use coordinator::VersionedCheckpoint;
pub use coordinator::{RunCoordinator, RunRecord, RunStatus};
pub use durable::{
    DurableMutation, DurableRecord, DurableStore, FileDurableStore, InMemoryDurableStore,
    InboxEntry, OutboxEntry, RecordKey, RevisionGuard,
};
pub use effect::{
    DurableEffectExecutor, DurableEffectState, EffectClass, EffectError, EffectExecutor,
    EffectRequest, EffectResponse, EffectSpec, ExecutionClock, IdempotencyPolicy,
    MemoizingEffectExecutor, RetryPolicy, TokioClock, VirtualClock,
};
pub use engine::{DurableWorkflowRun, EngineLimits, VisualScriptEngine, WorkflowStart};
pub use error::{VsError, VsResult};
pub use execution::{
    ExecutionEvent, ExecutionEventKind, ExecutionEventSink, ExecutionReplay, InMemoryEventLog,
    ReplayDivergence, ReplayReport,
};
pub use ir::{Block, ExecutionPlan, Instr, SlotProgram};
pub use migration::{migration_digest, MigrationDescriptor, MigrationFn, MigrationRegistry};
pub use model::{
    ControlNode, Edge, Endpoint, Graph, GraphCall, JoinPolicy, Node, NodeDefRef, NodeId, Pin,
    PinDir, PinIndex, Variable,
};
pub use prism_adapter::{
    diagnostic_for_error, diagnostic_for_finding, diagnostics_for_report, value_from_reflect,
    value_to_reflect, JsonValue, PrismHost, ReflectValueError, TaskId, TaskPoll,
};
pub use registry::{
    send_chunk, AsyncNodeRunner, BudgetMeter, Chunk, ChunkTx, DeterminismGate, ExecNext,
    ExecutionEnv, FnRunner, HostServices, InMemoryHost, InvokeCtx, LogLevel, NodeCallInfo,
    NodeDefinition, NodeLibrary, NodeMiddleware, NodePhase, NodeRegistry, NodeRunner,
    PermissionGate, Purity, RunnerClass, RunnerKind, TraceSink,
};
pub use saga::{
    CompensationIntent, PendingCompensation, SagaOrchestrator, SagaState, SagaStatus, SagaStep,
};
pub use scheduler::{AdmitResult, FairScheduler, ScheduledRun, SchedulerLimits, SchedulerSnapshot};
pub use simulation::{SimulationDeterminismGate, SimulationHarness, SimulationTrace};
pub use stream::{
    AggregationSpec, BackpressurePolicy, ClosedWindow, LateEvent, ReductionOutcome, StreamEvent,
    StreamIngest, StreamIngestError, WindowAggregate, WindowReducer, WindowSpec, WindowedStream,
};
pub use stream_advanced::{
    IntervalJoin, JoinSide, JoinedEvent, KeyedStreamEvent, PartitionProgress, PartitionWatermarks,
    SessionWindow, SessionWindowStream, SourceOffset, SourceOffsets,
};
pub use upgrade::{ShadowExecutor, UpgradeCoordinator, UpgradeRecord, UpgradeStatus};
pub use value::{Value, ValueType};
pub use vm::{Suspension, SuspensionResolution, Vm, VmRunStatus, VmState};
#[cfg(feature = "wasm")]
pub use wasm::{WasmEffectExecutor, WasmHost};
pub use workflow::{
    WorkflowCompletion, WorkflowProgram, WorkflowRunStatus, WorkflowVm, WorkflowVmState,
};
