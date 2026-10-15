//! A deterministic, pure-data StateTree ST0 semantic kernel.
//!
//! The crate implements hierarchical atomic/compound/parallel states,
//! eventless and event-triggered transitions, LCA lifecycle ordering,
//! shallow/deep history, a bounded event queue, and run-to-completion
//! microsteps. It deliberately has no ECS, renderer, networking, editor, or
//! task-system dependency.
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod behavior_pack;
mod binding;
mod codec;
mod counterfactual;
mod debugger;
mod frame;
mod group_contract;
mod model;
mod model_verify;
mod replication_contract;
mod replay;
mod resilience;
mod revision;
mod runtime;
mod schema;
mod spatial_query;
mod snapshot;
mod task_contract;
mod telemetry;
mod validator;

pub use behavior_pack::{
    BehaviorPack, ContractKind, DependencyId, DependencyRequirement, GroupSelectionError,
    InstalledDependency, InstanceGroupId, InstanceGroupSelector, InstanceId, PackDefinitionError,
    PackLoadContext, PackLoadError, PackLoadReport, PlanCompatibility, ProvidedEvent, ProvidedSlot,
    RolloutConfig, RolloutController, RolloutError, RolloutEvent, RolloutState, SemVer,
    VersionRequirement, BASIS_POINTS, MAX_INSTANCE_GROUPS, MAX_PACK_DEPENDENCIES, MAX_PACK_EVENTS,
    MAX_PACK_MIGRATIONS, MAX_PACK_SLOTS, MAX_ROLLOUT_STEPS,
};
pub use binding::{
    BindingPath, BindingPathError, BindingSlot, DependencyError, DependencyPlan,
    EvaluatorDependency,
};
pub use codec::{
    FrameMigration, FrameSnapshot, SlotCodec, SlotDecl, SlotGuid, SlotType, SlotValue,
};
pub use counterfactual::{
    CounterfactualDiff, CounterfactualError, CounterfactualOverride, CounterfactualRecord,
    CounterfactualTrace, DecisionTag, DecisionTagEdit, MAX_COUNTERFACTUAL_OVERRIDES,
    MAX_COUNTERFACTUAL_RECORDS,
};
pub use debugger::{
    Breakpoint, CapacityKind as DebuggerCapacityKind, DebugEvent, DebugObservation, Debugger,
    DebuggerCapacity, DebuggerError, HitRecord, MigrationCause, MigrationCauseId, MigrationChain,
    TaskId, TaskOutcome, TraceCursor, MAX_DEBUGGER_BREAKPOINTS, MAX_DEBUGGER_HITS,
    MAX_DEBUGGER_MIGRATION_CAUSES,
};
pub use frame::{FrameError, FrameField, FrameLayout, InstanceFrame};
pub use group_contract::{
    AccessContractError, AccessGrant, AccessMode, ActorId, BlackboardError, BlackboardKey, ClaimId,
    ClaimLease, ClaimMode, ConcurrentWritePolicy, GroupAccessContract, GroupNode, GroupNodeId,
    GroupTree, GroupTreeError, NegotiationDecision, NegotiationDecisionKind, NegotiationOperation,
    NegotiationRejectReason, NegotiationRequest, RequestId, SharedBlackboard, SharedBlackboardKey,
};
pub use model::{
    EventId, GuardId, HistoryMode, NodeKind, Plan, PlanError, StateId, StateNode, Transition,
    TransitionKind, TransitionTarget, TreeDef, Trigger,
};
pub use model_verify::{
    verify, Diagnostic as ModelDiagnostic, DiagnosticCode as ModelDiagnosticCode,
    DiagnosticSeverity as ModelDiagnosticSeverity, PathStep, PathStepKind, TransitionCoverage,
    VerificationLimits, VerificationPath, VerificationReport,
};
pub use replication_contract::{
    AuthorityMode, PlanRevisionHandshake, PlanRevisionHandshakeError,
    PlanRevisionHandshakePolicy, PlanRevisionHandshakeStatus, PredictionSafety,
    PredictionSafetyReason, ReplicationContractError, ReplicationPolicy, RollbackInput,
    RollbackInputBuffer, RollbackInputContract, StateTreeReplicationPolicy, MAX_ROLLBACK_INPUTS,
    MAX_ROLLBACK_INPUT_BYTES, MAX_ROLLBACK_INPUT_SIZE,
};
pub use replay::{
    active_state_digest, ActiveStateDigest, DivergenceReport, PlanDigest, ReplayError,
    ReplayReport, TraceInput, TraceLog, TraceLogError, TraceRecord,
};
pub use resilience::{
    BudgetChargeError, BudgetExhaustion, BudgetUsage, CircuitAdmission, CircuitBreaker,
    CircuitBreakerConfig, CircuitConfigError, CircuitState, CircuitTransition, EventBuffer,
    EventBufferError, FailureReason, FrameAdmission, FrameBudget, FrameBudgetLimits, FrameIndex,
    FramePermit, FrameReport, FrameResult, ResilienceError, ResilienceEvent, ResilienceEventKind,
    SafeMode, TreeId, TreeResilience, MAX_EVENT_CAPACITY,
};
pub use revision::{
    CompatibilityClass, MigrationAction, MigrationDecision, MigrationDecisionSet, MigrationKey,
    MigrationRejection, MigrationVerdict, PlanRevision, RevisionError, MAX_MIGRATION_DECISIONS,
};
pub use runtime::{
    AlwaysGuard, Event, EventQueueError, Guard, GuardContext, Instance, LifecycleCallback,
    LogEntry, LogKind, MicrostepLimit, NoopLifecycle, RuntimeError, StepReport,
};
pub use schema::{
    ContextRef, Schema, SchemaCapabilities, SchemaDescriptor, SchemaError, SchemaId, SchemaVersion,
};
pub use snapshot::{RestoreError, Snapshot, SnapshotCodecError};
pub use spatial_query::{
    FixedVec3, QueryCandidate, QueryError, QueryEvaluator, QueryKind, QueryRequest, QueryResult,
    QuerySample, QueryWeights, SampleSet, ScoredCandidate, MAX_QUERY_CANDIDATES, MAX_QUERY_RESULTS,
    MAX_QUERY_SAMPLES, QUERY_SCORE_SCALE,
};
pub use task_contract::{
    AbortReason, AccessSet, Cadence, CadenceError, CancelWindow, CancelWindowError, InterruptTier,
    InterruptTierError, InterruptTierSet, Outcome, ReplaySafety, RequestKind, RequestQueue,
    RequestQueueError,
};
pub use telemetry::{
    BudgetMetric, GuardMetric, LodMetric, MigrationMetric, ResidencyMetric, TelemetryCapacity,
    TelemetryCollector, TelemetryError, TelemetryMetric, TelemetryReport,
};
pub use validator::{
    validate, Diagnostic, DiagnosticCode, DiagnosticSeverity, ValidationLimits, ValidationReport,
    Validator,
};

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod binding_tests;
#[cfg(test)]
mod tests;
