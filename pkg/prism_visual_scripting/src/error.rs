//! 错误类型。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VsError {
    #[error("validation error: {0}")]
    Validate(String),
    #[error("compile error: {0}")]
    Compile(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("retryable runtime error: {0}")]
    RetryableRuntime(String),
    #[error("type mismatch: expected {expected:?}, got {got:?} at {location}")]
    TypeMismatch {
        expected: crate::value::ValueType,
        got: crate::value::ValueType,
        location: String,
    },
    #[error("unknown node def: {0}")]
    UnknownDef(String),
    #[error("unknown node id: {0}")]
    UnknownNode(u32),
    #[error("cycle detected involving node {0}")]
    Cycle(u32),
    #[error("execution cancelled")]
    Cancelled,
    #[error("execution suspended at {0}")]
    Suspended(String),
    #[error("async runner invoked from synchronous Vm path")]
    AsyncRunnerInSyncVm,
    #[error("effect runner invoked from synchronous Vm path")]
    EffectRunnerInSyncVm,
    #[error("effect execution requires ExecutionEnv::with_execution_id")]
    MissingExecutionId,
    #[error("effect execution requires ExecutionEnv::with_effect_executor")]
    MissingEffectExecutor,
    #[error("effect execution failed: {0}")]
    Effect(String),
}

pub type VsResult<T> = Result<T, VsError>;
