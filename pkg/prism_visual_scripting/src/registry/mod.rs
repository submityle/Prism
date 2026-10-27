//! 节点注册表与执行抽象。
//!
//! - [`runner`]：[`NodeRunner`] trait 与 [`FnRunner`] 适配器。
//! - [`host`]：[`HostServices`] trait + [`InvokeCtx`] + 内存实现 [`InMemoryHost`]。
//! - [`library`]：[`NodeDefinition`] / [`NodeLibrary`] / [`Purity`] / [`ExecNext`]。
//!
//! 设计要点：节点的"运行时实现"是 trait 对象，可以是纯 Rust 闭包、可以是
//! 包装 `WasmHost` 的 WASM effect runner，对编译器和 VM 透明。

pub mod host;
pub mod library;
pub mod runner;

pub use host::{
    BudgetMeter, Chunk, ChunkTx, DeterminismGate, ExecutionEnv, HostServices, InMemoryHost,
    InvokeCtx, LogLevel, NodeCallInfo, NodeMiddleware, NodePhase, PermissionGate, RunnerClass,
    TraceSink,
};
pub use library::{
    ConcurrentNodeLibrary, ExecNext, NodeDefinition, NodeLibrary, NodeRegistry, Purity,
};
pub use runner::{send_chunk, AsyncNodeRunner, FnRunner, NodeRunner, RunnerKind, SharedRunner};
