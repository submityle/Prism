//! `runtime` —— 沙箱运行时层。
//!
//! - [`engine`]   ：底层 wasmtime 引擎 + Component / InstancePre 缓存
//! - [`sandbox`]  ：单沙箱（一份策略 + Linker 配置 + 证明者）
//! - [`registry`] ：多沙箱（多租户）注册表
//! - [`canary`]   ：金丝雀发布框架 + 自愈机制

pub mod canary;
pub mod engine;
pub mod registry;
pub mod sandbox;

// 把核心类型扁平化重导出，外部既可以用 `crate::runtime::Sandbox`
// 也可以继续用 `crate::Sandbox`（顶层 lib.rs 再 re-export 一次）。
pub use canary::{CanaryRouter, RouteDecision, SelfHealing, VersionStats};
pub use engine::{default_wasi_ctx, no_wasi_ctx, SandboxEngine, StoreState};
pub use registry::{SandboxRegistry, SharedRegistry};
pub use sandbox::{CallReceipt, Sandbox};
