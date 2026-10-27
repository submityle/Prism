//! prism_wasm —— 通用 WebAssembly 沙箱引擎（Component Model + WASI Preview 2 + 多沙箱）。
//!
//! # 目录结构
//! ```text
//! src/
//!   ├─ lib.rs
//!   ├─ runtime/      运行时层
//!   │    ├─ engine.rs      引擎 + Component / InstancePre 缓存 + WASI p2 状态
//!   │    ├─ sandbox.rs     单沙箱（policy + linker + attestor）
//!   │    ├─ registry.rs    多沙箱（多租户）注册表
//!   │    └─ canary.rs      金丝雀发布框架 + 自愈机制
//!   ├─ loader/       入口层
//!   │    ├─ mod.rs         Loader trait + FileLoader / MemoryLoader / ChainLoader
//!   │    └─ hotswap.rs     mtime 轮询 + 原子指针热插拔
//!   ├─ security/     安全层
//!   │    ├─ policy.rs       零信任能力策略 + 资源上限
//!   │    ├─ attestation.rs  「零知识风格」执行回执
//!   │    └─ ebpf_bridge.rs  eBPF + WASM 双重可信链验证
//!   └─ debug/        调试层
//!        └─ timetravel.rs   时间旅行调试：快照/倒带/差分/重放
//! ```

pub mod debug;
pub mod loader;
pub mod runtime;
pub mod security;

// ---- 顶层扁平 re-export，保持外部使用习惯 ----

// 加载器
pub use loader::{ChainLoader, FileLoader, HotSwap, Loader, MemoryLoader, ModuleSource};

// 运行时
pub use runtime::{
    default_wasi_ctx, no_wasi_ctx, CallReceipt, CanaryRouter, RouteDecision, Sandbox,
    SandboxEngine, SandboxRegistry, SelfHealing, SharedRegistry, StoreState, VersionStats,
};

// 安全
pub use security::{
    Attestor, Capability, CheckItem, EbpfBridge, HostCallEvent, Policy, PolicyBuilder, Receipt,
    ResourceCaps, SyscallEvent, VerifyResult,
};

// 调试
pub use debug::{ReplayResult, Snapshot, SnapshotDiff, TimeTravelSession};

// ---- 兼容旧路径，外部代码若已用 `prism_wasm::engine::xxx` 等深路径仍可工作 ----
pub use debug::timetravel;
pub use loader::hotswap;
pub use runtime::canary;
pub use runtime::engine;
pub use runtime::registry;
pub use runtime::sandbox;
pub use security::attestation;
pub use security::ebpf_bridge;
pub use security::policy;
