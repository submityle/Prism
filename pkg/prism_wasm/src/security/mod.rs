//! `security` —— 沙箱安全层。
//!
//! - [`policy`]       ：零信任能力策略 + 资源上限
//! - [`attestation`]  ：「零知识风格」执行回执
//! - [`ebpf_bridge`]  ：eBPF + WASM 双重可信链验证

pub mod attestation;
pub mod ebpf_bridge;
pub mod policy;

pub use attestation::{Attestor, Receipt};
pub use ebpf_bridge::{CheckItem, EbpfBridge, HostCallEvent, SyscallEvent, VerifyResult};
pub use policy::{Capability, Policy, PolicyBuilder, ResourceCaps};
