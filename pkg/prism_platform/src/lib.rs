//! # prism_platform
//!
//! Prism's platform-abstraction kernel. It sits at the root of the dependency
//! graph alongside `prism_math` and `prism_utils` and is depended on by
//! `prism_time`, `prism_tasks`, `prism_diagnostic`, and the app/asset layers.
//!
//! ## M0 scope (this build) — the "no-OS" probe layer
//! - [`atomics`]: atomic/ordering/`pause`/fence facade over `core`.
//! - [`cpu::CpuInfo`]: instruction-set (SSE2/AVX2/AVX-512/NEON) and core-count
//!   probing feeding SIMD dispatch and thread-pool sizing.
//! - [`clock::now`]: a monotonic nanosecond clock.
//! - [`platform::Platform`]/[`platform::PlatformCaps`]: OS family + capability
//!   skeleton.
//!
//! Later milestones add filesystem, threads/affinity/sync, virtual memory,
//! dynamic libraries, process control, and crash/minidump backends per OS.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod atomics;
pub mod clock;
pub mod cpu;
pub mod platform;
pub mod prelude;

pub use clock::{now, MonotonicNanos};
pub use cpu::CpuInfo;
pub use platform::{Os, Platform, PlatformCaps};

#[cfg(test)]
mod tests;
