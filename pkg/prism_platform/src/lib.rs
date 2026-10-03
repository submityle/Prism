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
//! ## M1 scope — the desktop filesystem layer
//! - [`fs`]: cross-platform file/dir/path helpers and per-OS standard
//!   directories, gated behind the `std` feature.
//!
//! Later milestones add threads/affinity/sync, virtual memory, dynamic
//! libraries, process control, and crash/minidump backends per OS.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

pub mod atomics;
pub mod clock;
pub mod cpu;
/// Cross-platform filesystem facade (file/dir/path/standard directories).
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod fs;
pub mod platform;
pub mod prelude;

pub use clock::{now, MonotonicNanos};
pub use cpu::CpuInfo;
#[cfg(feature = "std")]
pub use fs::{FsError, Result as FsResult};
pub use platform::{Os, Platform, PlatformCaps};

#[cfg(test)]
mod tests;
