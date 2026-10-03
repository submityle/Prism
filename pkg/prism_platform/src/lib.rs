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
//! ## M2 scope — threads and synchronization
//! - [`thread`]: thread spawning/join, a registry-based thread-local, CPU-core
//!   affinity (Linux/Windows pinning, honest `Unsupported` on macOS), a spin
//!   lock + backoff + one-time init, and a token-based parker, all gated behind
//!   the `std` feature.
//!
//! Later milestones add virtual memory, dynamic libraries, process control,
//! and crash/minidump backends per OS.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.
//!
//! The crate uses no `unsafe` except narrowly scoped, documented FFI in
//! [`thread::affinity`]; the workspace-level `unsafe_code = "deny"` lint stays
//! in force and is overridden only there via `#[expect(unsafe_code, reason)]`.

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
/// Threads, thread-local storage, affinity, lightweight sync, and park/unpark.
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod thread;

pub use clock::{now, MonotonicNanos};
pub use cpu::CpuInfo;
#[cfg(feature = "std")]
pub use fs::{FsError, Result as FsResult};
pub use platform::{Os, Platform, PlatformCaps};
#[cfg(feature = "std")]
pub use thread::{
    affinity_supported, current_id, hardware_concurrency, set_current_thread_affinity,
    set_current_thread_affinity_mask, sleep as thread_sleep, spawn, yield_now, AffinityError,
    Backoff, Builder as ThreadBuilder, JoinHandle, Once, Parker, SpinLock, SpinLockGuard,
    ThreadId, ThreadLocal, Unparker,
};

#[cfg(test)]
mod tests;
