//! # `prism_platform`
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
//! ## M3 scope — virtual memory
//! - [`vm`]: page-level reserve/commit/decommit/release, aligned reservations,
//!   guard pages, optional large (huge) pages (Linux/Windows; honest
//!   `Unsupported` on macOS), page protection, and physical-memory info, gated
//!   behind the `std` feature. This is the page substrate under the
//!   `prism_utils` allocators.
//!
//! ## M4 scope — mmap, file watching, and dynamic libraries
//! - [`fs::mmap`]: zero-copy memory-mapped files (`mmap`/`munmap`/`msync` on
//!   Unix, `CreateFileMapping`/`MapViewOfFile` on Windows) with an honest
//!   read-into-buffer fallback where mapping is unavailable. Part of [`fs`]
//!   (requires `std`).
//! - [`fs::watch`]: file-system watching for hot-reload — a native `kqueue`
//!   backend on macOS/BSD plus a portable `stat`-based polling fallback that
//!   covers Linux/Windows/everywhere. Requires the `watch` feature.
//! - [`dynlib`]: dynamic-library load / symbol lookup / unload with hot-reload
//!   via a temp copy and a per-load version (`dlopen` family on Unix,
//!   `LoadLibrary` family on Windows). Requires the `dynlib` feature.
//!
//! ## M5 scope — process, environment, standard streams, and wall clock
//! - [`process`]: raw command-line arguments ([`process::args`]), environment
//!   variable read/iterate/set/remove ([`process::env`]), and child-process
//!   spawning with pipes/wait/exit-status/kill ([`process::child`]). Requires
//!   the `std` feature.
//! - [`stdio`]: standard stdin/stdout/stderr handles plus `is_terminal`
//!   probes for wiring the app CLI and diagnostic logging. Requires `std`.
//! - [`wallclock`]: a real system wall-clock source (UTC / Unix epoch) kept
//!   type-distinct from the monotonic [`clock`], with a paired
//!   wall+monotonic [`wallclock::sample`] for log timestamps. Requires `std`.
//!
//! ## M6 scope — crash capture + Web/mobile backends + `mock`
//! - [`crash`]: async-signal-safe `POSIX` signal handlers
//!   (`SIGSEGV`/`SIGABRT`/`SIGBUS`/`SIGILL`/`SIGFPE`) on desktop Linux/macOS
//!   that capture a pre-allocated [`crash::CrashContext`] (signal, faulting
//!   address, frame-pointer backtrace, pre-registered build metadata) with
//!   no allocation and no locking in the handler, hand it to a registered
//!   handler, then re-raise so the process still dies correctly. A
//!   feature-gated in-process [`crash::mock`] backend drives the same
//!   capture pipeline without crashing, and Web/Android/iOS targets compile
//!   to an honest [`crash::CrashError::Unsupported`] with the
//!   [`platform::PlatformCaps::has_crash_capture`] bit cleared.
//!
//! This is the final milestone (M0–M6); minidump serialization and
//! cross-process handlers (design doc §24.4) remain future work.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.
//!
//! The crate uses no `unsafe` except narrowly scoped, documented FFI in
//! [`thread::affinity`], the [`vm`] virtual-memory backends, the M4
//! [`fs::mmap`] / [`fs::watch`] / [`dynlib`] backends, the two edition-2024
//! `unsafe` environment mutators in [`process::env`], and the M6 [`crash`]
//! signal backend (`sigaction` FFI plus the lock-free, pre-allocated,
//! async-signal-safe capture store); the workspace-level
//! `unsafe_code = "deny"` lint stays in force and is overridden only there via
//! per-site `#[expect(unsafe_code, reason)]` with a `// SAFETY:` comment on
//! every `unsafe` block.

extern crate alloc;

pub mod atomics;
/// Platform capability database and declarative degradation (design doc §24.6).
///
/// Pure `core`+`alloc`; available in every build configuration.
pub mod capability;
pub mod clock;
pub mod cpu;
/// Crash capture: async-signal-safe `POSIX` signal handlers, an in-process
/// `mock` backend, and honest `Unsupported` degradation on Web/mobile
/// (design doc §16, §22 M6).
pub mod crash;
/// Cross-platform filesystem facade (file/dir/path/standard directories).
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod fs;
pub mod platform;
pub mod prelude;

/// Real system wall-clock time source (UTC / Unix epoch), kept type-distinct
/// from the monotonic [`clock`].
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod wallclock;

/// Process, environment, and command-line facade (`argv`, env vars, child
/// processes).
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod process;

/// Standard stream handles (stdin/stdout/stderr) with `is_terminal` probes.
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod stdio;
/// Threads, thread-local storage, affinity, lightweight sync, and park/unpark.
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod thread;
/// Virtual memory: reserve/commit/decommit/release, aligned reservations,
/// guard pages, optional large (huge) pages, page protection, and physical
/// memory information.
///
/// Requires the `std` feature; absent in `no_std` builds.
#[cfg(feature = "std")]
pub mod vm;

/// Dynamic-library loading, symbol lookup, and hot-reload.
///
/// Requires the `dynlib` feature (which implies `std`); absent otherwise.
#[cfg(feature = "dynlib")]
pub mod dynlib;

pub use clock::{now, MonotonicNanos};
pub use cpu::CpuInfo;
pub use capability::{
    Capability, CapabilityDatabase, Category, Selection, Support, SupportLevel,
};
pub use crash::{
    install as install_crash_handler, last_context as last_crash_context,
    supported as crash_capture_supported, uninstall as uninstall_crash_handler, Backtrace,
    BuildMetadata, CrashContext, CrashError, Signal,
};
#[cfg(feature = "dynlib")]
pub use dynlib::{supported as dynlib_supported, DynlibError, Library, Symbol};
#[cfg(feature = "std")]
pub use fs::{mmap_supported, FsError, Mmap, MmapError, MmapMut, Result as FsResult};
#[cfg(feature = "watch")]
pub use fs::{Event, EventKind, WatchBackend, WatchError, Watcher};
pub use platform::{Os, Platform, PlatformCaps};
#[cfg(feature = "std")]
pub use process::{Child, Command, ExitStatus, Output, Stdio};
#[cfg(feature = "std")]
pub use stdio::{StandardError, StandardInput, StandardOutput, Stream};
#[cfg(feature = "std")]
pub use thread::{
    affinity_supported, current_id, hardware_concurrency, set_current_thread_affinity,
    set_current_thread_affinity_mask, sleep as thread_sleep, spawn, yield_now, AffinityError,
    Backoff, Builder as ThreadBuilder, JoinHandle, Once, Parker, SpinLock, SpinLockGuard, ThreadId,
    ThreadLocal, Unparker,
};
#[cfg(feature = "std")]
pub use vm::{
    huge_pages_supported, large_page_size, memory_info, page_size, virtual_memory_supported,
    MemoryInfo, Protection, Reservation, VmError,
};
#[cfg(feature = "std")]
pub use wallclock::{now as wall_now, sample as wall_sample, WallClock, WallClockSample, WallTime};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_m6_crash;

#[cfg(test)]
mod tests_capability;
