//! The catalog of AAA-relevant platform capabilities and their static metadata.
//!
//! A [`Capability`] names one platform service that upper layers may require
//! (for example zero-copy [`Capability::Mmap`] or [`Capability::HugePages`]).
//! Each variant carries static, allocation-free metadata — a stable key, a
//! short human summary, and a [`Category`] — so the capability database can
//! render a startup report (design doc §24.6) without any per-platform `cfg`
//! sprinkled across call sites.

/// A coarse grouping used when rendering the capability report so related
/// capabilities print together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    /// Clocks and timing sources.
    Timing,
    /// Virtual memory and mapping.
    Memory,
    /// Filesystem and I/O services.
    Io,
    /// Threads, affinity, and parallelism substrate.
    Concurrency,
    /// Crash capture and diagnostics plumbing.
    Diagnostics,
    /// Process / environment services.
    Process,
}

impl Category {
    /// A stable lowercase key for the category.
    pub const fn key(self) -> &'static str {
        match self {
            Category::Timing => "timing",
            Category::Memory => "memory",
            Category::Io => "io",
            Category::Concurrency => "concurrency",
            Category::Diagnostics => "diagnostics",
            Category::Process => "process",
        }
    }
}

/// One AAA-relevant platform capability.
///
/// The set is intentionally closed and ordered: [`Capability::ALL`] lists every
/// variant exactly once in a deterministic order, which the database and report
/// iterate so output is byte-stable across runs and platforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// The `std` library is linked (OS services can be built at all).
    Std,
    /// A monotonic nanosecond clock for measuring elapsed time.
    MonotonicClock,
    /// A real UTC/calendar wall clock.
    WallClock,
    /// Zero-copy memory-mapped files.
    Mmap,
    /// A native OS file-watcher backend (as opposed to `stat` polling).
    NativeFileWatch,
    /// Runtime load **and unload** of dynamic libraries.
    DynlibUnload,
    /// Child-process spawning.
    Subprocess,
    /// Async-signal-safe crash capture.
    CrashCapture,
    /// Large / huge memory pages for TLB-friendly big allocations.
    HugePages,
    /// Pinning a thread to a specific CPU core.
    ThreadAffinity,
}

impl Capability {
    /// Every capability, in a fixed, stable order.
    pub const ALL: &'static [Capability] = &[
        Capability::Std,
        Capability::MonotonicClock,
        Capability::WallClock,
        Capability::Mmap,
        Capability::NativeFileWatch,
        Capability::DynlibUnload,
        Capability::Subprocess,
        Capability::CrashCapture,
        Capability::HugePages,
        Capability::ThreadAffinity,
    ];

    /// A stable lowercase key, suitable for logs and config.
    pub const fn key(self) -> &'static str {
        match self {
            Capability::Std => "std",
            Capability::MonotonicClock => "monotonic_clock",
            Capability::WallClock => "wall_clock",
            Capability::Mmap => "mmap",
            Capability::NativeFileWatch => "native_file_watch",
            Capability::DynlibUnload => "dynlib_unload",
            Capability::Subprocess => "subprocess",
            Capability::CrashCapture => "crash_capture",
            Capability::HugePages => "huge_pages",
            Capability::ThreadAffinity => "thread_affinity",
        }
    }

    /// A short human-readable summary of what the capability provides.
    pub const fn summary(self) -> &'static str {
        match self {
            Capability::Std => "standard library / OS services linked",
            Capability::MonotonicClock => "monotonic nanosecond elapsed-time clock",
            Capability::WallClock => "UTC/calendar wall clock",
            Capability::Mmap => "zero-copy memory-mapped files",
            Capability::NativeFileWatch => "native OS file-change notification backend",
            Capability::DynlibUnload => "runtime dynamic-library load and unload",
            Capability::Subprocess => "child-process spawning",
            Capability::CrashCapture => "async-signal-safe crash capture",
            Capability::HugePages => "large/huge memory pages",
            Capability::ThreadAffinity => "pin a thread to a specific CPU core",
        }
    }

    /// The category this capability belongs to.
    pub const fn category(self) -> Category {
        match self {
            Capability::MonotonicClock | Capability::WallClock => Category::Timing,
            Capability::Mmap | Capability::HugePages => Category::Memory,
            Capability::NativeFileWatch | Capability::DynlibUnload => Category::Io,
            Capability::ThreadAffinity => Category::Concurrency,
            Capability::CrashCapture => Category::Diagnostics,
            Capability::Std | Capability::Subprocess => Category::Process,
        }
    }
}
