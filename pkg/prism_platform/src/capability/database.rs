//! The capability database and degradation matrix (design doc §24.6).
//!
//! [`CapabilityDatabase`] resolves every [`Capability`] to a [`Support`] status
//! for one platform, derived from the coarse [`Os`] family and the compiled-in
//! [`PlatformCaps`] bits. It is a pure, allocation-free data structure: build it
//! once from facts, then query it anywhere instead of re-deriving `cfg!(...)`
//! logic at each call site. [`CapabilityDatabase::from_platform`] accepts the
//! facts explicitly so the full matrix is unit-testable for *every* platform,
//! not only the host that happens to run the tests; [`CapabilityDatabase::current`]
//! fills those facts in from the running build.

use super::catalog::Capability;
use super::support::{Support, SupportLevel};
use crate::platform::{Os, PlatformCaps};

/// Resolved support for every [`Capability`] on a single platform.
///
/// The entries are stored in [`Capability::ALL`] order, so iteration is
/// deterministic and byte-stable across runs and platforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityDatabase {
    entries: [Support; Capability::ALL.len()],
}

impl CapabilityDatabase {
    /// Resolve the full matrix from an OS family and a capability-flag snapshot.
    ///
    /// This is the single source of truth for degradation: a capability that is
    /// not natively available is mapped either to its documented
    /// [`SupportLevel::Degraded`] fallback or to [`SupportLevel::Unsupported`]
    /// with a reason. Huge pages and thread affinity are not individual
    /// [`PlatformCaps`] bits, so they are derived from the OS family the same
    /// way `prism_platform`'s `vm` and `thread::affinity` backends decide at
    /// compile time.
    pub const fn from_platform(os: Os, caps: PlatformCaps) -> Self {
        Self {
            entries: [
                Self::resolve_std(caps),
                Self::resolve_monotonic(caps),
                Self::resolve_wall(caps),
                Self::resolve_mmap(caps),
                Self::resolve_file_watch(caps),
                Self::resolve_dynlib(caps),
                Self::resolve_subprocess(caps),
                Self::resolve_crash(caps),
                Self::resolve_huge_pages(os, caps),
                Self::resolve_affinity(os, caps),
            ],
        }
    }

    /// Resolve the matrix for the current build.
    pub fn current() -> Self {
        Self::from_platform(Os::current(), PlatformCaps::current())
    }

    /// The resolved support status for `capability`.
    pub const fn support(&self, capability: Capability) -> Support {
        // `Capability::ALL` and `entries` share the same fixed order, so the
        // enum's position is a direct index. A linear scan keeps this a `const
        // fn` without unstable features and is trivial for ten entries.
        let mut i = 0;
        while i < self.entries.len() {
            if cap_eq(self.entries[i].capability, capability) {
                return self.entries[i];
            }
            i += 1;
        }
        // Unreachable in practice: every variant is present in `entries`.
        self.entries[0]
    }

    /// The support level for `capability`.
    pub const fn level(&self, capability: Capability) -> SupportLevel {
        self.support(capability).level
    }

    /// Whether `capability` is backed by its native implementation.
    pub const fn is_native(&self, capability: Capability) -> bool {
        self.support(capability).is_native()
    }

    /// Whether `capability` can be used at all (native or degraded).
    pub const fn is_usable(&self, capability: Capability) -> bool {
        self.support(capability).is_usable()
    }

    /// All resolved statuses, in [`Capability::ALL`] order.
    pub const fn entries(&self) -> &[Support] {
        &self.entries
    }

    /// Iterate the statuses that are not native (degraded or unsupported) — the
    /// "degradation matrix" view used to flag which AAA paths run degraded on
    /// this platform.
    pub fn degraded(&self) -> impl Iterator<Item = Support> + '_ {
        self.entries.iter().copied().filter(|s| !s.is_native())
    }

    /// Count of capabilities at the given support level.
    pub fn count_at(&self, level: SupportLevel) -> usize {
        self.entries.iter().filter(|s| s.level == level).count()
    }

    const fn resolve_std(caps: PlatformCaps) -> Support {
        if caps.has_std {
            Support::native(Capability::Std)
        } else {
            Support::unsupported(Capability::Std, "no_std build: OS services are not linked")
        }
    }

    const fn resolve_monotonic(caps: PlatformCaps) -> Support {
        if caps.has_monotonic_clock {
            Support::native(Capability::MonotonicClock)
        } else {
            Support::unsupported(
                Capability::MonotonicClock,
                "no monotonic clock source on this target",
            )
        }
    }

    const fn resolve_wall(caps: PlatformCaps) -> Support {
        if caps.has_wall_clock {
            Support::native(Capability::WallClock)
        } else {
            Support::unsupported(Capability::WallClock, "wall clock requires std")
        }
    }

    const fn resolve_mmap(caps: PlatformCaps) -> Support {
        if caps.has_mmap {
            Support::native(Capability::Mmap)
        } else if caps.has_std {
            Support::degraded(
                Capability::Mmap,
                "zero-copy mapping unavailable; falls back to buffered read into a heap buffer",
            )
        } else {
            Support::unsupported(
                Capability::Mmap,
                "mapping and the buffered fallback both require std",
            )
        }
    }

    const fn resolve_file_watch(caps: PlatformCaps) -> Support {
        if caps.has_native_file_watch {
            Support::native(Capability::NativeFileWatch)
        } else if caps.has_std {
            Support::degraded(
                Capability::NativeFileWatch,
                "no native backend; falls back to portable stat-based polling (requires the watch feature)",
            )
        } else {
            Support::unsupported(Capability::NativeFileWatch, "file watching requires std")
        }
    }

    const fn resolve_dynlib(caps: PlatformCaps) -> Support {
        if caps.has_dynlib_unload {
            Support::native(Capability::DynlibUnload)
        } else {
            Support::unsupported(
                Capability::DynlibUnload,
                "no runtime loader (dynlib feature off, or target has no loader)",
            )
        }
    }

    const fn resolve_subprocess(caps: PlatformCaps) -> Support {
        if caps.has_subprocess {
            Support::native(Capability::Subprocess)
        } else {
            Support::unsupported(
                Capability::Subprocess,
                "no process model on this target (e.g. wasm) or std off",
            )
        }
    }

    const fn resolve_crash(caps: PlatformCaps) -> Support {
        if caps.has_crash_capture {
            Support::native(Capability::CrashCapture)
        } else {
            Support::unsupported(
                Capability::CrashCapture,
                "no async-signal-safe crash backend on this target",
            )
        }
    }

    const fn resolve_huge_pages(os: Os, caps: PlatformCaps) -> Support {
        if !caps.has_std {
            return Support::unsupported(
                Capability::HugePages,
                "virtual memory services require std",
            );
        }
        match os {
            Os::Windows | Os::Linux | Os::Android => Support::native(Capability::HugePages),
            Os::Apple => Support::degraded(
                Capability::HugePages,
                "no explicit huge-page API; falls back to normal pages (correct, no TLB win)",
            ),
            Os::Web | Os::Unknown => Support::degraded(
                Capability::HugePages,
                "no huge-page concept; falls back to normal allocation",
            ),
        }
    }

    const fn resolve_affinity(os: Os, caps: PlatformCaps) -> Support {
        if !caps.has_std {
            return Support::unsupported(
                Capability::ThreadAffinity,
                "threads/affinity require std",
            );
        }
        match os {
            Os::Windows | Os::Linux | Os::Android => Support::native(Capability::ThreadAffinity),
            Os::Apple => Support::unsupported(
                Capability::ThreadAffinity,
                "macOS/iOS expose no thread-to-core pinning API; the scheduler decides",
            ),
            Os::Web => Support::unsupported(
                Capability::ThreadAffinity,
                "web workers cannot be pinned to a core",
            ),
            Os::Unknown => {
                Support::unsupported(Capability::ThreadAffinity, "no affinity API on this target")
            }
        }
    }
}

/// `const fn`-compatible equality for the small [`Capability`] enum.
///
/// `PartialEq::eq` is not a `const fn` on stable, so the `const fn` lookups
/// above compare the stable string keys byte for byte instead.
const fn cap_eq(a: Capability, b: Capability) -> bool {
    let (x, y) = (a.key().as_bytes(), b.key().as_bytes());
    if x.len() != y.len() {
        return false;
    }
    let mut i = 0;
    while i < x.len() {
        if x[i] != y[i] {
            return false;
        }
        i += 1;
    }
    true
}
