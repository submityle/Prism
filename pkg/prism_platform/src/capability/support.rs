//! Support levels and the degradation path attached to each capability.
//!
//! A capability is either available natively, available only through a
//! documented degraded path, or entirely unavailable. [`Support`] encodes that
//! three-way answer together with a static description of the fallback so the
//! declarative degradation API (see [`crate::capability::require`]) and the
//! startup report can explain *why* a path is degraded without re-deriving the
//! per-platform `cfg` logic.

use super::catalog::Capability;

/// How well a capability is supported on the resolved platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SupportLevel {
    /// The capability is backed by the real, first-class native implementation.
    Native,
    /// The capability is usable, but only through a documented degraded path
    /// (for example `mmap` falling back to a buffered read, or huge pages
    /// falling back to normal pages). Correctness holds; some property (speed,
    /// latency, memory) is lost.
    Degraded,
    /// The capability is not available at all on this platform; callers must
    /// take an alternative branch or disable the dependent feature.
    Unsupported,
}

impl SupportLevel {
    /// A stable lowercase key for logs.
    pub const fn key(self) -> &'static str {
        match self {
            SupportLevel::Native => "native",
            SupportLevel::Degraded => "degraded",
            SupportLevel::Unsupported => "unsupported",
        }
    }

    /// Whether the capability can be used *at all* (native or degraded).
    pub const fn is_usable(self) -> bool {
        matches!(self, SupportLevel::Native | SupportLevel::Degraded)
    }
}

/// The resolved support status of one capability on one platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Support {
    /// Which capability this describes.
    pub capability: Capability,
    /// The resolved support level.
    pub level: SupportLevel,
    /// A static, human-readable description of the degraded path or the reason
    /// for being unavailable. Empty for [`SupportLevel::Native`].
    pub detail: &'static str,
}

impl Support {
    /// A natively supported capability.
    pub const fn native(capability: Capability) -> Self {
        Self { capability, level: SupportLevel::Native, detail: "" }
    }

    /// A capability usable only through the documented degraded path `detail`.
    pub const fn degraded(capability: Capability, detail: &'static str) -> Self {
        Self { capability, level: SupportLevel::Degraded, detail }
    }

    /// A capability that is unavailable; `detail` explains why.
    pub const fn unsupported(capability: Capability, detail: &'static str) -> Self {
        Self { capability, level: SupportLevel::Unsupported, detail }
    }

    /// Whether this capability can be used at all (native or degraded).
    pub const fn is_usable(self) -> bool {
        self.level.is_usable()
    }

    /// Whether this capability is backed by the native implementation.
    pub const fn is_native(self) -> bool {
        matches!(self.level, SupportLevel::Native)
    }
}
