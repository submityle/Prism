//! Declarative degradation: `require(cap).select(native, fallback)`.
//!
//! Instead of scattering `if db.is_native(cap) { .. } else { .. }` across the
//! codebase, callers express intent once:
//!
//! ```
//! use prism_platform::capability::{Capability, CapabilityDatabase};
//!
//! let db = CapabilityDatabase::current();
//! // "Use huge pages if the platform supports them natively, otherwise fall
//! // back to normal allocation." Only the chosen branch is evaluated.
//! let selection = db
//!     .require(Capability::HugePages)
//!     .select(|| "huge", || "normal");
//! assert!(selection.value == "huge" || selection.value == "normal");
//! ```
//!
//! A [`Requirement`] borrows the database and resolves against it lazily, so the
//! fallback closure never runs when the native path is taken and vice versa.

use super::catalog::Capability;
use super::database::CapabilityDatabase;
use super::support::{Support, SupportLevel};

/// A pending requirement for one capability, bound to a [`CapabilityDatabase`].
///
/// Created by [`CapabilityDatabase::require`].
#[derive(Clone, Copy, Debug)]
pub struct Requirement<'a> {
    db: &'a CapabilityDatabase,
    capability: Capability,
}

/// The outcome of resolving a [`Requirement`]: the chosen value plus whether the
/// native path was taken and the underlying [`Support`] status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection<T> {
    /// The value produced by the chosen branch.
    pub value: T,
    /// `true` when the fallback branch was taken because the capability was not
    /// natively available.
    pub degraded: bool,
    /// The resolved support status that drove the choice.
    pub support: Support,
}

impl CapabilityDatabase {
    /// Begin a declarative requirement for `capability`.
    pub const fn require(&self, capability: Capability) -> Requirement<'_> {
        Requirement { db: self, capability }
    }
}

impl Requirement<'_> {
    /// The capability this requirement is for.
    pub const fn capability(&self) -> Capability {
        self.capability
    }

    /// The resolved support status.
    pub const fn support(&self) -> Support {
        self.db.support(self.capability)
    }

    /// The resolved support level.
    pub const fn level(&self) -> SupportLevel {
        self.support().level
    }

    /// Whether the native implementation is available.
    pub const fn is_native(&self) -> bool {
        self.support().is_native()
    }

    /// Whether the capability is usable at all (native or degraded).
    pub const fn is_usable(&self) -> bool {
        self.support().is_usable()
    }

    /// Choose between a native and a fallback branch, evaluating only the one
    /// that is actually taken.
    ///
    /// The native branch runs when the capability is [`SupportLevel::Native`];
    /// otherwise the fallback branch runs and [`Selection::degraded`] is `true`.
    pub fn select<T, N, F>(&self, native: N, fallback: F) -> Selection<T>
    where
        N: FnOnce() -> T,
        F: FnOnce() -> T,
    {
        let support = self.support();
        if support.is_native() {
            Selection { value: native(), degraded: false, support }
        } else {
            Selection { value: fallback(), degraded: true, support }
        }
    }

    /// Value-based convenience over [`Requirement::select`]: pick `native` when
    /// the capability is native, otherwise `fallback`. Both values are
    /// constructed eagerly; use [`Requirement::select`] when a branch is
    /// expensive or must not run unless chosen.
    pub fn or_fallback<T>(&self, native: T, fallback: T) -> Selection<T> {
        self.select(move || native, move || fallback)
    }

    /// Run `native` and return `Some` only when the capability is native;
    /// otherwise return `None` without evaluating anything. Useful when there is
    /// no fallback and the dependent feature should simply be disabled.
    pub fn native_only<T, N>(&self, native: N) -> Option<T>
    where
        N: FnOnce() -> T,
    {
        if self.is_native() { Some(native()) } else { None }
    }
}
