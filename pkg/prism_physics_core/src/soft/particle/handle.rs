//! Handles into a [`ParticleStorage`](crate::soft::particle::storage::ParticleStorage).
//!
//! A [`ParticleHandle`] is a light-weight index newtype identifying one
//! particle in the unified soft-body / cloth / rope solver. Unlike the
//! rigid-body [`BodyHandle`](crate::state::handle::BodyHandle), particles are
//! never individually freed during a simulation step (a whole soft body is
//! created or discarded as a unit), so a plain index is sufficient and cheaper
//! than a generational handle.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. An index
//! newtype over a Structure-of-Arrays store is a standard, publicly documented
//! data-structure pattern.

/// A stable reference to a particle stored in a
/// [`ParticleStorage`](crate::soft::particle::storage::ParticleStorage).
///
/// Handles are cheap to copy and compare. A handle is valid for the lifetime of
/// the store that minted it; particles are not individually recycled, so no
/// generation counter is needed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParticleHandle(u32);

impl ParticleHandle {
    /// A sentinel handle that never refers to a live particle.
    pub const INVALID: ParticleHandle = ParticleHandle(u32::MAX);

    /// Creates a handle from a raw slot index.
    ///
    /// This is `pub(crate)` because only the storage should mint handles that
    /// are guaranteed to refer to a live slot.
    #[must_use]
    pub(crate) const fn from_index(index: u32) -> ParticleHandle {
        ParticleHandle(index)
    }

    /// Returns the slot index this handle refers to, as a `usize` suitable for
    /// indexing the Structure-of-Arrays columns.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the raw `u32` slot index this handle refers to.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl Default for ParticleHandle {
    fn default() -> Self {
        ParticleHandle::INVALID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_handle_reports_sentinel() {
        assert_eq!(ParticleHandle::INVALID.raw(), u32::MAX);
    }

    #[test]
    fn index_matches_raw() {
        let h = ParticleHandle::from_index(7);
        assert_eq!(h.raw(), 7);
        assert_eq!(h.index(), 7usize);
    }

    #[test]
    fn handles_are_value_equal() {
        assert_eq!(ParticleHandle::from_index(3), ParticleHandle::from_index(3));
        assert_ne!(ParticleHandle::from_index(3), ParticleHandle::from_index(4));
    }
}
