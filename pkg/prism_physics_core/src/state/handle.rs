//! Generational handles into [`BodyStorage`](crate::state::storage::BodyStorage).
//!
//! A [`BodyHandle`] combines a slot index with a generation counter. When a
//! slot is freed and later reused, its generation is incremented, so a stale
//! handle to the old occupant no longer validates. This makes use-after-free
//! of body slots detectable rather than silently aliasing a new body.

/// A stable, generational reference to a body stored in a
/// [`BodyStorage`](crate::state::storage::BodyStorage).
///
/// Handles are cheap to copy and compare. A handle is only valid while the
/// generation stored in the target slot matches the handle's generation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BodyHandle {
    index: u32,
    generation: u32,
}

impl BodyHandle {
    /// A sentinel handle that never refers to a live body.
    pub const INVALID: BodyHandle = BodyHandle {
        index: u32::MAX,
        generation: u32::MAX,
    };

    /// Creates a handle from raw parts. This is `pub(crate)` because only the
    /// storage should mint valid handles.
    #[must_use]
    pub(crate) const fn new(index: u32, generation: u32) -> BodyHandle {
        BodyHandle { index, generation }
    }

    /// Returns the slot index this handle refers to.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation this handle was minted with.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }
}

impl Default for BodyHandle {
    fn default() -> Self {
        BodyHandle::INVALID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_handle_reports_sentinel_parts() {
        assert_eq!(BodyHandle::INVALID.index(), u32::MAX);
        assert_eq!(BodyHandle::INVALID.generation(), u32::MAX);
    }

    #[test]
    fn handles_are_value_equal() {
        let a = BodyHandle::new(3, 1);
        let b = BodyHandle::new(3, 1);
        let c = BodyHandle::new(3, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
