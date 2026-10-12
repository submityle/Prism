//! [`WorldId`]: a process-unique identity for a [`World`](crate::world::World).
//!
//! Query state resolves every component and archetype to a dense id *within one
//! world's registries* (design §7). Reusing a state built against world A on
//! world B would therefore reinterpret B's columns as A's component types — a
//! soundness hole an unchecked safe API must not permit. [`WorldId`] closes it:
//! [`World`](crate::world::World) stamps a fresh id on construction and every
//! safe [`QueryState`](crate::query::QueryState) entry point verifies the world
//! it is handed carries the same id before dereferencing anything.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A process-unique identifier for a [`World`](crate::world::World).
///
/// Allocated from a monotonic global counter, so no two live or dead worlds in
/// one process ever share an id. Cheap to copy and compare, and stable for the
/// lifetime of the world (moving the world does not change its id).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct WorldId(usize);

impl WorldId {
    /// Allocate the next process-unique id.
    ///
    /// `Ordering::Relaxed` is sufficient: the id is only ever compared for
    /// equality, never used to publish other memory.
    pub(crate) fn next() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw numeric value, for diagnostics/logging.
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }
}
