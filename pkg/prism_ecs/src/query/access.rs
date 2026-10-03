//! Component-access sets and the self-conflict check that keeps queries sound.
//!
//! Every [`QueryData`](crate::query::QueryData) term contributes the set of
//! components it reads and/or writes. [`Access`] accumulates those sets and
//! rejects a query whose own terms would alias mutably (e.g.
//! `Query<(&mut A, &mut A)>` or `Query<(&mut A, &A)>`). Catching this at state
//! construction turns a would-be unsound aliasing `&mut` into an immediate,
//! deterministic panic — the M0 analogue of the scheduler conflict graph the
//! design doc describes for inter-system parallelism (§8.2).

use alloc::vec::Vec;

use crate::component::ComponentId;

/// The set of components a single query reads and writes.
///
/// `writes` are exclusive: a component may be written by at most one term and
/// must not also be read by another term of the same query. `reads` may repeat
/// freely (shared `&T` access never conflicts with itself).
#[derive(Clone, Debug, Default)]
pub struct Access {
    reads: Vec<ComponentId>,
    writes: Vec<ComponentId>,
}

impl Access {
    /// An empty access set.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Components this query reads (shared `&T`). Does not include writes.
    #[inline]
    pub fn reads(&self) -> &[ComponentId] {
        &self.reads
    }

    /// Components this query writes (exclusive `&mut T`).
    #[inline]
    pub fn writes(&self) -> &[ComponentId] {
        &self.writes
    }

    /// Register a shared read of `id`.
    ///
    /// # Panics
    /// Panics if `id` is already written by this query — a read aliasing a
    /// mutable borrow is unsound.
    #[inline]
    pub fn add_read(&mut self, id: ComponentId) {
        assert!(
            !self.writes.contains(&id),
            "query conflict: component #{} is borrowed both mutably and immutably in the same query",
            id.index()
        );
        self.reads.push(id);
    }

    /// Register an exclusive write of `id`.
    ///
    /// # Panics
    /// Panics if `id` is already read or written by this query — two mutable
    /// borrows (or a mutable + shared borrow) of one component in a single
    /// query would alias.
    #[inline]
    pub fn add_write(&mut self, id: ComponentId) {
        assert!(
            !self.writes.contains(&id),
            "query conflict: component #{} is borrowed mutably more than once in the same query",
            id.index()
        );
        assert!(
            !self.reads.contains(&id),
            "query conflict: component #{} is borrowed both mutably and immutably in the same query",
            id.index()
        );
        self.writes.push(id);
    }

    /// Whether this access set and `other` conflict (one writes a component the
    /// other reads or writes). Used by the future scheduler (M1) to decide
    /// whether two systems may run in parallel.
    pub fn is_compatible(&self, other: &Access) -> bool {
        for w in &self.writes {
            if other.writes.contains(w) || other.reads.contains(w) {
                return false;
            }
        }
        for w in &other.writes {
            if self.reads.contains(w) {
                return false;
            }
        }
        true
    }
}
