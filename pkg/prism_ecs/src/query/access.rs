//! Access sets and the conflict checks that keep queries and systems sound.
//!
//! Every [`QueryData`](crate::query::QueryData) term and every
//! [`SystemParam`](crate::system::SystemParam) contributes the set of resources
//! and components it reads and/or writes. [`Access`] accumulates those sets and
//! serves two purposes:
//!
//! 1. **Intra-query / intra-system soundness** — [`add_read`](Access::add_read)
//!    and [`add_write`](Access::add_write) reject a single query or system whose
//!    own terms would alias mutably (e.g. `Query<(&mut A, &mut A)>` or a system
//!    taking both `ResMut<A>` and `Res<A>`). Catching this at construction turns
//!    a would-be unsound aliasing `&mut` into an immediate, deterministic panic.
//! 2. **Inter-system scheduling** — [`is_compatible`](Access::is_compatible)
//!    reports whether two systems' access sets conflict, which the scheduler
//!    conflict graph uses to decide whether they may run in parallel
//!    (design §8.2).
//!
//! Component access and resource access are tracked in parallel: a `&mut T`
//! query term is a component write, a `ResMut<R>` param is a resource write, and
//! so on. An *exclusive* system that takes `&mut World`
//! ([`writes_everything`](Access::writes_everything)) conflicts with every other
//! access and runs alone.

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::resource::ResourceId;

/// The set of components and resources a query or system reads and writes.
///
/// `writes` are exclusive: an item may be written by at most one term and must
/// not also be read by another term of the same query/system. `reads` may
/// repeat freely (shared `&T` access never conflicts with itself).
#[derive(Clone, Debug, Default)]
pub struct Access {
    reads: Vec<ComponentId>,
    writes: Vec<ComponentId>,
    resource_reads: Vec<ResourceId>,
    resource_writes: Vec<ResourceId>,
    /// `true` for an exclusive system that borrows the whole world (`&mut
    /// World`): it conflicts with every other access.
    writes_everything: bool,
}

impl Access {
    /// An empty access set.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Components this query/system reads (shared `&T`). Does not include writes.
    #[inline]
    pub fn reads(&self) -> &[ComponentId] {
        &self.reads
    }

    /// Components this query/system writes (exclusive `&mut T`).
    #[inline]
    pub fn writes(&self) -> &[ComponentId] {
        &self.writes
    }

    /// Resources this system reads (`Res<T>`). Does not include writes.
    #[inline]
    pub fn resource_reads(&self) -> &[ResourceId] {
        &self.resource_reads
    }

    /// Resources this system writes (`ResMut<T>`).
    #[inline]
    pub fn resource_writes(&self) -> &[ResourceId] {
        &self.resource_writes
    }

    /// Whether this access set borrows the entire world exclusively (an
    /// exclusive system). Such a set is incompatible with every other.
    #[inline]
    pub fn writes_everything(&self) -> bool {
        self.writes_everything
    }

    /// Register a shared read of component `id`.
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

    /// Register a shared read of component `id` performed by a *filter* term
    /// (`Added<T>` / `Changed<T>`), which inspects the component's change ticks
    /// but yields no data.
    ///
    /// Unlike [`add_read`](Access::add_read) this never panics: a filter may
    /// legitimately read the ticks of a component the query also writes (e.g.
    /// `Query<&mut A, Changed<A>>`), and a filter read already covered by an
    /// existing read or write adds no new borrow. The id is pushed only when it
    /// is not already present in `reads`/`writes`, keeping the access set tidy.
    #[inline]
    pub fn add_filter_read(&mut self, id: ComponentId) {
        if !self.reads.contains(&id) && !self.writes.contains(&id) {
            self.reads.push(id);
        }
    }

    /// Register an exclusive write of component `id`.
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

    /// Register a shared read of resource `id`.
    ///
    /// # Panics
    /// Panics if `id` is already written by this system — a `Res<T>` aliasing a
    /// `ResMut<T>` in one system is unsound.
    #[inline]
    pub fn add_resource_read(&mut self, id: ResourceId) {
        assert!(
            !self.resource_writes.contains(&id),
            "system conflict: resource #{} is borrowed both mutably and immutably in the same system",
            id.index()
        );
        self.resource_reads.push(id);
    }

    /// Register an exclusive write of resource `id`.
    ///
    /// # Panics
    /// Panics if `id` is already read or written by this system — two
    /// `ResMut<T>` (or a `ResMut<T>` + `Res<T>`) in one system would alias.
    #[inline]
    pub fn add_resource_write(&mut self, id: ResourceId) {
        assert!(
            !self.resource_writes.contains(&id),
            "system conflict: resource #{} is borrowed mutably more than once in the same system",
            id.index()
        );
        assert!(
            !self.resource_reads.contains(&id),
            "system conflict: resource #{} is borrowed both mutably and immutably in the same system",
            id.index()
        );
        self.resource_writes.push(id);
    }

    /// Mark this access set as borrowing the entire world exclusively. Used by
    /// exclusive systems, which the scheduler must run with no other system in
    /// flight.
    #[inline]
    pub fn set_writes_everything(&mut self) {
        self.writes_everything = true;
    }

    /// Merge `other` into `self`, combining both reads/writes.
    ///
    /// Unlike [`add_read`](Access::add_read) / [`add_write`](Access::add_write)
    /// this does **not** panic on overlap: it is used to accumulate the access
    /// of several independent system params, which may legitimately read the
    /// same component. Soundness of such sharing is a scheduling concern handled
    /// by [`is_compatible`](Access::is_compatible), not an intra-system alias.
    pub fn extend(&mut self, other: &Access) {
        self.reads.extend_from_slice(&other.reads);
        self.writes.extend_from_slice(&other.writes);
        self.resource_reads.extend_from_slice(&other.resource_reads);
        self.resource_writes
            .extend_from_slice(&other.resource_writes);
        self.writes_everything |= other.writes_everything;
    }

    /// Whether this access set and `other` are compatible — i.e. neither writes
    /// anything the other reads or writes. Two compatible systems may run in
    /// parallel; incompatible ones must be ordered (design §8.2).
    pub fn is_compatible(&self, other: &Access) -> bool {
        if self.writes_everything || other.writes_everything {
            return false;
        }
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
        for w in &self.resource_writes {
            if other.resource_writes.contains(w) || other.resource_reads.contains(w) {
                return false;
            }
        }
        for w in &other.resource_writes {
            if self.resource_reads.contains(w) {
                return false;
            }
        }
        true
    }
}
