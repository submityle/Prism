//! Concrete registry of virtual resources and their lifecycle state.
//!
//! [`VirtualResourceTable`] is the shared, cross-subsystem store the scheduler
//! reads: it maps each key to its [`RequestPriority`], optional parent (for
//! dependency chains), byte cost, [`ResidencyState`], and invalidation epoch.
//! Texture streaming, virtual shadows, virtual geometry, and any other consumer
//! register their demands here, and the greedy pass in
//! [`crate::virtual_resource::scheduler`] turns the table into a deterministic
//! plan.
//!
//! The table is keyed on an ordered key type and iterates in key order, so every
//! query and plan is reproducible. It holds no `GPU` handle: residency here is
//! `CPU`-side truth, and the backend drives the real uploads, pending the `GPU`
//! backend.

use super::state::IllegalTransition;
use super::{RequestPriority, ResidencyState, VirtualResourceClient};
use alloc::collections::BTreeMap;
use core::hash::Hash;

/// Key requirements for a resource in the table.
///
/// Ordering is mandatory so the underlying [`BTreeMap`] iterates deterministically
/// and the scheduler's tie-breaks are exact.
pub trait ResourceKey: Copy + Eq + Ord + Hash {}
impl<T: Copy + Eq + Ord + Hash> ResourceKey for T {}

/// Why a [`VirtualResourceTable::transition`] call failed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TransitionError {
    /// No resource with the given key is registered.
    Unknown,
    /// The resource exists but the requested lifecycle edge is illegal.
    Illegal(IllegalTransition),
}

/// Per-resource bookkeeping held by the table.
///
/// Pure integer / enum state, so equality and ordering are exact and no
/// floating-point comparison is ever involved.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceEntry<K> {
    /// Scheduling priority; greater is admitted first.
    pub priority: RequestPriority,
    /// Parent whose residency this resource depends on, if any.
    pub parent: Option<K>,
    /// Physical byte cost once resident.
    pub byte_cost: u64,
    /// Current lifecycle state.
    pub state: ResidencyState,
    /// Latest invalidation epoch applied to this resource.
    pub epoch: u64,
    /// Frame index this resource was last touched, for recency tie-breaks.
    pub last_used_frame: u64,
}

impl<K> ResourceEntry<K> {
    /// Whether the resource currently occupies physical storage.
    #[must_use]
    pub const fn is_resident(&self) -> bool {
        self.state.is_resident()
    }
}

/// Session-lifetime table of virtual resources.
#[derive(Clone, Debug)]
pub struct VirtualResourceTable<K: ResourceKey> {
    entries: BTreeMap<K, ResourceEntry<K>>,
}

impl<K: ResourceKey> Default for VirtualResourceTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: ResourceKey> VirtualResourceTable<K> {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Number of registered resources.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no resources are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `key` is registered.
    #[must_use]
    pub fn contains(&self, key: K) -> bool {
        self.entries.contains_key(&key)
    }

    /// Registers `key` (or updates its priority and byte cost), starting a fresh
    /// resource in [`ResidencyState::Missing`].
    ///
    /// Re-declaring an existing resource keeps its current state, epoch, and
    /// parent while refreshing priority and byte cost, so a subsystem can re-post
    /// its demand each frame without resetting the lifecycle.
    pub fn declare(&mut self, key: K, priority: RequestPriority, byte_cost: u64) {
        self.entries
            .entry(key)
            .and_modify(|e| {
                e.priority = priority;
                e.byte_cost = byte_cost;
            })
            .or_insert(ResourceEntry {
                priority,
                parent: None,
                byte_cost,
                state: ResidencyState::Missing,
                epoch: 0,
                last_used_frame: 0,
            });
    }

    /// Sets (or clears, with `None`) the parent `key` depends on.
    ///
    /// Does nothing if `key` is not registered.
    pub fn set_parent(&mut self, key: K, parent: Option<K>) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.parent = parent;
        }
    }

    /// Records that `key` was touched on `frame`, updating its recency.
    pub fn touch(&mut self, key: K, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = frame;
        }
    }

    /// Moves a `Missing` resource to `Requested`.
    ///
    /// A convenience wrapper over [`Self::transition`] for the common "someone
    /// needs this" edge. Returns the [`TransitionError`] if the resource is
    /// unknown or is not currently `Missing`.
    ///
    /// # Errors
    ///
    /// Propagates the error from [`Self::transition`].
    pub fn request(&mut self, key: K) -> Result<(), TransitionError> {
        self.transition(key, ResidencyState::Requested)
    }

    /// Applies a validated lifecycle transition to `key`.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError::Unknown`] if `key` is not registered, or
    /// [`TransitionError::Illegal`] if the edge violates the state machine in
    /// [`crate::virtual_resource::state`].
    pub fn transition(&mut self, key: K, to: ResidencyState) -> Result<(), TransitionError> {
        let entry = self.entries.get_mut(&key).ok_or(TransitionError::Unknown)?;
        let next = entry
            .state
            .try_transition(to)
            .map_err(TransitionError::Illegal)?;
        entry.state = next;
        Ok(())
    }

    /// Forces `key` to [`ResidencyState::Resident`], modelling a completed
    /// upload the backend has confirmed.
    ///
    /// This is the settled end-of-upload callback, so it bypasses the
    /// intermediate `Requested`/`Uploading` edges the caller would otherwise
    /// drive; the real upload is issued by the backend, pending the `GPU`
    /// backend. Does nothing for an unknown key.
    pub fn mark_resident(&mut self, key: K) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.state = ResidencyState::Resident;
        }
    }

    /// Forces `key` to [`ResidencyState::Missing`], modelling a completed
    /// eviction. Does nothing for an unknown key.
    pub fn mark_evicted(&mut self, key: K) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.state = ResidencyState::Missing;
        }
    }

    /// Current state of `key`, or [`ResidencyState::Missing`] if unknown.
    #[must_use]
    pub fn state(&self, key: K) -> ResidencyState {
        self.entries
            .get(&key)
            .map_or(ResidencyState::Missing, |e| e.state)
    }

    /// Priority of `key`, or [`RequestPriority`] zero if unknown.
    #[must_use]
    pub fn priority(&self, key: K) -> RequestPriority {
        self.entries
            .get(&key)
            .map_or(RequestPriority(0), |e| e.priority)
    }

    /// Parent of `key`, if registered and it has one.
    #[must_use]
    pub fn parent(&self, key: K) -> Option<K> {
        self.entries.get(&key).and_then(|e| e.parent)
    }

    /// Invalidation epoch last applied to `key`, or `0` if unknown.
    #[must_use]
    pub fn epoch(&self, key: K) -> u64 {
        self.entries.get(&key).map_or(0, |e| e.epoch)
    }

    /// Borrows the full entry for `key`.
    #[must_use]
    pub fn entry(&self, key: K) -> Option<&ResourceEntry<K>> {
        self.entries.get(&key)
    }

    /// Iterates every `(key, entry)` in ascending key order.
    pub fn iter(&self) -> impl Iterator<Item = (K, &ResourceEntry<K>)> {
        self.entries.iter().map(|(k, v)| (*k, v))
    }

    /// Whether `key` is resident, walking no ancestors.
    #[must_use]
    pub fn is_resident(&self, key: K) -> bool {
        self.state(key).is_resident()
    }

    /// Whether every ancestor of `key` up the parent chain is resident.
    ///
    /// A resource with no parent trivially satisfies this. Cycles are broken by
    /// bounding the walk to the number of registered resources, so a malformed
    /// parent link can never loop forever.
    #[must_use]
    pub fn parents_resident(&self, key: K) -> bool {
        let mut cursor = self.parent(key);
        let mut guard = self.entries.len();
        while let Some(parent) = cursor {
            if guard == 0 {
                return false;
            }
            guard -= 1;
            if !self.is_resident(parent) {
                return false;
            }
            cursor = self.parent(parent);
        }
        true
    }

    /// Applies an invalidation stamped with `epoch`.
    ///
    /// If `epoch` is newer than the resource's last epoch, the underlying data
    /// changed, so the resource is forced back to [`ResidencyState::Missing`] and
    /// its epoch advanced; the scheduler will re-request and re-upload it. A
    /// stale or equal `epoch` is ignored, which makes invalidation idempotent and
    /// order-independent for a given frame. Unknown keys are ignored.
    pub fn invalidate_epoch(&mut self, key: K, epoch: u64) {
        if let Some(entry) = self.entries.get_mut(&key)
            && epoch > entry.epoch
        {
            entry.epoch = epoch;
            entry.state = ResidencyState::Missing;
        }
    }
}

impl<K: ResourceKey> VirtualResourceClient for VirtualResourceTable<K> {
    type Key = K;

    fn priority(&self, key: Self::Key) -> RequestPriority {
        VirtualResourceTable::priority(self, key)
    }

    fn parent(&self, key: Self::Key) -> Option<Self::Key> {
        VirtualResourceTable::parent(self, key)
    }

    fn invalidate(&mut self, key: Self::Key, epoch: u64) {
        self.invalidate_epoch(key, epoch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> VirtualResourceTable<u32> {
        VirtualResourceTable::new()
    }

    #[test]
    fn declare_starts_missing_and_refreshes_without_reset() {
        let mut t = table();
        t.declare(1, RequestPriority(10), 100);
        assert_eq!(t.state(1), ResidencyState::Missing);
        t.mark_resident(1);
        // Re-declaring keeps the resident state but updates priority / bytes.
        t.declare(1, RequestPriority(20), 250);
        assert_eq!(t.state(1), ResidencyState::Resident);
        assert_eq!(t.priority(1), RequestPriority(20));
        assert_eq!(t.entry(1).unwrap().byte_cost, 250);
    }

    #[test]
    fn transition_validates_edges() {
        let mut t = table();
        t.declare(1, RequestPriority(1), 10);
        assert!(t.request(1).is_ok());
        assert_eq!(t.state(1), ResidencyState::Requested);
        // Illegal skip is rejected and leaves the state untouched.
        let err = t.transition(1, ResidencyState::Resident).unwrap_err();
        assert!(matches!(err, TransitionError::Illegal(_)));
        assert_eq!(t.state(1), ResidencyState::Requested);
        assert!(t.transition(1, ResidencyState::Uploading).is_ok());
        assert!(t.transition(1, ResidencyState::Resident).is_ok());
    }

    #[test]
    fn transition_unknown_key_errors() {
        let mut t = table();
        assert_eq!(
            t.transition(9, ResidencyState::Requested),
            Err(TransitionError::Unknown)
        );
    }

    #[test]
    fn parents_resident_walks_the_chain() {
        let mut t = table();
        t.declare(1, RequestPriority(1), 10);
        t.declare(2, RequestPriority(1), 10);
        t.declare(3, RequestPriority(1), 10);
        t.set_parent(2, Some(1));
        t.set_parent(3, Some(2));
        // Nothing resident yet: grandchild's chain is not satisfied.
        assert!(!t.parents_resident(3));
        t.mark_resident(1);
        assert!(t.parents_resident(2));
        assert!(!t.parents_resident(3));
        t.mark_resident(2);
        assert!(t.parents_resident(3));
    }

    #[test]
    fn parents_resident_survives_a_cycle() {
        let mut t = table();
        t.declare(1, RequestPriority(1), 10);
        t.declare(2, RequestPriority(1), 10);
        t.set_parent(1, Some(2));
        t.set_parent(2, Some(1));
        // A cycle can never be fully resident; the bounded walk returns false.
        assert!(!t.parents_resident(1));
    }

    #[test]
    fn invalidate_epoch_resets_only_on_newer_epoch() {
        let mut t = table();
        t.declare(1, RequestPriority(1), 10);
        t.mark_resident(1);
        // Stale epoch is ignored.
        t.invalidate_epoch(1, 0);
        assert_eq!(t.state(1), ResidencyState::Resident);
        // Newer epoch forces a re-request.
        t.invalidate_epoch(1, 5);
        assert_eq!(t.state(1), ResidencyState::Missing);
        assert_eq!(t.epoch(1), 5);
        // Equal epoch is idempotent.
        t.mark_resident(1);
        t.invalidate_epoch(1, 5);
        assert_eq!(t.state(1), ResidencyState::Resident);
    }

    #[test]
    fn client_trait_delegates_to_inherent_methods() {
        let mut t = table();
        t.declare(1, RequestPriority(7), 10);
        t.declare(2, RequestPriority(3), 10);
        t.set_parent(2, Some(1));
        assert_eq!(VirtualResourceClient::priority(&t, 2), RequestPriority(3));
        assert_eq!(VirtualResourceClient::parent(&t, 2), Some(1));
        t.mark_resident(1);
        VirtualResourceClient::invalidate(&mut t, 1, 2);
        assert_eq!(t.state(1), ResidencyState::Missing);
    }

    #[test]
    fn iter_is_key_ordered() {
        let mut t = table();
        for k in [5u32, 1, 3, 2, 4] {
            t.declare(k, RequestPriority(1), 10);
        }
        let keys: Vec<u32> = t.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, alloc::vec![1, 2, 3, 4, 5]);
    }
}
