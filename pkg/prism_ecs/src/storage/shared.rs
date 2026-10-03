//! Unity-style *SharedComponent* value de-duplication (design §6 存储模型 四态
//! "SharedComponent"，§15 GPU 驱动批次键).
//!
//! A **shared component** is a component whose *value* is stored once per
//! distinct value and shared by every entity that carries that value, rather
//! than once per entity. Classic uses are render/physics *batch keys* — e.g. a
//! `(mesh, material)` pair — where millions of entities fall into a handful of
//! distinct buckets. Storing the value once and tagging entities with a dense
//! [`SharedValueId`] turns "group entities by batch" into an O(1) key compare
//! and feeds instancing/合批 directly (design §15).
//!
//! This module is the **value store**: the de-duplicating, reference-counted
//! pool that maps each distinct shared value to a stable [`SharedValueId`] and
//! back. It is deliberately independent of the archetype split that routes
//! entities by that id — the store answers only "what is the canonical id for
//! this value, and how many references does it have". One [`SharedValuePool`]
//! exists per shared [`ComponentId`]; [`SharedComponents`] is the per-world
//! registry of those pools.
//!
//! ### De-duplication without a second copy
//!
//! Shared values are type-erased behind [`SharedValue`] (a `dyn`-dispatched
//! `Hash + Eq`). Each pool keeps the single authoritative boxed value in a slot
//! vector and a `hash -> candidate slot ids` index; a lookup hashes the probe
//! value, then compares it against the few candidates in its bucket with
//! [`SharedValue::dyn_eq`]. There is exactly one stored copy per distinct
//! value, and the id of an interned value is stable until its last reference is
//! released.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;
use core::hash::{BuildHasher, Hash, Hasher};

use hashbrown::DefaultHashBuilder;

use crate::collections::HashMap;
use crate::component::ComponentId;

/// A dense, stable identifier for one distinct value interned in a shared
/// component's [`SharedValuePool`].
///
/// Ids are scoped to a single [`ComponentId`]: the same raw index in two
/// different pools names unrelated values. An id stays valid (and keeps naming
/// the same value) until its reference count drops to zero, after which the
/// slot may be reused for a later value (design §6 批次键去重).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SharedValueId(u32);

impl SharedValueId {
    /// Construct a [`SharedValueId`] from its raw index. Primarily for internal
    /// use and tests; ids are normally minted by [`SharedValuePool::insert`].
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The raw dense index of this id.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A type-erased shared component value that can be de-duplicated by content.
///
/// Blanket-implemented for every `T: Send + Sync + Eq + Hash + 'static`, so any
/// such component is usable as a shared value with no extra boilerplate. The
/// `dyn`-dispatched hash/eq let the pool store heterogeneous erased values while
/// still comparing them exactly within one component's pool.
pub trait SharedValue: Send + Sync {
    /// Feed this value into `state`, mirroring its [`Hash`] implementation.
    fn dyn_hash(&self, state: &mut dyn Hasher);
    /// Whether this value equals `other`. Returns `false` when `other` is a
    /// different concrete type (so cross-type collisions never compare equal).
    fn dyn_eq(&self, other: &dyn SharedValue) -> bool;
    /// Upcast for the concrete-type downcast used by [`SharedValue::dyn_eq`]
    /// and [`SharedComponents::get`].
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + Eq + Hash + 'static> SharedValue for T {
    #[inline]
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        self.hash(&mut state);
    }

    #[inline]
    fn dyn_eq(&self, other: &dyn SharedValue) -> bool {
        other.as_any().downcast_ref::<T>() == Some(self)
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One interned value plus its live reference count and cached hash.
struct Slot {
    value: Box<dyn SharedValue>,
    hash: u64,
    refcount: u32,
}

/// The de-duplicating, reference-counted value pool for one shared
/// [`ComponentId`].
///
/// Interning the same value repeatedly returns the same [`SharedValueId`] and
/// increments a reference count; [`SharedValuePool::release`] decrements it and
/// frees the slot (and the stored value) when the count reaches zero. Freed
/// slots are recycled, keeping ids dense without ever aliasing a live value.
#[derive(Default)]
pub struct SharedValuePool {
    /// `slots[id] = Some(slot)` while id names a live value; `None` when free.
    slots: Vec<Option<Slot>>,
    /// Recycled ids whose slot is currently free.
    free: Vec<u32>,
    /// `hash -> slot ids` bucket index for content de-duplication.
    buckets: HashMap<u64, Vec<u32>>,
    /// Fixed hasher so a value's bucket is stable across calls.
    hasher: DefaultHashBuilder,
}

impl SharedValuePool {
    /// An empty pool.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct live values currently interned.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Whether no value is currently interned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hash `value` with this pool's fixed hasher.
    fn hash_of(&self, value: &dyn SharedValue) -> u64 {
        let mut state = self.hasher.build_hasher();
        value.dyn_hash(&mut state);
        state.finish()
    }

    /// Intern `value`, returning the stable id for its content.
    ///
    /// If an equal value is already interned, its reference count is bumped and
    /// its existing id returned (no second copy is stored). Otherwise a fresh
    /// slot — recycled from the free list when possible — is allocated with a
    /// reference count of one.
    pub fn insert(&mut self, value: Box<dyn SharedValue>) -> SharedValueId {
        let hash = self.hash_of(value.as_ref());
        if let Some(ids) = self.buckets.get(&hash) {
            for &id in ids {
                if let Some(slot) = &self.slots[id as usize]
                    && slot.value.dyn_eq(value.as_ref())
                {
                    let slot = self.slots[id as usize].as_mut().expect("live slot");
                    slot.refcount += 1;
                    return SharedValueId(id);
                }
            }
        }
        let slot = Slot {
            value,
            hash,
            refcount: 1,
        };
        let id = if let Some(id) = self.free.pop() {
            self.slots[id as usize] = Some(slot);
            id
        } else {
            let id = self.slots.len() as u32;
            self.slots.push(Some(slot));
            id
        };
        self.buckets.entry(hash).or_default().push(id);
        SharedValueId(id)
    }

    /// Add one reference to an already-interned id. No-op for a stale id.
    pub fn increment(&mut self, id: SharedValueId) {
        if let Some(Some(slot)) = self.slots.get_mut(id.index() as usize) {
            slot.refcount += 1;
        }
    }

    /// Remove one reference from `id`; returns `true` iff that was the last
    /// reference and the value was dropped and its slot freed. A stale id
    /// returns `false`.
    pub fn release(&mut self, id: SharedValueId) -> bool {
        let index = id.index() as usize;
        let Some(Some(slot)) = self.slots.get_mut(index) else {
            return false;
        };
        slot.refcount -= 1;
        if slot.refcount > 0 {
            return false;
        }
        let hash = slot.hash;
        self.slots[index] = None;
        self.free.push(index as u32);
        if let Some(ids) = self.buckets.get_mut(&hash) {
            ids.retain(|&other| other != index as u32);
            if ids.is_empty() {
                self.buckets.remove(&hash);
            }
        }
        true
    }

    /// The current reference count of `id`, or `0` for a stale id.
    pub fn refcount(&self, id: SharedValueId) -> u32 {
        match self.slots.get(id.index() as usize) {
            Some(Some(slot)) => slot.refcount,
            _ => 0,
        }
    }

    /// Borrow the interned value for `id`, if it is live.
    #[inline]
    pub fn get(&self, id: SharedValueId) -> Option<&dyn SharedValue> {
        match self.slots.get(id.index() as usize) {
            Some(Some(slot)) => Some(slot.value.as_ref()),
            _ => None,
        }
    }
}

/// The per-world registry of shared-component value pools (design §6 四态
/// "SharedComponent").
///
/// One [`SharedValuePool`] is created lazily the first time a value is interned
/// for a given shared [`ComponentId`]; a declared-but-unused shared component
/// costs nothing. The registry owns no entity routing — it only interns values
/// and hands back stable [`SharedValueId`]s for the archetype layer to tag
/// entities with.
#[derive(Default)]
pub struct SharedComponents {
    pools: HashMap<ComponentId, SharedValuePool>,
}

impl SharedComponents {
    /// A new, empty registry.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of shared components that currently have a backing pool.
    #[inline]
    pub fn len(&self) -> usize {
        self.pools.len()
    }

    /// Whether no shared component has a backing pool yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    /// Intern `value` for shared component `component`, returning the stable id
    /// of its content within that component's pool. Equal values collapse to
    /// one id with a shared reference count (design §6 批次键去重).
    pub fn insert<T: Send + Sync + Eq + Hash + 'static>(
        &mut self,
        component: ComponentId,
        value: T,
    ) -> SharedValueId {
        self.pools
            .entry(component)
            .or_default()
            .insert(Box::new(value))
    }

    /// Add one reference to `id` in `component`'s pool. No-op if the pool or id
    /// does not exist.
    pub fn increment(&mut self, component: ComponentId, id: SharedValueId) {
        if let Some(pool) = self.pools.get_mut(&component) {
            pool.increment(id);
        }
    }

    /// Remove one reference to `id` in `component`'s pool; returns `true` iff
    /// the value was dropped. No-op (returns `false`) for an unknown pool/id.
    pub fn release(&mut self, component: ComponentId, id: SharedValueId) -> bool {
        match self.pools.get_mut(&component) {
            Some(pool) => pool.release(id),
            None => false,
        }
    }

    /// The reference count of `id` in `component`'s pool, or `0` if unknown.
    pub fn refcount(&self, component: ComponentId, id: SharedValueId) -> u32 {
        self.pools
            .get(&component)
            .map_or(0, |pool| pool.refcount(id))
    }

    /// Borrow the interned value for `id` in `component`'s pool, downcast to
    /// `T`. Returns `None` if the pool/id is unknown or `T` is the wrong type.
    pub fn get<T: 'static>(&self, component: ComponentId, id: SharedValueId) -> Option<&T> {
        self.pools
            .get(&component)?
            .get(id)?
            .as_any()
            .downcast_ref::<T>()
    }

    /// Borrow `component`'s pool, if one has been created.
    #[inline]
    pub fn pool(&self, component: ComponentId) -> Option<&SharedValuePool> {
        self.pools.get(&component)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct BatchKey {
        mesh: u32,
        material: u32,
    }

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct Layer(u16);

    fn comp(i: u32) -> ComponentId {
        ComponentId::new(i)
    }

    #[test]
    fn equal_values_dedupe_to_one_id_and_refcount() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, BatchKey { mesh: 1, material: 2 });
        let b = shared.insert(c, BatchKey { mesh: 1, material: 2 });
        assert_eq!(a, b);
        assert_eq!(shared.refcount(c, a), 2);
        assert_eq!(shared.pool(c).unwrap().len(), 1);
    }

    #[test]
    fn distinct_values_get_distinct_ids() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, BatchKey { mesh: 1, material: 2 });
        let b = shared.insert(c, BatchKey { mesh: 1, material: 3 });
        assert_ne!(a, b);
        assert_eq!(shared.pool(c).unwrap().len(), 2);
        assert_eq!(shared.get::<BatchKey>(c, a), Some(&BatchKey { mesh: 1, material: 2 }));
        assert_eq!(shared.get::<BatchKey>(c, b), Some(&BatchKey { mesh: 1, material: 3 }));
    }

    #[test]
    fn release_frees_on_last_reference() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, Layer(7));
        shared.insert(c, Layer(7)); // refcount 2
        assert!(!shared.release(c, a));
        assert_eq!(shared.refcount(c, a), 1);
        assert!(shared.release(c, a));
        assert_eq!(shared.refcount(c, a), 0);
        assert!(shared.get::<Layer>(c, a).is_none());
        assert!(shared.pool(c).unwrap().is_empty());
    }

    #[test]
    fn freed_slot_is_recycled() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, Layer(1));
        assert!(shared.release(c, a));
        // A later distinct value should reuse the freed slot index.
        let b = shared.insert(c, Layer(2));
        assert_eq!(a.index(), b.index());
        assert_eq!(shared.get::<Layer>(c, b), Some(&Layer(2)));
    }

    #[test]
    fn pools_are_scoped_per_component() {
        let mut shared = SharedComponents::new();
        let c0 = comp(0);
        let c1 = comp(1);
        let a = shared.insert(c0, Layer(5));
        let b = shared.insert(c1, Layer(5));
        // Same value, different components => independent pools and ids.
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 0);
        assert_eq!(shared.len(), 2);
        // Releasing one pool's id leaves the other untouched.
        assert!(shared.release(c0, a));
        assert_eq!(shared.refcount(c1, b), 1);
    }

    #[test]
    fn wrong_type_downcast_returns_none() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, Layer(9));
        assert!(shared.get::<BatchKey>(c, a).is_none());
    }

    #[test]
    fn increment_tracks_external_references() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, Layer(3));
        shared.increment(c, a);
        assert_eq!(shared.refcount(c, a), 2);
        assert!(!shared.release(c, a));
        assert!(shared.release(c, a));
        assert_eq!(shared.refcount(c, a), 0);
    }

    #[test]
    fn stale_id_operations_are_safe() {
        let mut shared = SharedComponents::new();
        let c = comp(0);
        let a = shared.insert(c, Layer(1));
        assert!(shared.release(c, a));
        // All operations on the now-stale id are no-ops / defined.
        assert_eq!(shared.refcount(c, a), 0);
        assert!(!shared.release(c, a));
        shared.increment(c, a);
        assert_eq!(shared.refcount(c, a), 0);
        // Unknown component pool.
        assert_eq!(shared.refcount(comp(99), a), 0);
        assert!(!shared.release(comp(99), a));
    }
}
