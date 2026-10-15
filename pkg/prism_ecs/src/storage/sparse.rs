//! SparseSet component storage: the second of the design's four storage states
//! (design §6 存储模型 四态, EnTT-style sparse set).
//!
//! Where a [`Table`](crate::storage::Table) column is dense and archetype-bound
//! — adding or removing the component moves the entity between archetypes — a
//! sparse set stores the component *out of band*, keyed directly by
//! [`Entity`]. Insert and remove are O(1) and never relocate any other
//! component or fragment the archetype graph, which is exactly what frequently
//! toggled tag / transient components want (design §6: "增删不搬迁 archetype,
//! 适合频繁 toggle 的 tag/临时组件").
//!
//! # Layout
//!
//! ```text
//! sparse:  entity.index() ───▶ dense index   (u32, NOT_PRESENT sentinel)
//! dense:   component bytes (BlobVec)          packed, no holes
//! entities: dense index  ───▶ Entity          parallel to dense
//! ```
//!
//! The `dense` column is kept hole-free by swap-remove: removing a row moves the
//! current last row into the freed slot and fixes up the moved entity's sparse
//! entry. The full [`Entity`] (index **and** generation) is stored per dense
//! row, so a stale handle whose index has since been recycled is rejected on
//! lookup — the same dangling-safe discipline the rest of the kernel relies on
//! (design §5.1).
//!
//! Change detection mirrors [`Column`](crate::storage::Column): one `added` and
//! one `changed` [`Tick`] per dense row, behind [`UnsafeCell`] so the `&mut T` /
//! [`Mut`](crate::change::Mut) fetch can stamp the changed tick through a shared
//! `&ComponentSparseSet` (design §10). A sparse set is *not* chunk-windowed —
//! chunk versions (design §7) are a dense-table concept; the fine per-row ticks
//! are the whole change-detection story here.

use alloc::vec::Vec;
use core::alloc::Layout;
use core::cell::UnsafeCell;

use crate::change::{ComponentTicks, Tick};
use crate::component::DropFn;
use crate::entity::Entity;
use crate::storage::blob_vec::BlobVec;

/// Sentinel meaning "this entity index has no dense row in this set".
const NOT_PRESENT: u32 = u32::MAX;

/// A type-erased sparse-set column for a single component type, keyed by
/// [`Entity`] (design §6).
///
/// All typed access goes through `unsafe` methods that trust the caller to pass
/// the component type this set was created for, exactly like
/// [`Column`](crate::storage::Column).
pub struct ComponentSparseSet {
    /// Packed component bytes; `dense` row `d` belongs to `entities[d]`.
    dense: BlobVec,
    /// Dense-row → owning entity (full handle, for generation validation).
    entities: Vec<Entity>,
    /// Dense-row → tick the value was first inserted.
    added_ticks: Vec<UnsafeCell<Tick>>,
    /// Dense-row → tick the value was last written.
    changed_ticks: Vec<UnsafeCell<Tick>>,
    /// Snapshot identity follows each value through swap-removal. Unlike ticks,
    /// these revisions cannot be reset by replay or change-detection overrides.
    snapshot_revisions: Vec<SparseSnapshotRevision>,
    /// Allocated only with exclusive set access; never reset by remove/clear.
    next_snapshot_incarnation: u64,
    /// Entity index → dense row, or [`NOT_PRESENT`]. Grows to cover the largest
    /// entity index ever inserted.
    sparse: Vec<u32>,
}

struct SparseSnapshotRevision {
    incarnation: u64,
    writes: UnsafeCell<u64>,
}

// SAFETY: identical discipline to `Column` (which is `unsafe impl Sync`): the
// tick and snapshot-revision cells are mutated solely under unique per-row
// access, and no reader observes a cell mid-write. Distinct rows never mutate a
// shared revision counter; insertion identities require exclusive set access.
unsafe impl Sync for ComponentSparseSet {}

impl ComponentSparseSet {
    /// A new, empty sparse set storing values of layout `layout`, invoking
    /// `drop` (if any) on each value when it is removed or the set is dropped.
    #[inline]
    pub fn new(layout: Layout, drop: Option<DropFn>) -> Self {
        Self {
            dense: BlobVec::new(layout, drop),
            entities: Vec::new(),
            added_ticks: Vec::new(),
            changed_ticks: Vec::new(),
            snapshot_revisions: Vec::new(),
            next_snapshot_incarnation: 0,
            sparse: Vec::new(),
        }
    }

    /// Number of entities with a value in this set.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether the set holds no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// The dense row owning `entity`, validating generation, or `None`.
    #[inline]
    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let slot = *self.sparse.get(entity.index() as usize)?;
        if slot == NOT_PRESENT {
            return None;
        }
        let dense = slot as usize;
        // Reject a stale handle whose index was recycled into a new generation.
        if self.entities[dense] == entity {
            Some(dense)
        } else {
            None
        }
    }

    /// Whether `entity` currently has a value in this set.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.dense_index(entity).is_some()
    }

    /// The dense-row entities, in packed order (test / iteration helper).
    #[inline]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Raw pointer to `entity`'s value, or `None` if absent.
    ///
    /// # Safety
    /// The returned pointer must only be accessed as the component type this set
    /// stores. Writes must also obtain [`changed_tick_ptr`](Self::changed_tick_ptr)
    /// or call [`set_changed_tick`](Self::set_changed_tick) before snapshot
    /// capture, even if the desired changed tick is unchanged.
    #[inline]
    pub unsafe fn get_ptr(&self, entity: Entity) -> Option<*mut u8> {
        let dense = self.dense_index(entity)?;
        // SAFETY: `dense < self.dense.len()` by construction.
        Some(unsafe { self.dense.get_ptr(dense) })
    }

    /// Typed shared reference to `entity`'s value, or `None`.
    ///
    /// # Safety
    /// `T` must be the exact component type this set stores.
    #[inline]
    pub unsafe fn get<T>(&self, entity: Entity) -> Option<&T> {
        // SAFETY: forwarded contract on `T`; pointer valid for a `T`.
        let ptr = unsafe { self.get_ptr(entity) }?;
        // SAFETY: `ptr` is valid and aligned for `T`.
        Some(unsafe { &*ptr.cast::<T>() })
    }

    /// Typed unique reference to `entity`'s value, or `None`.
    ///
    /// # Safety
    /// `T` must be the exact component type this set stores, and the caller must
    /// hold unique access to this entity's value.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut<T>(&self, entity: Entity) -> Option<&mut T> {
        let dense = self.dense_index(entity)?;
        // SAFETY: caller guarantees exclusive access to this row.
        unsafe { self.record_snapshot_write(dense) };
        // SAFETY: forwarded contract on `T` and unique access.
        let ptr = unsafe { self.get_ptr(entity) }?;
        // SAFETY: caller guarantees unique access, so `&mut` does not alias.
        Some(unsafe { &mut *ptr.cast::<T>() })
    }

    /// Grow `sparse` so `index` is addressable.
    #[inline]
    fn ensure_sparse(&mut self, index: usize) {
        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, NOT_PRESENT);
        }
    }

    fn push_snapshot_revision(&mut self) {
        self.next_snapshot_incarnation = self.next_snapshot_incarnation.saturating_add(1);
        self.snapshot_revisions.push(SparseSnapshotRevision {
            incarnation: self.next_snapshot_incarnation,
            writes: UnsafeCell::new(0),
        });
    }

    /// Invalidate at write-capable access acquisition, independently of whether
    /// the caller subsequently stamps a current, old, or identical tick. This
    /// conservatively recaptures fetched-but-unwritten rows too.
    ///
    /// # Safety
    /// The caller must hold exclusive access to the live row at `dense`.
    #[inline]
    unsafe fn record_snapshot_write(&self, dense: usize) {
        // SAFETY: the caller owns this row; other rows have independent cells.
        let writes = unsafe { &mut *self.snapshot_revisions[dense].writes.get() };
        *writes = writes.saturating_add(1);
    }

    /// Independent insertion/write identity, or None once either counter is
    /// exhausted. An exhausted row must be freshly cloned on every capture;
    /// saturation must never turn into a reusable ABA identity.
    #[inline]
    pub(crate) fn snapshot_revision(&self, entity: Entity) -> Option<(u64, u64)> {
        let revision = &self.snapshot_revisions[self.dense_index(entity)?];
        // SAFETY: snapshot readers exclude concurrent writers to this row.
        let writes = unsafe { *revision.writes.get() };
        (revision.incarnation != u64::MAX && writes != u64::MAX)
            .then_some((revision.incarnation, writes))
    }

    /// Insert or overwrite `entity`'s value, stamping change ticks with
    /// `change_tick`. Returns `true` if this was a fresh insert, `false` if an
    /// existing value was overwritten (last-wins, added tick preserved).
    ///
    /// # Safety
    /// `value` points to a valid, initialized value of this set's component
    /// type; ownership transfers into the set.
    pub unsafe fn insert(&mut self, entity: Entity, value: *const u8, change_tick: Tick) -> bool {
        if let Some(dense) = self.dense_index(entity) {
            // SAFETY: exclusive set access also excludes all row readers.
            unsafe { self.record_snapshot_write(dense) };
            // SAFETY: `dense < len` and `value` is a valid owned value of the
            // stored type (forwarded contract); `replace` drops the old value.
            unsafe { self.dense.replace(dense, value) };
            *self.changed_ticks[dense].get_mut() = change_tick;
            return false;
        }
        let index = entity.index() as usize;
        self.ensure_sparse(index);
        let dense = self.dense.len();
        // SAFETY: `value` is a valid owned value of the stored type (forwarded).
        unsafe { self.dense.push(value) };
        self.entities.push(entity);
        self.added_ticks.push(UnsafeCell::new(change_tick));
        self.changed_ticks.push(UnsafeCell::new(change_tick));
        self.push_snapshot_revision();
        self.sparse[index] = dense as u32;
        true
    }

    /// Insert `entity`'s value stamping its added and changed ticks
    /// *independently*, for exact world [`restore`](crate::World::restore)
    /// (design §14). `entity` must be absent (a fresh snapshot restore places
    /// each entity exactly once); unlike [`insert`](Self::insert) this never
    /// overwrites.
    ///
    /// # Safety
    /// `value` points to a valid, initialized value of this set's component
    /// type; ownership transfers into the set. `entity` must not already be
    /// present.
    pub unsafe fn insert_with_ticks(
        &mut self,
        entity: Entity,
        value: *const u8,
        added: Tick,
        changed: Tick,
    ) {
        debug_assert!(
            !self.contains(entity),
            "insert_with_ticks requires an absent entity"
        );
        let index = entity.index() as usize;
        self.ensure_sparse(index);
        let dense = self.dense.len();
        // SAFETY: `value` is a valid owned value of the stored type (forwarded).
        unsafe { self.dense.push(value) };
        self.entities.push(entity);
        self.added_ticks.push(UnsafeCell::new(added));
        self.changed_ticks.push(UnsafeCell::new(changed));
        self.push_snapshot_revision();
        self.sparse[index] = dense as u32;
    }

    /// Remove and drop `entity`'s value if present. Returns `true` if a value
    /// was removed.
    pub fn remove(&mut self, entity: Entity) -> bool {
        let Some(dense) = self.dense_index(entity) else {
            return false;
        };
        // SAFETY: `dense < len`; swap-remove drops the value at `dense` and moves
        // the current last row into its place.
        unsafe { self.dense.swap_remove_and_drop(dense) };
        self.entities.swap_remove(dense);
        self.added_ticks.swap_remove(dense);
        self.changed_ticks.swap_remove(dense);
        self.snapshot_revisions.swap_remove(dense);
        self.sparse[entity.index() as usize] = NOT_PRESENT;
        // If a row moved into `dense`, repoint its sparse entry to the new slot.
        if dense < self.entities.len() {
            let moved = self.entities[dense];
            self.sparse[moved.index() as usize] = dense as u32;
        }
        true
    }

    /// Move an owned value out without running user drop glue. Repairs dense
    /// indices and tick arrays before returning, so subsequent cleanup may
    /// unwind without leaving this set inconsistent.
    ///
    /// # Safety
    /// `dst` must be aligned, uninitialized storage for this set's exact type,
    /// disjoint from the set's allocations. On success the caller owns the
    /// value written there; on absence nothing is written.
    pub(crate) unsafe fn remove_and_copy_out(&mut self, entity: Entity, dst: *mut u8) -> bool {
        let Some(dense) = self.dense_index(entity) else {
            return false;
        };
        // SAFETY: live dense index and destination supplied by the caller.
        unsafe { self.dense.swap_remove_and_copy_out(dense, dst) };
        self.entities.swap_remove(dense);
        self.added_ticks.swap_remove(dense);
        self.changed_ticks.swap_remove(dense);
        self.snapshot_revisions.swap_remove(dense);
        self.sparse[entity.index() as usize] = NOT_PRESENT;
        if dense < self.entities.len() {
            let moved = self.entities[dense];
            self.sparse[moved.index() as usize] = dense as u32;
        }
        true
    }

    /// The tick at which `entity`'s value was first inserted, or `None`.
    #[inline]
    pub fn added_tick(&self, entity: Entity) -> Option<Tick> {
        let dense = self.dense_index(entity)?;
        // SAFETY: shared read; writers hold unique access to the row.
        Some(unsafe { *self.added_ticks[dense].get() })
    }

    /// The tick at which `entity`'s value was last changed, or `None`.
    #[inline]
    pub fn changed_tick(&self, entity: Entity) -> Option<Tick> {
        let dense = self.dense_index(entity)?;
        // SAFETY: shared read; see [`ComponentSparseSet::added_tick`].
        Some(unsafe { *self.changed_ticks[dense].get() })
    }

    /// The added/changed [`ComponentTicks`] pair for `entity`, or `None`.
    #[inline]
    pub fn component_ticks(&self, entity: Entity) -> Option<ComponentTicks> {
        let dense = self.dense_index(entity)?;
        Some(ComponentTicks {
            // SAFETY: shared read; writers hold unique per-row access.
            added: unsafe { *self.added_ticks[dense].get() },
            // SAFETY: shared read; writers hold unique per-row access.
            changed: unsafe { *self.changed_ticks[dense].get() },
        })
    }

    /// Raw pointer to `entity`'s changed-tick cell, for interior-mutable
    /// stamping through a shared `&ComponentSparseSet` (the `&mut T` /
    /// [`Mut`](crate::change::Mut) fetch), or `None` if absent.
    /// Granting this write-capable access also invalidates snapshot reuse for
    /// the row, without modifying its added/changed ticks.
    ///
    /// # Safety
    /// The caller must hold unique access to `entity`'s row and only *write*
    /// the pointer (never form a shared `&Tick` that outlives a concurrent
    /// write), mirroring
    /// [`Column::changed_tick_ptr`](crate::storage::Column::changed_tick_ptr).
    #[inline]
    pub unsafe fn changed_tick_ptr(&self, entity: Entity) -> Option<*mut Tick> {
        let dense = self.dense_index(entity)?;
        // SAFETY: the caller guarantees exclusive row access.
        unsafe { self.record_snapshot_write(dense) };
        Some(self.changed_ticks[dense].get())
    }

    /// Stamp `entity`'s changed tick (used by structural writes already holding
    /// `&mut self`). Returns `false` if `entity` is absent.
    #[inline]
    pub fn set_changed_tick(&mut self, entity: Entity, change_tick: Tick) -> bool {
        let Some(dense) = self.dense_index(entity) else {
            return false;
        };
        // SAFETY: exclusive set access also excludes all row readers.
        unsafe { self.record_snapshot_write(dense) };
        *self.changed_ticks[dense].get_mut() = change_tick;
        true
    }

    /// Clamp every stored tick against `this_run` so none can wrap around and
    /// masquerade as recent (mirrors
    /// [`Column::check_change_ticks`](crate::storage::Column::check_change_ticks)).
    pub fn check_change_ticks(&mut self, this_run: Tick) {
        for cell in &mut self.added_ticks {
            cell.get_mut().check_tick(this_run);
        }
        for cell in &mut self.changed_ticks {
            cell.get_mut().check_tick(this_run);
        }
    }

    /// Drop every value and reset the set to empty, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.dense.clear();
        self.entities.clear();
        self.added_ticks.clear();
        self.changed_ticks.clear();
        self.snapshot_revisions.clear();
        // Reset sparse to all-absent without giving back the allocation.
        for slot in &mut self.sparse {
            *slot = NOT_PRESENT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;

    /// Build a sparse set storing `u32` values with a no-op drop.
    fn u32_set() -> ComponentSparseSet {
        ComponentSparseSet::new(Layout::new::<u32>(), None)
    }

    /// Insert a `u32` by pointer (the set is type-erased).
    fn insert_u32(set: &mut ComponentSparseSet, e: Entity, v: u32, tick: Tick) -> bool {
        // SAFETY: `&v` is a valid, initialized `u32`, matching the set's layout.
        unsafe { set.insert(e, (&raw const v).cast::<u8>(), tick) }
    }

    fn read_u32(set: &ComponentSparseSet, e: Entity) -> Option<u32> {
        // SAFETY: the set stores `u32`.
        unsafe { set.get::<u32>(e) }.copied()
    }

    #[test]
    fn insert_get_contains_roundtrip() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();
        let b = es.alloc();

        assert!(insert_u32(&mut set, a, 10, Tick::new(1)));
        assert!(insert_u32(&mut set, b, 20, Tick::new(1)));
        assert_eq!(set.len(), 2);
        assert!(set.contains(a));
        assert!(set.contains(b));
        assert_eq!(read_u32(&set, a), Some(10));
        assert_eq!(read_u32(&set, b), Some(20));
    }

    #[test]
    fn reinsert_overwrites_and_preserves_added_tick() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();

        assert!(insert_u32(&mut set, a, 1, Tick::new(2)));
        // A second insert overwrites in place (not a fresh insert).
        assert!(!insert_u32(&mut set, a, 99, Tick::new(5)));
        assert_eq!(set.len(), 1);
        assert_eq!(read_u32(&set, a), Some(99));
        let ticks = set.component_ticks(a).unwrap();
        assert_eq!(ticks.added, Tick::new(2), "added tick is preserved");
        assert_eq!(ticks.changed, Tick::new(5), "changed tick advances");
    }

    #[test]
    fn remove_keeps_dense_packed_and_fixes_sparse() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();
        let b = es.alloc();
        let c = es.alloc();
        insert_u32(&mut set, a, 1, Tick::new(1));
        insert_u32(&mut set, b, 2, Tick::new(1));
        insert_u32(&mut set, c, 3, Tick::new(1));

        // Remove the middle entity: the last row (c) swaps into its slot.
        assert!(set.remove(b));
        assert!(!set.contains(b));
        assert_eq!(set.len(), 2);
        // a and c are still correctly addressable after the swap.
        assert_eq!(read_u32(&set, a), Some(1));
        assert_eq!(read_u32(&set, c), Some(3));
        // Dense stayed hole-free.
        assert_eq!(set.entities().len(), 2);

        // Removing an absent entity is a no-op.
        assert!(!set.remove(b));
    }

    #[test]
    fn stale_generation_handle_is_rejected() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();
        insert_u32(&mut set, a, 7, Tick::new(1));
        // Free and recycle the index: the new handle shares `a`'s index but has
        // a bumped generation.
        es.free(a);
        let a2 = es.alloc();
        assert_eq!(a2.index(), a.index(), "index recycled");
        assert_ne!(a2, a, "generation differs");

        // The stale handle must not resolve to the recycled index's row.
        assert!(!set.contains(a2), "new-generation handle has no value yet");
        assert_eq!(read_u32(&set, a2), None);
        // The old handle still maps to the stored row (same generation).
        assert_eq!(read_u32(&set, a), Some(7));
    }

    #[test]
    fn changed_tick_ptr_stamps_through_shared_ref() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();
        insert_u32(&mut set, a, 1, Tick::new(1));

        // Simulate the `Mut<T>` fetch stamping the changed tick via raw pointer.
        // SAFETY: unique access (no other borrow), writing a newer tick only.
        let ptr = unsafe { set.changed_tick_ptr(a) }.unwrap();
        // SAFETY: `ptr` is a live, uniquely-accessed tick cell.
        unsafe { *ptr = Tick::new(4) };
        assert_eq!(set.changed_tick(a), Some(Tick::new(4)));
    }

    #[test]
    fn check_change_ticks_clamps() {
        let mut es = Entities::new();
        let mut set = u32_set();
        let a = es.alloc();
        insert_u32(&mut set, a, 1, Tick::new(1));
        let now = Tick::new(Tick::MAX_CHANGE_AGE + 50);
        set.check_change_ticks(now);
        assert_eq!(
            set.changed_tick(a).unwrap().age_since(now),
            Tick::MAX_CHANGE_AGE
        );
    }

    #[test]
    fn snapshot_revision_tracks_mutable_access_not_tick_identity() {
        let mut entities = Entities::new();
        let entity = entities.alloc();
        let mut set = u32_set();
        let tick = Tick::new(7);
        insert_u32(&mut set, entity, 1, tick);
        let inserted = set.snapshot_revision(entity).unwrap();
        assert_eq!(read_u32(&set, entity), Some(1));
        assert_eq!(set.snapshot_revision(entity), Some(inserted));

        // SAFETY: u32 storage, unique access to the row.
        unsafe { *set.get_mut::<u32>(entity).unwrap() = 2 };
        let mutated = set.snapshot_revision(entity).unwrap();
        assert_eq!(mutated.0, inserted.0);
        assert!(mutated.1 > inserted.1);
        // SAFETY: unique access to the row and its tick cell.
        unsafe { *set.changed_tick_ptr(entity).unwrap() = tick };
        let fetched = set.snapshot_revision(entity).unwrap();
        assert_eq!(fetched.0, inserted.0);
        assert!(fetched.1 > mutated.1);
        assert_eq!(set.changed_tick(entity), Some(tick));

        set.set_changed_tick(entity, tick);
        let stamped = set.snapshot_revision(entity).unwrap();
        assert!(stamped.1 > fetched.1);
        assert!(!insert_u32(&mut set, entity, 3, tick));
        let replaced = set.snapshot_revision(entity).unwrap();
        assert_eq!(replaced.0, inserted.0);
        assert!(replaced.1 > stamped.1);
        assert_eq!(set.added_tick(entity), Some(tick));
        assert_eq!(set.changed_tick(entity), Some(tick));
    }

    #[test]
    fn snapshot_revision_follows_swap_removal_and_never_reuses_insertions() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        let c = entities.alloc();
        let mut set = u32_set();
        let tick = Tick::new(7);
        for (entity, value) in [(a, 1), (b, 2), (c, 3)] {
            insert_u32(&mut set, entity, value, tick);
        }
        let a_revision = set.snapshot_revision(a).unwrap();
        let b_revision = set.snapshot_revision(b).unwrap();
        let c_revision = set.snapshot_revision(c).unwrap();
        assert!(set.remove(b));
        assert_eq!(set.entities(), &[a, c]);
        assert_eq!(set.snapshot_revision(c), Some(c_revision));
        insert_u32(&mut set, b, 20, tick);
        assert_ne!(set.snapshot_revision(b), Some(b_revision));

        let mut removed = core::mem::MaybeUninit::<u32>::uninit();
        let reinserted_b = set.snapshot_revision(b);
        // SAFETY: disjoint, aligned uninitialized u32 storage.
        assert!(unsafe { set.remove_and_copy_out(a, removed.as_mut_ptr().cast()) });
        // SAFETY: successful removal initialized the destination with a u32.
        assert_eq!(unsafe { removed.assume_init() }, 1);
        assert_eq!(set.entities(), &[b, c]);
        assert_eq!(set.snapshot_revision(b), reinserted_b);
        assert_eq!(set.snapshot_revision(c), Some(c_revision));

        set.clear();
        let replacement = 10u32;
        // SAFETY: initialized u32 value, absent entity, matching set layout.
        unsafe { set.insert_with_ticks(a, (&raw const replacement).cast(), tick, tick) };
        assert_ne!(set.snapshot_revision(a), Some(a_revision));
        assert_eq!(read_u32(&set, a), Some(10));
    }

    #[test]
    fn snapshot_revision_exhaustion_disables_reuse_without_wrapping() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        let mut set = u32_set();
        set.next_snapshot_incarnation = u64::MAX - 2;
        insert_u32(&mut set, a, 1, Tick::ZERO);
        assert!(set.snapshot_revision(a).is_some());
        insert_u32(&mut set, b, 2, Tick::ZERO);
        assert_eq!(set.snapshot_revision(b), None);
        set.clear();
        insert_u32(&mut set, b, 3, Tick::ZERO);
        assert_eq!(set.snapshot_revision(b), None);

        let mut set = u32_set();
        insert_u32(&mut set, a, 1, Tick::ZERO);
        *set.snapshot_revisions[0].writes.get_mut() = u64::MAX - 1;
        assert!(set.snapshot_revision(a).is_some());
        set.set_changed_tick(a, Tick::ZERO);
        assert_eq!(set.snapshot_revision(a), None);
        set.set_changed_tick(a, Tick::ZERO);
        assert_eq!(set.snapshot_revision(a), None);
        assert_eq!(*set.snapshot_revisions[0].writes.get_mut(), u64::MAX);
    }
}
