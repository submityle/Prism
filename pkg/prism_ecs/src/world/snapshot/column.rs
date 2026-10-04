//! A type-erased, owned snapshot column: one cloned value per entity that holds
//! a given component, kept densely with its exact change ticks (design §14).
//!
//! The live world stores component values it owns; a snapshot must own an
//! independent copy so the world can keep mutating (predict) while the snapshot
//! stays frozen (authoritative), and so a snapshot can be restored repeatedly
//! (rollback replays the same confirmed frame many times). We therefore clone
//! each captured value through the component's type-erased [`CloneFn`] glue into
//! our own [`BlobVec`], which also carries the component's [`DropFn`] so the
//! cloned values are released when the snapshot is dropped.

use alloc::alloc::{alloc, dealloc};
use alloc::vec::Vec;
use core::alloc::Layout;

use crate::change::Tick;
use crate::component::{CloneFn, ComponentId, DropFn, SnapshotHashFn, StorageType};
use crate::storage::BlobVec;

/// Allocate scratch storage for one value of `layout`, tolerating zero-sized
/// types (for which `alloc` is UB) by returning a dangling-but-aligned pointer.
///
/// # Safety
/// The returned pointer is valid for a single `layout`-sized write and must be
/// released with [`free_scratch`] using the same `layout`.
unsafe fn alloc_scratch(layout: Layout) -> *mut u8 {
    if layout.size() == 0 {
        // A ZST write touches no bytes; any non-null aligned address is fine.
        core::ptr::without_provenance_mut(layout.align())
    } else {
        // SAFETY: `layout` has non-zero size (checked above).
        let p = unsafe { alloc(layout) };
        assert!(!p.is_null(), "snapshot scratch allocation failed");
        p
    }
}

/// Release scratch storage obtained from [`alloc_scratch`] for the same
/// `layout`. The value it held must already have been moved out (not dropped
/// here).
///
/// # Safety
/// `ptr` must come from [`alloc_scratch`] with this exact `layout` and must not
/// be used afterwards.
unsafe fn free_scratch(ptr: *mut u8, layout: Layout) {
    if layout.size() != 0 {
        // SAFETY: `ptr` came from `alloc(layout)` (forwarded contract).
        unsafe { dealloc(ptr, layout) }
    }
}

/// One type-erased column of a [`WorldSnapshot`](super::WorldSnapshot): the
/// cloned values of a single [`ComponentId`] for every entity that holds it.
///
/// Values are stored densely in `data`, parallel to `added`/`changed` ticks and
/// `rows`. `rows[i]` is the index — into the snapshot's sorted entity list — of
/// the entity owning `data[i]`, and `rows` is strictly ascending, so a column is
/// a sparse, deterministically-ordered map from entity to value.
pub(super) struct SnapshotColumn {
    /// Which component this column captures.
    pub(super) component: ComponentId,
    /// How the component is stored in the live world (table vs sparse set).
    pub(super) storage: StorageType,
    /// The component's memory layout (also the stride of `data`).
    pub(super) layout: Layout,
    /// Type-erased clone glue used both to capture (clone live → snapshot) and
    /// restore (clone snapshot → live).
    pub(super) clone: CloneFn,
    /// Type-erased drop glue, carried by `data` so cloned values are released.
    pub(super) drop: Option<DropFn>,
    /// Optional deterministic hash glue for [`state_hash`](super::WorldSnapshot::state_hash).
    pub(super) hash: Option<SnapshotHashFn>,
    /// The cloned, owned component values, one per holder.
    pub(super) data: BlobVec,
    /// Per-value "added" tick, parallel to `data`.
    pub(super) added: Vec<Tick>,
    /// Per-value "changed" tick, parallel to `data`.
    pub(super) changed: Vec<Tick>,
    /// Per-value owning-entity index into the snapshot's entity list; ascending.
    pub(super) rows: Vec<u32>,
}

impl SnapshotColumn {
    /// Create an empty column for `component` with its captured glue + layout.
    pub(super) fn new(
        component: ComponentId,
        storage: StorageType,
        layout: Layout,
        clone: CloneFn,
        drop: Option<DropFn>,
        hash: Option<SnapshotHashFn>,
    ) -> Self {
        Self {
            component,
            storage,
            layout,
            clone,
            drop,
            hash,
            data: BlobVec::new(layout, drop),
            added: Vec::new(),
            changed: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Number of stored values (holders of this component).
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }

    /// Clone a *live* value at `src` into this column, recording its ticks and
    /// the owning entity's `row` index. Callers must push in ascending `row`
    /// order to keep the column deterministic.
    ///
    /// # Safety
    /// `src` points at a valid, initialized value of this column's component
    /// type (the live world's copy, which is left untouched).
    pub(super) unsafe fn push_cloned(
        &mut self,
        src: *const u8,
        added: Tick,
        changed: Tick,
        row: u32,
    ) {
        // SAFETY: scratch is a single `layout`-sized, aligned slot.
        let tmp = unsafe { alloc_scratch(self.layout) };
        // SAFETY: `src` is a valid value (contract); `tmp` is uninitialized,
        // aligned storage for one value — the clone glue initializes it.
        unsafe { (self.clone)(src, tmp) };
        // SAFETY: `tmp` now holds an initialized, owned value of the column's
        // type; `push` moves (memcpy) it into `data`, which takes ownership.
        unsafe { self.data.push(tmp) };
        // SAFETY: the value was moved into `data`; free the raw scratch without
        // dropping it (ownership already transferred).
        unsafe { free_scratch(tmp, self.layout) };
        self.added.push(added);
        self.changed.push(changed);
        self.rows.push(row);
    }

    /// Clone the value stored at `src_slot` of another column `src` into this
    /// column, carrying over its exact added/changed ticks and recording the
    /// owning-entity index `row`. Used by [`SnapshotDelta`](super::SnapshotDelta)
    /// to assemble a column from a mix of base-snapshot and freshly-captured
    /// cells — cloning (never moving) so both `src` and this column stay valid.
    ///
    /// # Safety
    /// `src_slot < src.len()`, and `src` must share this column's component
    /// layout and clone/drop glue (same [`ComponentId`]). Callers must push in
    /// ascending `row` order to keep the column deterministic.
    pub(super) unsafe fn push_cloned_from(
        &mut self,
        src: &SnapshotColumn,
        src_slot: usize,
        row: u32,
    ) {
        // SAFETY: scratch is a single `layout`-sized aligned slot.
        let tmp = unsafe { alloc_scratch(self.layout) };
        // SAFETY: `src_slot < src.len()` (contract); `tmp` is uninitialized,
        // aligned storage for one value, which the clone glue initializes.
        unsafe { src.clone_value_into(src_slot, tmp) };
        // SAFETY: `tmp` now holds an initialized owned value of the column's
        // type; `push` moves (memcpy) it into `data`, which takes ownership.
        unsafe { self.data.push(tmp) };
        // SAFETY: the value was moved into `data`; free the raw scratch without
        // dropping it (ownership already transferred).
        unsafe { free_scratch(tmp, self.layout) };
        self.added.push(src.added[src_slot]);
        self.changed.push(src.changed[src_slot]);
        self.rows.push(row);
    }

    /// Clone the stored value at `slot` into caller-provided destination `dst`,
    /// leaving the snapshot's own copy intact (so restore can replay repeatedly).
    ///
    /// # Safety
    /// `slot < len()`. `dst` is aligned, writable, uninitialized storage for one
    /// value of this column's type; on return it owns a fresh clone the caller
    /// is responsible for.
    pub(super) unsafe fn clone_value_into(&self, slot: usize, dst: *mut u8) {
        // SAFETY: `slot < len()` (contract) so the pointer is valid.
        let src = unsafe { self.data.get_ptr(slot) };
        // SAFETY: `src` is a valid stored value; `dst` is uninitialized aligned
        // storage for one value (contract).
        unsafe { (self.clone)(src, dst) };
    }

    /// Clone the stored value at `slot` into fresh scratch, hand the raw
    /// pointer to `f` (which must *move* the value out exactly once, e.g. via
    /// [`Column::push_with_ticks`](crate::storage::Column::push_with_ticks)
    /// or [`ComponentSparseSet::insert_with_ticks`](crate::storage::ComponentSparseSet::insert_with_ticks)),
    /// then release the scratch without dropping it (ownership already left).
    /// The snapshot's own copy is untouched, so restore can replay repeatedly.
    ///
    /// # Safety
    /// `slot < len()`. `f` must consume the value behind the pointer exactly
    /// once (move it); it must not leave the value in place or drop it twice.
    pub(super) unsafe fn with_cloned_value<R>(
        &self,
        slot: usize,
        f: impl FnOnce(*const u8) -> R,
    ) -> R {
        // SAFETY: scratch is a single `layout`-sized aligned slot.
        let tmp = unsafe { alloc_scratch(self.layout) };
        // SAFETY: `slot < len()` (contract); `tmp` is uninitialized aligned
        // storage for one value, which the clone glue initializes.
        unsafe { self.clone_value_into(slot, tmp) };
        let out = f(tmp as *const u8);
        // SAFETY: `f` moved the value out of `tmp`; free the raw bytes without
        // running drop (ownership already transferred into the live store).
        unsafe { free_scratch(tmp, self.layout) };
        out
    }

    /// Raw bytes of the stored value at `slot`, for byte-exact delta/equality
    /// comparison (design §14 delta encoding is conservative: any byte change
    /// marks the cell changed).
    ///
    /// # Safety
    /// `slot < len()`.
    pub(super) unsafe fn value_bytes(&self, slot: usize) -> &[u8] {
        // SAFETY: `slot < len()` (contract).
        let ptr = unsafe { self.data.get_ptr(slot) };
        // SAFETY: `ptr` is valid for `layout.size()` initialized bytes.
        unsafe { core::slice::from_raw_parts(ptr, self.layout.size()) }
    }

    /// Fold the value at `slot` into `hasher` via the component's hash glue, if
    /// registered. Returns `true` if a value contribution was made.
    ///
    /// # Safety
    /// `slot < len()`.
    pub(super) unsafe fn hash_value(
        &self,
        slot: usize,
        hasher: &mut dyn core::hash::Hasher,
    ) -> bool {
        match self.hash {
            Some(hash) => {
                // SAFETY: `slot < len()` (contract) so the pointer is valid.
                let ptr = unsafe { self.data.get_ptr(slot) };
                // SAFETY: `ptr` is a valid stored value of the hashed type.
                unsafe { hash(ptr, hasher) };
                true
            }
            None => false,
        }
    }
}
