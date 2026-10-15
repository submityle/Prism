//! A type-erased snapshot column: one immutable value reference per holder,
//! kept densely with its exact change ticks (design §14).
//!
//! The live world stores component values it owns; a snapshot must own an
//! independent copy so the world can keep mutating (predict) while the snapshot
//! stays frozen (authoritative), and so a snapshot can be restored repeatedly
//! (rollback replays the same confirmed frame many times). Clone-only values
//! live in independent [`BlobVec`] storage. Explicit SnapshotValue registration
//! also permits sharing frozen, snapshot-owned chunk buffers across captures,
//! deltas and reconstructed history. These buffers never alias live storage.

use alloc::alloc::{alloc, dealloc};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::alloc::Layout;

use crate::change::Tick;
use crate::component::{CloneFn, ComponentId, DropFn, SnapshotHashFn, StorageType};
use crate::storage::BlobVec;

use super::hash::SnapshotEq;
use super::reuse::ChunkOrigin;

/// Values in this buffer are never exposed mutably or moved out. Arc shares
/// ownership only after SnapshotValue registration, not arbitrary Clone data.
pub(super) struct SnapshotChunk {
    data: BlobVec,
    pub(super) origin: Option<ChunkOrigin>,
}

struct SharedCell {
    chunk: Arc<SnapshotChunk>,
    slot: usize,
}

enum SnapshotData {
    Owned(BlobVec),
    Shared(Vec<SharedCell>),
}

impl SnapshotData {
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::Shared(cells) => cells.len(),
        }
    }

    /// # Safety
    /// slot must index an initialized value.
    unsafe fn get_ptr(&self, slot: usize) -> *const u8 {
        match self {
            // SAFETY: forwarded initialized-slot contract.
            Self::Owned(data) => unsafe { data.get_ptr(slot) },
            Self::Shared(cells) => {
                let cell = &cells[slot];
                // SAFETY: shared cells retain their buffer and a valid slot.
                unsafe { cell.chunk.data.get_ptr(cell.slot) }
            }
        }
    }
}

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
    /// The component's memory layout (also the stride within value buffers).
    pub(super) layout: Layout,
    /// Type-erased clone glue used both to capture (clone live → snapshot) and
    /// restore (clone snapshot → live).
    pub(super) clone: CloneFn,
    /// Type-erased drop glue, carried by `data` so cloned values are released.
    pub(super) drop: Option<DropFn>,
    /// Optional deterministic hash glue for [`state_hash`](super::WorldSnapshot::state_hash).
    pub(super) hash: Option<SnapshotHashFn>,
    /// Optional field-semantic comparison, captured alongside the value glue.
    pub(super) equality: Option<SnapshotEq>,
    /// Independent values or immutable chunk references, one per holder.
    data: SnapshotData,
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
            equality: None,
            data: SnapshotData::Owned(BlobVec::new(layout, drop)),
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

    /// An empty owning column with the same captured type/value contracts.
    /// Delta fresh storage and reconstructed columns must preserve these even
    /// after the original world and target snapshot have been dropped.
    pub(super) fn empty_like(&self) -> Self {
        let mut column = Self::new(
            self.component,
            self.storage,
            self.layout,
            self.clone,
            self.drop,
            self.hash,
        );
        column.equality = self.equality;
        if matches!(&self.data, SnapshotData::Shared(_)) {
            column.enable_sharing();
        }
        column
    }

    pub(super) fn enable_sharing(&mut self) {
        assert_eq!(self.len(), 0, "sharing must be selected before capture");
        self.data = SnapshotData::Shared(Vec::new());
    }

    pub(super) fn uses_shared_values(&self) -> bool {
        matches!(&self.data, SnapshotData::Shared(_))
    }

    pub(super) fn shared_chunks(&self) -> impl Iterator<Item = &Arc<SnapshotChunk>> {
        let cells = match &self.data {
            SnapshotData::Shared(cells) => cells.as_slice(),
            SnapshotData::Owned(_) => &[],
        };
        cells.iter().map(|cell| &cell.chunk)
    }

    pub(super) fn into_chunk(self, origin: Option<ChunkOrigin>) -> Arc<SnapshotChunk> {
        let SnapshotData::Owned(data) = self.data else {
            panic!("capture buffers must be independently owned before freezing");
        };
        Arc::new(SnapshotChunk { data, origin })
    }

    pub(super) fn push_shared(
        &mut self,
        chunk: Arc<SnapshotChunk>,
        slot: usize,
        added: Tick,
        changed: Tick,
        row: u32,
    ) {
        assert!(
            slot < chunk.data.len(),
            "shared snapshot slot out of bounds"
        );
        let SnapshotData::Shared(cells) = &mut self.data else {
            panic!("shared capture requires value-semantic registration");
        };
        cells.push(SharedCell { chunk, slot });
        self.added.push(added);
        self.changed.push(changed);
        self.rows.push(row);
    }

    /// Identity of the very same frozen value proves reuse without invoking
    /// user comparison or inspecting value bytes (including heap payloads).
    pub(super) fn shares_value(&self, slot: usize, other: &Self, other_slot: usize) -> bool {
        match (&self.data, &other.data) {
            (SnapshotData::Shared(a), SnapshotData::Shared(b)) => {
                a[slot].slot == b[other_slot].slot
                    && Arc::ptr_eq(&a[slot].chunk, &b[other_slot].chunk)
            }
            _ => false,
        }
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
        match &mut self.data {
            // SAFETY: clone glue initialized tmp with this column's type.
            SnapshotData::Owned(data) => unsafe { data.push(tmp) },
            SnapshotData::Shared(cells) => {
                let mut data = BlobVec::new(self.layout, self.drop);
                // SAFETY: tmp owns one initialized value, transferred to data.
                unsafe { data.push(tmp) };
                cells.push(SharedCell {
                    chunk: Arc::new(SnapshotChunk { data, origin: None }),
                    slot: 0,
                });
            }
        }
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
    /// cells. Explicit immutable buffers are shared; other values are cloned
    /// (never moved), so both `src` and this column stay valid.
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
        if let (SnapshotData::Shared(target), SnapshotData::Shared(source)) =
            (&mut self.data, &src.data)
        {
            let cell = &source[src_slot];
            target.push(SharedCell {
                chunk: cell.chunk.clone(),
                slot: cell.slot,
            });
            self.added.push(src.added[src_slot]);
            self.changed.push(src.changed[src_slot]);
            self.rows.push(row);
            return;
        }
        // SAFETY: scratch is a single `layout`-sized aligned slot.
        let tmp = unsafe { alloc_scratch(self.layout) };
        // SAFETY: `src_slot < src.len()` (contract); `tmp` is uninitialized,
        // aligned storage for one value, which the clone glue initializes.
        unsafe { src.clone_value_into(src_slot, tmp) };
        // SAFETY: `tmp` now holds an initialized owned value of the column's
        // type; `push` moves (memcpy) it into `data`, which takes ownership.
        match &mut self.data {
            // SAFETY: tmp contains a clone of the same component type.
            SnapshotData::Owned(data) => unsafe { data.push(tmp) },
            SnapshotData::Shared(cells) => {
                let mut data = BlobVec::new(self.layout, self.drop);
                // SAFETY: tmp owns one initialized value, transferred to data.
                unsafe { data.push(tmp) };
                cells.push(SharedCell {
                    chunk: Arc::new(SnapshotChunk { data, origin: None }),
                    slot: 0,
                });
            }
        }
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

    /// Compare two owned values through captured typed glue. `None` means no
    /// compatible comparison contract is available: callers must not infer
    /// equality from matching ticks, hashes, layouts, or pointer addresses.
    /// In particular, delta encoding must clone a fresh value in that case.
    pub(super) fn value_eq(&self, slot: usize, other: &Self, other_slot: usize) -> Option<bool> {
        assert!(slot < self.len(), "snapshot comparison slot out of bounds");
        assert!(
            other_slot < other.len(),
            "snapshot comparison slot out of bounds"
        );
        if self.component != other.component || self.layout != other.layout {
            return Some(false);
        }
        let (left, right) = (self.equality?, other.equality?);
        if !left.compatible(&right) {
            return None;
        }
        // SAFETY: the checked slots contain initialized owned values. Capture
        // validates the callbacks against their concrete type and layout, and
        // empty_like preserves that contract across delta reconstruction.
        let (a, b) = unsafe { (self.data.get_ptr(slot), other.data.get_ptr(other_slot)) };
        // SAFETY: both captured callbacks describe the same concrete Rust type.
        Some(unsafe { left.compare_ptr(a, b) })
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

#[cfg(test)]
#[path = "value_semantics_tests.rs"]
mod value_semantics_tests;
