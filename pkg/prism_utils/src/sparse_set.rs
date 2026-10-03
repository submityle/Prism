//! A sparse set (`SparseSet`): `O(1)` insert/remove/contains keyed by a `u32`
//! index, with cache-friendly dense iteration over the stored values.
//!
//! The structure keeps a `sparse` array mapping an id to its position in a
//! packed `dense` array, alongside the parallel `values`. Removal uses
//! swap-remove on the dense arrays and patches the moved element's sparse
//! entry, keeping values densely packed for fast linear scans. This is the
//! classic ECS component-storage layout (EnTT/flecs style). The implementation
//! is fully safe.

extern crate alloc;

use alloc::vec::Vec;

/// Filler stored in freshly grown `sparse` entries. Lookups never trust it
/// directly; they always re-validate against the dense id array.
const UNUSED: u32 = u32::MAX;

/// A sparse-to-dense set storing a value of type `T` per `u32` key.
///
/// Keys (ids) may be arbitrary `u32` values; the `sparse` layer grows to cover
/// the largest id seen. Iteration visits values in dense (insertion-order,
/// minus swap-removals) order, which is stable and address-independent.
pub struct SparseSet<T> {
    /// `sparse[id]` is the dense slot holding `id` (validated via `dense_ids`).
    sparse: Vec<u32>,
    /// Dense slot -> owning id; parallel to `values`.
    dense_ids: Vec<u32>,
    /// Dense slot -> value; parallel to `dense_ids`.
    values: Vec<T>,
}

impl<T> SparseSet<T> {
    /// Create an empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_ids: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create an empty sparse set with pre-reserved dense capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            sparse: Vec::new(),
            dense_ids: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
    }

    /// Whether `id` currently has a value stored.
    pub fn contains(&self, id: u32) -> bool {
        self.dense_slot(id).is_some()
    }

    /// Resolve `id` to its dense slot if present, validating against the dense
    /// id array so stale `sparse` entries never produce a false positive.
    fn dense_slot(&self, id: u32) -> Option<usize> {
        let slot = *self.sparse.get(id as usize)? as usize;
        if slot < self.dense_ids.len() && self.dense_ids[slot] == id {
            Some(slot)
        } else {
            None
        }
    }

    /// Insert or overwrite the value for `id`.
    ///
    /// Returns the previous value if `id` was already present.
    pub fn insert(&mut self, id: u32, value: T) -> Option<T> {
        if let Some(slot) = self.dense_slot(id) {
            return Some(core::mem::replace(&mut self.values[slot], value));
        }
        let idx = id as usize;
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, UNUSED);
        }
        let slot = self.dense_ids.len() as u32;
        self.sparse[idx] = slot;
        self.dense_ids.push(id);
        self.values.push(value);
        None
    }

    /// Borrow the value for `id`, if present.
    pub fn get(&self, id: u32) -> Option<&T> {
        let slot = self.dense_slot(id)?;
        Some(&self.values[slot])
    }

    /// Mutably borrow the value for `id`, if present.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut T> {
        let slot = self.dense_slot(id)?;
        Some(&mut self.values[slot])
    }

    /// Remove and return the value for `id` via dense swap-remove.
    pub fn remove(&mut self, id: u32) -> Option<T> {
        let slot = self.dense_slot(id)?;
        let last = self.dense_ids.len() - 1;
        let value = self.values.swap_remove(slot);
        self.dense_ids.swap_remove(slot);
        if slot != last {
            // The former last element now lives at `slot`; fix its sparse link.
            let moved_id = self.dense_ids[slot];
            self.sparse[moved_id as usize] = slot as u32;
        }
        Some(value)
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.dense_ids.len()
    }

    /// Whether the set stores no values.
    pub fn is_empty(&self) -> bool {
        self.dense_ids.is_empty()
    }

    /// Dense value capacity currently allocated.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    /// Remove all values, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.sparse.clear();
        self.dense_ids.clear();
        self.values.clear();
    }

    /// Iterate over shared references to the values in dense order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    /// Iterate over mutable references to the values in dense order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.values.iter_mut()
    }

    /// Iterate over the stored ids in dense order.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.dense_ids.iter().copied()
    }

    /// Iterate over `(id, &value)` pairs in dense order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.dense_ids
            .iter()
            .copied()
            .zip(self.values.iter())
    }

    /// Iterate over `(id, &mut value)` pairs in dense order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.dense_ids
            .iter()
            .copied()
            .zip(self.values.iter_mut())
    }
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self::new()
    }
}
