//! Authoritative material record store keyed by a generational handle.
//!
//! [`MaterialRecord`] on its own is just a value; the renderer needs a stable,
//! persistent place to keep every registered material so that GPU scene
//! instances can reference one by [`MaterialHandle`] and the deferred resolve
//! stage can recover its [`MaterialExecutionPath`]. [`MaterialRegistry`] is that
//! place: a generational arena that hands out reusable slots, validates stale
//! handles, and answers the "which resolve path does this material take?"
//! question that [`resolve`](super::resolve) buckets on.
//!
//! Slots are recycled through a free list, and each recycle bumps the slot's
//! generation so a handle to a removed material can never silently resolve to
//! the material that later took its place. This mirrors the generational
//! handles used across the rest of the architecture contract.

use alloc::vec::Vec;

use super::resolve::MaterialResolveBins;
use super::{MaterialExecutionPath, MaterialHandle, MaterialRecord};

#[derive(Clone, Debug)]
struct MaterialSlot {
    generation: u32,
    record: Option<MaterialRecord>,
}

/// A generational arena mapping [`MaterialHandle`]s to [`MaterialRecord`]s.
///
/// Insertion reuses freed slots before growing, keeping the backing storage
/// compact so a GPU mirror can upload it as a dense table indexed by slot.
#[derive(Clone, Debug, Default)]
pub struct MaterialRegistry {
    slots: Vec<MaterialSlot>,
    free: Vec<u32>,
    live: usize,
}

impl MaterialRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Creates a registry pre-sized for `capacity` materials.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Number of live materials currently registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns `true` when no material is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Number of slots backing the registry, including recycled free slots.
    ///
    /// This is the width a dense GPU mirror must allocate, since freed slots
    /// keep their index until reused.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Registers `record`, returning a handle that resolves to it.
    ///
    /// A freed slot is reused before the arena grows; reusing a slot bumps its
    /// generation so previously issued handles to that slot stop resolving.
    pub fn insert(&mut self, record: MaterialRecord) -> MaterialHandle {
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.generation = slot.generation.wrapping_add(1);
            slot.record = Some(record);
            return MaterialHandle {
                index,
                generation: slot.generation,
            };
        }
        let index = self.slots.len() as u32;
        self.slots.push(MaterialSlot {
            generation: 1,
            record: Some(record),
        });
        MaterialHandle {
            index,
            generation: 1,
        }
    }

    /// Returns the record for `handle`, or `None` if the handle is stale or the
    /// slot is empty.
    #[must_use]
    pub fn get(&self, handle: MaterialHandle) -> Option<&MaterialRecord> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.record.as_ref()
    }

    /// Returns a mutable reference to the record for `handle`, validating the
    /// generation first.
    pub fn get_mut(&mut self, handle: MaterialHandle) -> Option<&mut MaterialRecord> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.record.as_mut()
    }

    /// Returns `true` when `handle` resolves to a live material.
    #[must_use]
    pub fn contains(&self, handle: MaterialHandle) -> bool {
        self.get(handle).is_some()
    }

    /// Returns the current live record occupying `slot`, ignoring generation.
    ///
    /// GPU scene instance rows carry a bare `material_index`, so the resolve
    /// classifier reads the current occupant of a slot rather than validating a
    /// handle generation it does not carry. An empty slot yields `None`.
    #[must_use]
    pub fn record_at(&self, slot: u32) -> Option<&MaterialRecord> {
        self.slots.get(slot as usize)?.record.as_ref()
    }

    /// Removes the material referenced by `handle`, returning its record.
    ///
    /// The freed slot is recycled on a later [`insert`](Self::insert); its
    /// generation is bumped at reuse so this handle cannot resolve again.
    pub fn remove(&mut self, handle: MaterialHandle) -> Option<MaterialRecord> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let record = slot.record.take()?;
        self.free.push(handle.index);
        self.live -= 1;
        Some(record)
    }

    /// Classifies the visible material slots into deferred resolve buckets.
    ///
    /// `visible_slots` is the set of `material_index` values referenced by the
    /// frame's visible instances (from `GpuSceneInstance`). Slots are
    /// deduplicated — each material resolves once as a screen-space pass — while
    /// preserving first-seen order for deterministic submission. A slot with no
    /// live record is routed to the diagnostic fallback so missing materials
    /// stay visible instead of being dropped.
    #[must_use]
    pub fn classify_visible(&self, visible_slots: &[u32]) -> MaterialResolveBins {
        let mut bins = MaterialResolveBins::default();
        let mut seen: Vec<u32> = Vec::new();
        for &slot in visible_slots {
            if let Err(pos) = seen.binary_search(&slot) {
                seen.insert(pos, slot);
            } else {
                continue;
            }
            let path = self
                .record_at(slot)
                .map_or(MaterialExecutionPath::DiagnosticFallback, |record| {
                    record.execution
                });
            bins.push(path, slot);
        }
        bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialDomain;

    fn record(execution: MaterialExecutionPath) -> MaterialRecord {
        MaterialRecord {
            domain: MaterialDomain::Surface,
            execution,
            parameter_offset: 0,
            shader: None,
        }
    }

    #[test]
    fn insert_then_get_round_trips() {
        let mut reg = MaterialRegistry::new();
        let h = reg.insert(record(MaterialExecutionPath::FixedPbr));
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(h));
        assert_eq!(
            reg.get(h).map(|r| r.execution),
            Some(MaterialExecutionPath::FixedPbr)
        );
    }

    #[test]
    fn remove_frees_slot_and_invalidates_handle() {
        let mut reg = MaterialRegistry::new();
        let h = reg.insert(record(MaterialExecutionPath::FixedNpr));
        assert!(reg.remove(h).is_some());
        assert_eq!(reg.len(), 0);
        assert!(!reg.contains(h));
        assert!(reg.get(h).is_none());
        // Double remove is a no-op.
        assert!(reg.remove(h).is_none());
    }

    #[test]
    fn reused_slot_bumps_generation_and_rejects_stale_handle() {
        let mut reg = MaterialRegistry::new();
        let stale = reg.insert(record(MaterialExecutionPath::FixedPbr));
        reg.remove(stale);
        let fresh = reg.insert(record(MaterialExecutionPath::ClosureTable));
        // Slot index is recycled but the generation advanced.
        assert_eq!(fresh.index, stale.index);
        assert_ne!(fresh.generation, stale.generation);
        // The stale handle must not resolve to the new occupant.
        assert!(reg.get(stale).is_none());
        assert_eq!(
            reg.get(fresh).map(|r| r.execution),
            Some(MaterialExecutionPath::ClosureTable)
        );
        assert_eq!(reg.slot_count(), 1);
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut reg = MaterialRegistry::new();
        let h = reg.insert(record(MaterialExecutionPath::FixedPbr));
        reg.get_mut(h).expect("live handle").execution = MaterialExecutionPath::FixedNpr;
        assert_eq!(
            reg.get(h).map(|r| r.execution),
            Some(MaterialExecutionPath::FixedNpr)
        );
    }

    #[test]
    fn classify_visible_dedups_and_buckets_by_path() {
        let mut reg = MaterialRegistry::new();
        let pbr = reg.insert(record(MaterialExecutionPath::FixedPbr));
        let npr = reg.insert(record(MaterialExecutionPath::FixedNpr));
        // A hybrid mesh references both, plus a repeat of the PBR sub-material.
        let visible = [pbr.index, npr.index, pbr.index];
        let bins = reg.classify_visible(&visible);
        assert_eq!(bins.fixed_pbr, [pbr.index]);
        assert_eq!(bins.fixed_npr, [npr.index]);
        assert_eq!(bins.total(), 2);
    }

    #[test]
    fn classify_visible_routes_empty_slot_to_fallback() {
        let mut reg = MaterialRegistry::new();
        let h = reg.insert(record(MaterialExecutionPath::FixedPbr));
        reg.remove(h);
        // Slot 0 is now empty; a stale visibility list still references it.
        let bins = reg.classify_visible(&[0]);
        assert_eq!(bins.diagnostic_fallback, [0]);
        assert_eq!(bins.total(), 1);
    }

    #[test]
    fn classify_visible_preserves_first_seen_order() {
        let mut reg = MaterialRegistry::new();
        let a = reg.insert(record(MaterialExecutionPath::FixedPbr));
        let b = reg.insert(record(MaterialExecutionPath::FixedPbr));
        let c = reg.insert(record(MaterialExecutionPath::FixedPbr));
        let bins = reg.classify_visible(&[c.index, a.index, b.index, a.index]);
        assert_eq!(bins.fixed_pbr, [c.index, a.index, b.index]);
    }
}
