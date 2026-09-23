//! Render-world store of compute-friendly surface tables keyed by geometry.
//!
//! Mirrors [`super::runtime::RenderGeometryRegistry`] but retains the stable
//! vertex/primitive rows required by the compute resolve pass, which the
//! rasterizer-oriented geometry registry does not keep.

use alloc::collections::BTreeMap;

use prism_render_architecture::abi::GenerationalHandle;

use super::shading::RenderShadingGeometry;

/// One resident geometry's surface table plus its generational identity.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderShadingGeometryEntry {
    pub handle: GenerationalHandle,
    pub revision: u32,
    pub geometry: RenderShadingGeometry,
}

/// Sparse, generation-addressed registry of surface tables.
#[derive(Debug, Default)]
pub struct RenderShadingGeometryRegistry {
    entries: BTreeMap<u32, RenderShadingGeometryEntry>,
    dirty: bool,
}

impl RenderShadingGeometryRegistry {
    /// Inserts or replaces the surface table for one geometry handle. Returns
    /// `true` when the stored payload actually changed.
    pub fn upsert(
        &mut self,
        handle: GenerationalHandle,
        revision: u32,
        geometry: RenderShadingGeometry,
    ) -> bool {
        let entry = RenderShadingGeometryEntry { handle, revision, geometry };
        if self.entries.get(&handle.index) == Some(&entry) {
            return false;
        }
        self.entries.insert(handle.index, entry);
        self.dirty = true;
        true
    }

    /// Removes a geometry slot, e.g. when its asset is unloaded.
    pub fn remove(&mut self, index: u32) -> Option<RenderShadingGeometryEntry> {
        let removed = self.entries.remove(&index);
        if removed.is_some() {
            self.dirty = true;
        }
        removed
    }

    /// Returns the entry stored at a sparse slot, if any.
    pub fn get(&self, index: u32) -> Option<&RenderShadingGeometryEntry> {
        self.entries.get(&index)
    }

    /// Number of resident geometries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no geometries are resident.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `true` when the registry changed since the last [`Self::take_dirty`].
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears and returns the dirty flag so callers can rebuild buffers once.
    pub fn take_dirty(&mut self) -> bool {
        core::mem::replace(&mut self.dirty, false)
    }

    /// Entries in ascending slot order, ready for deterministic packing.
    pub fn entries_for_upload(&self) -> impl Iterator<Item = &RenderShadingGeometryEntry> {
        self.entries.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingGeometry, RenderShadingPrimitive, RenderShadingVertex};

    fn geometry(marker: f32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![RenderShadingVertex { position: [marker; 3], ..Default::default() }],
            primitives: vec![RenderShadingPrimitive { indices: [0, 0, 0], flags: 0 }],
            flags: 0,
        }
    }

    #[test]
    fn upsert_tracks_change_and_dirty_state() {
        let mut registry = RenderShadingGeometryRegistry::default();
        let handle = GenerationalHandle { index: 2, generation: 1 };
        assert!(registry.upsert(handle, 1, geometry(1.0)));
        assert!(registry.is_dirty());
        assert!(registry.take_dirty());
        assert!(!registry.is_dirty());
        // Identical payload is a no-op and does not re-dirty the registry.
        assert!(!registry.upsert(handle, 1, geometry(1.0)));
        assert!(!registry.is_dirty());
        // Changed payload re-dirties.
        assert!(registry.upsert(handle, 2, geometry(2.0)));
        assert!(registry.is_dirty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn entries_are_returned_in_ascending_slot_order() {
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(GenerationalHandle { index: 7, generation: 1 }, 1, geometry(7.0));
        registry.upsert(GenerationalHandle { index: 3, generation: 1 }, 1, geometry(3.0));
        let indices: Vec<u32> = registry.entries_for_upload().map(|e| e.handle.index).collect();
        assert_eq!(indices, vec![3, 7]);
        assert!(registry.remove(3).is_some());
        assert_eq!(registry.len(), 1);
    }
}
