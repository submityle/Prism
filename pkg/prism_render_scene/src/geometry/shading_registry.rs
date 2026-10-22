//! Render-world store of compute-friendly surface tables keyed by geometry.
//!
//! Mirrors [`super::runtime::RenderGeometryRegistry`] but retains the stable
//! vertex/primitive rows required by the compute resolve pass, which the
//! rasterizer-oriented geometry registry does not keep.

use alloc::collections::BTreeMap;

use bevy_ecs::resource::Resource;

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
#[derive(Resource, Debug, Default)]
pub struct RenderShadingGeometryRegistry {
    entries: BTreeMap<u32, RenderShadingGeometryEntry>,
    lod_entries: BTreeMap<(u32, u32), RenderShadingGeometryEntry>,
    lod_payload_revisions: BTreeMap<(u32, u32), u32>,
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
        let entry = RenderShadingGeometryEntry {
            handle,
            revision,
            geometry,
        };
        if self.entries.get(&handle.index) == Some(&entry) {
            return false;
        }
        self.entries.insert(handle.index, entry);
        self.lod_entries
            .retain(|(index, _), _| *index != handle.index);
        self.lod_payload_revisions
            .retain(|(index, _), _| *index != handle.index);
        self.dirty = true;
        true
    }

    /// Inserts or replaces one resident LOD surface table for a geometry.
    /// LOD tables are kept separate from the base upload table because the
    /// compute shading ABI has one header row per geometry generation, while
    /// ray consumers need CPU triangles for the exact selected representation.
    pub fn upsert_lod(
        &mut self,
        handle: GenerationalHandle,
        level: u32,
        revision: u32,
        payload_revision: u32,
        geometry: RenderShadingGeometry,
    ) -> bool {
        if level == 0 || payload_revision == 0 {
            return false;
        }
        let key = (handle.index, level);
        let entry = RenderShadingGeometryEntry {
            handle,
            revision,
            geometry,
        };
        if self.lod_entries.get(&key) == Some(&entry)
            && self.lod_payload_revisions.get(&key) == Some(&payload_revision)
        {
            return false;
        }
        self.lod_entries.insert(key, entry);
        self.lod_payload_revisions.insert(key, payload_revision);
        self.dirty = true;
        true
    }

    /// Removes a geometry slot, e.g. when its asset is unloaded.
    pub fn remove(&mut self, index: u32) -> Option<RenderShadingGeometryEntry> {
        let removed = self.entries.remove(&index);
        let lod_count_before = self.lod_entries.len();
        self.lod_entries
            .retain(|(lod_index, _), _| *lod_index != index);
        self.lod_payload_revisions
            .retain(|(lod_index, _), _| *lod_index != index);
        if removed.is_some() || self.lod_entries.len() != lod_count_before {
            self.dirty = true;
        }
        removed
    }

    /// Returns the entry stored at a sparse slot, if any.
    pub fn get(&self, index: u32) -> Option<&RenderShadingGeometryEntry> {
        self.entries.get(&index)
    }

    /// Returns a resident LOD surface table for the exact generation.
    pub fn lod(
        &self,
        handle: GenerationalHandle,
        level: u32,
    ) -> Option<&RenderShadingGeometryEntry> {
        self.lod_entries
            .get(&(handle.index, level))
            .filter(|entry| entry.handle == handle)
    }

    /// Returns the content revision of one resident LOD surface table.
    #[must_use]
    pub fn lod_payload_revision(&self, handle: GenerationalHandle, level: u32) -> Option<u32> {
        self.lod_entries
            .get(&(handle.index, level))
            .filter(|entry| entry.handle == handle)
            .and_then(|_| {
                self.lod_payload_revisions
                    .get(&(handle.index, level))
                    .copied()
            })
    }

    /// Stable source identities for acceleration-cache invalidation.
    pub(crate) fn lod_payload_revisions_for_signature(
        &self,
    ) -> impl Iterator<Item = (u32, u32, u32, u32)> + '_ {
        self.lod_payload_revisions
            .iter()
            .filter_map(|(&(index, level), &payload_revision)| {
                let entry = self.lod_entries.get(&(index, level))?;
                Some((index, entry.handle.generation, level, payload_revision))
            })
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

    /// Drops every slot whose index is no longer reported live by the scene,
    /// e.g. after a geometry asset is retired. Dirties the registry when it
    /// actually removes something so buffers repack exactly once.
    pub fn retain_live(&mut self, is_live: impl Fn(u32) -> bool) {
        let before = self.entries.len() + self.lod_entries.len();
        self.entries.retain(|index, _| is_live(*index));
        self.lod_entries.retain(|(index, _), _| is_live(*index));
        self.lod_payload_revisions
            .retain(|(index, _), _| is_live(*index));
        if self.entries.len() + self.lod_entries.len() != before {
            self.dirty = true;
        }
    }

    /// Retains only the LOD keys still published by the scene mirror.
    pub fn retain_live_lods(&mut self, is_live: impl Fn(u32, u32) -> bool) {
        let before = self.lod_entries.len();
        self.lod_entries
            .retain(|(index, level), _| is_live(*index, *level));
        self.lod_payload_revisions
            .retain(|(index, level), _| is_live(*index, *level));
        if self.lod_entries.len() != before {
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingGeometry, RenderShadingPrimitive, RenderShadingVertex};

    fn geometry(marker: f32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![RenderShadingVertex {
                position: [marker; 3],
                ..Default::default()
            }],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 0, 0],
                flags: 0,
            }],
            flags: 0,
        }
    }

    #[test]
    fn upsert_tracks_change_and_dirty_state() {
        let mut registry = RenderShadingGeometryRegistry::default();
        let handle = GenerationalHandle {
            index: 2,
            generation: 1,
        };
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
        registry.upsert(
            GenerationalHandle {
                index: 7,
                generation: 1,
            },
            1,
            geometry(7.0),
        );
        registry.upsert(
            GenerationalHandle {
                index: 3,
                generation: 1,
            },
            1,
            geometry(3.0),
        );
        let indices: Vec<u32> = registry
            .entries_for_upload()
            .map(|e| e.handle.index)
            .collect();
        assert_eq!(indices, vec![3, 7]);
        assert!(registry.remove(3).is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn lod_payload_revision_is_generation_scoped_and_removal_dirties_registry() {
        let mut registry = RenderShadingGeometryRegistry::default();
        let handle = GenerationalHandle::new(5, 2);
        assert!(registry.upsert(handle, 9, geometry(0.0)));
        assert!(registry.upsert_lod(handle, 1, 9, 3, geometry(1.0)));
        assert_eq!(registry.lod_payload_revision(handle, 1), Some(3));
        assert!(registry.take_dirty());

        assert!(registry.upsert_lod(handle, 1, 9, 4, geometry(1.0)));
        assert_eq!(registry.lod_payload_revision(handle, 1), Some(4));
        assert!(registry.take_dirty());

        assert!(registry.remove(handle.index).is_some());
        assert!(registry.is_dirty());
        assert_eq!(registry.lod_payload_revision(handle, 1), None);
    }
}
