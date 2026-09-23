use std::collections::HashMap;

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_mesh::Mesh;
use prism_render_architecture::{geometry::GeometryRecord, gpu_scene::GeometryHandle};

/// CPU owner of stable geometry metadata; physical buffers stay backend-owned.
#[derive(Resource, Default)]
pub struct RenderGeometryRegistry {
    records: HashMap<u32, GeometryRecord>,
    assets: HashMap<AssetId<Mesh>, GeometryHandle>,
    dirty: Vec<u32>,
    next_buffer_class: u32,
    buffer_classes: HashMap<bevy_render::render_resource::BufferId, u32>,
}

impl RenderGeometryRegistry {
    pub fn upsert(&mut self, asset: AssetId<Mesh>, record: GeometryRecord) {
        self.assets.insert(asset, record.handle);
        let index = record.handle.index;
        self.records.insert(index, record);
        self.dirty.push(index);
    }

    pub fn retire(&mut self, asset: AssetId<Mesh>) -> Option<GeometryHandle> {
        let handle = self.assets.remove(&asset)?;
        self.records.remove(&handle.index);
        self.dirty.push(handle.index);
        Some(handle)
    }

    pub fn record(&self, handle: GeometryHandle) -> Option<&GeometryRecord> {
        self.records
            .get(&handle.index)
            .filter(|record| record.handle.generation == handle.generation)
    }

    pub fn handle(&self, asset: AssetId<Mesh>) -> Option<GeometryHandle> {
        self.assets.get(&asset).copied()
    }

    pub(crate) fn take_dirty(&mut self) -> Vec<u32> {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        std::mem::take(&mut self.dirty)
    }

    pub(crate) fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn version(&self) -> u32 {
        1
    }

    pub(crate) fn records_for_upload(&self) -> impl Iterator<Item = &GeometryRecord> {
        self.records.values()
    }

    pub(crate) fn buffer_class(
        &mut self,
        buffer: &bevy_render::render_resource::Buffer,
    ) -> u32 {
        let id = buffer.id();
        if let Some(class) = self.buffer_classes.get(&id) {
            return *class;
        }
        self.next_buffer_class = self.next_buffer_class.saturating_add(1).max(1);
        self.buffer_classes.insert(id, self.next_buffer_class);
        self.next_buffer_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use prism_render_architecture::abi::GenerationalHandle;

    #[test]
    fn registry_rejects_stale_generation_and_deduplicates_dirty_rows() {
        let asset = AssetId::Uuid {
            uuid: Uuid::from_u128(7),
        };
        let handle = GenerationalHandle {
            index: 3,
            generation: 2,
        };
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(asset, GeometryRecord { handle, ..Default::default() });
        registry.upsert(asset, GeometryRecord { handle, revision: 2, ..Default::default() });
        assert_eq!(registry.record(handle).unwrap().revision, 2);
        assert_eq!(registry.take_dirty(), vec![3]);
        assert!(registry.record(GenerationalHandle { generation: 1, ..handle }).is_none());
        assert_eq!(registry.retire(asset), Some(handle));
        assert_eq!(registry.take_dirty(), vec![3]);
    }
}
