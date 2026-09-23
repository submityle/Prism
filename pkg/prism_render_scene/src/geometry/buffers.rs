use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::{
    rows::{RenderGeometryHeader, RenderGeometryLod, GEOMETRY_FLAG_ACTIVE},
    runtime::RenderGeometryRegistry,
};

#[derive(Resource)]
pub(crate) struct RenderGeometryBuffers {
    headers: RawBufferVec<RenderGeometryHeader>,
    lods: RawBufferVec<RenderGeometryLod>,
    version: u32,
}

impl FromWorld for RenderGeometryBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut headers = RawBufferVec::new(BufferUsages::STORAGE);
        headers.set_label(Some("prism geometry headers"));
        let mut lods = RawBufferVec::new(BufferUsages::STORAGE);
        lods.set_label(Some("prism geometry lods"));
        Self {
            headers,
            lods,
            version: 1,
        }
    }
}

impl RenderGeometryBuffers {
    pub(crate) fn rebuild(&mut self, registry: &RenderGeometryRegistry) {
        self.headers.clear();
        self.lods.clear();
        self.headers.push(RenderGeometryHeader::default());
        self.lods.push(RenderGeometryLod::default());
        let mut records: Vec<_> = registry.records_for_upload().collect();
        records.sort_by_key(|record| record.handle.index);
        for record in records {
            while self.headers.len() <= record.handle.index as usize {
                self.headers.push(RenderGeometryHeader::default());
            }
            let lod_offset = self.lods.len() as u32;
            self.lods.extend(record.lods.iter().copied().map(|lod| {
                RenderGeometryLod::from_record(
                    lod,
                    record.vertex_buffer_class,
                    record.index_buffer_class,
                )
            }));
            self.headers.values_mut()[record.handle.index as usize] = RenderGeometryHeader {
                generation: record.handle.generation,
                revision: record.revision,
                lod_offset,
                lod_count: record.lods.len() as u32,
                primitive_start: record.primitive_start,
                primitive_count: record.primitive_count,
                flags: GEOMETRY_FLAG_ACTIVE,
                _padding: 0,
            };
        }
        self.version = self.version.wrapping_add(1).max(1);
    }

    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.headers.write_buffer(device, queue);
        self.lods.write_buffer(device, queue);
    }

    pub(crate) fn buffers(&self) -> Option<(&Buffer, &Buffer)> {
        Some((self.headers.buffer()?, self.lods.buffer()?))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, AssetId};
    use prism_render_architecture::{
        abi::GenerationalHandle,
        geometry::{GeometryLodRecord, GeometryRecord},
    };

    #[test]
    fn table_preserves_sparse_generational_addresses() {
        let mut world = World::new();
        let mut buffers = RenderGeometryBuffers::from_world(&mut world);
        let mut registry = RenderGeometryRegistry::default();
        let handle = GenerationalHandle { index: 5, generation: 3 };
        registry.upsert(
            AssetId::Uuid { uuid: Uuid::from_u128(5) },
            GeometryRecord {
                handle,
                lods: vec![GeometryLodRecord { resident: true, ..Default::default() }],
                ..Default::default()
            },
        );
        buffers.rebuild(&registry);
        assert_eq!(buffers.headers.values().len(), 6);
        assert_eq!(buffers.headers.values()[5].generation, 3);
        assert_eq!(buffers.headers.values()[5].lod_offset, 1);
        assert_eq!(buffers.lods.values().len(), 2);
    }
}
