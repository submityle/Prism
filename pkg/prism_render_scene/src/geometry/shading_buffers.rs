//! GPU storage buffers backing the compute-friendly surface tables.
//!
//! The compute resolve pass binds three parallel storage buffers: a sparse,
//! generation-addressed header table plus densely packed vertex and primitive
//! arrays.  Headers reference their rows by absolute offset so a single
//! dispatch can resolve any resident geometry.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::{
    shading::{
        RenderShadingGeometryHeader, RenderShadingPrimitive, RenderShadingVertex,
        SHADING_GEOMETRY_FLAG_ACTIVE,
    },
    shading_registry::RenderShadingGeometryRegistry,
};

#[derive(Resource)]
pub struct RenderShadingGeometryBuffers {
    headers: RawBufferVec<RenderShadingGeometryHeader>,
    vertices: RawBufferVec<RenderShadingVertex>,
    primitives: RawBufferVec<RenderShadingPrimitive>,
    version: u32,
}

impl FromWorld for RenderShadingGeometryBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut headers = RawBufferVec::new(BufferUsages::STORAGE);
        headers.set_label(Some("prism shading geometry headers"));
        let mut vertices = RawBufferVec::new(BufferUsages::STORAGE);
        vertices.set_label(Some("prism shading geometry vertices"));
        let mut primitives = RawBufferVec::new(BufferUsages::STORAGE);
        primitives.set_label(Some("prism shading geometry primitives"));
        Self {
            headers,
            vertices,
            primitives,
            version: 1,
        }
    }
}

impl RenderShadingGeometryBuffers {
    /// Repacks every resident geometry into the parallel storage arrays.
    ///
    /// Slot zero is reserved as an inactive fallback so a stale header index
    /// resolves to an empty geometry rather than reading a neighbour's rows.
    pub fn rebuild(&mut self, registry: &RenderShadingGeometryRegistry) {
        self.headers.clear();
        self.vertices.clear();
        self.primitives.clear();
        self.headers.push(RenderShadingGeometryHeader::default());

        for entry in registry.entries_for_upload() {
            let slot = entry.handle.index as usize;
            while self.headers.len() <= slot {
                self.headers.push(RenderShadingGeometryHeader::default());
            }
            let vertex_offset = self.vertices.len() as u32;
            let primitive_offset = self.primitives.len() as u32;
            self.vertices
                .extend(entry.geometry.vertices.iter().copied());
            self.primitives
                .extend(entry.geometry.primitives.iter().copied());
            self.headers.values_mut()[slot] = RenderShadingGeometryHeader {
                generation: entry.handle.generation,
                revision: entry.revision,
                vertex_offset,
                vertex_count: entry.geometry.vertices.len() as u32,
                primitive_offset,
                primitive_count: entry.geometry.primitives.len() as u32,
                flags: SHADING_GEOMETRY_FLAG_ACTIVE | entry.geometry.flags,
                _padding: 0,
            };
        }
        self.version = self.version.wrapping_add(1).max(1);
    }

    /// Streams the packed arrays to the GPU. Empty arrays are padded so the
    /// bind group always has a valid, non-null storage binding.
    pub fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.vertices.is_empty() {
            self.vertices.push(RenderShadingVertex::default());
        }
        if self.primitives.is_empty() {
            self.primitives.push(RenderShadingPrimitive::default());
        }
        self.headers.write_buffer(device, queue);
        self.vertices.write_buffer(device, queue);
        self.primitives.write_buffer(device, queue);
    }

    /// Monotonic version bumped on every rebuild, for bind-group caching.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The three storage buffers once they have been uploaded at least once.
    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.headers.buffer()?,
            self.vertices.buffer()?,
            self.primitives.buffer()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingGeometry, RenderShadingPrimitive, RenderShadingVertex};
    use prism_render_architecture::abi::GenerationalHandle;

    fn geometry(vertices: usize, primitives: usize) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![RenderShadingVertex::default(); vertices],
            primitives: vec![RenderShadingPrimitive::default(); primitives],
            flags: 0,
        }
    }

    #[test]
    fn packs_sparse_headers_with_absolute_row_offsets() {
        let mut world = World::new();
        let mut buffers = RenderShadingGeometryBuffers::from_world(&mut world);
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(
            GenerationalHandle {
                index: 1,
                generation: 4,
            },
            2,
            geometry(3, 1),
        );
        registry.upsert(
            GenerationalHandle {
                index: 3,
                generation: 7,
            },
            5,
            geometry(4, 2),
        );
        buffers.rebuild(&registry);

        // Slot 0 reserved + sparse growth up to slot 3 -> four headers.
        assert_eq!(buffers.headers.values().len(), 4);
        let first = buffers.headers.values()[1];
        let second = buffers.headers.values()[3];
        assert_eq!(first.generation, 4);
        assert_eq!(first.vertex_offset, 0);
        assert_eq!(first.vertex_count, 3);
        assert_eq!(first.primitive_offset, 0);
        assert_eq!(first.primitive_count, 1);
        assert_eq!(first.flags, SHADING_GEOMETRY_FLAG_ACTIVE);
        // Second geometry's rows are appended after the first's.
        assert_eq!(second.vertex_offset, 3);
        assert_eq!(second.vertex_count, 4);
        assert_eq!(second.primitive_offset, 1);
        assert_eq!(second.primitive_count, 2);
        // The reserved slot stays inactive.
        assert_eq!(buffers.headers.values()[0].flags, 0);
        assert_eq!(buffers.headers.values()[2].flags, 0);
        assert_eq!(buffers.vertices.values().len(), 7);
        assert_eq!(buffers.primitives.values().len(), 3);
    }

    #[test]
    fn rebuild_bumps_version_monotonically() {
        let mut world = World::new();
        let mut buffers = RenderShadingGeometryBuffers::from_world(&mut world);
        let registry = RenderShadingGeometryRegistry::default();
        let before = buffers.version();
        buffers.rebuild(&registry);
        assert_ne!(buffers.version(), before);
        assert_eq!(buffers.headers.values().len(), 1);
    }
}
