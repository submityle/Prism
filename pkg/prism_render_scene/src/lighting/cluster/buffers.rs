//! GPU storage buffers backing the clustered-light tables.
//!
//! The clustered resolve pass binds three parallel storage buffers: the
//! single-element [`GpuClusterGrid`] record, the per-cluster `[offset, count]`
//! table, and the flat light-index list.  They are repacked from
//! [`ClusterCpuData`] every frame; empty tables are padded with one neutral
//! element so the bindings are never zero-sized in an unlit scene.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::abi::GpuClusterGrid;
use super::build::ClusterCpuData;

/// The three storage buffers mirroring the clustered tables onto the GPU.
#[derive(Resource)]
pub struct ClusterGpuBuffers {
    grid: RawBufferVec<GpuClusterGrid>,
    offsets_and_counts: RawBufferVec<[u32; 2]>,
    light_indices: RawBufferVec<u32>,
    version: u32,
}

impl FromWorld for ClusterGpuBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut grid = RawBufferVec::new(BufferUsages::STORAGE);
        grid.set_label(Some("prism cluster grid"));
        let mut offsets_and_counts = RawBufferVec::new(BufferUsages::STORAGE);
        offsets_and_counts.set_label(Some("prism cluster offsets"));
        let mut light_indices = RawBufferVec::new(BufferUsages::STORAGE);
        light_indices.set_label(Some("prism cluster light indices"));
        Self {
            grid,
            offsets_and_counts,
            light_indices,
            version: 1,
        }
    }
}

impl ClusterGpuBuffers {
    /// Repacks the built clustered tables into the staging arrays.
    ///
    /// The grid buffer always holds exactly one record, so its cluster count
    /// stays authoritative even when the scene has no lights.
    pub fn rebuild(&mut self, data: &ClusterCpuData) {
        self.grid.clear();
        self.offsets_and_counts.clear();
        self.light_indices.clear();

        self.grid.push(data.grid);
        self.offsets_and_counts
            .extend(data.offsets_and_counts.iter().copied());
        self.light_indices
            .extend(data.light_indices.iter().copied());

        self.version = self.version.wrapping_add(1).max(1);
    }

    /// Streams the packed arrays to the GPU. Empty tables are padded with a
    /// single neutral element so the storage bindings are never zero-sized.
    pub fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.grid.is_empty() {
            self.grid.push(GpuClusterGrid::default());
        }
        if self.offsets_and_counts.is_empty() {
            self.offsets_and_counts.push([0, 0]);
        }
        if self.light_indices.is_empty() {
            self.light_indices.push(0);
        }
        self.grid.write_buffer(device, queue);
        self.offsets_and_counts.write_buffer(device, queue);
        self.light_indices.write_buffer(device, queue);
    }

    /// Monotonic version bumped on every rebuild, for bind-group caching.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The three storage buffers once they have been uploaded at least once.
    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.grid.buffer()?,
            self.offsets_and_counts.buffer()?,
            self.light_indices.buffer()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_packs_grid_table_and_indices() {
        let mut world = World::new();
        let mut buffers = ClusterGpuBuffers::from_world(&mut world);
        let data = ClusterCpuData {
            grid: GpuClusterGrid::default(),
            offsets_and_counts: alloc::vec![[0, 2], [2, 1]],
            light_indices: alloc::vec![0, 1, 0],
        };
        buffers.rebuild(&data);
        assert_eq!(buffers.grid.values().len(), 1);
        assert_eq!(buffers.offsets_and_counts.values().len(), 2);
        assert_eq!(buffers.light_indices.values().len(), 3);
        assert_eq!(buffers.offsets_and_counts.values()[1], [2, 1]);
    }

    #[test]
    fn rebuild_bumps_version_monotonically() {
        let mut world = World::new();
        let mut buffers = ClusterGpuBuffers::from_world(&mut world);
        let before = buffers.version();
        buffers.rebuild(&ClusterCpuData::default());
        assert_ne!(buffers.version(), before);
        // The grid record is always present, even in an unlit scene.
        assert_eq!(buffers.grid.values().len(), 1);
    }
}
