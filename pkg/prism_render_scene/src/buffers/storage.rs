use alloc::sync::Arc;

use bevy_ecs::{resource::Resource, world::FromWorld};
use bevy_math::Vec4;
use bevy_render::render_resource::{AtomicPod, AtomicSparseBufferVec, Buffer, BufferUsages};
use prism_render_architecture::gpu_scene::{
    bounds_row, current_transform_row, instance_row, previous_transform_row, CpuRenderScene,
    DirtySceneSlot,
};

use super::rows::{RenderGpuSceneBounds, RenderGpuSceneInstance, RenderGpuSceneTransform};

#[derive(Resource)]
pub struct GpuSceneBuffers {
    pub(crate) instances: AtomicSparseBufferVec<RenderGpuSceneInstance>,
    pub(crate) current_transforms: AtomicSparseBufferVec<RenderGpuSceneTransform>,
    pub(crate) previous_transforms: AtomicSparseBufferVec<RenderGpuSceneTransform>,
    pub(crate) bounds: AtomicSparseBufferVec<RenderGpuSceneBounds>,
}

impl FromWorld for GpuSceneBuffers {
    fn from_world(_: &mut bevy_ecs::world::World) -> Self {
        Self {
            instances: sparse_storage("prism gpu scene instances"),
            current_transforms: sparse_storage("prism gpu scene current transforms"),
            previous_transforms: sparse_storage("prism gpu scene previous transforms"),
            bounds: sparse_storage("prism gpu scene bounds"),
        }
    }
}

impl GpuSceneBuffers {
    pub fn instances(&self) -> Option<&Buffer> {
        self.instances.buffer()
    }

    pub fn current_transforms(&self) -> Option<&Buffer> {
        self.current_transforms.buffer()
    }

    pub fn previous_transforms(&self) -> Option<&Buffer> {
        self.previous_transforms.buffer()
    }

    pub fn bounds(&self) -> Option<&Buffer> {
        self.bounds.buffer()
    }

    pub(crate) fn binding_buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.instances()?,
            self.current_transforms()?,
            self.previous_transforms()?,
            self.bounds()?,
        ))
    }

    pub(crate) fn apply_dirty_slots(
        &mut self,
        mirror: &CpuRenderScene,
        dirty_slots: &[DirtySceneSlot],
    ) {
        for dirty in dirty_slots {
            self.write_slot(mirror, dirty.handle.index);
        }
    }

    pub(crate) fn rebuild_from_mirror(&mut self, mirror: &CpuRenderScene) {
        for index in 0..mirror.capacity() as u32 {
            self.write_slot(mirror, index);
        }
    }

    fn write_slot(&mut self, mirror: &CpuRenderScene, index: u32) {
        if let Some(row) = instance_row(mirror, index) {
            self.instances.grow_and_set(index, row.into());
        }
        if let Some(row) = current_transform_row(mirror, index) {
            self.current_transforms.grow_and_set(index, row.into());
        }
        if let Some(row) = previous_transform_row(mirror, index) {
            self.previous_transforms.grow_and_set(index, row.into());
        }
        if let Some(row) = bounds_row(mirror, index) {
            self.bounds.grow_and_set(index, row.into());
        }
    }
}

fn sparse_storage<T: AtomicPod>(label: &'static str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, Arc::from(label))
}

impl From<prism_render_architecture::gpu_scene::GpuSceneInstance> for RenderGpuSceneInstance {
    fn from(row: prism_render_architecture::gpu_scene::GpuSceneInstance) -> Self {
        Self {
            generation: row.generation,
            flags: row.flags,
            geometry_index: row.geometry_index,
            geometry_generation: row.geometry_generation,
            material_index: row.material_index,
            material_generation: row.material_generation,
            render_layers: row.render_layers,
            active: row.active,
        }
    }
}

impl From<prism_render_architecture::gpu_scene::GpuSceneTransform> for RenderGpuSceneTransform {
    fn from(row: prism_render_architecture::gpu_scene::GpuSceneTransform) -> Self {
        Self {
            row_0: Vec4::from_array(row.rows[0]),
            row_1: Vec4::from_array(row.rows[1]),
            row_2: Vec4::from_array(row.rows[2]),
        }
    }
}

impl From<prism_render_architecture::gpu_scene::GpuSceneBounds> for RenderGpuSceneBounds {
    fn from(row: prism_render_architecture::gpu_scene::GpuSceneBounds) -> Self {
        Self {
            center_radius: Vec4::from_array(row.center_radius),
            half_extents: Vec4::from_array(row.half_extents),
        }
    }
}
