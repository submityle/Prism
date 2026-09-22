//! Unified, stable GPU scene storage.
//!
//! The main and render worlds remain ECS-driven. This module owns the retained
//! CPU mirror and sparse GPU tables used by all rendering consumers.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_math::Vec4;
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::{
    bounds_row, current_transform_row, instance_row, previous_transform_row, CpuRenderScene,
    GpuCompletionValue, GpuSceneSnapshot, SceneApplyReport, SceneCapacityError, SceneHandle,
    SceneHandleAllocator, SceneHandleError, SceneTransaction,
};

use crate::{
    impl_atomic_pod,
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, Buffer, BufferUsages, PipelineCache,
        SparseBufferUpdateBindGroups, SparseBufferUpdateJobs, SparseBufferUpdatePipelines,
    },
    renderer::{RenderGraph, RenderGraphSystems, RenderQueue},
    GpuResourceAppExt, Render, RenderApp, RenderSystems,
};

const DEFAULT_MAX_SCENE_SLOTS: u32 = 1 << 24;

/// Installs the retained GPU scene and its upload lifecycle in the render app.
pub struct GpuScenePlugin;

impl Plugin for GpuScenePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gpu_scene.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<RenderGpuScene>()
            .init_gpu_resource::<GpuSceneBuffers>()
            .init_resource::<GpuCompletionTracker>()
            .add_systems(
                Render,
                write_gpu_scene_buffers.in_set(RenderSystems::PrepareResourcesFlush),
            )
            .add_systems(
                RenderGraph,
                reclaim_completed_scene_handles.in_set(RenderGraphSystems::Finish),
            );
    }
}

/// Render-world owner of stable handles, the authoritative CPU mirror, and
/// the last snapshot published to GPU consumers.
#[derive(Resource)]
pub struct RenderGpuScene {
    allocator: SceneHandleAllocator,
    mirror: CpuRenderScene,
    snapshot: GpuSceneSnapshot,
    buffer_version: u32,
}

impl Default for RenderGpuScene {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SCENE_SLOTS)
    }
}

impl RenderGpuScene {
    pub fn new(max_slots: u32) -> Self {
        Self {
            allocator: SceneHandleAllocator::new(max_slots),
            mirror: CpuRenderScene::default(),
            snapshot: GpuSceneSnapshot::default(),
            buffer_version: 1,
        }
    }

    pub fn allocate(&mut self) -> Result<SceneHandle, SceneCapacityError> {
        self.allocator.allocate()
    }

    pub fn apply_transaction(
        &mut self,
        buffers: &mut GpuSceneBuffers,
        transaction: &SceneTransaction,
    ) -> SceneApplyReport {
        let report = self.mirror.apply(transaction);
        if report.errors.is_empty() {
            buffers.apply_dirty_slots(&self.mirror, &report.dirty_slots);
            self.snapshot = GpuSceneSnapshot {
                frame_epoch: transaction.frame_epoch,
                scene_epoch: self.mirror.scene_epoch(),
                instance_count: self.mirror.live_count(),
                buffer_version: self.buffer_version,
            };
        }
        report
    }

    /// Retires a destroyed handle only after the next frame submission has
    /// completed on the GPU.
    pub fn retire(
        &mut self,
        handle: SceneHandle,
        completion: &GpuCompletionTracker,
    ) -> Result<(), SceneHandleError> {
        self.allocator
            .retire(handle, completion.next_submission_value())
    }

    pub fn snapshot(&self) -> GpuSceneSnapshot {
        self.snapshot
    }

    pub fn mirror(&self) -> &CpuRenderScene {
        &self.mirror
    }

    fn reclaim_completed(&mut self, completed: GpuCompletionValue) -> u32 {
        self.allocator.reclaim_completed(completed)
    }
}

/// GPU representation of one scene instance. This stays compact and contains
/// only hot identity and classification data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneInstance {
    pub generation: u32,
    pub flags: u32,
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub material_index: u32,
    pub material_generation: u32,
    pub render_layers: u32,
    pub active: u32,
}

impl_atomic_pod!(RenderGpuSceneInstance, RenderGpuSceneInstanceBlob);

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneTransform {
    pub row_0: Vec4,
    pub row_1: Vec4,
    pub row_2: Vec4,
}

impl_atomic_pod!(RenderGpuSceneTransform, RenderGpuSceneTransformBlob);

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneBounds {
    pub center_radius: Vec4,
    pub half_extents: Vec4,
}

impl_atomic_pod!(RenderGpuSceneBounds, RenderGpuSceneBoundsBlob);

/// Sparse structure-of-arrays GPU storage. Slot indices are identical across
/// all tables and match [`SceneHandle::index`].
#[derive(Resource)]
pub struct GpuSceneBuffers {
    instances: AtomicSparseBufferVec<RenderGpuSceneInstance>,
    current_transforms: AtomicSparseBufferVec<RenderGpuSceneTransform>,
    previous_transforms: AtomicSparseBufferVec<RenderGpuSceneTransform>,
    bounds: AtomicSparseBufferVec<RenderGpuSceneBounds>,
}

impl FromWorld for GpuSceneBuffers {
    fn from_world(_: &mut World) -> Self {
        Self {
            instances: sparse_storage("gpu scene instances"),
            current_transforms: sparse_storage("gpu scene current transforms"),
            previous_transforms: sparse_storage("gpu scene previous transforms"),
            bounds: sparse_storage("gpu scene bounds"),
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

    fn apply_dirty_slots(
        &mut self,
        mirror: &CpuRenderScene,
        dirty_slots: &[prism_render_architecture::gpu_scene::DirtySceneSlot],
    ) {
        for dirty in dirty_slots {
            let index = dirty.handle.index;
            if let Some(row) = instance_row(mirror, index) {
                self.instances.grow_and_set(
                    index,
                    RenderGpuSceneInstance {
                        generation: row.generation,
                        flags: row.flags,
                        geometry_index: row.geometry_index,
                        geometry_generation: row.geometry_generation,
                        material_index: row.material_index,
                        material_generation: row.material_generation,
                        render_layers: row.render_layers,
                        active: row.active,
                    },
                );
            }
            if let Some(row) = current_transform_row(mirror, index) {
                self.current_transforms.grow_and_set(
                    index,
                    RenderGpuSceneTransform {
                        row_0: Vec4::from_array(row.rows[0]),
                        row_1: Vec4::from_array(row.rows[1]),
                        row_2: Vec4::from_array(row.rows[2]),
                    },
                );
            }
            if let Some(row) = previous_transform_row(mirror, index) {
                self.previous_transforms.grow_and_set(
                    index,
                    RenderGpuSceneTransform {
                        row_0: Vec4::from_array(row.rows[0]),
                        row_1: Vec4::from_array(row.rows[1]),
                        row_2: Vec4::from_array(row.rows[2]),
                    },
                );
            }
            if let Some(row) = bounds_row(mirror, index) {
                self.bounds.grow_and_set(
                    index,
                    RenderGpuSceneBounds {
                        center_radius: Vec4::from_array(row.center_radius),
                        half_extents: Vec4::from_array(row.half_extents),
                    },
                );
            }
        }
    }
}

fn sparse_storage<T: AtomicPod>(label: &'static str) -> AtomicSparseBufferVec<T> {
    AtomicSparseBufferVec::new(BufferUsages::STORAGE, Arc::from(label))
}

fn write_gpu_scene_buffers(
    mut buffers: ResMut<GpuSceneBuffers>,
    render_device: Res<crate::renderer::RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut jobs: ResMut<SparseBufferUpdateJobs>,
    mut bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    pipelines: Res<SparseBufferUpdatePipelines>,
) {
    let GpuSceneBuffers {
        instances,
        current_transforms,
        previous_transforms,
        bounds,
    } = &mut *buffers;

    write_and_prepare(
        instances,
        &render_device,
        &render_queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    write_and_prepare(
        current_transforms,
        &render_device,
        &render_queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    write_and_prepare(
        previous_transforms,
        &render_device,
        &render_queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    write_and_prepare(
        bounds,
        &render_device,
        &render_queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
}

fn write_and_prepare<T: AtomicPod>(
    buffer: &mut AtomicSparseBufferVec<T>,
    render_device: &crate::renderer::RenderDevice,
    render_queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    jobs: &mut SparseBufferUpdateJobs,
    bind_groups: &mut SparseBufferUpdateBindGroups,
    pipelines: &SparseBufferUpdatePipelines,
) {
    buffer.write_buffers(render_device, render_queue);
    buffer.prepare_to_populate_buffers(render_device, pipeline_cache, jobs, bind_groups, pipelines);
}

/// Tracks queue completion without assuming a fixed number of frames in
/// flight. A callback for submission N runs only after all work submitted up to
/// N has completed.
#[derive(Resource, Clone)]
pub struct GpuCompletionTracker {
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
}

impl Default for GpuCompletionTracker {
    fn default() -> Self {
        Self {
            submitted: Arc::new(AtomicU64::new(0)),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl GpuCompletionTracker {
    pub fn next_submission_value(&self) -> GpuCompletionValue {
        GpuCompletionValue(self.submitted.load(Ordering::Acquire) + 1)
    }

    pub fn completed_value(&self) -> GpuCompletionValue {
        GpuCompletionValue(self.completed.load(Ordering::Acquire))
    }

    pub fn track_submission(&self, queue: &RenderQueue) -> GpuCompletionValue {
        let value = self.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let completed = Arc::clone(&self.completed);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(value, Ordering::Release);
        });
        GpuCompletionValue(value)
    }
}

fn reclaim_completed_scene_handles(
    mut scene: ResMut<RenderGpuScene>,
    tracker: Res<GpuCompletionTracker>,
) {
    scene.reclaim_completed(tracker.completed_value());
}

/// Marks the final queue submission for this frame so retired scene handles
/// can be reclaimed from an actual completion callback.
pub(crate) fn track_frame_submission(world: &World) {
    let tracker = world.resource::<GpuCompletionTracker>();
    let queue = world.resource::<RenderQueue>();
    tracker.track_submission(queue);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_scene_layouts_match_architecture_contract() {
        assert_eq!(size_of::<RenderGpuSceneInstance>(), 32);
        assert_eq!(size_of::<RenderGpuSceneTransform>(), 48);
        assert_eq!(size_of::<RenderGpuSceneBounds>(), 32);
    }
}
