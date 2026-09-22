use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, PipelineCache, SparseBufferUpdateBindGroups,
        SparseBufferUpdateJobs, SparseBufferUpdatePipelines,
    },
    renderer::{RenderDevice, RenderQueue},
};

use super::{bindings::GpuSceneBindGroup, storage::GpuSceneBuffers};

pub(crate) fn write_gpu_scene_buffers(
    mut buffers: ResMut<GpuSceneBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
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
        ..
    } = &mut *buffers;

    upload(
        instances,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    upload(
        current_transforms,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    upload(
        previous_transforms,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    upload(
        bounds,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
}

pub(crate) fn prepare_gpu_scene_bind_group(
    buffers: Res<GpuSceneBuffers>,
    mut bindings: ResMut<GpuSceneBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &buffers);
}

fn upload<T: AtomicPod>(
    buffer: &mut AtomicSparseBufferVec<T>,
    device: &RenderDevice,
    queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    jobs: &mut SparseBufferUpdateJobs,
    bind_groups: &mut SparseBufferUpdateBindGroups,
    pipelines: &SparseBufferUpdatePipelines,
) {
    buffer.write_buffers(device, queue);
    buffer.prepare_to_populate_buffers(device, pipeline_cache, jobs, bind_groups, pipelines);
}
