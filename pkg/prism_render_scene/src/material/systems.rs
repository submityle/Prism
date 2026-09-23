use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
};
use crate::completion::GpuCompletionTracker;
use bevy_asset::AssetEvent;
use bevy_ecs::{message::MessageReader, prelude::*};
use bevy_pbr::StandardMaterial;
use bevy_render::{
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, PipelineCache, SparseBufferUpdateBindGroups,
        SparseBufferUpdateJobs, SparseBufferUpdatePipelines,
    },
    renderer::{RenderDevice, RenderQueue},
    Extract,
};

pub(crate) fn extract_standard_materials(
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
    materials: Extract<Res<bevy_asset::Assets<StandardMaterial>>>,
    completion: Res<GpuCompletionTracker>,
    mut runtime: ResMut<RenderMaterialRegistry>,
    mut diagnostics: ResMut<PrismMaterialDiagnostics>,
) {
    runtime.dirty_assets.clear();
    diagnostics.created = 0;
    diagnostics.updated = 0;
    diagnostics.retired = 0;
    diagnostics.errors = 0;
    for event in events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let Some(material) = materials.get(id) else {
                    continue;
                };
                let exists = runtime.material_handle(id).is_some();
                if runtime.publish_asset(id, material).is_ok() {
                    if exists {
                        diagnostics.updated += 1;
                    } else {
                        diagnostics.created += 1;
                    }
                } else {
                    diagnostics.errors += 1;
                }
            }
            AssetEvent::Unused { id } | AssetEvent::Removed { id } => {
                match runtime.retire_asset(id, completion.next_submission_value()) {
                    Ok(Some(_)) => {
                        diagnostics.retired += 1;
                    }
                    Ok(None) => {}
                    Err(_) => diagnostics.errors += 1,
                }
            }
            AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
    let snapshot = runtime.registry.snapshot();
    diagnostics.active = snapshot.active_materials;
    diagnostics.epoch = snapshot.epoch;
    diagnostics.buffer_version = snapshot.buffer_version;
}

pub(crate) fn invalidate_scene_materials(
    runtime: Res<RenderMaterialRegistry>,
    mut instances: Query<&mut crate::extract::ExtractedSceneInstance>,
) {
    if runtime.dirty_assets.is_empty() {
        return;
    }
    for mut instance in &mut instances {
        if instance.material == prism_render_material::FALLBACK_MATERIAL_HANDLE
            && instance
                .material_asset
                .is_some_and(|asset| runtime.dirty_assets.contains(&asset))
        {
            instance.set_changed();
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_asset::Assets;
    use bevy_ecs::world::World;
    use bevy_pbr::StandardMaterial;
    use prism_render_material::{MaterialShadingModel, FALLBACK_MATERIAL_HANDLE};

    use super::*;

    fn asset_world() -> (World, bevy_asset::AssetId<StandardMaterial>) {
        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let handle = assets.add(StandardMaterial::default());
        let id = handle.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        (world, id)
    }

    #[test]
    fn asset_lifecycle_updates_retires_and_reuses_after_completion() {
        let (mut world, id) = asset_world();
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(id, materials.get(id).unwrap())
                .unwrap();
        });
        let first = world
            .resource::<RenderMaterialRegistry>()
            .material_handle(id)
            .unwrap();
        assert_ne!(first, FALLBACK_MATERIAL_HANDLE);
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .registry
                .get(first)
                .unwrap()
                .shading_model,
            MaterialShadingModel::Principled
        );

        let completion = prism_render_architecture::gpu_scene::GpuCompletionValue(7);
        world
            .resource_mut::<RenderMaterialRegistry>()
            .retire_asset(id, completion)
            .unwrap();
        assert!(world
            .resource::<RenderMaterialRegistry>()
            .material_handle(id)
            .is_none());
        assert_eq!(
            world
                .resource_mut::<RenderMaterialRegistry>()
                .registry
                .reclaim_completed(completion),
            1
        );
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(id, materials.get(id).unwrap())
                .unwrap();
        });
        let second = world
            .resource::<RenderMaterialRegistry>()
            .material_handle(id)
            .unwrap();
        assert_eq!(second.index, first.index);
        assert_ne!(second.generation, first.generation);
    }
}

pub(crate) fn stage_material_uploads(
    mut runtime: ResMut<RenderMaterialRegistry>,
    mut buffers: ResMut<MaterialGpuBuffers>,
    mut diagnostics: ResMut<PrismMaterialDiagnostics>,
) {
    let (rows, bytes) = buffers.apply_dirty(&mut runtime);
    diagnostics.uploaded_rows = rows;
    diagnostics.uploaded_bytes = bytes;
}

pub(crate) fn write_material_buffers(
    mut buffers: ResMut<MaterialGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut jobs: ResMut<SparseBufferUpdateJobs>,
    mut bind_groups: ResMut<SparseBufferUpdateBindGroups>,
    pipelines: Res<SparseBufferUpdatePipelines>,
) {
    upload(
        &mut buffers.headers,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    upload(
        &mut buffers.parameters,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
    upload(
        &mut buffers.textures,
        &device,
        &queue,
        &pipeline_cache,
        &mut jobs,
        &mut bind_groups,
        &pipelines,
    );
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

pub(crate) fn reclaim_completed_materials(
    mut runtime: ResMut<RenderMaterialRegistry>,
    tracker: Res<GpuCompletionTracker>,
    mut diagnostics: ResMut<PrismMaterialDiagnostics>,
) {
    diagnostics.reclaimed = diagnostics.reclaimed.saturating_add(
        runtime
            .registry
            .reclaim_completed(tracker.completed_value()),
    );
}

pub(crate) fn prepare_material_bind_group(
    buffers: Res<MaterialGpuBuffers>,
    runtime: Res<RenderMaterialRegistry>,
    mut bindings: ResMut<MaterialBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &buffers, &runtime);
}

pub(crate) fn rebuild_material_buffers(
    mut runtime: ResMut<RenderMaterialRegistry>,
    mut buffers: ResMut<MaterialGpuBuffers>,
    mut diagnostics: ResMut<PrismMaterialDiagnostics>,
) {
    runtime.registry.mark_all_dirty();
    let (rows, bytes) = buffers.apply_dirty(&mut runtime);
    diagnostics.uploaded_rows = rows;
    diagnostics.uploaded_bytes = bytes;
    diagnostics.buffer_rebuilds = diagnostics.buffer_rebuilds.saturating_add(1);
}
