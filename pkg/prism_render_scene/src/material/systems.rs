use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
    texture_upload::MaterialTextureArrays,
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
    let texture_stats = runtime.texture_heap_stats();
    diagnostics.texture_slots_capacity = texture_stats.capacity;
    diagnostics.texture_images_resident = texture_stats.live_images;
    diagnostics.texture_overflow = texture_stats.overflow;
    diagnostics.texture_slots_live = runtime.texture_high_water();
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
    use prism_render_material::{Illumination, FALLBACK_MATERIAL_HANDLE};

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
                .illumination,
            Illumination::Lit
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

    #[test]
    fn publishing_reference_counts_and_reconciles_material_textures() {
        use bevy_asset::Handle;
        use bevy_image::Image;

        fn image_handle(tag: u128) -> Handle<Image> {
            Handle::<Image>::from(bevy_asset::uuid::Uuid::from_u128(tag))
        }

        let base = image_handle(0xB0);
        let normal = image_handle(0x0A);

        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let handle = assets.add(StandardMaterial {
            base_color_texture: Some(base.clone()),
            normal_map_texture: Some(normal.clone()),
            ..StandardMaterial::default()
        });
        let id = handle.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());

        // Initial publish acquires one slot per referenced image.
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(id, materials.get(id).unwrap())
                .unwrap();
        });
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .texture_heap_stats()
                .live_images,
            2
        );

        // Drop the normal map and republish: the stale reference is released so
        // exactly one image stays resident, and the freed slot returns to the
        // free list rather than leaking.
        world.resource_mut::<Assets<StandardMaterial>>().get_mut(id).unwrap().normal_map_texture =
            None;
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(id, materials.get(id).unwrap())
                .unwrap();
        });
        let stats = world
            .resource::<RenderMaterialRegistry>()
            .texture_heap_stats();
        assert_eq!(stats.live_images, 1, "the dropped normal map must be released");
        assert_eq!(stats.free_slots, 1, "its slot must return to the free list");

        // Retiring the material releases its remaining textures.
        let completion = prism_render_architecture::gpu_scene::GpuCompletionValue(3);
        world
            .resource_mut::<RenderMaterialRegistry>()
            .retire_asset(id, completion)
            .unwrap();
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .texture_heap_stats()
                .live_images,
            0
        );
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
    arrays: Res<MaterialTextureArrays>,
    mut bindings: ResMut<MaterialBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &buffers, &runtime, &arrays);
}

/// Clamps the bindless texture heap to the device-supported binding-array
/// capacity before any material is published, so `acquire` can never hand out a
/// slot the bound `binding_array` cannot address. Runs once at render startup
/// and is a no-op on non-bindless devices, which keep the heap's default
/// capacity for CPU-side bookkeeping only.
pub(crate) fn configure_material_texture_capacity(
    arrays: Res<MaterialTextureArrays>,
    mut runtime: ResMut<RenderMaterialRegistry>,
) {
    if arrays.bindless() {
        runtime.set_texture_capacity(arrays.capacity());
    }
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
