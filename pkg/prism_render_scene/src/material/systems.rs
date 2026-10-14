use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
    texture_upload::MaterialTextureArrays,
    vt_source::{
        MaterialVtMaterialSourceQueries, MaterialVtSourceChangeQueue, MaterialVtSourceError,
        MaterialVtSourceQuery, MaterialVtSourceRegistry,
    },
};
use crate::completion::GpuCompletionTracker;
use bevy_asset::{AssetEvent, Assets};
use bevy_ecs::{message::MessageReader, prelude::*};
use bevy_image::Image;
use bevy_pbr::StandardMaterial;
use bevy_render::{
    render_resource::{
        AtomicPod, AtomicSparseBufferVec, PipelineCache, SparseBufferUpdateBindGroups,
        SparseBufferUpdateJobs, SparseBufferUpdatePipelines,
    },
    renderer::{RenderDevice, RenderQueue},
    Extract,
};

/// Captures changed main-world images before the generic `GpuImage` extractor
/// is allowed to move CPU bytes out of `Image::data`.
pub(crate) fn extract_material_vt_sources(
    mut events: Extract<MessageReader<AssetEvent<Image>>>,
    images: Extract<Res<Assets<Image>>>,
    mut sources: ResMut<MaterialVtSourceRegistry>,
    mut changed: ResMut<MaterialVtSourceChangeQueue>,
) {
    if !sources.initial_scan_complete() {
        for (id, image) in images.iter() {
            sources.upsert_asset_image(id, image);
        }
        sources.mark_initial_scan_complete();
    }
    for event in events.read() {
        let id = match *event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id }
            | AssetEvent::Unused { id }
            | AssetEvent::Removed { id } => id,
        };
        changed.record(id);
        sync_material_vt_source_event(*event, &images, &mut sources);
    }
}

fn sync_material_vt_source_event(
    event: AssetEvent<Image>,
    images: &Assets<Image>,
    sources: &mut MaterialVtSourceRegistry,
) {
    match event {
        AssetEvent::Added { id }
        | AssetEvent::Modified { id }
        | AssetEvent::LoadedWithDependencies { id } => {
            if let Some(image) = images.get(id) {
                sources.upsert_asset_image(id, image);
            } else {
                sources.upsert_unavailable(id, MaterialVtSourceError::MissingAsset);
            }
        }
        AssetEvent::Unused { id } | AssetEvent::Removed { id } => {
            sources.remove(id);
        }
    }
}

/// Returns the source state of all image inputs referenced by a material.
///
/// This is intentionally a query-only bridge. It does not allocate a bindless
/// slot, publish a page-table entry, or claim atlas residency; those actions
/// remain the responsibility of the future material-VT producer.
pub(crate) fn query_standard_material_vt_sources(
    material: &StandardMaterial,
    sources: &MaterialVtSourceRegistry,
) -> MaterialVtSourceQuery {
    sources.query_material(material)
}

pub(crate) fn extract_standard_materials(
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    sources: Res<MaterialVtSourceRegistry>,
    mut source_queries: ResMut<MaterialVtMaterialSourceQueries>,
    completion: Res<GpuCompletionTracker>,
    mut runtime: ResMut<RenderMaterialRegistry>,
    mut diagnostics: ResMut<PrismMaterialDiagnostics>,
    mut changed_images: ResMut<MaterialVtSourceChangeQueue>,
) {
    runtime.dirty_assets.clear();
    diagnostics.created = 0;
    diagnostics.updated = 0;
    diagnostics.retired = 0;
    diagnostics.errors = 0;
    let mut published = Vec::new();
    for event in events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let Some(material) = materials.get(id) else {
                    continue;
                };
                // Keep source lookup adjacent to material publication. The
                // query is observational until the real producer consumes the
                // immutable snapshot and publishes residency.
                let exists = runtime.material_handle(id).is_some();
                let source_query = query_standard_material_vt_sources(material, &sources);
                if runtime
                    .publish_asset_with_sources(
                        id,
                        material,
                        completion.next_submission_value(),
                        Some(&sources),
                    )
                    .is_ok()
                {
                    published.push(id);
                    source_queries.update(id, source_query);
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
                source_queries.remove(id);
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
    for image in changed_images.take() {
        let material_ids: Vec<_> = runtime.materials_referencing_image(image).collect();
        for id in material_ids {
            if published.contains(&id) {
                continue;
            }
            let Some(material) = materials.get(id) else {
                continue;
            };
            let source_query = query_standard_material_vt_sources(material, &sources);
            if runtime
                .publish_asset_with_sources(
                    id,
                    material,
                    completion.next_submission_value(),
                    Some(&sources),
                )
                .is_ok()
            {
                source_queries.update(id, source_query);
                diagnostics.updated += 1;
            } else {
                diagnostics.errors += 1;
            }
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
    use bevy_asset::{uuid::Uuid, AssetId, Assets, Handle, RenderAssetUsages};
    use bevy_ecs::world::World;
    use bevy_image::Image;
    use bevy_pbr::StandardMaterial;
    use bevy_render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    use prism_render_material::{BuiltinProgram, ProgramRef, ShadingPlan, FALLBACK_MATERIAL_HANDLE};

    use super::*;

    fn asset_world() -> (World, AssetId<StandardMaterial>) {
        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let handle = assets.add(StandardMaterial::default());
        let id = handle.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        (world, id)
    }

    fn complete_image(value: u8) -> Image {
        let mut image = Image::new(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![value; 16],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.mip_level_count = 3;
        image.data = Some(vec![value; 21]);
        image
    }

    fn complete_rgba_image(value: u8) -> Image {
        let mut image = Image::new(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![value; 4 * 4 * 4],
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.mip_level_count = 3;
        image.data = Some(vec![value; (4 * 4 * 4) + (2 * 2 * 4) + 4]);
        image
    }

    #[test]
    fn image_events_capture_snapshot_and_remove_it_without_claiming_residency() {
        let id: AssetId<Image> = Handle::from(Uuid::from_u128(0xA1)).id();
        let mut images = Assets::<Image>::default();
        images.insert(id, complete_image(7)).unwrap();
        let mut sources = MaterialVtSourceRegistry::default();

        sync_material_vt_source_event(AssetEvent::Added { id }, &images, &mut sources);
        let query = sources.query_images([id]);
        assert_eq!(query.referenced, 1);
        assert_eq!(query.available, 1);
        assert_eq!(query.missing, 0);
        assert_eq!(query.unavailable, 0);
        assert_eq!(sources.len(), 1);

        sync_material_vt_source_event(AssetEvent::Removed { id }, &images, &mut sources);
        assert!(matches!(
            sources.lookup(id),
            super::super::vt_source::MaterialVtSourceLookup::Missing
        ));
    }

    #[test]
    fn image_event_without_asset_records_explicit_unavailable_state() {
        let id: AssetId<Image> = Handle::from(Uuid::from_u128(0xA5)).id();
        let images = Assets::<Image>::default();
        let mut sources = MaterialVtSourceRegistry::default();

        sync_material_vt_source_event(AssetEvent::Modified { id }, &images, &mut sources);

        assert!(matches!(
            sources.lookup(id),
            super::super::vt_source::MaterialVtSourceLookup::Unavailable {
                error: MaterialVtSourceError::MissingAsset,
                ..
            }
        ));
    }

    #[test]
    fn material_texture_query_distinguishes_available_missing_and_unavailable_sources() {
        let available: Handle<Image> = Handle::from(Uuid::from_u128(0xA2));
        let unavailable: Handle<Image> = Handle::from(Uuid::from_u128(0xA3));
        let missing: Handle<Image> = Handle::from(Uuid::from_u128(0xA4));
        let material = StandardMaterial {
            base_color_texture: Some(available.clone()),
            normal_map_texture: Some(unavailable.clone()),
            occlusion_texture: Some(missing.clone()),
            ..StandardMaterial::default()
        };
        let mut sources = MaterialVtSourceRegistry::default();
        sources.upsert_asset_image(available.id(), &complete_image(1));
        sources.upsert_unavailable(unavailable.id(), MaterialVtSourceError::MissingData);

        let query = query_standard_material_vt_sources(&material, &sources);
        assert_eq!(query.referenced, 3);
        assert_eq!(query.available, 1);
        assert_eq!(query.unavailable, 1);
        assert_eq!(query.missing, 1);
        assert_eq!(
            query.availability(),
            super::super::vt_source::MaterialVtSourceAvailability::Missing
        );

        let mut cached = MaterialVtMaterialSourceQueries::default();
        cached.update(AssetId::default(), query);
        assert_eq!(cached.get(AssetId::default()), Some(query));
        cached.remove(AssetId::default());
        assert_eq!(cached.len(), 0);
    }

    #[test]
    fn material_rows_advertise_vt_only_when_the_default_producer_accepts_the_source() {
        let compatible: Handle<Image> = Handle::from(Uuid::from_u128(0xB1));
        let incompatible: Handle<Image> = Handle::from(Uuid::from_u128(0xB2));
        let mut sources = MaterialVtSourceRegistry::default();
        sources.upsert_asset_image(compatible.id(), &complete_rgba_image(1));
        // Advance the source identity independently of the bindless heap. The
        // two counters intentionally have different ownership/lifetime
        // semantics; a resolver that copies the source revision into the GPU
        // row would be stale as soon as this source is refreshed in place.
        sources.upsert_asset_image(compatible.id(), &complete_rgba_image(3));
        sources.upsert_asset_image(incompatible.id(), &complete_image(2));

        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let material = assets.add(StandardMaterial {
            base_color_texture: Some(compatible.clone()),
            ..StandardMaterial::default()
        });
        let material_id = material.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset_with_sources(
                    material_id,
                    materials.get(material_id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(1),
                    Some(&sources),
                )
                .unwrap();
        });
        let handle = world
            .resource::<RenderMaterialRegistry>()
            .material_handle(material_id)
            .unwrap();
        let record = world
            .resource::<RenderMaterialRegistry>()
            .registry
            .get(handle)
            .unwrap();
        assert_eq!(record.textures.len(), 1);
        assert_ne!(
            record.textures[0].virtual_texture_id,
            prism_render_material::NO_VIRTUAL_TEXTURE
        );
        let texture = record.textures[0];
        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(
            texture.generation,
            runtime.texture_generation_table()[texture.index as usize],
            "VT rows must use the bindless heap generation, not the source revision"
        );
        let source_revision = match sources.lookup(compatible.id()) {
            super::super::vt_source::MaterialVtSourceLookup::Available { identity, .. } => {
                identity.revision()
            }
            other => panic!("expected compatible source, got {other:?}"),
        };
        assert_ne!(
            texture.generation as u64, source_revision,
            "the regression fixture must distinguish heap generations from source revisions"
        );

        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let material = assets.add(StandardMaterial {
            base_color_texture: Some(incompatible.clone()),
            ..StandardMaterial::default()
        });
        let material_id = material.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset_with_sources(
                    material_id,
                    materials.get(material_id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(1),
                    Some(&sources),
                )
                .unwrap();
        });
        let handle = world
            .resource::<RenderMaterialRegistry>()
            .material_handle(material_id)
            .unwrap();
        let record = world
            .resource::<RenderMaterialRegistry>()
            .registry
            .get(handle)
            .unwrap();
        assert_eq!(
            record.textures[0].virtual_texture_id,
            prism_render_material::NO_VIRTUAL_TEXTURE
        );
    }

    #[test]
    fn asset_lifecycle_updates_retires_and_reuses_after_completion() {
        let (mut world, id) = asset_world();
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(
                    id,
                    materials.get(id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(0),
                )
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
                .response_plan,
            ShadingPlan::Single { program: ProgramRef::Builtin(BuiltinProgram::Lit) }
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
                .publish_asset(
                    id,
                    materials.get(id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(0),
                )
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
            Handle::<Image>::from(Uuid::from_u128(tag))
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
                .publish_asset(
                    id,
                    materials.get(id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(0),
                )
                .unwrap();
        });
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .texture_heap_stats()
                .live_images,
            2
        );

        // Drop the normal map and republish: the stale reference is retired
        // until the GPU completion proof, so exactly one image is live while
        // the old slot remains quarantined.
        world
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(id)
            .unwrap()
            .normal_map_texture = None;
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(
                    id,
                    materials.get(id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(0),
                )
                .unwrap();
        });
        let stats = world
            .resource::<RenderMaterialRegistry>()
            .texture_heap_stats();
        assert_eq!(
            stats.live_images, 2,
            "the dropped normal map remains resident until GPU completion"
        );
        assert_eq!(
            stats.pending_releases, 2,
            "both the old shared base reference and normal reference are quarantined"
        );
        assert_eq!(
            stats.free_slots, 0,
            "a GPU-visible slot must not be reused early"
        );

        world
            .resource_mut::<RenderMaterialRegistry>()
            .reclaim_completed_textures(prism_render_architecture::gpu_scene::GpuCompletionValue(
                0,
            ));
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .texture_heap_stats()
                .free_slots,
            1,
            "the retired slot becomes reusable only after completion"
        );

        // Retiring the material releases its remaining textures.
        let completion = prism_render_architecture::gpu_scene::GpuCompletionValue(3);
        world
            .resource_mut::<RenderMaterialRegistry>()
            .retire_asset(id, completion)
            .unwrap();
        world
            .resource_mut::<RenderMaterialRegistry>()
            .reclaim_completed_textures(completion);
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
    diagnostics.reclaimed = diagnostics
        .reclaimed
        .saturating_add(runtime.reclaim_completed_textures(tracker.completed_value()));
}

pub(crate) fn prepare_material_bind_group(
    buffers: Res<MaterialGpuBuffers>,
    runtime: Res<RenderMaterialRegistry>,
    arrays: Res<MaterialTextureArrays>,
    virtual_texture: Res<super::virtual_texture::MaterialVirtualTextureGpuBundle>,
    mut bindings: ResMut<MaterialBindGroup>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    bindings.prepare(
        &device,
        &buffers,
        &runtime,
        &arrays,
        &virtual_texture,
        &queue,
    );
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
