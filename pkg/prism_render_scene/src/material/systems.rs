use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
    texture_upload::MaterialTextureArrays,
    vt_source::{
        material_texture_ids, MaterialVtActivePublication, MaterialVtMaterialSourceQueries,
        MaterialVtSourceChangeQueue, MaterialVtSourceError, MaterialVtSourceQuery,
        MaterialVtSourceRegistry,
    },
};
use crate::completion::GpuCompletionTracker;
use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_ecs::{message::MessageReader, prelude::*};
use bevy_image::Image;
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::HashSet;
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
/// remain the responsibility of the material-VT producer.
pub(crate) fn query_standard_material_vt_sources(
    material: &StandardMaterial,
    sources: &MaterialVtSourceRegistry,
) -> MaterialVtSourceQuery {
    sources.query_material(material)
}

#[allow(clippy::too_many_arguments)]
fn publish_standard_material(
    id: AssetId<StandardMaterial>,
    material: &StandardMaterial,
    sources: &MaterialVtSourceRegistry,
    active_vt: &MaterialVtActivePublication,
    source_queries: &mut MaterialVtMaterialSourceQueries,
    completion: &GpuCompletionTracker,
    runtime: &mut RenderMaterialRegistry,
    diagnostics: &mut PrismMaterialDiagnostics,
) -> bool {
    // Keep source lookup adjacent to material publication. The query is
    // observational until the producer consumes the immutable snapshot and
    // publishes residency.
    let exists = runtime.material_handle(id).is_some();
    let source_query = query_standard_material_vt_sources(material, sources);
    if runtime
        .publish_asset_with_sources(
            id,
            material,
            completion.next_submission_value(),
            Some(sources),
            Some(active_vt),
        )
        .is_err()
    {
        diagnostics.errors += 1;
        return false;
    }

    source_queries.update(id, source_query, material_texture_ids(material));
    if exists {
        diagnostics.updated += 1;
    } else {
        diagnostics.created += 1;
    }
    true
}

fn process_pending_material_retirements(
    materials: &Assets<StandardMaterial>,
    completion: &GpuCompletionTracker,
    runtime: &mut RenderMaterialRegistry,
    source_queries: &mut MaterialVtMaterialSourceQueries,
    diagnostics: &mut PrismMaterialDiagnostics,
    attempted: &mut HashSet<(AssetId<StandardMaterial>, u64)>,
) {
    let pending_count = runtime.pending_retire_count();
    for _ in 0..pending_count {
        let Some((id, token)) = runtime.pop_pending_retire() else {
            break;
        };
        if !runtime.pending_retire_token_is_current(id, token) {
            continue;
        }
        // The event stream is only a change hint.  A material that was
        // removed and re-added before this extraction pass must remain live;
        // invalidate this retirement and let the publish queue reconcile the
        // final Assets state instead.
        if materials.get(id).is_some() {
            runtime.queue_publish(id);
            continue;
        }
        if !attempted.insert((id, token)) {
            runtime.retry_pending_retire(id, token);
            continue;
        }
        match runtime.retire_asset(id, completion.next_submission_value()) {
            Ok(Some(_)) => diagnostics.retired += 1,
            Ok(None) => {}
            Err(_) => {
                diagnostics.errors += 1;
                runtime.retry_pending_retire(id, token);
                continue;
            }
        }
        source_queries.remove(id);
        runtime.finish_pending_retire(id, token);
    }
}

pub(crate) fn extract_standard_materials(
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    sources: Res<MaterialVtSourceRegistry>,
    active_vt: Res<MaterialVtActivePublication>,
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
    let mut changed_materials: HashSet<AssetId<StandardMaterial>> = HashSet::default();
    for event in events.read() {
        let id = match *event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id }
            | AssetEvent::Unused { id }
            | AssetEvent::Removed { id } => id,
        };
        changed_materials.insert(id);
    }

    if !runtime.initial_asset_scan_started() {
        // Queue a deterministic first-frame reconciliation. Failed rows stay in
        // the persistent queue; successful rows are removed and are not republished.
        // The bootstrap set is then maintained from lifecycle events, avoiding a
        // full asset enumeration on every retry frame.
        let mut asset_ids: Vec<_> = materials.iter().map(|(id, _)| id).collect();
        asset_ids.sort_by_key(|id| id.to_string());
        runtime.begin_initial_asset_scan(asset_ids);
    }

    // Asset events are not the source of truth: several lifecycle events for
    // one ID can be delivered in one extraction pass, and an event can race
    // the final mutation of Assets. Reconcile each affected ID exactly once
    // against the final asset presence before touching either queue.
    let mut changed_materials: Vec<_> = changed_materials.into_iter().collect();
    changed_materials.sort_by_key(|id| id.to_string());
    for id in changed_materials {
        if materials.get(id).is_some() {
            runtime.queue_publish(id);
        } else {
            runtime.queue_retire(id);
        }
    }

    let mut affected_materials: HashSet<AssetId<StandardMaterial>> = HashSet::default();
    for image in changed_images.take() {
        affected_materials.extend(runtime.materials_referencing_image(image));
    }
    let mut affected_materials: Vec<_> = affected_materials.into_iter().collect();
    affected_materials.sort_by_key(|id| id.to_string());
    for id in affected_materials {
        if !runtime.is_pending_retire(id) {
            runtime.queue_publish(id);
        }
    }

    let mut attempted_retirements = HashSet::default();
    process_pending_material_retirements(
        &materials,
        &completion,
        &mut runtime,
        &mut source_queries,
        &mut diagnostics,
        &mut attempted_retirements,
    );

    let pending_count = runtime.pending_publish_count();
    for _ in 0..pending_count {
        let Some((id, token)) = runtime.pop_pending_publish() else {
            break;
        };
        if !runtime.pending_publish_token_is_current(id, token) {
            continue;
        }
        let Some(material) = materials.get(id) else {
            runtime.queue_retire(id);
            continue;
        };
        if publish_standard_material(
            id,
            material,
            &sources,
            &active_vt,
            &mut source_queries,
            &completion,
            &mut runtime,
            &mut diagnostics,
        ) {
            runtime.finish_pending_publish(id, token);
        } else {
            runtime.retry_pending_publish(id, token);
        }
    }

    runtime.finish_initial_asset_scan_if_ready();

    // Reconcile retirements once more after publication. A pending publish can
    // discover that its final asset state is absent and supersede itself with a
    // retirement in the same extraction pass.
    process_pending_material_retirements(
        &materials,
        &completion,
        &mut runtime,
        &mut source_queries,
        &mut diagnostics,
        &mut attempted_retirements,
    );
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
        if instance
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
    use bevy_ecs::{message::Messages, schedule::Schedule, world::World};
    use bevy_image::Image;
    use bevy_pbr::StandardMaterial;
    use bevy_render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        MainWorld,
    };
    use prism_render_material::{
        BuiltinProgram, GpuMaterialTexture, MaterialRegistry, ProgramRef, ShadingPlan,
        FALLBACK_MATERIAL_HANDLE, NO_VIRTUAL_TEXTURE,
    };

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

    fn material_extraction_world(
        materials: Assets<StandardMaterial>,
        events: Messages<AssetEvent<StandardMaterial>>,
    ) -> World {
        let mut main_world = MainWorld::default();
        main_world.insert_resource(materials);
        main_world.insert_resource(events);

        let mut world = World::new();
        world.insert_resource(main_world);
        world.insert_resource(MaterialVtSourceRegistry::default());
        world.insert_resource(MaterialVtActivePublication::default());
        world.insert_resource(MaterialVtMaterialSourceQueries::default());
        world.insert_resource(GpuCompletionTracker::default());
        world.insert_resource(RenderMaterialRegistry::default());
        world.insert_resource(PrismMaterialDiagnostics::default());
        world.insert_resource(MaterialVtSourceChangeQueue::default());
        world
    }

    #[test]
    fn extraction_bootstraps_existing_materials_without_asset_events_exactly_once() {
        let mut materials = Assets::<StandardMaterial>::default();
        let material = materials.add(StandardMaterial::default());
        let material_id = material.id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert!(runtime.initial_asset_scan_complete());
        assert!(runtime.material_handle(material_id).is_some());
        assert_eq!(runtime.revisions.get(&material_id), Some(&1));
        let diagnostics = world.resource::<PrismMaterialDiagnostics>();
        assert_eq!(diagnostics.created, 1);
        assert_eq!(diagnostics.updated, 0);
        assert_eq!(diagnostics.errors, 0);

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(
            runtime.revisions.get(&material_id),
            Some(&1),
            "the steady-state extractor must not enumerate and republish every asset"
        );
        let diagnostics = world.resource::<PrismMaterialDiagnostics>();
        assert_eq!(diagnostics.created, 0);
        assert_eq!(diagnostics.updated, 0);
        assert_eq!(diagnostics.errors, 0);
    }

    #[test]
    fn loaded_with_dependencies_publishes_material_after_initial_scan() {
        let mut world = material_extraction_world(
            Assets::<StandardMaterial>::default(),
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);
        schedule.run(&mut world);
        assert!(world
            .resource::<RenderMaterialRegistry>()
            .initial_asset_scan_complete());

        let material_id = {
            let mut main_world = world.resource_mut::<MainWorld>();
            let material_id = main_world
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial::default())
                .id();
            main_world
                .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
                .write(AssetEvent::LoadedWithDependencies { id: material_id });
            material_id
        };

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert!(runtime.material_handle(material_id).is_some());
        assert_eq!(runtime.revisions.get(&material_id), Some(&1));
        let diagnostics = world.resource::<PrismMaterialDiagnostics>();
        assert_eq!(diagnostics.created, 1);
        assert_eq!(diagnostics.updated, 0);
        assert_eq!(diagnostics.errors, 0);
    }

    #[test]
    fn failed_bootstrap_retries_only_unpublished_materials_and_remains_incomplete() {
        let mut materials = Assets::<StandardMaterial>::default();
        let first = materials.add(StandardMaterial::default()).id();
        let second = materials.add(StandardMaterial::default()).id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        // Force one deterministic capacity failure while allowing the other
        // material to publish. The stable asset-ID scan selects the first ID.
        world.resource_mut::<RenderMaterialRegistry>().registry = MaterialRegistry::new(1);
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);

        schedule.run(&mut world);

        let (published_id, published_revision, epoch) = {
            let runtime = world.resource::<RenderMaterialRegistry>();
            let published = runtime
                .revisions
                .iter()
                .next()
                .map(|(id, revision)| (*id, *revision))
                .expect("one material fits in the registry");
            assert!(published.0 == first || published.0 == second);
            assert_eq!(runtime.revisions.len(), 1);
            assert!(!runtime.initial_asset_scan_complete());
            (published.0, published.1, runtime.registry.snapshot().epoch)
        };
        assert_eq!(published_revision, 1);
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 1);

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(runtime.revisions.get(&published_id), Some(&1));
        assert_eq!(runtime.registry.snapshot().epoch, epoch);
        assert!(
            !runtime.initial_asset_scan_complete(),
            "an asset that still cannot obtain a registry slot must keep bootstrap incomplete"
        );
        assert_eq!(runtime.revisions.len(), 1);
        assert_eq!(runtime.pending_publish_count(), 1);
        assert_eq!(
            world.resource::<PrismMaterialDiagnostics>().errors,
            1,
            "the still-unpublished material must be retried"
        );
    }

    #[test]
    fn extraction_coalesces_same_frame_lifecycle_events_to_the_final_action() {
        let mut materials = Assets::<StandardMaterial>::default();
        let handle = materials.add(StandardMaterial::default());
        let material_id = handle.id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        {
            let mut main_world = world.resource_mut::<MainWorld>();
            let mut events = main_world.resource_mut::<Messages<AssetEvent<StandardMaterial>>>();
            events.write(AssetEvent::Added { id: material_id });
            events.write(AssetEvent::Removed { id: material_id });
            events.write(AssetEvent::Added { id: material_id });
        }
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(runtime.revisions.get(&material_id), Some(&1));
        assert_eq!(runtime.registry.snapshot().active_materials, 1);
        let diagnostics = world.resource::<PrismMaterialDiagnostics>();
        assert_eq!(diagnostics.created, 1);
        assert_eq!(diagnostics.retired, 0);
        assert_eq!(diagnostics.errors, 0);
        drop(handle);
    }

    #[test]
    fn extraction_reconciles_final_absence_after_add_remove_add_events() {
        let mut materials = Assets::<StandardMaterial>::default();
        let handle = materials.add(StandardMaterial::default());
        let material_id = handle.id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);
        schedule.run(&mut world);

        {
            let mut main_world = world.resource_mut::<MainWorld>();
            main_world
                .resource_mut::<Assets<StandardMaterial>>()
                .remove(material_id);
            let mut events = main_world.resource_mut::<Messages<AssetEvent<StandardMaterial>>>();
            events.write(AssetEvent::Added { id: material_id });
            events.write(AssetEvent::Removed { id: material_id });
            events.write(AssetEvent::Added { id: material_id });
        }

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert!(runtime.material_handle(material_id).is_none());
        assert_eq!(runtime.registry.snapshot().active_materials, 0);
        assert!(!runtime.is_pending_retire(material_id));
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().retired, 1);
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 0);
        drop(handle);
    }

    #[test]
    fn removal_of_a_material_absent_from_assets_leaves_no_registry_row() {
        let mut materials = Assets::<StandardMaterial>::default();
        let handle = materials.add(StandardMaterial::default());
        let material_id = handle.id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);
        schedule.run(&mut world);
        assert!(world
            .resource::<RenderMaterialRegistry>()
            .material_handle(material_id)
            .is_some());

        {
            let mut main_world = world.resource_mut::<MainWorld>();
            let mut materials = main_world.resource_mut::<Assets<StandardMaterial>>();
            materials.remove(material_id);
            drop(materials);
            main_world
                .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
                .write(AssetEvent::Removed { id: material_id });
        }
        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert!(runtime.material_handle(material_id).is_none());
        assert!(!runtime.has_pending_publish(material_id));
        assert!(!runtime.is_pending_retire(material_id));
        assert_eq!(runtime.registry.snapshot().active_materials, 0);
        drop(handle);
    }

    #[test]
    fn failed_event_publication_is_retried_after_the_event_is_consumed() {
        let mut world = material_extraction_world(
            Assets::<StandardMaterial>::default(),
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);
        schedule.run(&mut world);

        {
            let mut runtime = world.resource_mut::<RenderMaterialRegistry>();
            runtime.registry = MaterialRegistry::new(0);
            // Zero live capacity deterministically exercises retry-on-capacity
            // without relying on a synthetic allocation to exhaust the pool.
        }
        let (material_id, _material_handle) = {
            let mut main_world = world.resource_mut::<MainWorld>();
            let handle = main_world
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial::default());
            let id = handle.id();
            main_world
                .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
                .write(AssetEvent::Added { id });
            (id, handle)
        };

        schedule.run(&mut world);
        assert!(world
            .resource::<RenderMaterialRegistry>()
            .has_pending_publish(material_id));
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 1);

        schedule.run(&mut world);
        assert!(world
            .resource::<RenderMaterialRegistry>()
            .has_pending_publish(material_id));
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 1);
    }

    #[test]
    fn one_material_capacity_publishes_exactly_one_asset() {
        let mut materials = Assets::<StandardMaterial>::default();
        let first = materials.add(StandardMaterial::default()).id();
        let second = materials.add(StandardMaterial::default()).id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        world.resource_mut::<RenderMaterialRegistry>().registry = MaterialRegistry::new(1);
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);

        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(runtime.registry.snapshot().active_materials, 1);
        assert_ne!(
            runtime.material_handle(first).is_some(),
            runtime.material_handle(second).is_some()
        );
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 1);
    }

    #[test]
    fn failed_retirement_keeps_source_query_for_retry() {
        let image: Handle<Image> = Handle::from(Uuid::from_u128(0xC1));
        let mut materials = Assets::<StandardMaterial>::default();
        let handle = materials.add(StandardMaterial {
            base_color_texture: Some(image),
            ..StandardMaterial::default()
        });
        let material_id = handle.id();
        let mut world = material_extraction_world(
            materials,
            Messages::<AssetEvent<StandardMaterial>>::default(),
        );
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_standard_materials);
        schedule.run(&mut world);
        assert!(world
            .resource::<MaterialVtMaterialSourceQueries>()
            .get(material_id)
            .is_some());

        // Leave the render registry's asset map intact but remove its row, so
        // the lifecycle retirement fails with a stale handle. The source query
        // must remain available while the retry is pending.
        let published = world
            .resource::<RenderMaterialRegistry>()
            .material_handle(material_id)
            .unwrap();
        world
            .resource_mut::<RenderMaterialRegistry>()
            .registry
            .retire(
                published,
                prism_render_architecture::gpu_scene::GpuCompletionValue(7),
            )
            .unwrap();

        {
            let mut main_world = world.resource_mut::<MainWorld>();
            main_world
                .resource_mut::<Assets<StandardMaterial>>()
                .remove(material_id);
            main_world
                .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
                .write(AssetEvent::Removed { id: material_id });
        }
        schedule.run(&mut world);

        let runtime = world.resource::<RenderMaterialRegistry>();
        assert!(runtime.is_pending_retire(material_id));
        assert!(world
            .resource::<MaterialVtMaterialSourceQueries>()
            .get(material_id)
            .is_some());
        assert_eq!(world.resource::<PrismMaterialDiagnostics>().errors, 1);
        drop(handle);
    }

    fn publish_texture_row(
        world: &mut World,
        material_id: AssetId<StandardMaterial>,
        sources: &MaterialVtSourceRegistry,
        active_vt: &MaterialVtActivePublication,
    ) -> GpuMaterialTexture {
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset_with_sources(
                    material_id,
                    materials.get(material_id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(1),
                    Some(sources),
                    Some(active_vt),
                )
                .unwrap();
        });
        let runtime = world.resource::<RenderMaterialRegistry>();
        let handle = runtime.material_handle(material_id).unwrap();
        let record = runtime.registry.get(handle).unwrap();
        assert_eq!(record.textures.len(), 1);
        record.textures[0]
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
        cached.update(AssetId::default(), query, std::iter::empty());
        assert_eq!(cached.get(AssetId::default()), Some(query));
        cached.remove(AssetId::default());
        assert_eq!(cached.len(), 0);
    }

    #[test]
    fn material_rows_require_exact_completion_published_source_revision() {
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
        let compatible_identity = match sources.lookup(compatible.id()) {
            super::super::vt_source::MaterialVtSourceLookup::Available { identity, .. } => identity,
            other => panic!("expected compatible source, got {other:?}"),
        };
        let mut active_vt = MaterialVtActivePublication::default();

        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let material = assets.add(StandardMaterial {
            base_color_texture: Some(compatible.clone()),
            ..StandardMaterial::default()
        });
        let material_id = material.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        let initial = publish_texture_row(&mut world, material_id, &sources, &active_vt);
        assert_eq!(initial.virtual_texture_id, NO_VIRTUAL_TEXTURE);
        assert_eq!(initial.virtual_texture_width, 0);
        assert_eq!(initial.virtual_texture_height, 0);
        assert_eq!(initial.virtual_texture_page_size, 0);

        active_vt.replace([compatible_identity]);
        let texture = publish_texture_row(&mut world, material_id, &sources, &active_vt);
        assert_ne!(texture.virtual_texture_id, NO_VIRTUAL_TEXTURE);
        let runtime = world.resource::<RenderMaterialRegistry>();
        assert_eq!(
            texture.generation,
            runtime.texture_generation_table()[texture.index as usize],
            "VT rows must use the bindless heap generation, not the source revision"
        );
        let source_revision = compatible_identity.revision();
        assert_ne!(
            texture.generation as u64, source_revision,
            "the regression fixture must distinguish heap generations from source revisions"
        );

        let next_identity = sources.upsert_asset_image(compatible.id(), &complete_rgba_image(4));
        let transition = publish_texture_row(&mut world, material_id, &sources, &active_vt);
        assert_eq!(transition.virtual_texture_id, NO_VIRTUAL_TEXTURE);
        assert_eq!(transition.virtual_texture_width, 0);
        assert_eq!(transition.virtual_texture_height, 0);
        assert_eq!(transition.virtual_texture_page_size, 0);
        assert_eq!(transition.index, texture.index);
        assert_eq!(transition.generation, texture.generation);

        active_vt.replace([next_identity]);
        let republished = publish_texture_row(&mut world, material_id, &sources, &active_vt);
        assert_ne!(republished.virtual_texture_id, NO_VIRTUAL_TEXTURE);
        assert_eq!(republished.index, texture.index);
        assert_eq!(republished.generation, texture.generation);

        let mut world = World::new();
        let mut assets = Assets::<StandardMaterial>::default();
        let material = assets.add(StandardMaterial {
            base_color_texture: Some(incompatible.clone()),
            ..StandardMaterial::default()
        });
        let material_id = material.id();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        let incompatible_row = publish_texture_row(&mut world, material_id, &sources, &active_vt);
        assert_eq!(incompatible_row.virtual_texture_id, NO_VIRTUAL_TEXTURE);
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
            ShadingPlan::Single {
                program: ProgramRef::Builtin(BuiltinProgram::Lit)
            }
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
    fn image_reverse_index_preserves_other_materials_when_one_reference_changes() {
        fn image_handle(tag: u128) -> Handle<Image> {
            Handle::<Image>::from(Uuid::from_u128(tag))
        }

        let shared = image_handle(0xB1);
        let mut assets = Assets::<StandardMaterial>::default();
        let first = assets.add(StandardMaterial {
            base_color_texture: Some(shared.clone()),
            ..StandardMaterial::default()
        });
        let second = assets.add(StandardMaterial {
            base_color_texture: Some(shared.clone()),
            ..StandardMaterial::default()
        });
        let first_id = first.id();
        let second_id = second.id();

        let mut world = World::new();
        world.insert_resource(assets);
        world.insert_resource(RenderMaterialRegistry::default());
        for id in [first_id, second_id] {
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
        }

        let runtime = world.resource::<RenderMaterialRegistry>();
        let referenced: HashSet<_> = runtime.materials_referencing_image(shared.id()).collect();
        assert_eq!(referenced.len(), 2);
        assert!(referenced.contains(&first_id));
        assert!(referenced.contains(&second_id));
        drop(runtime);

        world
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(first_id)
            .unwrap()
            .base_color_texture = None;
        world.resource_scope(|world, mut runtime: Mut<RenderMaterialRegistry>| {
            let materials = world.resource::<Assets<StandardMaterial>>();
            runtime
                .publish_asset(
                    first_id,
                    materials.get(first_id).unwrap(),
                    prism_render_architecture::gpu_scene::GpuCompletionValue(0),
                )
                .unwrap();
        });

        let runtime = world.resource::<RenderMaterialRegistry>();
        let referenced: Vec<_> = runtime.materials_referencing_image(shared.id()).collect();
        assert_eq!(referenced, vec![second_id]);
        drop(runtime);

        world
            .resource_mut::<RenderMaterialRegistry>()
            .retire_asset(
                second_id,
                prism_render_architecture::gpu_scene::GpuCompletionValue(0),
            )
            .unwrap();
        assert_eq!(
            world
                .resource::<RenderMaterialRegistry>()
                .materials_referencing_image(shared.id())
                .count(),
            0
        );
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
        {
            let runtime = world.resource::<RenderMaterialRegistry>();
            assert!(runtime
                .materials_referencing_image(base.id())
                .any(|item| item == id));
            assert!(runtime
                .materials_referencing_image(normal.id())
                .any(|item| item == id));
        }

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
        {
            let runtime = world.resource::<RenderMaterialRegistry>();
            assert!(runtime
                .materials_referencing_image(base.id())
                .any(|item| item == id));
            assert!(!runtime
                .materials_referencing_image(normal.id())
                .any(|item| item == id));
        }

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
        assert!(!world
            .resource::<RenderMaterialRegistry>()
            .materials_referencing_image(base.id())
            .any(|item| item == id));
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
    upload(
        &mut buffers.surface_contexts,
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
    virtual_texture: Res<super::virtual_texture::MaterialVirtualTextureGpuBundle>,
    mut runtime: ResMut<RenderMaterialRegistry>,
) {
    runtime.set_texture_capacity(if arrays.bindless() {
        arrays.capacity()
    } else {
        virtual_texture.sizes().page_table_entry_capacity
    });
    runtime.set_portable_atlas(!arrays.bindless());
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
