//! Observe images after every ordinary extractor, without consuming its batch.

use std::collections::{BTreeMap, BTreeSet};

use prism_asset::{AssetId, AssetServer, Assets};
use prism_ecs::{prelude::Resource, world::World};
use prism_image::Image;

use super::{
    virtual_texture::{NativeVtError, NativeVtSourceRequest, NativeVtSources},
    RenderMaterials,
};
use crate::{render_asset::ExtractedAssets, GpuImage, RenderAppConfig};

/// Why a referenced image cannot currently supply native material VT pages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeVtSourceIssue {
    /// No resident image exists in the authoritative asset arena.
    MissingImage,
    /// This extraction explicitly removed the image; removal wins over changes.
    Removed,
    /// The image cannot be represented by the configured VT source contract.
    Invalid(NativeVtError),
}

/// Source readiness for final published material references, not all loaded images.
#[derive(Default, Resource)]
pub struct NativeVtSourceStatus {
    requests: BTreeMap<AssetId<Image>, NativeVtSourceRequest>,
    issues: BTreeMap<AssetId<Image>, NativeVtSourceIssue>,
}

impl NativeVtSourceStatus {
    /// Unavailable sources, pruned when no published material references them.
    pub fn issues(&self) -> &BTreeMap<AssetId<Image>, NativeVtSourceIssue> {
        &self.issues
    }

    /// Number of distinct images demanded by published, live texture rows.
    pub fn requested_images(&self) -> usize {
        self.requests.len()
    }
}

/// Runs as an extraction finalizer so later user extractors can publish material
/// rows in the same frame. Generic image preparation retains its entire batch.
pub(crate) fn extract_native_vt_sources(main: &mut World, render: &mut World) {
    let page_size = render
        .resource::<RenderAppConfig>()
        .material_virtual_texture
        .page_size;
    let requests: BTreeMap<_, _> = render
        .resource::<RenderMaterials>()
        .referenced_texture_images()
        .into_iter()
        .map(|(image, slot)| {
            (
                image,
                NativeVtSourceRequest {
                    texture_id: slot.index,
                    page_size,
                },
            )
        })
        .collect();
    let mut sources = render
        .remove_resource::<NativeVtSources>()
        .unwrap_or_default();
    let mut status = render
        .remove_resource::<NativeVtSourceStatus>()
        .unwrap_or_default();
    status
        .issues
        .retain(|image, _| requests.contains_key(image));
    let empty = ExtractedAssets::<GpuImage>::new();
    let extracted = render
        .get_resource::<ExtractedAssets<GpuImage>>()
        .unwrap_or(&empty);
    let changed: BTreeSet<_> = extracted.extracted().iter().map(|(id, _)| *id).collect();
    let removed: BTreeSet<_> = extracted.removed().iter().copied().collect();
    for image in &changed {
        status.issues.remove(image);
    }
    for (image, error) in sources.observe_images(extracted, &requests) {
        status
            .issues
            .insert(image, NativeVtSourceIssue::Invalid(error));
    }
    for image in &removed {
        if requests.contains_key(image) {
            status.issues.insert(*image, NativeVtSourceIssue::Removed);
        }
    }

    // Valid quiet-frame sources require neither a clone nor a byte comparison.
    // Invalid images retry only on an event or a request edit. Missing images
    // may become resident without a new material publication, so retry those.
    // A removed image needs a new extraction, not just a request edit: an
    // arena may still contain an older copy that must not revive its token.
    let pending: Vec<_> = requests
        .iter()
        .filter_map(|(&image, &request)| {
            if sources.get(request.texture_id).is_some() || removed.contains(&image) {
                return None;
            }
            let unchanged_request = status.requests.get(&image) == Some(&request);
            if matches!(
                status.issues.get(&image),
                Some(NativeVtSourceIssue::Removed)
            ) || (unchanged_request
                && matches!(
                    status.issues.get(&image),
                    Some(NativeVtSourceIssue::Invalid(_))
                ))
            {
                return None;
            }
            // A changed image was already captured above, including its precise
            // failure. Do not reread an authoritative arena to mask that failure.
            if changed.contains(&image) {
                return None;
            }
            Some((image, request))
        })
        .collect();
    if !pending.is_empty() {
        let mut capture = |assets: &Assets<Image>| {
            for &(id, request) in &pending {
                let result = assets
                    .get(id)
                    .map(|image| sources.capture_image(id, image, request));
                match result {
                    Some(Ok(_)) => {
                        status.issues.remove(&id);
                    }
                    Some(Err(error)) => {
                        status
                            .issues
                            .insert(id, NativeVtSourceIssue::Invalid(error));
                    }
                    None => {
                        status.issues.insert(id, NativeVtSourceIssue::MissingImage);
                    }
                }
            }
        };
        let server_arena = main
            .get_resource::<AssetServer>()
            .and_then(|server| server.with_asset_arena::<Image, _>(&mut capture));
        if server_arena.is_none() {
            if let Some(assets) = main.get_resource::<Assets<Image>>() {
                capture(assets);
            } else {
                for &(id, _) in &pending {
                    status.issues.insert(id, NativeVtSourceIssue::MissingImage);
                }
            }
        }
    }
    status.requests = requests;
    render.insert_resource(sources);
    render.insert_resource(status);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prism_asset::AssetIndex;
    use prism_render_driver::{Extent3d, TextureFormat};
    use prism_render_material::fallback_material_record;

    use super::*;
    use crate::material::TextureRequest;

    fn image(value: u8) -> Image {
        Image::new(
            Extent3d::new_2d(1, 1),
            TextureFormat::Rgba8Unorm,
            vec![value; 4],
        )
        .unwrap()
    }

    fn invalid_image() -> Image {
        Image::new(
            Extent3d::new_2d(1, 1),
            TextureFormat::Rgba32Float,
            vec![0; 16],
        )
        .unwrap()
    }

    fn render_for(id: AssetId<Image>) -> (World, u32) {
        let mut materials = RenderMaterials::new(8);
        let handle = materials.registry_mut().allocate().unwrap();
        let mut record = fallback_material_record(handle, 1);
        record.textures = materials
            .resolve_textures(handle.index, [(11, TextureRequest::Image(id))])
            .expect("fixture image fits the texture heap");
        let slot = record.textures[0].index;
        materials.registry_mut().publish(record).unwrap();
        let mut render = World::new();
        render.insert_resource(RenderAppConfig::DEFAULT);
        render.insert_resource(materials);
        (render, slot)
    }

    #[test]
    fn source_stage_duplicate_events_use_final_snapshot_without_draining() {
        let id = AssetId::new(AssetIndex::from_parts(41, 1));
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, invalid_image());
        batch.push_extracted(id, image(9));
        render.insert_resource(batch);

        extract_native_vt_sources(&mut main, &mut render);
        let source = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        assert_eq!(source.cpu_snapshot().unwrap().bytes(), &[9; 4]);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        extract_native_vt_sources(&mut main, &mut render);
        assert!(Arc::ptr_eq(
            &source,
            render.resource::<NativeVtSources>().get(slot).unwrap(),
        ));
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert_eq!(
            render
                .resource::<ExtractedAssets<GpuImage>>()
                .extracted()
                .len(),
            2
        );
    }

    #[test]
    fn source_stage_invalid_final_event_cannot_be_masked_by_resident_arena() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(7));
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, image(8));
        batch.push_extracted(id, invalid_image());
        render.insert_resource(batch);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::Invalid(
                NativeVtError::UnsupportedFormat
            )),
        );

        render.insert_resource(ExtractedAssets::<GpuImage>::new());
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        let mut recovered = ExtractedAssets::<GpuImage>::new();
        recovered.push_extracted(id, invalid_image());
        recovered.push_extracted(id, image(7));
        render.insert_resource(recovered);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert!(!Arc::ptr_eq(
            &old,
            render.resource::<NativeVtSources>().get(slot).unwrap(),
        ));
    }

    #[test]
    fn source_stage_removal_wins_and_quiet_arena_does_not_resurrect_source() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(7));
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, invalid_image());
        batch.push_extracted(id, image(8));
        batch.push_removed(id);
        render.insert_resource(batch);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::Removed),
        );
        assert_eq!(
            render.resource::<ExtractedAssets<GpuImage>>().removed(),
            &[id]
        );
        render.insert_resource(ExtractedAssets::<GpuImage>::new());
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::Removed),
        );

        render
            .resource_mut::<RenderAppConfig>()
            .material_virtual_texture
            .page_size /= 2;
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::Removed),
        );
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());

        // A later explicit replacement, unlike a quiet arena read, restores
        // readiness without reviving the removed source's publication token.
        let mut recovered = ExtractedAssets::<GpuImage>::new();
        recovered.push_extracted(id, image(7));
        render.insert_resource(recovered);
        extract_native_vt_sources(&mut main, &mut render);
        let current = render.resource::<NativeVtSources>().get(slot).unwrap();
        assert!(!Arc::ptr_eq(&old, current));
        assert!(current.revision() > old.revision());
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
    }

    #[test]
    fn source_stage_missing_arena_retries_without_material_republication() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(6));
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let before = render.resource::<RenderMaterials>().registry().snapshot();
        let mut main = World::new();
        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::MissingImage),
        );
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<NativeVtSources>()
                .get(slot)
                .unwrap()
                .cpu_snapshot()
                .expect("CPU source snapshot")
                .bytes(),
            &[6; 4]
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );
    }

    #[test]
    fn source_stage_server_arena_is_authoritative_without_consuming_events() {
        let server = AssetServer::new();
        let server_handle = server.insert_runtime(image(3), []).unwrap();
        let mut compatibility = Assets::new();
        let compatibility_handle = compatibility.insert(image(8));
        assert_eq!(server_handle.id(), compatibility_handle.id());
        let id = server_handle.id();
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(server);
        main.insert_resource(compatibility);

        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<NativeVtSources>()
                .get(slot)
                .unwrap()
                .cpu_snapshot()
                .expect("CPU source snapshot")
                .bytes(),
            &[3; 4],
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        let batch = render.resource::<ExtractedAssets<GpuImage>>();
        assert_eq!(batch.extracted().len(), 1);
        assert_eq!(batch.extracted()[0].0, id);
        assert_eq!(batch.extracted()[0].1.mip_data(0), Some(&[3; 4][..]));

        // An existing server arena is authoritative even for an absent ID.
        let missing = main.resource_mut::<Assets<Image>>().insert(image(9));
        assert!(main
            .resource::<AssetServer>()
            .with_asset::<Image, _>(missing.id(), |_| ())
            .is_none());
        let (mut missing_render, missing_slot) = render_for(missing.id());
        extract_native_vt_sources(&mut main, &mut missing_render);
        assert!(missing_render
            .resource::<NativeVtSources>()
            .get(missing_slot)
            .is_none());
        assert_eq!(
            missing_render
                .resource::<NativeVtSourceStatus>()
                .issues()
                .get(&missing.id()),
            Some(&NativeVtSourceIssue::MissingImage),
        );
    }

    #[test]
    fn source_stage_absent_server_arena_uses_compatibility_without_consuming_events() {
        let mut compatibility = Assets::new();
        let handle = compatibility.insert(image(4));
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(AssetServer::new());
        main.insert_resource(compatibility);

        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<NativeVtSources>()
                .get(slot)
                .unwrap()
                .cpu_snapshot()
                .expect("CPU source snapshot")
                .bytes(),
            &[4; 4],
        );
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        let batch = render.resource::<ExtractedAssets<GpuImage>>();
        assert_eq!(batch.extracted().len(), 1);
        assert_eq!(batch.extracted()[0].0, id);
        assert_eq!(batch.extracted()[0].1.mip_data(0), Some(&[4; 4][..]));
    }

    #[test]
    fn source_stage_request_edit_revokes_source_and_retries_failed_capture() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(2));
        let (mut render, slot) = render_for(handle.id());
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        let page_size = old.layout().descriptor().page_size();

        render
            .resource_mut::<RenderAppConfig>()
            .material_virtual_texture
            .page_size = 0;
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert!(matches!(
            render
                .resource::<NativeVtSourceStatus>()
                .issues()
                .get(&handle.id()),
            Some(NativeVtSourceIssue::Invalid(_)),
        ));
        render
            .resource_mut::<RenderAppConfig>()
            .material_virtual_texture
            .page_size = page_size;
        extract_native_vt_sources(&mut main, &mut render);
        let sources = render.resource::<NativeVtSources>();
        assert!(!Arc::ptr_eq(&old, sources.get(slot).unwrap()));
        assert_eq!(
            sources.get(slot).unwrap().cpu_snapshot().unwrap().bytes(),
            &[2; 4]
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
    }

    #[test]
    fn source_stage_server_hot_reload_revokes_then_recovers_without_republishing_rows() {
        let server = AssetServer::new();
        let handle = server.insert_runtime(image(3), []).unwrap();
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let before = render.resource::<RenderMaterials>().registry().snapshot();
        let mut main = World::new();
        main.insert_resource(server);

        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        render.resource_mut::<ExtractedAssets<GpuImage>>().take();

        // A no-op hot reload preserves the publication token; changed valid
        // bytes replace it even though the material's authored rows are quiet.
        main.resource::<AssetServer>()
            .replace_runtime(&handle, image(3), [])
            .unwrap();
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(Arc::ptr_eq(
            &old,
            render.resource::<NativeVtSources>().get(slot).unwrap(),
        ));
        render.resource_mut::<ExtractedAssets<GpuImage>>().take();
        main.resource::<AssetServer>()
            .replace_runtime(&handle, image(8), [])
            .unwrap();
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        extract_native_vt_sources(&mut main, &mut render);
        let replaced = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        assert!(!Arc::ptr_eq(&old, &replaced));
        assert!(replaced.revision() > old.revision());
        assert_eq!(replaced.cpu_snapshot().unwrap().bytes(), &[8; 4]);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        render.resource_mut::<ExtractedAssets<GpuImage>>().take();

        main.resource::<AssetServer>()
            .replace_runtime(&handle, invalid_image(), [])
            .unwrap();
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::Invalid(
                NativeVtError::UnsupportedFormat
            )),
        );
        assert_eq!(render.resource::<ExtractedAssets<GpuImage>>().len(), 1);
        render.resource_mut::<ExtractedAssets<GpuImage>>().take();
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert!(matches!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(NativeVtSourceIssue::Invalid(_)),
        ));

        // Restoring identical bytes after an invalid replacement must mint a
        // new token, not qualify jobs staged against the original source.
        main.resource::<AssetServer>()
            .replace_runtime(&handle, image(8), [])
            .unwrap();
        crate::render_asset::extract_assets::<GpuImage>(&mut main, &mut render);
        extract_native_vt_sources(&mut main, &mut render);
        let current = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        assert!(!Arc::ptr_eq(&old, &current));
        assert!(!Arc::ptr_eq(&replaced, &current));
        assert!(current.revision() > replaced.revision());
        assert_eq!(current.cpu_snapshot().unwrap().bytes(), &[8; 4]);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        render.resource_mut::<ExtractedAssets<GpuImage>>().take();
        extract_native_vt_sources(&mut main, &mut render);
        assert!(Arc::ptr_eq(
            &current,
            render.resource::<NativeVtSources>().get(slot).unwrap(),
        ));
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );
    }

    #[test]
    fn source_stage_quiet_frame_keeps_snapshot_without_an_arena_or_extracted_batch() {
        let id = AssetId::new(AssetIndex::from_parts(41, 1));
        let (mut render, slot) = render_for(id);
        let before = render.resource::<RenderMaterials>().registry().snapshot();
        let mut main = World::new();
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, image(5));
        render.insert_resource(batch);
        extract_native_vt_sources(&mut main, &mut render);
        let source = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());
        render.remove_resource::<ExtractedAssets<GpuImage>>();

        for _ in 0..3 {
            extract_native_vt_sources(&mut main, &mut render);
            assert!(Arc::ptr_eq(
                &source,
                render.resource::<NativeVtSources>().get(slot).unwrap(),
            ));
            assert!(render
                .resource::<NativeVtSourceStatus>()
                .issues()
                .is_empty());
        }
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );
    }

    #[test]
    fn source_stage_texture_request_remap_revokes_old_slot_without_an_image_event() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(6));
        let id = handle.id();
        let (mut render, old_slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(old_slot).unwrap());

        let materials = render.resource_mut::<RenderMaterials>();
        let mut record = materials.registry().record_at(1).unwrap().clone();
        materials.release_textures(1);
        // Occupy the freed slot so the same image's next request changes ID.
        let other = AssetId::new(AssetIndex::from_parts(99, 1));
        let occupied = materials
            .textures_mut()
            .acquire(other)
            .expect("fixture image fits the texture heap");
        assert_eq!(occupied.index, old_slot);
        record.textures = materials
            .resolve_textures(1, [(11, TextureRequest::Image(id))])
            .expect("released texture slot is reusable");
        let new_slot = record.textures[0].index;
        assert_ne!(new_slot, old_slot);
        record.revision += 1;
        materials.registry_mut().publish(record).unwrap();
        let before = materials.registry().snapshot();

        extract_native_vt_sources(&mut main, &mut render);
        let sources = render.resource::<NativeVtSources>();
        assert!(sources.get(old_slot).is_none());
        let current = sources.get(new_slot).unwrap();
        assert!(!Arc::ptr_eq(&old, current));
        assert!(current.revision() > old.revision());
        assert_eq!(current.image(), id);
        assert_eq!(current.layout().descriptor().texture_id(), new_slot);
        assert_eq!(current.cpu_snapshot().unwrap().bytes(), &[6; 4]);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );
    }

    #[test]
    fn source_stage_pending_image_retries_when_fulfilled_without_an_extraction_event() {
        let mut assets = Assets::<Image>::new();
        let handle = assets.reserve();
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let before = render.resource::<RenderMaterials>().registry().snapshot();
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().issues().get(&id),
            Some(&NativeVtSourceIssue::MissingImage),
        );

        assert!(main.resource_mut::<Assets<Image>>().fulfill(id, image(8)));
        let pending_events = main.resource::<Assets<Image>>().pending_event_count();
        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<NativeVtSources>()
                .get(slot)
                .unwrap()
                .cpu_snapshot()
                .expect("CPU source snapshot")
                .bytes(),
            &[8; 4],
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert_eq!(
            main.resource::<Assets<Image>>().pending_event_count(),
            pending_events
        );
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );
    }

    #[test]
    fn source_stage_unpublished_request_waits_for_final_material_rows() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(6));
        let id = handle.id();
        let pending_events = assets.pending_event_count();
        let mut materials = RenderMaterials::new(8);
        let material = materials.registry_mut().allocate().unwrap();
        let mut record = fallback_material_record(material, 1);
        record.textures = materials
            .resolve_textures(material.index, [(11, TextureRequest::Image(id))])
            .expect("fixture image fits the texture heap");
        let slot = record.textures[0].index;
        let before = materials.registry().snapshot();
        let mut render = World::new();
        render.insert_resource(RenderAppConfig::DEFAULT);
        render.insert_resource(materials);
        let mut main = World::new();
        main.insert_resource(assets);

        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().requested_images(),
            0
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );

        // A late material publication, without a new image event or batch,
        // makes the already resident CPU image eligible for capture.
        render
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .publish(record)
            .unwrap();
        let published = render.resource::<RenderMaterials>().registry().snapshot();
        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render
                .resource::<NativeVtSources>()
                .get(slot)
                .unwrap()
                .cpu_snapshot()
                .expect("CPU source snapshot")
                .bytes(),
            &[6; 4]
        );
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().requested_images(),
            1
        );
        assert_eq!(
            main.resource::<Assets<Image>>().pending_event_count(),
            pending_events
        );
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            published
        );
    }

    #[test]
    fn source_stage_quiet_stale_lease_revokes_source_despite_same_image_and_slot() {
        let mut assets = Assets::new();
        let handle = assets.insert(image(6));
        let id = handle.id();
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        main.insert_resource(assets);
        extract_native_vt_sources(&mut main, &mut render);
        let old = Arc::clone(render.resource::<NativeVtSources>().get(slot).unwrap());

        // Simulate a heap release/reacquisition that leaves the published row
        // and its tracked lease naming an obsolete slot generation.
        let materials = render.resource_mut::<RenderMaterials>();
        let old_slot = materials.texture_slot(id).unwrap();
        assert!(materials.textures_mut().release(id));
        let replacement = materials
            .textures_mut()
            .acquire(id)
            .expect("released texture slot is reusable");
        assert_eq!(replacement.index, old_slot.index);
        assert_ne!(replacement.generation, old_slot.generation);
        let before = materials.registry().snapshot();

        for _ in 0..2 {
            extract_native_vt_sources(&mut main, &mut render);
            assert!(render.resource::<NativeVtSources>().get(slot).is_none());
            assert_eq!(
                render.resource::<NativeVtSourceStatus>().requested_images(),
                0
            );
            assert!(render
                .resource::<NativeVtSourceStatus>()
                .issues()
                .is_empty());
        }
        assert_eq!(
            render.resource::<RenderMaterials>().registry().snapshot(),
            before
        );

        let materials = render.resource_mut::<RenderMaterials>();
        let mut record = materials.registry().record_at(1).unwrap().clone();
        record.textures = materials
            .resolve_textures(1, [(11, TextureRequest::Image(id))])
            .expect("released texture slot is reusable");
        assert_eq!(record.textures[0].generation, replacement.generation);
        record.revision += 1;
        materials.registry_mut().publish(record).unwrap();
        extract_native_vt_sources(&mut main, &mut render);
        let current = render.resource::<NativeVtSources>().get(slot).unwrap();
        assert!(!Arc::ptr_eq(&old, current));
        assert!(current.revision() > old.revision());
        assert_eq!(current.cpu_snapshot().unwrap().bytes(), &[6; 4]);
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().requested_images(),
            1
        );
    }

    #[test]
    fn source_stage_unreferenced_invalid_image_drops_its_issue() {
        let id = AssetId::new(AssetIndex::from_parts(41, 1));
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, invalid_image());
        render.insert_resource(batch);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .contains_key(&id));
        render.resource_mut::<RenderMaterials>().release_textures(1);
        extract_native_vt_sources(&mut main, &mut render);
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().requested_images(),
            0
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
    }

    #[test]
    fn source_stage_unreferenced_rows_prune_sources_and_issues_without_draining_dirty() {
        let id = AssetId::new(AssetIndex::from_parts(41, 1));
        let (mut render, slot) = render_for(id);
        let mut main = World::new();
        let mut batch = ExtractedAssets::<GpuImage>::new();
        batch.push_extracted(id, image(5));
        render.insert_resource(batch);
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_some());
        let materials = render.resource_mut::<RenderMaterials>();
        let mut record = materials.registry().record_at(1).unwrap().clone();
        record.revision += 1;
        record.textures.clear();
        materials.registry_mut().publish(record).unwrap();
        extract_native_vt_sources(&mut main, &mut render);
        assert!(render.resource::<NativeVtSources>().get(slot).is_none());
        assert_eq!(
            render.resource::<NativeVtSourceStatus>().requested_images(),
            0
        );
        assert!(render
            .resource::<NativeVtSourceStatus>()
            .issues()
            .is_empty());
        assert!(render
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .take_dirty()
            .contains(&1));
    }
}
