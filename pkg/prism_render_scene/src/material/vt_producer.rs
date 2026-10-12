//! Runtime material-VT producer.
//!
//! This is the first real residency producer for the material-VT contract. It
//! turns immutable CPU source snapshots into a deterministic full-source
//! residency table, builds guttered atlas tiles, writes the dedicated GPU
//! resources, and publishes only after the renderer's queue completion proof.
//! It intentionally has no visibility feedback yet: a later feedback-driven
//! producer can replace the page selection policy without changing the upload
//! or publication state machine.

use std::{collections::BTreeSet, sync::Arc};

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderQueue;
use prism_render_architecture::{gpu_scene::GpuCompletionValue, texture_streaming::TexturePageKey};

use crate::{
    completion::GpuCompletionTracker,
    shading::texel_cache::gpu::page_table::{
        MaterialVirtualTexturePageRecord, MaterialVirtualTexturePageTableSnapshot,
    },
};

use super::{
    virtual_texture::{
        MaterialVirtualTextureGpuBundle, MaterialVirtualTextureGpuConfig,
        MaterialVirtualTextureLifecycle, MaterialVirtualTextureStaging,
    },
    vt_source::MaterialVtSourceRegistry,
};

/// Maximum number of physical material-VT pages in the default producer.
///
/// The atlas is deliberately bounded. Sources larger than this limit remain
/// valid CPU sources but are not published until a streaming policy with a
/// larger budget is selected.
pub const DEFAULT_MATERIAL_VT_PAGE_CAPACITY: u32 = 256;

#[derive(Resource, Debug, Default)]
pub(crate) struct MaterialVtProducer {
    fingerprint: Vec<u64>,
    next_epoch: u64,
    next_generation: u64,
    lifecycle: MaterialVirtualTextureLifecycle,
    pending: Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
}

/// Builds and submits one deterministic source-residency update per source-set
/// revision, then completion-gates its publication.
pub(crate) fn prepare_material_virtual_texture(
    mut producer: ResMut<MaterialVtProducer>,
    sources: Res<MaterialVtSourceRegistry>,
    bundle: Res<MaterialVirtualTextureGpuBundle>,
    queue: Res<RenderQueue>,
    tracker: Res<GpuCompletionTracker>,
) {
    if producer.pending.is_some() {
        let completed = tracker.completed_value();
        let MaterialVtProducer {
            lifecycle, pending, ..
        } = &mut *producer;
        finish_completed_submission(lifecycle, pending, completed);
    }

    if producer.pending.is_some() {
        return;
    }

    let mut entries = Vec::new();
    for entry in sources.entries() {
        let identity = entry.identity();
        let Some(snapshot) = entry.state().snapshot_handle() else {
            entries.push((identity.asset_id(), identity.revision(), None));
            continue;
        };
        entries.push((identity.asset_id(), identity.revision(), Some(snapshot)));
    }
    entries.sort_by_key(|(_, revision, snapshot)| {
        (
            snapshot
                .as_ref()
                .map(|source| source.descriptor().texture_id())
                .unwrap_or(u32::MAX),
            *revision,
        )
    });
    let fingerprint: Vec<_> = entries.iter().map(|(_, revision, _)| *revision).collect();
    if fingerprint == producer.fingerprint && producer.lifecycle.active().is_some() {
        return;
    }

    let Some((mut records, page_sources)) = build_residency(
        &entries,
        bundle.atlas_config(),
        bundle.sizes().page_table_entry_capacity,
    ) else {
        return;
    };

    producer.next_epoch = producer.next_epoch.saturating_add(1);
    producer.next_generation = producer.next_generation.saturating_add(1).max(1);
    let epoch = producer.next_epoch;
    let generation = producer.next_generation;
    for record in &mut records {
        record.epoch = epoch;
        record.generation = generation;
    }
    let snapshot =
        match MaterialVirtualTexturePageTableSnapshot::try_from_records(epoch, generation, records)
        {
            Ok(snapshot) => snapshot,
            Err(_) => return,
        };
    let geometry = match bundle.atlas_config().geometry() {
        Ok(geometry) => geometry,
        Err(_) => return,
    };
    let coarsest_mip = snapshot
        .records()
        .iter()
        .map(|record| u32::from(record.key.mip))
        .max()
        .unwrap_or(0);
    let enabled = !snapshot.is_empty();
    let config = MaterialVirtualTextureGpuConfig::new(
        enabled,
        u32::try_from(snapshot.len()).unwrap_or(u32::MAX),
        bundle.atlas_config().page_size(),
        geometry.tile_extent_px(),
        bundle.atlas_config().tiles_per_row(),
        bundle.sizes().atlas_layers,
        bundle.atlas_config().gutter_px(),
        coarsest_mip,
    );
    let mut staging =
        match MaterialVirtualTextureStaging::new(bundle.atlas_config(), &snapshot, config) {
            Ok(staging) => staging,
            Err(_) => return,
        };
    let neighbors: BTreeSet<_> = page_sources.iter().map(|(key, _, _)| *key).collect();
    for (key, source, slot) in page_sources {
        if staging
            .push_source_page(bundle.atlas_config(), &source, key, slot, &neighbors)
            .is_err()
        {
            return;
        }
    }
    let ranges = match bundle.upload_staging(&queue, &staging) {
        Ok(ranges) => ranges,
        Err(_) => return,
    };
    let serial = match producer.lifecycle.enqueue(staging, snapshot, ranges) {
        Ok(serial) => serial,
        Err(_) => return,
    };
    let completion = tracker.next_submission_value();
    producer.pending = producer.lifecycle.mark_submitted(serial, completion).ok();
    if producer.pending.is_some() {
        producer.fingerprint = fingerprint;
    }
}

/// Publishes a completion-verified update and reclaims publications that are
/// no longer referenced by the GPU.
///
/// Both operations intentionally use the same observed completion value. This
/// keeps the hand-off atomic from the producer's point of view: an old
/// publication can only be reclaimed after the new submission has become
/// publishable, and no reclaim is attempted using a guessed frame delay.
fn finish_completed_submission(
    lifecycle: &mut MaterialVirtualTextureLifecycle,
    pending: &mut Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
    completed: GpuCompletionValue,
) -> usize {
    if let Some(submission) = *pending {
        if let Ok(ready) = lifecycle.complete(submission, completed) {
            let _ = lifecycle.publish(ready, submission.completion());
            *pending = None;
        }
    }
    lifecycle.reclaim(completed)
}

fn build_residency(
    entries: &[(
        bevy_asset::AssetId<bevy_image::Image>,
        u64,
        Option<Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    )],
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
    capacity: u32,
) -> Option<(
    Vec<MaterialVirtualTexturePageRecord>,
    Vec<(
        TexturePageKey,
        Arc<prism_render_material::MaterialVtSourceSnapshot>,
        u32,
    )>,
)> {
    let mut pages = Vec::new();
    for (_, _, snapshot) in entries {
        let Some(source) = snapshot else { continue };
        if source.descriptor().page_size() != atlas.page_size()
            || source.descriptor().format() != atlas.format()
        {
            continue;
        }
        for key in source.page_keys() {
            pages.push((key, Arc::clone(source)));
        }
    }
    pages.sort_by_key(|(key, _)| *key);
    if pages.len() > capacity as usize {
        return None;
    }
    let mut records = Vec::with_capacity(pages.len());
    let mut page_sources = Vec::with_capacity(pages.len());
    for (slot, (key, source)) in pages.into_iter().enumerate() {
        let slot = u32::try_from(slot).ok()?;
        records.push(MaterialVirtualTexturePageRecord {
            key,
            slot,
            generation: 1,
            epoch: 1,
            ready: true,
        });
        page_sources.push((key, source, slot));
    }
    Some((records, page_sources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, Handle};
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    };

    fn source(texture_id: u32, seed: u8) -> Arc<prism_render_material::MaterialVtSourceSnapshot> {
        let descriptor = MaterialVtSourceDescriptor::new(
            texture_id,
            4,
            4,
            1,
            4,
            3,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid source descriptor");
        let mut bytes = Vec::new();
        for mip in 0..3u32 {
            let extent = (4u32 >> mip).max(1);
            bytes.extend(std::iter::repeat_n(seed, (extent * extent * 4) as usize));
        }
        Arc::new(MaterialVtSourceSnapshot::new(descriptor, bytes).expect("valid source"))
    }

    #[test]
    fn full_source_residency_is_key_sorted_and_bounded() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid atlas");
        let entries = vec![
            (
                Handle::<bevy_image::Image>::from(Uuid::from_u128(1)).id(),
                2,
                Some(source(9, 1)),
            ),
            (
                Handle::<bevy_image::Image>::from(Uuid::from_u128(2)).id(),
                1,
                Some(source(3, 2)),
            ),
        ];
        let (records, pages) = build_residency(&entries, atlas, 8).expect("within capacity");
        assert_eq!(records.len(), 6);
        assert_eq!(pages.len(), records.len());
        assert!(records.windows(2).all(|pair| pair[0].key < pair[1].key));
        assert_eq!(records[0].slot, 0);
        assert_eq!(records[5].slot, 5);
        assert!(build_residency(&entries, atlas, 5).is_none());
    }

    fn page_table_snapshot(generation: u64) -> MaterialVirtualTexturePageTableSnapshot {
        MaterialVirtualTexturePageTableSnapshot::try_from_records(
            1,
            generation,
            [MaterialVirtualTexturePageRecord {
                key: TexturePageKey {
                    texture: 1,
                    mip: 0,
                    layer: 0,
                    x: 0,
                    y: 0,
                },
                slot: 0,
                generation,
                epoch: 1,
                ready: true,
            }],
        )
        .expect("valid material VT snapshot")
    }

    fn staging_for(
        atlas: super::super::virtual_texture::MaterialVirtualTextureAtlasConfig,
        snapshot: &MaterialVirtualTexturePageTableSnapshot,
    ) -> MaterialVirtualTextureStaging {
        MaterialVirtualTextureStaging::new(
            atlas,
            snapshot,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 0),
        )
        .expect("valid material VT staging")
    }

    #[test]
    fn completed_submission_publishes_and_reclaims_retired_publication() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            4,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid atlas");
        let ranges = super::super::virtual_texture::MaterialVirtualTextureUploadRanges {
            page_table: 0..16,
            config: 0..32,
            atlas: Vec::new(),
        };
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let first = page_table_snapshot(1);
        let first_serial = lifecycle
            .enqueue(staging_for(atlas, &first), first, ranges.clone())
            .expect("enqueue first update");
        let first_submission = lifecycle
            .mark_submitted(first_serial, GpuCompletionValue(1))
            .expect("submit first update");
        let mut pending = Some(first_submission);

        assert_eq!(
            finish_completed_submission(&mut lifecycle, &mut pending, GpuCompletionValue(0),),
            0
        );
        assert!(pending.is_some());
        assert!(lifecycle.active().is_none());

        assert_eq!(
            finish_completed_submission(&mut lifecycle, &mut pending, GpuCompletionValue(1),),
            0
        );
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(1));

        let second = page_table_snapshot(2);
        let second_serial = lifecycle
            .enqueue(staging_for(atlas, &second), second, ranges)
            .expect("enqueue second update");
        let second_submission = lifecycle
            .mark_submitted(second_serial, GpuCompletionValue(2))
            .expect("submit second update");
        pending = Some(second_submission);

        assert_eq!(
            finish_completed_submission(&mut lifecycle, &mut pending, GpuCompletionValue(1),),
            0
        );
        assert!(pending.is_some());
        assert_eq!(lifecycle.retired_len(), 0);

        assert_eq!(
            finish_completed_submission(&mut lifecycle, &mut pending, GpuCompletionValue(2),),
            1
        );
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(2));
        assert_eq!(lifecycle.retired_len(), 0);
    }
}
