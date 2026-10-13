//! Runtime material-VT producer.
//!
//! This is the first real residency producer for the material-VT contract. It
//! turns immutable CPU source snapshots into a deterministic full-source
//! residency table, builds guttered atlas tiles, writes the dedicated GPU
//! resources, and publishes only after the renderer's queue completion proof.
//! When real material visibility demands are present, selection is delegated to
//! the shared cross-frame streamer; the upload/publication state machine stays
//! completion-gated and unchanged.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderQueue;
use prism_render_architecture::{
    gpu_scene::GpuCompletionValue,
    texture_streaming::{
        PageDemand, StreamerConfig, StreamerFrame, TexturePageKey, VirtualTextureStreamer,
    },
};

use crate::{
    completion::GpuCompletionTracker,
    shading::texel_cache::gpu::page_table::{
        MaterialVirtualTexturePageRecord, MaterialVirtualTexturePageTableSnapshot,
    },
};

use super::{
    virtual_texture::MaterialVtDemandRegistry,
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

/// Stable producer input identity.  Revisions are normally sufficient, but
/// retaining the asset and descriptor identity makes the rebuild condition
/// explicit and prevents a future registry change from aliasing two source
/// sets that happen to reuse the same revision values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceFingerprint {
    asset_id: bevy_asset::AssetId<bevy_image::Image>,
    revision: u64,
    texture_id: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    page_size: Option<u32>,
    format: Option<prism_render_material::MaterialVtFormat>,
}

#[derive(Resource, Debug)]
pub(crate) struct MaterialVtProducer {
    fingerprint: Vec<SourceFingerprint>,
    next_epoch: u64,
    next_generation: u64,
    lifecycle: MaterialVirtualTextureLifecycle,
    pending: Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
    /// A queue write was recorded but assigning its completion token failed.
    /// Keep the lifecycle serial alive and retry instead of dropping the only
    /// handle to an in-flight update.
    unsubmitted_serial: Option<u64>,
    streamer: VirtualTextureStreamer,
    last_demand_revision: u64,
    last_feedback_mode: bool,
}

impl Default for MaterialVtProducer {
    fn default() -> Self {
        Self {
            fingerprint: Vec::new(),
            next_epoch: 0,
            next_generation: 0,
            lifecycle: MaterialVirtualTextureLifecycle::new(),
            pending: None,
            unsubmitted_serial: None,
            streamer: VirtualTextureStreamer::new(
                StreamerConfig::new(u64::from(DEFAULT_MATERIAL_VT_PAGE_CAPACITY) * 70_000)
                    .with_min_demand_frames(1)
                    .with_mip_tail_floor(None),
                DEFAULT_MATERIAL_VT_PAGE_CAPACITY,
            ),
            last_demand_revision: 0,
            last_feedback_mode: false,
        }
    }
}

/// Builds and submits one deterministic source-residency update per source-set
/// revision, then completion-gates its publication.
pub(crate) fn prepare_material_virtual_texture(
    mut producer: ResMut<MaterialVtProducer>,
    sources: Res<MaterialVtSourceRegistry>,
    demands: Res<MaterialVtDemandRegistry>,
    bundle: Res<MaterialVirtualTextureGpuBundle>,
    queue: Res<RenderQueue>,
    tracker: Res<GpuCompletionTracker>,
) {
    if producer.pending.is_some() {
        let completed = tracker.completed_value();
        let MaterialVtProducer {
            lifecycle, pending, ..
        } = &mut *producer;
        finish_completed_submission(lifecycle, pending, completed, |slot| {
            bundle.publish_slot(slot);
        });
    }

    if let Some(serial) = producer.unsubmitted_serial {
        let completion = tracker.next_submission_value();
        match producer.lifecycle.mark_submitted(serial, completion) {
            Ok(submission) => {
                producer.pending = Some(submission);
                producer.unsubmitted_serial = None;
            }
            Err(_) => return,
        }
    }

    if producer.pending.is_some() || producer.unsubmitted_serial.is_some() {
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
    entries.sort_by_key(|(asset_id, revision, snapshot)| {
        (
            snapshot
                .as_ref()
                .map(|source| source.descriptor().texture_id())
                .unwrap_or(u32::MAX),
            *revision,
            *asset_id,
        )
    });
    let fingerprint: Vec<_> = entries
        .iter()
        .map(|(asset_id, revision, snapshot)| {
            let descriptor = snapshot.as_ref().map(|source| source.descriptor());
            SourceFingerprint {
                asset_id: *asset_id,
                revision: *revision,
                texture_id: descriptor.map(|value| value.texture_id()),
                width: descriptor.map(|value| value.width()),
                height: descriptor.map(|value| value.height()),
                page_size: descriptor.map(|value| value.page_size()),
                format: descriptor.map(|value| value.format()),
            }
        })
        .collect();
    let source_changed = fingerprint != producer.fingerprint;
    let demand_changed = demands.revision() != producer.last_demand_revision;
    let feedback_mode_changed = demands.has_feedback() != producer.last_feedback_mode;
    if fingerprint == producer.fingerprint
        && producer.lifecycle.active().is_some()
        && !demand_changed
        && !feedback_mode_changed
    {
        return;
    }

    // A source revision or a feedback-mode transition invalidates every old
    // physical binding. The no-feedback fallback assigns slots independently,
    // so retaining streamer state across either boundary could pair new bytes
    // with an old page key.
    if source_changed || feedback_mode_changed {
        producer.streamer.clear();
    }
    if demand_changed {
        producer.last_demand_revision = demands.revision();
    }
    let source_pages = match source_page_map(&entries, bundle.atlas_config()) {
        Some(source_pages) => source_pages,
        // If the changed source set is structurally invalid (for example two
        // sources claim the same page key), publish an empty table rather than
        // leaving the old publication visible.
        None if source_changed => BTreeMap::new(),
        None => return,
    };
    let (mut records, page_sources, streamer_changed) = if demands.has_feedback() {
        let Some((records, page_sources, frame)) = select_feedback_residency(
            &mut producer.streamer,
            demands.demands(),
            &source_pages,
            bundle.atlas_config(),
        ) else {
            return;
        };
        // Stable feedback still advances the streamer's retention state, but
        // it must not rebuild and re-upload an unchanged residency table.
        (records, page_sources, !frame.is_steady())
    } else {
        let fallback = build_residency(
            &entries,
            bundle.atlas_config(),
            bundle
                .sizes()
                .page_table_entry_capacity
                .min(bundle.atlas_config().capacity()),
        );
        match fallback {
            Some((records, page_sources)) => (records, page_sources, true),
            // A changed source set must not retain stale pages just because
            // the new deterministic bootstrap set exceeds its capacity.
            None if source_changed => (Vec::new(), Vec::new(), true),
            None => return,
        }
    };
    if !streamer_changed && producer.lifecycle.active().is_some() {
        return;
    }

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
    // Reserve the lifecycle before touching the queue. If validation or the
    // upload itself fails, abort is safe because no queue write has been
    // recorded yet (plan_upload performs the same pre-write validation).
    let ranges = match bundle.plan_upload(&staging) {
        Ok(ranges) => ranges,
        Err(_) => return,
    };
    let serial = match producer
        .lifecycle
        .enqueue(staging.clone(), snapshot.clone(), ranges.clone())
    {
        Ok(serial) => serial,
        Err(_) => return,
    };
    if bundle.upload_staging(&queue, &staging).is_err() {
        let _ = producer.lifecycle.abort(serial);
        return;
    }
    producer.fingerprint = fingerprint;
    producer.last_feedback_mode = demands.has_feedback();
    let completion = tracker.next_submission_value();
    match producer.lifecycle.mark_submitted(serial, completion) {
        Ok(submission) => {
            producer.pending = Some(submission);
        }
        Err(_) => {
            // Keep the serial owned by the lifecycle. Dropping it here would
            // make the queued writes untrackable and allow slot reuse.
            producer.unsubmitted_serial = Some(serial);
        }
    }
}

fn normalized_demands(
    demands: &[PageDemand],
    sources: &BTreeMap<TexturePageKey, Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    streamer: &VirtualTextureStreamer,
) -> Vec<PageDemand> {
    demands
        .iter()
        .filter(|demand| sources.contains_key(&demand.key))
        .map(|demand| PageDemand {
            // Feedback can be produced before the previous publication has
            // completed. Reconcile against the streamer's actual published
            // table instead of charging an already resident page as missing.
            resident_mip: streamer
                .page_table()
                .resolve(demand.key, demand.key.mip)
                .map(|resolution| resolution.key.mip),
            ..*demand
        })
        .collect()
}

/// Applies one validated feedback frame to the producer's bounded streamer and
/// translates only its published page table into source-backed staging inputs.
///
/// Keeping this seam pure with respect to GPU resources makes the important
/// safety invariant testable: a demand that has no current source cannot create
/// a page record or an atlas upload.
fn select_feedback_residency(
    streamer: &mut VirtualTextureStreamer,
    demands: &[PageDemand],
    sources: &BTreeMap<TexturePageKey, Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
) -> Option<(
    Vec<MaterialVirtualTexturePageRecord>,
    Vec<(
        TexturePageKey,
        Arc<prism_render_material::MaterialVtSourceSnapshot>,
        u32,
    )>,
    StreamerFrame,
)> {
    let demand_set = normalized_demands(demands, sources, streamer);
    let frame = streamer.stream_demands(&demand_set);
    let table = streamer.page_table().clone();
    let (records, page_sources) = build_residency_for_table(&table, sources, atlas)?;
    Some((records, page_sources, frame))
}

fn source_page_map(
    entries: &[(
        bevy_asset::AssetId<bevy_image::Image>,
        u64,
        Option<Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    )],
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
) -> Option<BTreeMap<TexturePageKey, Arc<prism_render_material::MaterialVtSourceSnapshot>>> {
    let mut map = BTreeMap::new();
    for (_, _, snapshot) in entries {
        let Some(source) = snapshot else { continue };
        // Keep every VT producer entry point aligned with the row ABI: the
        // current shader can address only source layer zero.
        if source.descriptor().layers() != 1
            || source.descriptor().page_size() != atlas.page_size()
            || source.descriptor().format() != atlas.format()
        {
            continue;
        }
        for key in source.page_keys() {
            if map.insert(key, Arc::clone(source)).is_some() {
                return None;
            }
        }
    }
    Some(map)
}

fn build_residency_for_table(
    table: &prism_render_architecture::texture_streaming::GpuPageTable,
    sources: &BTreeMap<TexturePageKey, Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
) -> Option<(
    Vec<MaterialVirtualTexturePageRecord>,
    Vec<(
        TexturePageKey,
        Arc<prism_render_material::MaterialVtSourceSnapshot>,
        u32,
    )>,
)> {
    let mut records = Vec::with_capacity(table.len());
    let mut page_sources = Vec::with_capacity(table.len());
    for index in 0..table.len() {
        let (key, slot) = table.entry(index)?;
        let source = Arc::clone(sources.get(&key)?);
        records.push(MaterialVirtualTexturePageRecord {
            key,
            slot,
            generation: 1,
            epoch: 1,
            ready: true,
        });
        page_sources.push((key, source, slot));
    }
    if records.len() > atlas.capacity() as usize {
        return None;
    }
    Some((records, page_sources))
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
    publish_slot: impl Fn(u8),
) -> usize {
    if let Some(submission) = *pending {
        if let Ok(ready) = lifecycle.complete(submission, completed) {
            if lifecycle.publish(ready, submission.completion()).is_ok() {
                publish_slot(submission.resource_slot());
                *pending = None;
            }
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
        // The scene material row ABI carries no source-array layer, and the
        // shader requests layer zero. Do not publish unreachable layer>0 pages.
        if source.descriptor().layers() != 1
            || source.descriptor().page_size() != atlas.page_size()
            || source.descriptor().format() != atlas.format()
        {
            continue;
        }
        for key in source.page_keys() {
            pages.push((key, Arc::clone(source)));
        }
    }
    pages.sort_by_key(|(key, _)| *key);
    if pages.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return None;
    }
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

    #[test]
    fn array_sources_do_not_publish_unreachable_layer_pages() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid atlas");
        let descriptor =
            MaterialVtSourceDescriptor::new(9, 4, 4, 2, 4, 3, MaterialVtFormat::Rgba8Unorm)
                .expect("valid two-layer source");
        let bytes = vec![0_u8; (4 * 4 + 2 * 2 + 1) * 4 * 2];
        let source = Arc::new(
            MaterialVtSourceSnapshot::new(descriptor, bytes).expect("valid source payload"),
        );
        let entries = vec![(
            Handle::<bevy_image::Image>::from(Uuid::from_u128(1)).id(),
            1,
            Some(source),
        )];

        let (records, pages) = build_residency(&entries, atlas, 8)
            .expect("unsupported array source is skipped, not partially published");
        assert!(records.is_empty());
        assert!(pages.is_empty());
    }

    #[test]
    fn duplicate_page_keys_across_sources_are_rejected() {
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
                1,
                Some(source(9, 1)),
            ),
            (
                Handle::<bevy_image::Image>::from(Uuid::from_u128(2)).id(),
                2,
                Some(source(9, 2)),
            ),
        ];

        assert!(build_residency(&entries, atlas, 8).is_none());
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
            resource_slot: 1,
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
            finish_completed_submission(
                &mut lifecycle,
                &mut pending,
                GpuCompletionValue(0),
                |_| {}
            ),
            0
        );
        assert!(pending.is_some());
        assert!(lifecycle.active().is_none());

        assert_eq!(
            finish_completed_submission(
                &mut lifecycle,
                &mut pending,
                GpuCompletionValue(1),
                |_| {}
            ),
            0
        );
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(1));

        let second = page_table_snapshot(2);
        let second_ranges = super::super::virtual_texture::MaterialVirtualTextureUploadRanges {
            resource_slot: 0,
            page_table: 0..16,
            config: 0..32,
            atlas: Vec::new(),
        };
        let second_serial = lifecycle
            .enqueue(staging_for(atlas, &second), second, second_ranges)
            .expect("enqueue second update");
        let second_submission = lifecycle
            .mark_submitted(second_serial, GpuCompletionValue(2))
            .expect("submit second update");
        pending = Some(second_submission);

        assert_eq!(
            finish_completed_submission(
                &mut lifecycle,
                &mut pending,
                GpuCompletionValue(1),
                |_| {}
            ),
            0
        );
        assert!(pending.is_some());
        assert_eq!(lifecycle.retired_len(), 0);

        assert_eq!(
            finish_completed_submission(
                &mut lifecycle,
                &mut pending,
                GpuCompletionValue(2),
                |_| {}
            ),
            1
        );
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(2));
        assert_eq!(lifecycle.retired_len(), 0);
    }
}
