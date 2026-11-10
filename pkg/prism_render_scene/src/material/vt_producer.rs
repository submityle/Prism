//! Runtime material-VT producer.
//!
//! This is the first real residency producer for the material-VT contract. It
//! turns immutable CPU source snapshots and bounded page demand into a
//! residency table, builds guttered atlas tiles, writes the dedicated GPU
//! resources, and publishes only sources whose coarsest covering page exists
//! after the renderer's queue completion proof.

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
    portable_atlas::{portable_texture_fallback, supports_portable_texture},
    runtime::RenderMaterialRegistry,
    virtual_texture::MaterialVtDemandRegistry,
    virtual_texture::{
        MaterialVirtualTextureGpuBundle, MaterialVirtualTextureGpuConfig,
        MaterialVirtualTextureLifecycle, MaterialVirtualTextureStaging,
    },
    vt_source::{
        material_texture_ids, MaterialVtActivePublication, MaterialVtMaterialSourceQueries,
        MaterialVtSourceIdentity, MaterialVtSourceRegistry,
    },
};

/// Maximum number of physical material-VT pages in the default producer.
///
/// The atlas is deliberately bounded. A source may describe more virtual pages
/// than this limit; demand selection decides which subset is physically
/// resident.
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
    portable_slot: Option<super::texture_heap::BindlessSlot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourcePublicationCandidate {
    identity: MaterialVtSourceIdentity,
    coarsest_page: TexturePageKey,
}

type MaterialVtSourceEntry = (
    bevy_asset::AssetId<bevy_image::Image>,
    u64,
    Option<Arc<prism_render_material::MaterialVtSourceSnapshot>>,
);

#[derive(Resource, Debug)]
pub(crate) struct MaterialVtProducer {
    fingerprint: Vec<SourceFingerprint>,
    next_epoch: u64,
    next_generation: u64,
    lifecycle: MaterialVirtualTextureLifecycle,
    pending: Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
    /// Exact source revisions represented by the currently in-flight bundle.
    /// This set becomes visible to material lowering only after completion and
    /// the active-slice flip succeed.
    pending_sources: Vec<MaterialVtSourceIdentity>,
    pending_portable_slots: Vec<(
        bevy_asset::AssetId<bevy_image::Image>,
        super::texture_heap::BindlessSlot,
    )>,
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
            pending_sources: Vec::new(),
            pending_portable_slots: Vec::new(),
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
    mut material_sources: ResMut<MaterialVtMaterialSourceQueries>,
    demands: Res<MaterialVtDemandRegistry>,
    bundle: Res<MaterialVirtualTextureGpuBundle>,
    queue: Res<RenderQueue>,
    tracker: Res<GpuCompletionTracker>,
    mut active_publication: ResMut<MaterialVtActivePublication>,
    arrays: Res<super::texture_upload::MaterialTextureArrays>,
    mut runtime: ResMut<RenderMaterialRegistry>,
) {
    if producer.pending.is_some() {
        let completed = tracker.completed_value();
        let MaterialVtProducer {
            lifecycle,
            pending,
            pending_sources,
            pending_portable_slots,
            ..
        } = &mut *producer;
        finish_completed_publication_with_rows(
            lifecycle,
            pending,
            pending_sources,
            pending_portable_slots,
            completed,
            &mut active_publication,
            &sources,
            &mut material_sources,
            &mut runtime,
            &tracker,
            |slot| bundle.publish_slot(slot),
        );
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

    let mut referenced_images: Vec<_> = material_sources.referenced_images().collect();
    referenced_images.sort_unstable();
    let mut entries = Vec::with_capacity(referenced_images.len());
    for asset_id in referenced_images {
        let Some(entry) = sources.entry(asset_id) else {
            continue;
        };
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
                portable_slot: (!arrays.bindless())
                    .then(|| runtime.texture_slot(*asset_id))
                    .flatten(),
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
    let (source_pages, publication_candidates) =
        match source_page_map(&entries, bundle.atlas_config()) {
            Some(result) => result,
            // If the changed source set is structurally invalid (for example two
            // sources claim the same page key), publish an empty table rather than
            // leaving the old publication visible.
            None if source_changed => (BTreeMap::new(), Vec::new()),
            None => return,
        };
    let (mut records, page_sources, streamer_changed, portable_publication) = if !arrays.bindless()
    {
        let Some((records, page_sources, publication)) =
            build_portable_residency(&entries, &runtime, bundle.atlas_config())
        else {
            return;
        };
        (records, page_sources, true, publication)
    } else if demands.has_feedback() {
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
        (records, page_sources, !frame.is_steady(), Vec::new())
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
            Some((records, page_sources)) => (records, page_sources, true, Vec::new()),
            // A changed source set must not retain stale pages just because
            // the new deterministic bootstrap set exceeds its capacity.
            None if source_changed => (Vec::new(), Vec::new(), true, Vec::new()),
            None => return,
        }
    };
    if !streamer_changed
        && producer.lifecycle.active().is_some()
        && !source_changed
        && !feedback_mode_changed
    {
        return;
    }
    let published_sources = if arrays.bindless() {
        publication_for_residency(&publication_candidates, &records)
    } else {
        portable_publication
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
        let result = if arrays.bindless() {
            staging.push_source_page(bundle.atlas_config(), &source, key, slot, &neighbors)
        } else {
            staging.push_portable_source_page(bundle.atlas_config(), &source, key, slot)
        };
        if result.is_err() {
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
    producer.pending_sources = published_sources;
    producer.pending_portable_slots = if arrays.bindless() {
        Vec::new()
    } else {
        producer
            .pending_sources
            .iter()
            .filter_map(|identity| {
                runtime
                    .texture_slot(identity.asset_id())
                    .map(|slot| (identity.asset_id(), slot))
            })
            .collect()
    };
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

/// A completed upload is still private until every affected live material can
/// be republished. All fallible row checks precede the lifecycle transition;
/// exclusive system resource access keeps that proof valid through commit.
/// The normal dirty-row upload runs after this system in the same frame.
fn finish_completed_publication_with_rows(
    lifecycle: &mut MaterialVirtualTextureLifecycle,
    pending: &mut Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
    pending_sources: &mut Vec<MaterialVtSourceIdentity>,
    pending_portable_slots: &mut Vec<(
        bevy_asset::AssetId<bevy_image::Image>,
        super::texture_heap::BindlessSlot,
    )>,
    completed: GpuCompletionValue,
    active: &mut MaterialVtActivePublication,
    sources: &MaterialVtSourceRegistry,
    queries: &mut MaterialVtMaterialSourceQueries,
    runtime: &mut RenderMaterialRegistry,
    tracker: &GpuCompletionTracker,
    mut publish_slot: impl FnMut(u8),
) -> bool {
    let Some(submission) = *pending else {
        return false;
    };
    let Ok(ready) = lifecycle.complete(submission, completed) else {
        return false;
    };
    let mut next = active.clone();
    let changed = next.replace_with_portable_slots(
        pending_sources.iter().copied(),
        pending_portable_slots.iter().copied(),
    );
    let affected: BTreeSet<_> = changed
        .into_iter()
        .flat_map(|image| runtime.materials_referencing_image(image))
        .collect();
    if affected
        .iter()
        .any(|id| !runtime.can_republish_source_row(*id))
    {
        // Keep the completed submission and both public views unchanged. A
        // repaired/retired row can unblock this exact publication next frame.
        return false;
    }
    if lifecycle.publish(ready, submission.completion()).is_err() {
        return false;
    }
    for id in affected {
        let material = runtime
            .material_snapshot(id)
            .expect("preflight snapshot")
            .clone();
        let handle = runtime.material_handle(id).expect("preflight handle");
        let dynamic = runtime.registry.dynamic_channels_at(handle.index);
        let execution = runtime.registry.execution_binding_at(handle.index);
        runtime
            .publish_asset_with_sources(
                id,
                &material,
                tracker.next_submission_value(),
                Some(sources),
                Some(&next),
            )
            .expect("exclusive source-row preflight must remain valid");
        // Residency is not a material rebake: preserve frame-local state.
        runtime
            .registry
            .set_dynamic_channels(handle, dynamic)
            .expect("live source row");
        if let Some(binding) = execution {
            runtime
                .registry
                .set_execution_binding(handle, binding)
                .expect("unchanged execution tables");
        }
        queries.update(
            id,
            sources.query_material(&material),
            material_texture_ids(&material),
        );
    }
    // No consumer can run between these writes: they share this system's
    // exclusive resource borrows and precede row staging/bind-group prepare.
    *active = next;
    publish_slot(submission.resource_slot());
    *pending = None;
    pending_sources.clear();
    pending_portable_slots.clear();
    lifecycle.reclaim(completed);
    true
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
    entries: &[MaterialVtSourceEntry],
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
) -> Option<(
    BTreeMap<TexturePageKey, Arc<prism_render_material::MaterialVtSourceSnapshot>>,
    Vec<SourcePublicationCandidate>,
)> {
    let mut map = BTreeMap::new();
    let mut candidates = Vec::new();
    for (asset_id, revision, snapshot) in entries {
        let Some(source) = snapshot.as_ref() else {
            continue;
        };
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
        candidates.push(SourcePublicationCandidate {
            identity: MaterialVtSourceIdentity::new(*asset_id, *revision),
            coarsest_page: TexturePageKey {
                texture: source.descriptor().texture_id(),
                mip: source.descriptor().mip_count().saturating_sub(1),
                layer: 0,
                x: 0,
                y: 0,
            },
        });
    }
    Some((map, candidates))
}

fn build_portable_residency(
    entries: &[MaterialVtSourceEntry],
    runtime: &RenderMaterialRegistry,
    atlas: super::virtual_texture::MaterialVirtualTextureAtlasConfig,
) -> Option<(
    Vec<MaterialVirtualTexturePageRecord>,
    Vec<(
        TexturePageKey,
        Arc<prism_render_material::MaterialVtSourceSnapshot>,
        u32,
    )>,
    Vec<MaterialVtSourceIdentity>,
)> {
    let mut rows = Vec::new();
    for (asset_id, revision, snapshot) in entries {
        let Some(source) = snapshot.as_ref() else {
            continue;
        };
        if !supports_portable_texture(source) {
            continue;
        }
        let Some(slot) = runtime.texture_slot(*asset_id) else {
            // A referenced image can reach source discovery before its
            // material row has acquired an authored-texture slot. Keep the
            // portable batch productive for every other source; this source
            // remains unpublished and therefore resolves through the normal
            // authored-texture fallback when its material row is built.
            continue;
        };
        let slot = slot.index;
        if slot >= atlas.capacity() {
            return None;
        }
        let fallback = portable_texture_fallback(source)?;
        let key = fallback.key;
        rows.push((*asset_id, *revision, key, Arc::clone(source), slot));
    }
    rows.sort_by_key(|(_, _, key, _, _)| *key);
    if rows.windows(2).any(|pair| pair[0].2 == pair[1].2) {
        return None;
    }
    let mut records = Vec::with_capacity(rows.len());
    let mut pages = Vec::with_capacity(rows.len());
    let mut publication = Vec::with_capacity(rows.len());
    for (asset_id, revision, key, source, slot) in rows {
        records.push(MaterialVirtualTexturePageRecord {
            key,
            slot,
            generation: 1,
            epoch: 1,
            ready: true,
        });
        pages.push((key, source, slot));
        publication.push(MaterialVtSourceIdentity::new(asset_id, revision));
    }
    Some((records, pages, publication))
}

fn publication_for_residency(
    candidates: &[SourcePublicationCandidate],
    records: &[MaterialVirtualTexturePageRecord],
) -> Vec<MaterialVtSourceIdentity> {
    let resident_pages: BTreeSet<_> = records.iter().map(|record| record.key).collect();
    candidates
        .iter()
        .filter(|candidate| resident_pages.contains(&candidate.coarsest_page))
        .map(|candidate| candidate.identity)
        .collect()
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

/// Adapter for the no-material lifecycle regression. It calls the production
/// gate with an empty scene rather than maintaining a test-only commit path.
#[cfg(test)]
fn finish_completed_publication(
    lifecycle: &mut MaterialVirtualTextureLifecycle,
    pending: &mut Option<super::virtual_texture::MaterialVirtualTextureSubmission>,
    pending_sources: &mut Vec<MaterialVtSourceIdentity>,
    completed: GpuCompletionValue,
    active_publication: &mut MaterialVtActivePublication,
    publish_slot: impl FnMut(u8),
) -> (usize, Vec<bevy_asset::AssetId<bevy_image::Image>>) {
    let before_retired = lifecycle.retired_len();
    let had_active = lifecycle.active().is_some();
    let mut prospective = active_publication.clone();
    let changed_images = prospective.replace(pending_sources.iter().copied());
    if !finish_completed_publication_with_rows(
        lifecycle,
        pending,
        pending_sources,
        &mut Vec::new(),
        completed,
        active_publication,
        &MaterialVtSourceRegistry::default(),
        &mut MaterialVtMaterialSourceQueries::default(),
        &mut RenderMaterialRegistry::default(),
        &GpuCompletionTracker::default(),
        publish_slot,
    ) {
        return (0, Vec::new());
    }
    (
        before_retired + usize::from(had_active) - lifecycle.retired_len(),
        changed_images,
    )
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
    use bevy_pbr::StandardMaterial;
    use bevy_platform::collections::HashSet;
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    };

    fn source(texture_id: u32, seed: u8) -> Arc<MaterialVtSourceSnapshot> {
        source_with_extent(texture_id, 4, 4, seed)
    }

    fn source_with_extent(
        texture_id: u32,
        extent: u32,
        page_size: u32,
        seed: u8,
    ) -> Arc<MaterialVtSourceSnapshot> {
        source_with_extent_format(
            texture_id,
            extent,
            page_size,
            seed,
            MaterialVtFormat::Rgba8Unorm,
        )
    }

    fn source_with_extent_format(
        texture_id: u32,
        extent: u32,
        page_size: u32,
        seed: u8,
        format: MaterialVtFormat,
    ) -> Arc<MaterialVtSourceSnapshot> {
        let mip_count = extent.ilog2() as u8 + 1;
        let descriptor = MaterialVtSourceDescriptor::new(
            texture_id, extent, extent, 1, page_size, mip_count, format,
        )
        .expect("valid source descriptor");
        let mut bytes = Vec::new();
        for mip in 0..u32::from(mip_count) {
            let mip_extent = (extent >> mip).max(1);
            bytes.extend(std::iter::repeat_n(
                seed,
                (mip_extent * mip_extent * format.bytes_per_texel()) as usize,
            ));
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

        let (_, candidates) = source_page_map(&entries, atlas).expect("valid source map");
        let accepted: HashSet<_> = candidates
            .into_iter()
            .map(|candidate| candidate.identity)
            .collect();
        let expected: HashSet<_> = entries
            .iter()
            .map(|(asset_id, revision, _)| MaterialVtSourceIdentity::new(*asset_id, *revision))
            .collect();
        assert_eq!(accepted, expected);
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
        let (source_pages, candidates) =
            source_page_map(&entries, atlas).expect("unsupported source set remains valid");
        assert!(source_pages.is_empty());
        assert!(candidates.is_empty());
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
        assert!(source_page_map(&entries, atlas).is_none());
    }

    #[test]
    fn portable_residency_skips_slot_miss_without_blocking_other_sources() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            128,
            1,
            1,
            8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid portable atlas");
        let resident_handle = Handle::<bevy_image::Image>::from(Uuid::from_u128(0x91));
        let resident_image = resident_handle.id();
        let missing_handle = Handle::<bevy_image::Image>::from(Uuid::from_u128(0x92));
        let missing_image = missing_handle.id();
        let resident_source = source_with_extent(31, 128, 128, 1);
        let missing_source = source_with_extent(32, 128, 128, 2);
        let entries = vec![
            (resident_image, 4, Some(Arc::clone(&resident_source))),
            (missing_image, 5, Some(Arc::clone(&missing_source))),
        ];

        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        let resident_material_id = Handle::<StandardMaterial>::from(Uuid::from_u128(0x93)).id();
        runtime
            .publish_asset(
                resident_material_id,
                &StandardMaterial {
                    base_color_texture: Some(resident_handle),
                    ..StandardMaterial::default()
                },
                GpuCompletionValue(1),
            )
            .expect("resident material allocates its authored texture slot");
        let resident_slot = runtime
            .texture_slot(resident_image)
            .expect("resident image has a slot")
            .index;
        assert!(runtime.texture_slot(missing_image).is_none());

        let (records, pages, publication) = build_portable_residency(&entries, &runtime, atlas)
            .expect("one slot miss must not reject the portable batch");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].key.texture,
            resident_source.descriptor().texture_id()
        );
        assert_eq!(records[0].slot, resident_slot);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, records[0].key);
        assert_eq!(pages[0].2, resident_slot);
        assert_eq!(
            publication,
            [MaterialVtSourceIdentity::new(resident_image, 4)]
        );
        assert!(!publication.contains(&MaterialVtSourceIdentity::new(missing_image, 5)));

        let mut sources = MaterialVtSourceRegistry::default();
        let missing_identity = sources.insert_snapshot(missing_image, (*missing_source).clone());
        let active_publication = MaterialVtActivePublication::default();
        assert!(!active_publication.contains(missing_identity));
        let missing_material_id = Handle::<StandardMaterial>::from(Uuid::from_u128(0x94)).id();
        let missing_handle = runtime
            .publish_asset_with_sources(
                missing_material_id,
                &StandardMaterial {
                    base_color_texture: Some(missing_handle),
                    ..StandardMaterial::default()
                },
                GpuCompletionValue(2),
                Some(&sources),
                Some(&active_publication),
            )
            .expect("slot-miss source still publishes an authored material row");
        let missing_row = runtime.registry.get(missing_handle).unwrap().textures[0];
        assert_eq!(
            missing_row.virtual_texture_id,
            prism_render_material::NO_VIRTUAL_TEXTURE
        );
        assert_eq!(missing_row.virtual_texture_width, 0);
        assert_eq!(missing_row.index, missing_row.sampler_index);
        assert_eq!(
            missing_row.index,
            runtime
                .texture_slot(missing_image)
                .expect("authored fallback acquired a texture slot")
                .index
        );
    }

    #[test]
    fn portable_residency_uses_one_source_backed_fallback_mip_for_large_images() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            128,
            1,
            1,
            8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid portable atlas");
        let handle = Handle::<bevy_image::Image>::from(Uuid::from_u128(0x95));
        let image = handle.id();
        let source = source_with_extent(33, 256, 128, 3);
        let entries = vec![(image, 7, Some(Arc::clone(&source)))];
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        runtime
            .publish_asset(
                Handle::<StandardMaterial>::from(Uuid::from_u128(0x96)).id(),
                &StandardMaterial {
                    base_color_texture: Some(handle),
                    ..StandardMaterial::default()
                },
                GpuCompletionValue(1),
            )
            .expect("source image acquires a portable slot");

        let (records, pages, publication) =
            build_portable_residency(&entries, &runtime, atlas).expect("large source is portable");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key.mip, 1);
        assert_eq!(records[0].key.x, 0);
        assert_eq!(records[0].key.y, 0);
        assert_eq!(pages[0].0, records[0].key);
        assert_eq!(publication, [MaterialVtSourceIdentity::new(image, 7)]);
    }

    #[test]
    fn portable_residency_accepts_non_default_format_and_page_geometry() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            128,
            1,
            1,
            8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid portable atlas");
        let handle = Handle::<bevy_image::Image>::from(Uuid::from_u128(0x97));
        let image = handle.id();
        let source = source_with_extent_format(34, 4, 4, 3, MaterialVtFormat::Rgba8Srgb);
        let entries = vec![(image, 8, Some(Arc::clone(&source)))];
        let mut runtime = RenderMaterialRegistry::default();
        runtime.set_portable_atlas(true);
        runtime
            .publish_asset(
                Handle::<StandardMaterial>::from(Uuid::from_u128(0x98)).id(),
                &StandardMaterial {
                    base_color_texture: Some(handle),
                    ..StandardMaterial::default()
                },
                GpuCompletionValue(1),
            )
            .expect("source image acquires a portable slot");

        let (records, pages, publication) =
            build_portable_residency(&entries, &runtime, atlas).expect("source is portable");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].key, pages[0].0);
        assert_eq!(records[0].key.mip, 0);
        assert_eq!(pages[0].2, runtime.texture_slot(image).unwrap().index);
        assert_eq!(publication, [MaterialVtSourceIdentity::new(image, 8)]);
    }

    #[test]
    fn large_source_partial_residency_publishes_after_coarsest_demand() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            2,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid atlas");
        let image = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xA1)).id();
        let source = source_with_extent(17, 64, 4, 7);
        assert!(source.page_count() > u64::from(DEFAULT_MATERIAL_VT_PAGE_CAPACITY));
        let entries = vec![(image, 3, Some(source))];
        let (source_pages, candidates) =
            source_page_map(&entries, atlas).expect("large source map is valid");
        let coarsest = candidates[0].coarsest_page;
        let demand = PageDemand {
            key: coarsest,
            semantic: prism_render_architecture::texture_streaming::TextureSemantic::Mask,
            desired_mip: coarsest.mip,
            resident_mip: None,
            screen_importance: 1,
            byte_cost: 1,
            frame: 1,
        };
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(2).with_min_demand_frames(1),
            atlas.capacity(),
        );
        let (records, pages, frame) =
            select_feedback_residency(&mut streamer, &[demand], &source_pages, atlas)
                .expect("source-backed demand produces residency");

        assert_eq!(frame.loaded(), 1);
        assert_eq!(records.len(), 1);
        assert_eq!(pages.len(), 1);
        assert_eq!(records[0].key, coarsest);
        assert_eq!(
            publication_for_residency(&candidates, &records),
            [MaterialVtSourceIdentity::new(image, 3)]
        );
    }

    #[test]
    fn publication_only_contains_sources_with_resident_coarsest_pages() {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            2,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid atlas");
        let first_image = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xB1)).id();
        let second_image = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xB2)).id();
        let entries = vec![
            (first_image, 4, Some(source(10, 1))),
            (second_image, 5, Some(source(20, 2))),
        ];
        let (_, candidates) = source_page_map(&entries, atlas).expect("valid source map");
        let resident = [MaterialVirtualTexturePageRecord {
            key: candidates[0].coarsest_page,
            slot: 0,
            generation: 1,
            epoch: 1,
            ready: true,
        }];

        assert_eq!(
            publication_for_residency(&candidates, &resident),
            [candidates[0].identity]
        );
        assert!(publication_for_residency(&candidates, &[]).is_empty());
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
        let image = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xC1)).id();
        let first_identity = MaterialVtSourceIdentity::new(image, 1);
        let second_identity = MaterialVtSourceIdentity::new(image, 2);
        let mut pending_sources = vec![first_identity];
        let mut active_publication = MaterialVtActivePublication::default();
        let mut published_slots = Vec::new();

        assert_eq!(
            finish_completed_publication(
                &mut lifecycle,
                &mut pending,
                &mut pending_sources,
                GpuCompletionValue(0),
                &mut active_publication,
                |slot| published_slots.push(slot),
            ),
            (0, Vec::new())
        );
        assert!(pending.is_some());
        assert!(lifecycle.active().is_none());
        assert!(!active_publication.contains(first_identity));
        assert_eq!(pending_sources, [first_identity]);
        assert!(published_slots.is_empty());

        let (reclaimed, mut changed_images) = finish_completed_publication(
            &mut lifecycle,
            &mut pending,
            &mut pending_sources,
            GpuCompletionValue(1),
            &mut active_publication,
            |slot| published_slots.push(slot),
        );
        changed_images.sort_unstable();
        assert_eq!(reclaimed, 0);
        assert_eq!(changed_images, [image]);
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(1));
        assert!(active_publication.contains(first_identity));
        assert!(pending_sources.is_empty());
        assert_eq!(published_slots, [1]);

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
        pending_sources = vec![second_identity];

        assert_eq!(
            finish_completed_publication(
                &mut lifecycle,
                &mut pending,
                &mut pending_sources,
                GpuCompletionValue(1),
                &mut active_publication,
                |slot| published_slots.push(slot),
            ),
            (0, Vec::new())
        );
        assert!(pending.is_some());
        assert_eq!(lifecycle.retired_len(), 0);
        assert!(active_publication.contains(first_identity));
        assert!(!active_publication.contains(second_identity));
        assert_eq!(published_slots, [1]);

        let (reclaimed, mut changed_images) = finish_completed_publication(
            &mut lifecycle,
            &mut pending,
            &mut pending_sources,
            GpuCompletionValue(2),
            &mut active_publication,
            |slot| published_slots.push(slot),
        );
        changed_images.sort_unstable();
        assert_eq!(reclaimed, 1);
        assert_eq!(changed_images, [image]);
        assert!(pending.is_none());
        assert_eq!(lifecycle.active().map(|active| active.serial()), Some(2));
        assert_eq!(lifecycle.retired_len(), 0);
        assert!(active_publication.contains(second_identity));
        assert!(!active_publication.contains(first_identity));
        assert!(pending_sources.is_empty());
        assert_eq!(published_slots, [1, 0]);
    }

    #[test]
    fn completed_vt_publication_republishes_referencing_material_rows_same_frame() {
        let image_handle = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xD1));
        let image = image_handle.id();
        let material_id = Handle::<StandardMaterial>::from(Uuid::from_u128(0xD2)).id();
        let old_source = source_with_extent(17, 128, 128, 1);
        let new_source = source_with_extent(17, 256, 128, 2);

        let mut sources = MaterialVtSourceRegistry::default();
        let old_identity = sources.insert_snapshot(image, (*old_source).clone());
        let mut active_publication = MaterialVtActivePublication::default();
        active_publication.replace([old_identity]);
        let material = StandardMaterial {
            base_color_texture: Some(image_handle),
            ..StandardMaterial::default()
        };
        let mut runtime = RenderMaterialRegistry::default();
        let handle = runtime
            .publish_asset_with_sources(
                material_id,
                &material,
                GpuCompletionValue(1),
                Some(&sources),
                Some(&active_publication),
            )
            .expect("initial material publication");
        let old_row = runtime.registry.get(handle).unwrap().textures[0];
        assert_eq!(old_row.virtual_texture_width, 128);

        let new_identity = sources.insert_snapshot(image, (*new_source).clone());
        let mut source_queries = MaterialVtMaterialSourceQueries::default();
        let tracker = GpuCompletionTracker::default();
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let mut pending = Some(enqueue_publication(&mut lifecycle, 1, 1));
        let mut pending_sources = vec![new_identity];
        let mut pending_slots = Vec::new();
        let mut published_slots = Vec::new();
        let before = runtime.registry.snapshot();

        assert!(!finish_completed_publication_with_rows(
            &mut lifecycle,
            &mut pending,
            &mut pending_sources,
            &mut pending_slots,
            GpuCompletionValue(0),
            &mut active_publication,
            &sources,
            &mut source_queries,
            &mut runtime,
            &tracker,
            |slot| published_slots.push(slot),
        ));
        assert_eq!(runtime.registry.snapshot(), before);
        assert_eq!(runtime.registry.get(handle).unwrap().textures[0], old_row);
        assert!(active_publication.contains(old_identity));
        assert!(!active_publication.contains(new_identity));
        assert!(published_slots.is_empty());
        assert!(pending.is_some());
        assert_eq!(pending_sources, [new_identity]);
        assert!(lifecycle.active().is_none());

        assert!(finish_completed_publication_with_rows(
            &mut lifecycle,
            &mut pending,
            &mut pending_sources,
            &mut pending_slots,
            GpuCompletionValue(1),
            &mut active_publication,
            &sources,
            &mut source_queries,
            &mut runtime,
            &tracker,
            |slot| published_slots.push(slot),
        ));

        let new_row = &runtime.registry.get(handle).unwrap().textures[0];
        assert_eq!(new_row.virtual_texture_width, 256);
        assert_eq!(new_row.virtual_texture_height, 256);
        assert_eq!(new_row.virtual_texture_page_size, 128);
        assert_eq!(runtime.revisions.get(&material_id), Some(&2));
        assert!(active_publication.contains(new_identity));
        assert!(!active_publication.contains(old_identity));
        assert_eq!(published_slots, [1]);
        assert!(pending.is_none());
        assert!(pending_sources.is_empty());
        assert_eq!(lifecycle.active().unwrap().serial(), 1);
        assert_eq!(source_queries.get(material_id).unwrap().available, 1);
    }

    struct RowPublicationFixture {
        lifecycle: MaterialVirtualTextureLifecycle,
        pending: Option<super::super::virtual_texture::MaterialVirtualTextureSubmission>,
        pending_sources: Vec<MaterialVtSourceIdentity>,
        pending_slots: Vec<(
            bevy_asset::AssetId<bevy_image::Image>,
            super::super::texture_heap::BindlessSlot,
        )>,
        active: MaterialVtActivePublication,
        sources: MaterialVtSourceRegistry,
        queries: MaterialVtMaterialSourceQueries,
        runtime: RenderMaterialRegistry,
        tracker: GpuCompletionTracker,
        published_slots: Vec<u8>,
        ids: [bevy_asset::AssetId<StandardMaterial>; 2],
        handles: [prism_render_architecture::abi::GenerationalHandle; 2],
    }

    impl RowPublicationFixture {
        fn new() -> Self {
            let image = Handle::<bevy_image::Image>::from(Uuid::from_u128(0xDA));
            let material = StandardMaterial {
                base_color_texture: Some(image.clone()),
                ..Default::default()
            };
            let mut sources = MaterialVtSourceRegistry::default();
            let identity =
                sources.insert_snapshot(image.id(), (*source_with_extent(17, 4, 128, 3)).clone());
            let active = MaterialVtActivePublication::default();
            let mut runtime = RenderMaterialRegistry::default();
            runtime.set_portable_atlas(true);
            let ids =
                [0xDB, 0xDC].map(|id| Handle::<StandardMaterial>::from(Uuid::from_u128(id)).id());
            let handles = ids.map(|id| {
                runtime
                    .publish_asset_with_sources(
                        id,
                        &material,
                        GpuCompletionValue(1),
                        Some(&sources),
                        Some(&active),
                    )
                    .unwrap()
            });
            let slot = runtime.texture_slot(image.id()).unwrap();
            let mut lifecycle = MaterialVirtualTextureLifecycle::new();
            let pending = Some(enqueue_publication(&mut lifecycle, 1, 1));
            Self {
                lifecycle,
                pending,
                pending_sources: vec![identity],
                pending_slots: vec![(image.id(), slot)],
                active,
                sources,
                queries: MaterialVtMaterialSourceQueries::default(),
                runtime,
                tracker: GpuCompletionTracker::default(),
                published_slots: Vec::new(),
                ids,
                handles,
            }
        }

        fn finish(&mut self, completed: u64) -> bool {
            finish_completed_publication_with_rows(
                &mut self.lifecycle,
                &mut self.pending,
                &mut self.pending_sources,
                &mut self.pending_slots,
                GpuCompletionValue(completed),
                &mut self.active,
                &self.sources,
                &mut self.queries,
                &mut self.runtime,
                &self.tracker,
                |slot| self.published_slots.push(slot),
            )
        }
    }

    #[test]
    fn portable_publication_preflights_all_rows_and_retries_without_partial_commit() {
        use prism_render_material::{DynamicChannels, NO_VIRTUAL_TEXTURE, PORTABLE_TEXTURE_ATLAS};

        let mut fixture = RowPublicationFixture::new();
        let identity = fixture.pending_sources[0];
        let slot = fixture.pending_slots[0].1;
        let dynamic = DynamicChannels {
            wetness: 0.5,
            ..DynamicChannels::NONE
        };
        for handle in fixture.handles {
            fixture
                .runtime
                .registry
                .set_dynamic_channels(handle, dynamic)
                .unwrap();
        }
        fixture.runtime.registry.take_dirty();
        fixture.runtime.dirty_assets.clear();
        let before = fixture.runtime.registry.snapshot();
        assert!(!fixture.finish(0));
        // Overflow and stale-next-revision failures must both be checked
        // before *any* affected row or active publication is mutated.
        for blocked_revision in [u32::MAX, 0] {
            fixture
                .runtime
                .revisions
                .insert(fixture.ids[1], blocked_revision);
            assert!(!fixture.finish(1));
            assert_eq!(fixture.runtime.registry.snapshot(), before);
            assert!(fixture.runtime.registry.take_dirty().is_empty());
            assert!(fixture.runtime.dirty_assets.is_empty());
            assert!(!fixture.active.contains(identity));
            assert!(fixture.lifecycle.active().is_none());
            assert!(fixture.published_slots.is_empty());
            assert!(fixture.pending.is_some());
            assert_eq!(fixture.pending_sources, [identity]);
            assert_eq!(fixture.pending_slots, [(identity.asset_id(), slot)]);
            for (id, handle) in fixture.ids.into_iter().zip(fixture.handles) {
                let record = fixture.runtime.registry.get(handle).unwrap();
                assert_eq!(record.revision, 1);
                assert_eq!(record.textures[0].virtual_texture_id, NO_VIRTUAL_TEXTURE);
                assert_eq!(
                    fixture.runtime.registry.dynamic_channels_at(handle.index),
                    dynamic
                );
                assert!(fixture.queries.get(id).is_none());
            }
        }
        fixture.runtime.revisions.insert(fixture.ids[1], 1);
        assert!(fixture.finish(1));
        assert!(fixture.active.contains_portable(identity, slot));
        assert_eq!(fixture.published_slots, [1]);
        assert!(fixture.pending.is_none());
        assert!(fixture.pending_sources.is_empty());
        assert!(fixture.pending_slots.is_empty());
        let mut expected_dirty: Vec<_> = fixture.handles.map(|handle| handle.index).into();
        expected_dirty.sort_unstable();
        assert_eq!(fixture.runtime.registry.take_dirty(), expected_dirty);
        for (id, handle) in fixture.ids.into_iter().zip(fixture.handles) {
            let record = fixture.runtime.registry.get(handle).unwrap();
            assert_eq!(record.revision, 2);
            assert_eq!(
                record.textures[0].virtual_texture_id,
                PORTABLE_TEXTURE_ATLAS
            );
            assert_eq!(record.textures[0].index, slot.index);
            assert_eq!(record.textures[0].generation, slot.generation);
            assert_eq!(
                fixture.runtime.registry.dynamic_channels_at(handle.index),
                dynamic
            );
            assert_eq!(fixture.queries.get(id).unwrap().available, 1);
            assert!(fixture.runtime.dirty_assets.contains(&id));
        }
        let committed = fixture.runtime.registry.snapshot();
        assert!(!fixture.finish(1));
        assert_eq!(fixture.runtime.registry.snapshot(), committed);
        assert_eq!(fixture.published_slots, [1]);
    }

    #[test]
    fn completed_stale_source_or_slot_never_claims_a_portable_row() {
        use prism_render_material::NO_VIRTUAL_TEXTURE;

        for change in 0..3 {
            let mut fixture = RowPublicationFixture::new();
            let identity = fixture.pending_sources[0];
            let image = identity.asset_id();
            let original_slot = fixture.runtime.texture_slot(image).unwrap();
            match change {
                0 => {
                    fixture
                        .sources
                        .insert_snapshot(image, (*source_with_extent(17, 8, 128, 9)).clone());
                }
                1 => {
                    fixture.sources.remove(image);
                }
                _ => {
                    fixture.pending_slots[0].1.generation += 1;
                }
            }
            assert!(fixture.finish(1));
            for handle in fixture.handles {
                let row = fixture.runtime.registry.get(handle).unwrap().textures[0];
                assert_eq!(row.virtual_texture_id, NO_VIRTUAL_TEXTURE);
                assert_eq!(row.index, original_slot.index);
                assert_eq!(row.generation, original_slot.generation);
                assert_eq!(row.virtual_texture_width, 0);
            }
            assert_eq!(fixture.published_slots, [1]);
        }
    }

    #[test]
    fn completed_publication_removal_republishes_portable_rows_as_authored_fallback() {
        let mut fixture = RowPublicationFixture::new();
        let identity = fixture.pending_sources[0];
        let slot = fixture.pending_slots[0].1;
        assert!(fixture.finish(1));
        fixture.pending = Some(enqueue_publication(&mut fixture.lifecycle, 2, 0));
        assert!(!fixture.finish(1));
        assert!(fixture.active.contains_portable(identity, slot));
        assert!(fixture.finish(2));
        assert!(!fixture.active.contains(identity));
        assert_eq!(fixture.published_slots, [1, 0]);
        assert_eq!(fixture.lifecycle.retired_len(), 0);
        for handle in fixture.handles {
            let record = fixture.runtime.registry.get(handle).unwrap();
            assert_eq!(record.revision, 3);
            let row = record.textures[0];
            assert_eq!(
                row.virtual_texture_id,
                prism_render_material::NO_VIRTUAL_TEXTURE
            );
            assert_eq!((row.index, row.generation), (slot.index, slot.generation));
        }
    }

    // These lifecycle fixtures exercise the production commit gate without a
    // GPU. Atlas-byte construction is covered separately by residency tests.
    fn enqueue_publication(
        lifecycle: &mut MaterialVirtualTextureLifecycle,
        generation: u64,
        resource_slot: u8,
    ) -> super::super::virtual_texture::MaterialVirtualTextureSubmission {
        let atlas = super::super::virtual_texture::MaterialVirtualTextureAtlasConfig::new(
            4,
            0,
            2,
            4,
            MaterialVtFormat::Rgba8Unorm,
        )
        .unwrap();
        let snapshot = page_table_snapshot(generation);
        let ranges = super::super::virtual_texture::MaterialVirtualTextureUploadRanges {
            resource_slot,
            page_table: 0..16,
            config: 0..32,
            atlas: Vec::new(),
        };
        let serial = lifecycle
            .enqueue(staging_for(atlas, &snapshot), snapshot, ranges)
            .unwrap();
        lifecycle
            .mark_submitted(serial, GpuCompletionValue(generation))
            .unwrap()
    }
}
