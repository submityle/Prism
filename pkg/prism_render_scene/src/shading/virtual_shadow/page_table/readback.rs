//! GPU->CPU readback bridge closing the virtual-shadow-map paging loop.
//!
//! The GPU page-mark pass ([`super::super::page_mark`]) marks a resident-window
//! request bitmap every frame; this module copies that bitmap back (one frame
//! late), decodes it with [`super::decode::request_keys`], drives the golden
//! [`VirtualShadowMapDriver`] to make the requested pages resident, uploads the
//! resulting page table via [`VsmPageTableBufferCache`] and attaches the per-view
//! [`ViewVsmPageTable`] / [`ViewVsmRenderPages`] the resolve and raster-fill
//! slices consume.
//!
//! It runs the same three-stage `RenderGraph` state machine as the visibility
//! parity readback: `request` enqueues the copy, `map_submitted` starts the async
//! map after submission, and `collect` consumes the mapped bytes the next frame.
//! State is keyed by [`RetainedViewEntity`] because the render-world `Entity` is
//! rebuilt each frame and cannot be held across the one-frame latency.

use std::sync::{
    mpsc::{self, Receiver},
    Mutex,
};

use bevy_ecs::prelude::*;
use bevy_math::Vec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        Buffer, BufferAsyncError, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode,
    },
    renderer::{PendingCommandBuffers, RenderDevice, RenderQueue},
    view::{ExtractedView, RetainedViewEntity},
};

use super::super::super::runtime::PrismShadingSettings;
use super::super::bridge::{BridgePage, VsmBridgeCache};
use super::super::page_mark::ViewVsmPageRequests;
use super::super::settings::PrismVirtualShadowSettings;
use super::decode::request_keys;
use super::pages::VsmRenderPage;
use super::resources::{
    ViewVsmPageTable, ViewVsmRenderPages, VirtualShadowMapDriver, VsmPageTableBufferCache,
};

/// Render-world resource holding each view's in-flight page-request readback.
#[derive(Resource, Default)]
pub(crate) struct VsmPageRequestReadback {
    states: Mutex<HashMap<RetainedViewEntity, ReadbackState>>,
}

/// One view's readback lifecycle: a copy queued this frame, or a map in flight
/// awaiting the mapping callback.
enum ReadbackState {
    CopyQueued(PendingReadback),
    Mapping(InFlightReadback),
}

/// A queued (or mapping) readback: the mappable target buffer plus the driving
/// state the copy was captured against.
///
/// Only the camera snap and light id are retained (not the render-world
/// `Entity`, which is rebuilt each frame); the view is re-matched by
/// [`RetainedViewEntity`] at collect time.
struct PendingReadback {
    buffer: Buffer,
    camera_x: f32,
    camera_y: f32,
    light: u32,
}

/// A readback whose buffer map has been submitted, carrying the channel the
/// mapping callback signals completion on.
struct InFlightReadback {
    pending: PendingReadback,
    receiver: Receiver<Result<(), BufferAsyncError>>,
}

/// Enqueues a copy of each view's resident-window request bitmap into a
/// mappable buffer (`RenderGraphSystems::Render`, after `camera_driver`).
///
/// A view whose previous readback is still in flight is skipped (its frame is
/// dropped) so the one-frame-latency state machine never overlaps two copies
/// for the same view.
pub(crate) fn request_vsm_page_readback(
    settings: Res<PrismShadingSettings>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    readback: Res<VsmPageRequestReadback>,
    views: Query<(&ExtractedView, &ViewVsmPageRequests)>,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    let mut states = readback.states.lock().unwrap();
    for (view, requests) in &views {
        let retained = view.retained_view_entity;
        if states.contains_key(&retained) {
            continue;
        }
        let src = requests.requests_buffer();
        let size = src.size();
        let target = device.create_buffer(&BufferDescriptor {
            label: Some("prism VSM page request readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism VSM page request readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &target, 0, Some(size));
        pending.push_encoder(encoder, "prism VSM page request readback");
        let params = requests.params();
        states.insert(
            retained,
            ReadbackState::CopyQueued(PendingReadback {
                buffer: target,
                camera_x: params.camera_x,
                camera_y: params.camera_y,
                light: params.light,
            }),
        );
    }
}

/// Starts the async buffer map for every queued copy, once
/// `RenderGraphSystems::Finish` has submitted the command buffers.
pub(crate) fn map_submitted_vsm_page_readback(
    device: Res<RenderDevice>,
    readback: Res<VsmPageRequestReadback>,
) {
    let mut states = readback.states.lock().unwrap();
    let queued: Vec<RetainedViewEntity> = states
        .iter()
        .filter_map(|(retained, state)| {
            matches!(state, ReadbackState::CopyQueued(_)).then_some(*retained)
        })
        .collect();
    for retained in queued {
        let Some(ReadbackState::CopyQueued(pending)) = states.remove(&retained) else {
            continue;
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        device.map_buffer(&pending.buffer.slice(..), MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        states.insert(
            retained,
            ReadbackState::Mapping(InFlightReadback { pending, receiver }),
        );
    }
}

/// Consumes each mapped request bitmap (`RenderSystems::PrepareResources`, next frame):
/// decodes the requests, drives the golden allocator, uploads the resulting
/// page table and attaches the per-view [`ViewVsmPageTable`] /
/// [`ViewVsmRenderPages`].
///
/// When VSM is disabled every resource is cleared so no stale residency leaks
/// into a later re-enable. Driver / buffer state for views absent this frame is
/// dropped after the pass.
pub(crate) fn collect_vsm_page_readback(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    readback: Res<VsmPageRequestReadback>,
    mut driver: ResMut<VirtualShadowMapDriver>,
    mut table_cache: ResMut<VsmPageTableBufferCache>,
    mut bridge: ResMut<VsmBridgeCache>,
    views: Query<(Entity, &ExtractedView)>,
) {
    let mut states = readback.states.lock().unwrap();
    if !settings.enable_virtual_shadow {
        driver.clear();
        table_cache.clear();
        bridge.clear();
        states.clear();
        return;
    }

    let mut entity_by_retained: HashMap<RetainedViewEntity, Entity> = HashMap::default();
    for (entity, view) in &views {
        entity_by_retained.insert(view.retained_view_entity, entity);
    }

    let retained_keys: Vec<RetainedViewEntity> = states.keys().copied().collect();
    for retained in retained_keys {
        let Some(state) = states.remove(&retained) else {
            continue;
        };
        let ReadbackState::Mapping(in_flight) = state else {
            // Copy queued but not yet mapped: keep it for map_submitted.
            states.insert(retained, state);
            continue;
        };
        match in_flight.receiver.try_recv() {
            Ok(Ok(())) => {}
            Err(mpsc::TryRecvError::Empty) => {
                // Mapping not ready yet: re-insert and try again next frame.
                states.insert(retained, ReadbackState::Mapping(in_flight));
                continue;
            }
            // Mapping failed or the callback channel dropped: discard so a fresh
            // copy is queued next frame.
            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => continue,
        }

        let pending = &in_flight.pending;
        let bitmap: Vec<u32> = {
            let mapped = pending.buffer.slice(..).get_mapped_range().unwrap();
            let owned = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
            drop(mapped);
            owned
        };
        pending.buffer.unmap();

        let clipmap = vsm_settings.clipmap();
        let camera_light_space = Vec2::new(pending.camera_x, pending.camera_y);
        let keys = request_keys(&clipmap, camera_light_space, pending.light, &bitmap);
        let drive = driver.drive(
            retained,
            *vsm_settings,
            pending.light,
            camera_light_space,
            &keys,
        );
        let slot_count = drive.page_table.len() as u32;
        let buffer = table_cache.upload(&device, &queue, retained, &drive.page_table);
        let render_pages = drive.render_pages;
        // Record the driven page table + resident pages into the same-frame
        // [`VsmBridgeCache`] so [`bridge_vsm_view_resources`](super::super::bridge)
        // -- ordered after this system in `PrepareResources` -- can re-attach them
        // as the resolve/atlas-visible components before `PrepareBindGroups`, with
        // no extra frame of latency; see [`super::super::bridge`].
        let bridge_pages: Vec<BridgePage> = render_pages
            .iter()
            .map(|page: &VsmRenderPage| BridgePage {
                physical_page: page.physical_page,
                atlas_tile_origin: page.atlas_tile_origin,
                world_origin: page.world_origin,
                world_size: page.world_size,
                level: page.level,
            })
            .collect();
        bridge.store(retained, buffer.clone(), slot_count, bridge_pages);
        if let Some(&entity) = entity_by_retained.get(&retained) {
            commands.entity(entity).insert((
                ViewVsmPageTable { buffer, slot_count },
                ViewVsmRenderPages {
                    pages: render_pages,
                },
            ));
        }
    }

    let live: HashSet<RetainedViewEntity> = entity_by_retained.keys().copied().collect();
    driver.retain_seen(&live);
    table_cache.retain_seen(&live);
    bridge.retain_seen(&live);
}
