//! Per-view GPU resources for the virtual-shadow-map page-request (`page-mark`)
//! pass: the per-frame push-constant immediate block and the persistent
//! resident-window request bitmap the compute pass marks.
//!
//! The pass reads this frame's per-pixel receivers (owned by
//! [`super::super::resources::ViewVsmReceivers`]) and, for each one, marks every
//! clipmap page its soft-shadow footprint touches in a camera-snapped
//! resident-window request bitmap -- the flat `levels * edge^2` slot space both
//! VSM shaders index. Two things back that:
//!
//! * the per-frame [`super::super::abi::GpuVsmPageMarkParams`] immediate block,
//!   rebuilt every frame from [`PrismVirtualShadowSettings`] (the clipmap
//!   layout) plus this view's camera position projected onto the light's
//!   clipmap plane (the whole-page window snap origin) and its receiver count;
//!   and
//! * a persistent `array<atomic<u32>>` request bitmap sized to
//!   [`super::super::abi::window_slot_count`], cached across frames keyed by
//!   [`RetainedViewEntity`] and reallocated only when the slot count changes
//!   (a settings retune), so a steady-state camera never churns GPU allocations.
//!   The dispatch clears it to zero each frame before marking.
//!
//! The camera-plane projection reuses the same light basis
//! ([`prism_render_shading::ReceiverProjection`]) the receiver-generation pass
//! projects receivers with, so the request window and the receivers it marks
//! share one clipmap-plane origin -- exactly the golden `build_level` snap.

use bevy_ecs::prelude::*;
use bevy_math::{UVec2, Vec2};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::RenderDevice,
    view::{ExtractedView, RetainedViewEntity},
};
use prism_render_shading::ReceiverProjection;

use super::super::abi::{window_slot_count, GpuVsmPageMarkParams};
use super::super::extract::VsmPrimaryLight;
use super::super::resources::ViewVsmReceivers;
use super::super::settings::PrismVirtualShadowSettings;
use super::super::super::runtime::PrismShadingSettings;

/// Light id whose pages this dispatch marks. The subsystem drives a single
/// primary directional light (see [`VsmPrimaryLight`]); the id only rides the
/// immediate block for future multi-light routing and does not affect the
/// window-slot addressing.
const PRIMARY_LIGHT_ID: u32 = 0;

/// Per-view resources bound by the page-mark dispatch: the per-frame immediate
/// block and the persistent resident-window request bitmap.
///
/// Present only on views the page-mark prepare step ran this frame (VSM
/// enabled, a primary directional light present and a resident
/// [`ViewVsmReceivers`] receiver buffer).
#[derive(Component)]
pub(crate) struct ViewVsmPageRequests {
    /// [`GpuVsmPageMarkParams`] push-constant block for this frame; the dispatch
    /// uploads it with `set_immediates`.
    params: GpuVsmPageMarkParams,
    /// `window_slot_count` `array<atomic<u32>>` request bitmap, persistent across
    /// frames and cleared to zero by the dispatch before it marks.
    requests_buffer: Buffer,
    /// Number of valid receivers to dispatch over (`width * height`).
    receiver_count: u32,
}

impl ViewVsmPageRequests {
    /// The per-frame [`GpuVsmPageMarkParams`] immediate block.
    pub(crate) fn params(&self) -> &GpuVsmPageMarkParams {
        &self.params
    }

    /// The persistent resident-window request bitmap (bind group binding 1).
    pub(crate) fn requests_buffer(&self) -> &Buffer {
        &self.requests_buffer
    }

    /// The number of receivers the dispatch iterates (one per screen pixel).
    pub(crate) fn receiver_count(&self) -> u32 {
        self.receiver_count
    }
}

/// One cached request bitmap plus the slot count it was sized for, so a settings
/// retune that changes the clipmap layout can detect the mismatch and rebuild.
struct CachedPageRequests {
    buffer: Buffer,
    slot_count: u32,
}

/// Render-world cache of each view's persistent request bitmap, keyed by its
/// stable [`RetainedViewEntity`]. A view that persists across frames with an
/// unchanged clipmap reuses the same allocation; a retune rebuilds it and a
/// vanished view is dropped so buffers never leak.
#[derive(Resource, Default)]
pub(crate) struct VsmPageRequestBufferCache {
    buffers: HashMap<RetainedViewEntity, CachedPageRequests>,
}

impl VsmPageRequestBufferCache {
    /// Drops every cached buffer (used when the feature is disabled or no light
    /// drives the pass, so nothing lingers resident).
    fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Returns the cached request bitmap for `retained`, (re)allocating it when
    /// absent or sized for a different slot count. `byte_size` is the required
    /// `slot_count * size_of::<u32>()`.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        slot_count: u32,
        byte_size: u64,
    ) -> Buffer {
        let needs_new = self
            .buffers
            .get(&retained)
            .is_none_or(|cached| cached.slot_count != slot_count);
        if needs_new {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("prism VSM page requests"),
                size: byte_size,
                // COPY_DST so the dispatch can `clear_buffer` it to zero each
                // frame before the receivers accumulate requests with atomicOr.
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.buffers.insert(
                retained,
                CachedPageRequests {
                    buffer: buffer.clone(),
                    slot_count,
                },
            );
            buffer
        } else {
            self.buffers
                .get(&retained)
                .expect("buffer present after the is_none_or check")
                .buffer
                .clone()
        }
    }

    /// Drops any cached buffer whose view was not seen this frame.
    fn retain_seen(&mut self, seen: &HashSet<RetainedViewEntity>) {
        self.buffers.retain(|retained, _| seen.contains(retained));
    }
}

/// `PrepareResources` system: for every view with a resident receiver buffer,
/// rebuild the page-mark immediate block and ensure a correctly-sized persistent
/// request bitmap, then attach both as a [`ViewVsmPageRequests`] component.
///
/// Gated on [`PrismShadingSettings::enable_virtual_shadow`] and on the presence
/// of a primary directional light; when either is missing the cache is cleared
/// and no component is inserted, so the dispatch is a no-op that frame. Runs
/// after [`super::super::resources::prepare_vsm_receiver_resources`], whose
/// [`ViewVsmReceivers`] this consumes for the receiver buffer size and count.
pub(crate) fn prepare_vsm_page_requests(
    mut commands: Commands,
    device: Res<RenderDevice>,
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    primary_light: Res<VsmPrimaryLight>,
    mut cache: ResMut<VsmPageRequestBufferCache>,
    views: Query<(Entity, &ExtractedView, &ViewVsmReceivers)>,
) {
    if !settings.enable_virtual_shadow {
        cache.clear();
        return;
    }
    let Some(light_direction) = primary_light.direction else {
        cache.clear();
        return;
    };

    let clipmap = vsm_settings.clipmap();
    let slot_count = window_slot_count(&clipmap);
    let byte_size = u64::from(slot_count) * size_of::<u32>() as u64;

    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, receivers) in &views {
        let size: UVec2 = receivers.size;
        if size.x == 0 || size.y == 0 {
            continue;
        }

        let camera_world = view.world_from_view.translation();
        let projection = ReceiverProjection::from_light_direction(
            light_direction,
            camera_world,
            vsm_settings.pcf_radius as f32,
        );
        // Project the camera onto the light's clipmap plane exactly as the
        // receiver-generation shader projects each surface (light_space =
        // (dot(world, right), dot(world, up))), so the whole-page window snap
        // shares the receivers' plane origin.
        let camera_xy = Vec2::new(
            camera_world.dot(projection.light_right),
            camera_world.dot(projection.light_up),
        );

        let receiver_count = size.x.saturating_mul(size.y);
        let params = GpuVsmPageMarkParams::new(
            &clipmap,
            PRIMARY_LIGHT_ID,
            [camera_xy.x, camera_xy.y],
            receiver_count,
        );

        let retained = view.retained_view_entity;
        let requests_buffer = cache.get_or_create(&device, retained, slot_count, byte_size);

        commands.entity(entity).insert(ViewVsmPageRequests {
            params,
            requests_buffer,
            receiver_count,
        });
        seen.insert(retained);
    }
    cache.retain_seen(&seen);
}
