//! Per-view GPU resources for the virtual-shadow-map receiver-generation pass:
//! the per-frame uniform block and the persistent receiver storage buffer the
//! compute pass fills.
//!
//! The pass reads the camera device depth, reconstructs each pixel's world
//! position and writes one [`super::abi::GpuVsmReceiver`] per pixel into a
//! storage buffer the downstream page-request pass consumes. Two resources back
//! that:
//!
//! * a small per-frame **uniform** ([`super::abi::GpuVsmReceiverGenParams`])
//!   carrying the camera inverse view-projection, the light's clipmap-plane
//!   basis and the framebuffer extent, rebuilt and uploaded every frame exactly
//!   like [`super::super::resolve::prepare_resolve_motion`]; and
//! * a persistent **storage** buffer holding `width * height` receiver records,
//!   cached across frames keyed by [`RetainedViewEntity`] and only reallocated
//!   when the viewport resizes (mirroring the visibility subsystem's per-view
//!   caches) so a steady-state camera never churns GPU allocations.
//!
//! The depth source is the SSR geometry prepass's `R32Float` reverse-Z
//! `scene_depth` ([`super::super::ssr::ViewSsrTextures`]); only views that ran
//! the SSR prepass carry it, which is why the receiver-generation pass is
//! interim-gated behind `enable_ssr` (see
//! [`super::super::runtime::PrismShadingSettings::enable_virtual_shadow`]). A
//! dedicated shared depth prepass would lift that coupling in a later slice.

use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView, RetainedViewEntity},
};
use bevy_math::Mat4;
use prism_render_shading::ReceiverProjection;

use super::abi::{GpuVsmReceiver, GpuVsmReceiverGenParams};
use super::extract::VsmPrimaryLight;
use super::settings::PrismVirtualShadowSettings;
use super::super::runtime::PrismShadingSettings;
use super::super::ssr::ViewSsrTextures;

/// Per-view resources bound by the receiver-generation dispatch: the per-frame
/// uniform and the persistent per-pixel receiver storage buffer.
///
/// Present only on views the receiver-generation prepare step ran this frame
/// (VSM enabled, a primary directional light present and an SSR depth prepass
/// resident).
#[derive(Component)]
pub(crate) struct ViewVsmReceivers {
    /// [`GpuVsmReceiverGenParams`] uniform, re-uploaded every frame.
    params_buffer: Buffer,
    /// `width * height` `storage` array of [`GpuVsmReceiver`], persistent across
    /// frames and reallocated only on viewport resize.
    receivers_buffer: Buffer,
    /// Framebuffer extent in pixels; the dispatch derives its workgroup count
    /// from this and it doubles as the receiver-buffer sizing key.
    pub(crate) size: UVec2,
}

impl ViewVsmReceivers {
    /// The per-frame [`GpuVsmReceiverGenParams`] uniform (bind group binding 2).
    pub(crate) fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    /// The persistent per-pixel receiver storage buffer (bind group binding 1).
    pub(crate) fn receivers_buffer(&self) -> &Buffer {
        &self.receivers_buffer
    }
}

/// One cached receiver storage buffer plus the viewport extent it was sized
/// for, so a resize can detect the mismatch and reallocate.
struct CachedVsmReceivers {
    buffer: Buffer,
    size: UVec2,
}

/// Render-world cache of each view's persistent receiver storage buffer, keyed
/// by its stable [`RetainedViewEntity`]. A view that persists across frames
/// with an unchanged viewport reuses the same allocation; a resize rebuilds it
/// and a vanished view is dropped so buffers never leak.
#[derive(Resource, Default)]
pub(crate) struct VsmReceiverBufferCache {
    buffers: HashMap<RetainedViewEntity, CachedVsmReceivers>,
}

impl VsmReceiverBufferCache {
    /// Drops every cached buffer (used when the feature is disabled or no light
    /// drives the pass, so nothing lingers resident).
    fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Returns the cached receiver buffer for `retained`, (re)allocating it when
    /// absent or sized for a different viewport. `byte_size` is the required
    /// `width * height * size_of::<GpuVsmReceiver>()`.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        size: UVec2,
        byte_size: u64,
    ) -> Buffer {
        let needs_new = self
            .buffers
            .get(&retained)
            .is_none_or(|cached| cached.size != size);
        if needs_new {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("prism VSM receivers"),
                size: byte_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.buffers.insert(
                retained,
                CachedVsmReceivers {
                    buffer: buffer.clone(),
                    size,
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

/// `PrepareResources` system: for every view with a resident SSR depth prepass,
/// rebuild and upload the receiver-generation uniform and ensure a
/// correctly-sized persistent receiver storage buffer, then attach both as a
/// [`ViewVsmReceivers`] component.
///
/// Gated on [`PrismShadingSettings::enable_virtual_shadow`] and on the presence
/// of a primary directional light; when either is missing the cache is cleared
/// and no component is inserted, so the dispatch is a no-op that frame.
pub(crate) fn prepare_vsm_receiver_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    primary_light: Res<VsmPrimaryLight>,
    mut cache: ResMut<VsmReceiverBufferCache>,
    views: Query<(Entity, &ExtractedView, &ViewSsrTextures)>,
) {
    if !settings.enable_virtual_shadow {
        cache.clear();
        return;
    }
    let Some(light_direction) = primary_light.direction else {
        cache.clear();
        return;
    };

    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, ssr) in &views {
        let size = ssr.size;
        if size.x == 0 || size.y == 0 {
            continue;
        }

        // Same reconstruction the visibility / motion passes use: prefer the
        // explicit clip_from_world, else compose it from the projection and the
        // inverse view transform, then invert for the unproject matrix the
        // shader multiplies (column-major, matching GpuVsmReceiverGenParams).
        let clip_from_world: Mat4 = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * view.world_from_view.to_matrix().inverse()
        });
        let inverse_view_proj = clip_from_world.inverse().to_cols_array();
        let camera_world = view.world_from_view.translation();

        let projection = ReceiverProjection::from_light_direction(
            light_direction,
            camera_world,
            vsm_settings.pcf_radius as f32,
        );
        let params =
            GpuVsmReceiverGenParams::new(inverse_view_proj, &projection, [size.x, size.y]);

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism VSM receiver-gen params"),
            size: size_of::<GpuVsmReceiverGenParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let retained = view.retained_view_entity;
        let receiver_count = u64::from(size.x) * u64::from(size.y);
        let byte_size = receiver_count * size_of::<GpuVsmReceiver>() as u64;
        let receivers_buffer = cache.get_or_create(&device, retained, size, byte_size);

        commands.entity(entity).insert(ViewVsmReceivers {
            params_buffer,
            receivers_buffer,
            size,
        });
        seen.insert(retained);
    }
    cache.retain_seen(&seen);
}
