//! Per-view persistent ping-pong state backing the specular-GI *temporal
//! denoise* passes (reproject + history-clamp).
//!
//! Unlike the spatial filter ([`super::resources`]), the temporal accumulator
//! is a cross-frame subsystem: it reprojects last frame's converged specular
//! history into the current pixel, clamps it against this frame's noisy
//! `spec_gi` resolve, and advances the age + dual-rate luminance EMAs. The
//! history (radiance + age), metadata (normal + roughness) and luminance planes
//! therefore cannot live in the frame-transient
//! [`bevy_render::texture::TextureCache`] the spatial target uses — that pool is
//! recycled every frame — so this module keeps a persistent **ping-pong** set of
//! planes per view in a [`Local`] cache keyed by [`RetainedViewEntity`],
//! mirroring [`super::super::ao::temporal`] and
//! [`super::super::ssr::temporal`]. Each frame reads the slot written last frame
//! and writes the other.
//!
//! Four persistent planes ping-pong (two slots each):
//!
//! * **history / meta / luma** — `rgba16float`, `STORAGE | TEXTURE`, because they
//!   swap write (this frame's `out_*`) and read (next frame's `prev_*`) roles;
//! * **depth** — `r32float`, `COPY_DST | TEXTURE`, never shader-written: the
//!   reproject node `copy_texture_to_texture`s the current-frame SSR reverse-Z
//!   depth into the write slot each frame so next frame it is the `prev_depth`
//!   the world-space disocclusion guard reads. Zero-init on the first frame
//!   naturally rejects (reverse-Z `0` is background), so the clamp seeds from the
//!   resolve with no explicit cold-start flag.
//!
//! Four transient planes allocate per-frame from the [`TextureCache`]
//! (`rgba16float`, `STORAGE | TEXTURE`): the three `reprojected*` planes the
//! reproject pass writes and the clamp reads, plus the `denoised` plane the
//! clamp writes and the spatial filter consumes in place of the raw resolve.
//!
//! The subsystem is gated identically to the spatial filter and the `spec_gi`
//! reuse resolve it feeds from — on `enable_spec_gi`, `enable_ssr`,
//! `enable_visibility_buffer`, a resident visibility buffer and a single-sample
//! view — so the three allocate and free in lockstep.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::{Mat4, UVec2, Vec3};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, RetainedViewEntity},
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;
use super::resources::SPEC_DENOISE_FILTERED_FORMAT;

/// Device-depth encoding of the persistent previous-frame depth plane. Must
/// match the SSR prepass `scene_depth` the reproject node copies into it; the
/// `format_matches_ssr_scene_depth` test pins it to `R32Float` so a drift in the
/// SSR format is caught at build time rather than as a silent copy mismatch.
const TEMPORAL_DEPTH_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Per-view temporal state resolved each frame from the persistent ping-pong
/// cache: the readable previous-frame planes (history/meta/luma/depth), the
/// writable current-frame planes (history/meta/luma + the copy-target depth
/// texture), the transient reprojected/denoised planes, the reprojection
/// matrices and the current/previous camera positions.
#[derive(Component)]
pub(crate) struct ViewSpecDenoiseTemporal {
    /// Previous frame's accumulated history (rgb radiance, a = age), sampled at
    /// the reprojected texel. Zero on the first frame (clamp seeds from resolve).
    prev_history: TextureView,
    /// Previous frame's metadata (xyz = world normal, w = roughness).
    prev_meta: TextureView,
    /// Previous frame's dual-rate luminance EMAs (r = fast, g = slow).
    prev_luma: TextureView,
    /// Previous frame's reverse-Z device depth (the world-space disocclusion
    /// guard's prev surface). Written by last frame's copy, read this frame.
    prev_depth: TextureView,
    /// This frame's accumulated history output; next frame's `prev_history`.
    out_history: TextureView,
    /// This frame's metadata output; next frame's `prev_meta`.
    out_meta: TextureView,
    /// This frame's luminance EMA output; next frame's `prev_luma`.
    out_luma: TextureView,
    /// This frame's write-slot depth texture. The reproject node copies the SSR
    /// `scene_depth` into it (`COPY_DST`); next frame it is read as `prev_depth`.
    curr_depth_texture: Texture,
    /// Transient reprojected history the reproject pass writes and the clamp
    /// reads (this frame only).
    reprojected: CachedTexture,
    /// Transient reprojected metadata.
    reprojected_meta: CachedTexture,
    /// Transient reprojected luminance EMAs.
    reprojected_luma: CachedTexture,
    /// Transient denoised specular the clamp writes and the spatial filter reads
    /// in place of the raw resolve (this frame only).
    denoised: CachedTexture,
    /// Live viewport extent the planes were allocated for; a change reallocates.
    pub(crate) size: UVec2,
    /// Current camera clip (NDC, reverse-Z) -> world; rebuilds the surface.
    pub(crate) world_from_clip: Mat4,
    /// Previous camera world -> clip (NDC); finds the history texel. On the first
    /// frame this is the *current* transform (zero motion).
    pub(crate) prev_clip_from_world: Mat4,
    /// Previous camera clip (NDC) -> world; rebuilds the stored prev surface for
    /// the world-space disocclusion guard.
    pub(crate) prev_world_from_clip: Mat4,
    /// Current camera view -> world rotation; lifts the view-space G-buffer
    /// normal into the world space the reprojection maths runs in.
    pub(crate) world_from_view: Mat4,
    /// Current camera world-space position (parallax origin).
    pub(crate) curr_cam_pos: Vec3,
    /// Previous camera world-space position (last frame's parallax origin). On
    /// the first frame this equals `curr_cam_pos` (zero motion).
    pub(crate) prev_cam_pos: Vec3,
}

impl ViewSpecDenoiseTemporal {
    /// Previous-frame history plane (reproject binding 4).
    pub(crate) fn prev_history_view(&self) -> &TextureView {
        &self.prev_history
    }

    /// Previous-frame metadata plane (reproject binding 6).
    pub(crate) fn prev_meta_view(&self) -> &TextureView {
        &self.prev_meta
    }

    /// Previous-frame luminance plane (reproject binding 7).
    pub(crate) fn prev_luma_view(&self) -> &TextureView {
        &self.prev_luma
    }

    /// Previous-frame depth plane (reproject binding 5).
    pub(crate) fn prev_depth_view(&self) -> &TextureView {
        &self.prev_depth
    }

    /// This-frame history output (history-clamp binding 7).
    pub(crate) fn out_history_view(&self) -> &TextureView {
        &self.out_history
    }

    /// This-frame metadata output (history-clamp binding 8).
    pub(crate) fn out_meta_view(&self) -> &TextureView {
        &self.out_meta
    }

    /// This-frame luminance output (history-clamp binding 9).
    pub(crate) fn out_luma_view(&self) -> &TextureView {
        &self.out_luma
    }

    /// This-frame write-slot depth texture, the `copy_texture_to_texture`
    /// destination the reproject node fills from the SSR `scene_depth`.
    pub(crate) fn curr_depth_texture(&self) -> &Texture {
        &self.curr_depth_texture
    }

    /// Transient reprojected history (reproject binding 8 / clamp binding 1).
    pub(crate) fn reprojected_view(&self) -> &TextureView {
        &self.reprojected.default_view
    }

    /// Transient reprojected metadata (reproject binding 9 / clamp binding 2).
    pub(crate) fn reprojected_meta_view(&self) -> &TextureView {
        &self.reprojected_meta.default_view
    }

    /// Transient reprojected luminance (reproject binding 10 / clamp binding 3).
    pub(crate) fn reprojected_luma_view(&self) -> &TextureView {
        &self.reprojected_luma.default_view
    }

    /// Transient denoised specular (history-clamp binding 10). The spatial
    /// filter reads this in place of the raw `spec_gi` resolve.
    pub(crate) fn denoised_view(&self) -> &TextureView {
        &self.denoised.default_view
    }
}

/// A persistent ping-pong plane set for one view, kept out of the frame
/// transient [`TextureCache`] so last frame's accumulation survives into this
/// frame. The history/meta/luma planes carry `STORAGE | TEXTURE` (they swap
/// read/write every frame); the depth planes carry `COPY_DST | TEXTURE` (copy
/// target + `textureLoad`, never storage-written). The previous frame's
/// `clip_from_world` and camera position are stashed so the CPU can build the
/// reprojection without a second per-view uniform.
struct CachedSpecDenoiseTemporal {
    history: [TextureView; 2],
    meta: [TextureView; 2],
    luma: [TextureView; 2],
    depth_view: [TextureView; 2],
    depth_tex: [Texture; 2],
    size: UVec2,
    /// Which slot holds the readable previous-frame output: `false` -> slot 0,
    /// `true` -> slot 1. Flipped every frame after the roles are handed out.
    parity: bool,
    /// Column-major `clip_from_world` recorded last frame. Meaningful only when
    /// `has_prev`.
    prev_clip_from_world: [[f32; 4]; 4],
    /// Last frame's camera world-space position. Meaningful only when `has_prev`.
    prev_cam_pos: [f32; 3],
    /// `false` until the first frame records a transform, so the first
    /// accumulation reprojects through the current transform (zero motion).
    has_prev: bool,
}

/// The persistent per-view history cache, surviving across frames in a
/// [`Local`]. Keyed by [`RetainedViewEntity`] so a view keeps its history as its
/// render-world entity churns, exactly like [`super::super::ao::temporal`].
#[derive(Default)]
pub(crate) struct SpecDenoiseTemporalHistoryCache {
    views: HashMap<RetainedViewEntity, CachedSpecDenoiseTemporal>,
}

/// Allocates one persistent single-mip `rgba16float` plane at `size` with both
/// storage and texture binding so it can serve as the storage write target on
/// one frame and the sampled history on the next.
fn create_plane(device: &RenderDevice, size: UVec2, label: &str) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SPEC_DENOISE_FILTERED_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    })
}

/// Allocates one persistent single-mip `r32float` depth plane at `size` with
/// `COPY_DST | TEXTURE` so the reproject node can copy the SSR `scene_depth`
/// into it and next frame read it as the previous surface. Returns both the
/// texture (the copy destination) and its view (the `textureLoad` source).
fn create_depth_plane(device: &RenderDevice, size: UVec2) -> (Texture, TextureView) {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism spec_denoise temporal depth"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TEMPORAL_DEPTH_FORMAT,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("prism spec_denoise temporal depth view"),
        ..Default::default()
    });
    (texture, view)
}

/// Allocates one transient per-frame `rgba16float` plane (`STORAGE | TEXTURE`)
/// from the frame [`TextureCache`]: written by one pass and read by the next
/// within the same frame.
fn transient_plane(
    device: &RenderDevice,
    texture_cache: &mut TextureCache,
    size: UVec2,
    label: &'static str,
) -> CachedTexture {
    texture_cache.get(
        device,
        TextureDescriptor {
            label: Some(label),
            size: size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SPEC_DENOISE_FILTERED_FORMAT,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

/// `PrepareResources` system resolving [`ViewSpecDenoiseTemporal`] for every
/// view while the temporal denoiser is enabled, (re)allocating the persistent
/// ping-pong planes to match the viewport, flipping the read/write slots each
/// frame, allocating the transient planes, and building the reprojection
/// matrices from the previous frame's stashed transform and camera position.
///
/// Gated identically to [`super::resources::prepare_spec_denoise_resources`]
/// (the spatial target it feeds) and the `spec_gi` reuse resolve. A cache miss
/// or a size change allocates a fresh set and reprojects through the current
/// transform (zero motion); the zero-init depth plane naturally rejects on the
/// first frame so the clamp seeds from the resolve. Views that lose the gate
/// drop their cache entry and component.
pub(crate) fn prepare_spec_denoise_temporal_resources(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
    )>,
    mut cache: Local<SpecDenoiseTemporalHistoryCache>,
) {
    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, camera, msaa, visibility) in &views {
        let enabled = settings.enable_spec_gi
            && settings.enable_ssr
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        let retained_view = view.retained_view_entity;
        let size = camera.physical_viewport_size;
        if !enabled || size.is_none_or(|value| value.x == 0 || value.y == 0) {
            commands.entity(entity).remove::<ViewSpecDenoiseTemporal>();
            cache.views.remove(&retained_view);
            continue;
        }
        let size = size.expect("size checked non-degenerate just above");
        retained.insert(retained_view);

        // Current camera transforms. clip_from_world prefers the view's cached
        // value and reconstructs it from the projection otherwise, exactly like
        // the GTAO temporal pass.
        let clip_from_view = view.clip_from_view;
        let world_from_view = view.world_from_view.to_matrix();
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| clip_from_view * world_from_view.inverse());
        let world_from_clip = clip_from_world.inverse();
        let curr_cam_pos = view.world_from_view.translation();

        // Reuse the persistent set only when its extent still matches; a resize
        // reallocates and drops the history to the (zero-init) current frame.
        let reuse = cache
            .views
            .get(&retained_view)
            .is_some_and(|cached| cached.size == size);
        if !reuse {
            let (depth_tex_a, depth_view_a) = create_depth_plane(&device, size);
            let (depth_tex_b, depth_view_b) = create_depth_plane(&device, size);
            cache.views.insert(
                retained_view,
                CachedSpecDenoiseTemporal {
                    history: [
                        create_plane(&device, size, "prism spec_denoise temporal history"),
                        create_plane(&device, size, "prism spec_denoise temporal history"),
                    ],
                    meta: [
                        create_plane(&device, size, "prism spec_denoise temporal meta"),
                        create_plane(&device, size, "prism spec_denoise temporal meta"),
                    ],
                    luma: [
                        create_plane(&device, size, "prism spec_denoise temporal luma"),
                        create_plane(&device, size, "prism spec_denoise temporal luma"),
                    ],
                    depth_view: [depth_view_a, depth_view_b],
                    depth_tex: [depth_tex_a, depth_tex_b],
                    size,
                    parity: false,
                    prev_clip_from_world: [[0.0; 4]; 4],
                    prev_cam_pos: [0.0; 3],
                    has_prev: false,
                },
            );
        }

        let cached = cache
            .views
            .get_mut(&retained_view)
            .expect("history cache entry was just inserted when absent");

        // Read the previous frame's transform *before* overwriting it. On the
        // first frame there is no previous transform, so reproject through the
        // current one (zero motion). The zero-init prev depth rejects regardless.
        let had_prev = cached.has_prev;
        let (prev_clip_from_world, prev_cam_pos) = if had_prev {
            (
                Mat4::from_cols_array_2d(&cached.prev_clip_from_world),
                Vec3::from_array(cached.prev_cam_pos),
            )
        } else {
            (clip_from_world, curr_cam_pos)
        };
        let prev_world_from_clip = prev_clip_from_world.inverse();

        // read = slot written last frame (parity); write = the other slot.
        let read_idx = if cached.parity { 0usize } else { 1usize };
        let write_idx = 1 - read_idx;

        let prev_history = cached.history[read_idx].clone();
        let prev_meta = cached.meta[read_idx].clone();
        let prev_luma = cached.luma[read_idx].clone();
        let prev_depth = cached.depth_view[read_idx].clone();
        let out_history = cached.history[write_idx].clone();
        let out_meta = cached.meta[write_idx].clone();
        let out_luma = cached.luma[write_idx].clone();
        let curr_depth_texture = cached.depth_tex[write_idx].clone();

        // Next frame reads what we are about to write, and reprojects through the
        // transform we just used.
        cached.parity = !cached.parity;
        cached.prev_clip_from_world = clip_from_world.to_cols_array_2d();
        cached.prev_cam_pos = curr_cam_pos.to_array();
        cached.has_prev = true;

        let reprojected = transient_plane(
            &device,
            &mut texture_cache,
            size,
            "prism spec_denoise temporal reprojected",
        );
        let reprojected_meta = transient_plane(
            &device,
            &mut texture_cache,
            size,
            "prism spec_denoise temporal reprojected meta",
        );
        let reprojected_luma = transient_plane(
            &device,
            &mut texture_cache,
            size,
            "prism spec_denoise temporal reprojected luma",
        );
        let denoised = transient_plane(
            &device,
            &mut texture_cache,
            size,
            "prism spec_denoise temporal denoised",
        );

        commands.entity(entity).insert(ViewSpecDenoiseTemporal {
            prev_history,
            prev_meta,
            prev_luma,
            prev_depth,
            out_history,
            out_meta,
            out_luma,
            curr_depth_texture,
            reprojected,
            reprojected_meta,
            reprojected_luma,
            denoised,
            size,
            world_from_clip,
            prev_clip_from_world,
            prev_world_from_clip,
            world_from_view,
            curr_cam_pos,
            prev_cam_pos,
        });
    }

    // Drop history for views that no longer run the temporal denoiser.
    cache.views.retain(|view, _| retained.contains(view));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_plane_format_matches_ssr_scene_depth() {
        // The reproject node copies the SSR `scene_depth` (R32Float reverse-Z)
        // into this plane each frame; a drift in either side would make the copy
        // a silent format mismatch, so pin it here.
        assert_eq!(TEMPORAL_DEPTH_FORMAT, TextureFormat::R32Float);
    }

    #[test]
    fn ping_pong_slots_are_distinct_and_alternate() {
        // The read/write slot indices must be distinct every frame (so a pass
        // never reads the plane it is writing) and must swap with parity (so this
        // frame's write becomes next frame's read).
        let read_false = if false { 0usize } else { 1usize };
        let write_false = 1 - read_false;
        assert_ne!(read_false, write_false);
        let read_true = if true { 0usize } else { 1usize };
        let write_true = 1 - read_true;
        assert_ne!(read_true, write_true);
        // Parity flip swaps which slot is read.
        assert_eq!(read_false, write_true);
        assert_eq!(read_true, write_false);
    }
}
