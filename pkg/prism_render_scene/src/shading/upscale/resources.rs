//! Persistent per-view temporal-upscale history and this frame's resolved
//! [`ViewUpscale`] slots.
//!
//! Temporal upscaling accumulates across frames, so like [`super::super::taa`]
//! its history cannot live in the frame-transient texture pool. This module
//! keeps three persistent **ping-pong** pairs per view, all at the *display*
//! resolution, in a [`Local`] cache keyed by [`RetainedViewEntity`]:
//!
//! * `history_color` (`rgba16float`, `STORAGE | TEXTURE`) — the previous
//!   accumulated (pre-sharpen) colour, sampled at the reprojected UV;
//! * `history_meta` (`rgba16float`, `STORAGE | TEXTURE`) — the per-pixel
//!   accumulation count (`.r`) and thin-feature lock lifetime (`.g`);
//! * `history_depth` (`r32float`, `STORAGE | TEXTURE`) — the previous frame's
//!   depth at display resolution, which the reconstruction persists itself
//!   (its `depth_out`) so next frame's disocclusion test has a display-grid
//!   depth to compare against.
//!
//! Each frame the read slot is the pair written last frame and the write slot
//! is the other; the parity flips after the roles are handed out. A cache miss
//! or any extent change (re)allocates and marks the history invalid so the
//! reconstruction passes the freshly resolved current frame through instead of
//! ghosting.
//!
//! Alongside the history sits a per-frame display-resolution `upscale_out`
//! (`rgba16float`, `STORAGE | TEXTURE | COPY_SRC`): the RCAS pass writes the
//! finished display image here. It carries `COPY_SRC` so the graph wiring that
//! lands the `render_scale` render-target integration can blit it to the
//! swapchain-sized target; this self-contained slice produces it and leaves the
//! downstream routing to that later wiring.
//!
//! The render (low-resolution draw) extent is `ceil(display * render_scale)`,
//! so at the native default (`render_scale == 1.0`) render and display coincide
//! and the pass is a full-resolution temporal resolve + sharpen; below `1.0` it
//! reconstructs the display grid from the smaller render target the wiring
//! sizes. Gated on the [`UpscaleSettings`] resource being present together with
//! a resident [`ViewVisibilityBuffer`] (the `render_color` + `motion_vectors`)
//! and [`ViewSsrTextures`] (the reverse-Z device depth), on single-sample views.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        Texture, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::RenderDevice,
    view::{ExtractedView, Msaa, RetainedViewEntity},
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::ssr::ViewSsrTextures;
use super::pipeline::UPSCALE_DEPTH_FORMAT;
use super::settings::UpscaleSettings;

/// Per-view temporal-upscale state resolved each frame from the persistent
/// ping-pong cache: the render/display extents, the three history read/write
/// slot pairs, the display-resolution `upscale_out` the RCAS pass writes, and
/// whether the history is trustworthy this frame.
#[derive(Component)]
pub(crate) struct ViewUpscale {
    /// Low-resolution render (draw) extent, `ceil(display * render_scale)`.
    render_size: UVec2,
    /// Display (output / history) extent.
    display_size: UVec2,
    /// Previous-frame accumulated colour, sampled at the reprojected UV.
    history_color_read: TextureView,
    /// This frame's reconstructed colour (reconstruction `color_out`), then read
    /// by RCAS as its `input_color` and kept as next frame's `history_color`.
    history_color_write: TextureView,
    /// Previous-frame accumulation (`.r`) + lock lifetime (`.g`).
    history_meta_read: TextureView,
    /// This frame's accumulation + lock (reconstruction `meta_out`).
    history_meta_write: TextureView,
    /// Previous-frame display-resolution depth, tested for disocclusion.
    history_depth_read: TextureView,
    /// This frame's display-resolution depth (reconstruction `depth_out`).
    history_depth_write: TextureView,
    /// The finished display image written by RCAS (`rgba16float`, `COPY_SRC`).
    upscale_out_view: TextureView,
    /// The `upscale_out` GPU texture, for the `copy_texture_to_texture` the
    /// `render_scale` graph wiring blits to the display target.
    upscale_out_texture: Texture,
    /// `false` on the first frame, a resize, or a fresh allocation, so the
    /// reconstruction ignores the (garbage) history.
    valid: bool,
}

impl ViewUpscale {
    /// Low-resolution render (draw) extent as `(width, height)`.
    pub(crate) fn render_extent(&self) -> (u32, u32) {
        (self.render_size.x, self.render_size.y)
    }

    /// Display (output / history) extent as `(width, height)`.
    pub(crate) fn display_extent(&self) -> (u32, u32) {
        (self.display_size.x, self.display_size.y)
    }

    /// Display extent, for deriving the dispatch workgroup counts.
    pub(crate) fn display_size(&self) -> UVec2 {
        self.display_size
    }

    /// Previous-frame accumulated colour bound as the sampled history.
    pub(crate) fn history_color_read(&self) -> &TextureView {
        &self.history_color_read
    }

    /// This frame's reconstructed colour: the reconstruction `color_out` (and
    /// RCAS `input_color`, and next frame's `history_color`).
    pub(crate) fn history_color_write(&self) -> &TextureView {
        &self.history_color_write
    }

    /// Previous-frame accumulation/lock metadata bound for the disocclusion
    /// and lock tests.
    pub(crate) fn history_meta_read(&self) -> &TextureView {
        &self.history_meta_read
    }

    /// This frame's accumulation/lock metadata: the reconstruction `meta_out`.
    pub(crate) fn history_meta_write(&self) -> &TextureView {
        &self.history_meta_write
    }

    /// Previous-frame display-resolution depth bound for the disocclusion test.
    pub(crate) fn history_depth_read(&self) -> &TextureView {
        &self.history_depth_read
    }

    /// This frame's display-resolution depth: the reconstruction `depth_out`.
    pub(crate) fn history_depth_write(&self) -> &TextureView {
        &self.history_depth_write
    }

    /// The finished display image written by RCAS.
    pub(crate) fn upscale_out_view(&self) -> &TextureView {
        &self.upscale_out_view
    }

    /// The `upscale_out` GPU texture, for the downstream blit to the display
    /// target. The native-resolution wired path composites `upscale_out_view`
    /// directly (display and render extents match), so the blit only exists on
    /// the sub-resolution (`render_scale < 1`) follow-up, which is
    /// GPU-validation-gated.
    #[expect(
        dead_code,
        reason = "reserved for the sub-resolution blit to the display target (GPU-validation-gated follow-up)"
    )]
    pub(crate) fn upscale_out_texture(&self) -> &Texture {
        &self.upscale_out_texture
    }

    /// Whether the ping-pong history is trustworthy this frame.
    pub(crate) fn valid(&self) -> bool {
        self.valid
    }
}

/// The three persistent display-resolution ping-pong pairs for one view, plus
/// the per-frame `upscale_out`. All history slots carry `STORAGE | TEXTURE`
/// because they swap read/write roles every frame.
struct CachedUpscale {
    color_a: TextureView,
    color_b: TextureView,
    meta_a: TextureView,
    meta_b: TextureView,
    depth_a: TextureView,
    depth_b: TextureView,
    out_texture: Texture,
    out_view: TextureView,
    display_size: UVec2,
    render_size: UVec2,
    /// Which slot holds the readable previous-frame output: `false` -> A,
    /// `true` -> B. Flipped every frame after the roles are handed out.
    parity: bool,
}

/// The persistent per-view history cache, surviving across frames in a
/// [`Local`]. Keyed by [`RetainedViewEntity`] so a view keeps its history as
/// its render-world entity churns.
#[derive(Default)]
pub(crate) struct UpscaleHistoryCache {
    views: HashMap<RetainedViewEntity, CachedUpscale>,
}

/// Allocates one persistent single-mip storage+texture history slot of `format`
/// at `size`.
fn create_slot(
    device: &RenderDevice,
    size: UVec2,
    format: bevy_render::render_resource::TextureFormat,
    label: &str,
) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    })
}

/// Allocates the per-frame display-resolution `upscale_out` (`rgba16float`,
/// `STORAGE | TEXTURE | COPY_SRC`), returning the texture (for the downstream
/// blit) and its view (for the RCAS bind group).
fn create_output(device: &RenderDevice, size: UVec2) -> (Texture, TextureView) {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism upscale output"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SCENE_COLOR_FORMAT,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("prism upscale output view"),
        ..Default::default()
    });
    (texture, view)
}

/// `PrepareResources` system resolving [`ViewUpscale`] for every view with a
/// resident [`ViewVisibilityBuffer`] and [`ViewSsrTextures`] while the
/// [`UpscaleSettings`] resource is present, (re)allocating the three persistent
/// display-resolution ping-pong pairs plus `upscale_out` to match the viewport
/// and flipping the read/write slots each frame.
///
/// A cache miss or any extent change allocates fresh slots and marks the
/// history invalid (the reconstruction passes the current frame through); a hit
/// hands out last frame's write slots as the readable history. Views that lost
/// a backing buffer, went multi-sample, or that run without the settings drop
/// their [`ViewUpscale`] and cache entry.
pub(crate) fn prepare_upscale_textures(
    mut commands: Commands,
    settings: Option<Res<UpscaleSettings>>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
    )>,
    mut cache: Local<UpscaleHistoryCache>,
) {
    if settings.is_none() {
        // No settings resource: the feature is not wired this run. Drop any
        // stale per-view state and history so the textures free.
        for (entity, ..) in &views {
            commands.entity(entity).remove::<ViewUpscale>();
        }
        cache.views.clear();
        return;
    }

    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, msaa, visibility, ssr) in &views {
        let resident = visibility.is_some()
            && ssr.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        let Some(visibility) = visibility.filter(|_| resident) else {
            commands.entity(entity).remove::<ViewUpscale>();
            cache.views.remove(&view.retained_view_entity);
            continue;
        };

        let retained_view = view.retained_view_entity;
        let display_size = visibility.size;
        if display_size.x == 0 || display_size.y == 0 {
            commands.entity(entity).remove::<ViewUpscale>();
            cache.views.remove(&retained_view);
            continue;
        }
        // The reconstruction reads `scene_color` (allocated at the display
        // extent) as its `render_color`, so with a native-resolution draw the
        // render extent equals the display extent and the resolve is 1:1.
        // Driving `render_scale < 1` would require scaling the whole
        // visibility/compute chain and remapping the composite coverage test to
        // the render grid — a GPU-validation-gated follow-up — so the wired path
        // pins the render extent to the display extent here.
        let render_size = display_size;
        retained.insert(retained_view);

        // Reuse the persistent slots only when both extents still match; any
        // change reallocates and drops the history to the (invalid) current
        // frame.
        let reuse = cache.views.get(&retained_view).is_some_and(|cached| {
            cached.display_size == display_size && cached.render_size == render_size
        });
        if !reuse {
            let (out_texture, out_view) = create_output(&device, display_size);
            cache.views.insert(
                retained_view,
                CachedUpscale {
                    color_a: create_slot(
                        &device,
                        display_size,
                        SCENE_COLOR_FORMAT,
                        "prism upscale history colour",
                    ),
                    color_b: create_slot(
                        &device,
                        display_size,
                        SCENE_COLOR_FORMAT,
                        "prism upscale history colour",
                    ),
                    meta_a: create_slot(
                        &device,
                        display_size,
                        SCENE_COLOR_FORMAT,
                        "prism upscale history meta",
                    ),
                    meta_b: create_slot(
                        &device,
                        display_size,
                        SCENE_COLOR_FORMAT,
                        "prism upscale history meta",
                    ),
                    depth_a: create_slot(
                        &device,
                        display_size,
                        UPSCALE_DEPTH_FORMAT,
                        "prism upscale history depth",
                    ),
                    depth_b: create_slot(
                        &device,
                        display_size,
                        UPSCALE_DEPTH_FORMAT,
                        "prism upscale history depth",
                    ),
                    out_texture,
                    out_view,
                    display_size,
                    render_size,
                    parity: false,
                },
            );
        }

        let cached = cache
            .views
            .get_mut(&retained_view)
            .expect("history cache entry was just inserted when absent");

        // read = slots written last frame (parity); write = the other slots.
        let (
            history_color_read,
            history_color_write,
            history_meta_read,
            history_meta_write,
            history_depth_read,
            history_depth_write,
        ) = if cached.parity {
            (
                cached.color_a.clone(),
                cached.color_b.clone(),
                cached.meta_a.clone(),
                cached.meta_b.clone(),
                cached.depth_a.clone(),
                cached.depth_b.clone(),
            )
        } else {
            (
                cached.color_b.clone(),
                cached.color_a.clone(),
                cached.meta_b.clone(),
                cached.meta_a.clone(),
                cached.depth_b.clone(),
                cached.depth_a.clone(),
            )
        };

        // Next frame reads what we are about to write this frame.
        cached.parity = !cached.parity;

        commands.entity(entity).insert(ViewUpscale {
            render_size,
            display_size,
            history_color_read,
            history_color_write,
            history_meta_read,
            history_meta_write,
            history_depth_read,
            history_depth_write,
            upscale_out_view: cached.out_view.clone(),
            upscale_out_texture: cached.out_texture.clone(),
            valid: reuse,
        });
    }

    // Drop history for views that no longer run upscaling so their textures
    // free.
    cache.views.retain(|view, _| retained.contains(view));
}
