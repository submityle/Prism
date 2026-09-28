//! Persistent per-view TAA history ping-pong.
//!
//! The temporal resolve reads last frame's accumulated colour and writes this
//! frame's, so the history cannot live in the frame-transient [`TextureCache`]
//! the visibility targets use — that pool is recycled every frame. This module
//! keeps a persistent **ping-pong** pair of `rgba16float` textures per view in a
//! [`Local`] cache keyed by [`RetainedViewEntity`], mirroring
//! [`super::super::ssr`]'s temporal history and
//! [`crate::visibility::hzb::prepare_hzb_history`]. Each frame reads the slot
//! written last frame and writes the other; the shading composite then reads the
//! write slot in place of the raw `scene_color`.

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
    view::{ExtractedView, RetainedViewEntity},
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};

/// Per-view TAA state resolved each frame from the persistent ping-pong cache:
/// the readable previous-frame history, the writable current-frame output (which
/// the shading composite reads in place of `scene_color`), and whether that
/// history is trustworthy this frame.
#[derive(Component)]
pub(crate) struct ViewTaa {
    /// Previous frame's accumulated colour, sampled at the reprojected UV.
    read_view: TextureView,
    /// This frame's resolved output. Written by the resolve pass (storage) and
    /// read by the shading composite (sampled) in place of `scene_color`.
    write_view: TextureView,
    /// `false` on the first frame, a resize, or a fresh allocation, so the
    /// shader ignores the (garbage) history and passes the current frame through.
    valid: bool,
}

impl ViewTaa {
    /// Previous-frame accumulated colour bound as the sampled history.
    pub(crate) fn read_view(&self) -> &TextureView {
        &self.read_view
    }

    /// Current-frame resolved output bound as the storage write target (and read
    /// by the shading composite).
    pub(crate) fn write_view(&self) -> &TextureView {
        &self.write_view
    }

    /// Whether the ping-pong history is trustworthy this frame.
    pub(crate) fn valid(&self) -> bool {
        self.valid
    }
}

/// A persistent ping-pong history pair for one view. Both slots carry
/// `STORAGE_BINDING | TEXTURE_BINDING` because they swap read/write roles every
/// frame.
struct CachedTaa {
    view_a: TextureView,
    view_b: TextureView,
    size: UVec2,
    /// Which slot holds the readable previous-frame output: `false` -> A,
    /// `true` -> B. Flipped every frame after the roles are handed out.
    parity: bool,
}

/// The persistent per-view history cache, surviving across frames in a
/// [`Local`]. Keyed by [`RetainedViewEntity`] so a view keeps its history as its
/// render-world entity churns.
#[derive(Default)]
pub(crate) struct TaaHistoryCache {
    views: HashMap<RetainedViewEntity, CachedTaa>,
}

/// Allocates one persistent single-mip `rgba16float` history slot at `size`,
/// with both storage and texture binding so it can serve as the write target on
/// one frame and the sampled history on the next.
fn create_history(device: &RenderDevice, size: UVec2) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism TAA history"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SCENE_COLOR_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some("prism TAA history view"),
        ..Default::default()
    })
}

/// `PrepareResources` system resolving [`ViewTaa`] for every view with a
/// resident [`ViewVisibilityBuffer`], (re)allocating the persistent ping-pong
/// history to match the viewport and flipping the read/write slots each frame.
///
/// A cache miss or a size change allocates a fresh pair and marks the history
/// invalid (the shader passes the current frame through); a hit hands out last
/// frame's write slot as the readable history (the GPU reprojects it via the
/// motion buffer). Views that lost their visibility buffer drop their cache
/// entry. Skipped entirely when TAA is disabled.
pub(crate) fn prepare_taa_textures(
    mut commands: Commands,
    settings: Res<super::super::runtime::PrismShadingSettings>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView, &ViewVisibilityBuffer)>,
    mut cache: Local<TaaHistoryCache>,
) {
    if !settings.enable_taa {
        return;
    }

    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, visibility) in &views {
        let retained_view = view.retained_view_entity;
        let size = visibility.size;
        if size.x == 0 || size.y == 0 {
            commands.entity(entity).remove::<ViewTaa>();
            cache.views.remove(&retained_view);
            continue;
        }
        retained.insert(retained_view);

        // Reuse the persistent pair only when its extent still matches; a resize
        // reallocates and drops the history to the (invalid) current frame.
        let reuse = cache
            .views
            .get(&retained_view)
            .is_some_and(|cached| cached.size == size);
        if !reuse {
            cache.views.insert(
                retained_view,
                CachedTaa {
                    view_a: create_history(&device, size),
                    view_b: create_history(&device, size),
                    size,
                    parity: false,
                },
            );
        }

        let cached = cache
            .views
            .get_mut(&retained_view)
            .expect("history cache entry was just inserted when absent");

        // read = slot written last frame (parity); write = the other slot.
        let (read_view, write_view) = if cached.parity {
            (cached.view_a.clone(), cached.view_b.clone())
        } else {
            (cached.view_b.clone(), cached.view_a.clone())
        };

        // Next frame reads what we are about to write this frame.
        cached.parity = !cached.parity;

        commands.entity(entity).insert(ViewTaa {
            read_view,
            write_view,
            valid: reuse,
        });
    }

    // Drop history for views that no longer run TAA so their textures free.
    cache.views.retain(|view, _| retained.contains(view));
}
