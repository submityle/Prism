//! Per-view GPU textures backing the motion-blur passes.
//!
//! Motion blur runs as a three-step compute chain over the composited HDR
//! scene colour: `TileMax` reduces each screen tile to its longest velocity,
//! `NeighborMax` dilates that field by one ring, and the reconstruction gathers
//! along the dilated dominant velocity. This module owns the three cached
//! textures those steps read and write:
//!
//! 1. `tile_max` — the per-tile longest velocity (`rg16float`), one texel per
//!    `MOTION_BLUR_TILE_SIZE`-square tile, so sized `ceil(dim / tile)`;
//! 2. `neighbor` — the 3x3-dilated tile field (`rg16float`), same tile extent;
//!    and
//! 3. `blur_out` — the full-resolution blurred HDR output (`rgba16float`).
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates them to match the
//! viewport, mirroring [`super::super::ssr::resources::prepare_ssr_textures`].
//! It is gated on the motion-blur enable *and* on the presence of both the
//! visibility buffer (source of the resolved motion-vector G-buffer the tiles
//! reduce) and the SSR geometry prepass (source of the reverse-Z device depth
//! the reconstruction unprojects for its soft-depth term). That SSR coupling
//! mirrors how the VSM receiver pass leans on the same SSR depth: motion blur
//! reuses the geometry prepass rather than duplicating a depth pass.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages, TextureView},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::ssr::ViewSsrTextures;
use super::abi::MOTION_BLUR_TILE_SIZE;
use super::pipeline::MOTION_BLUR_TILE_FORMAT;
use super::settings::PrismMotionBlurSettings;

/// The three per-view motion-blur textures, present only while motion blur is
/// enabled and the backing visibility + SSR buffers are resident.
#[derive(Component)]
pub(crate) struct ViewMotionBlur {
    /// Per-tile longest shuttered/clamped velocity written by `tile_max`
    /// (`rg16float`, one texel per tile).
    tile_max: CachedTexture,
    /// 3x3-dilated dominant velocity written by `neighbor_max` (`rg16float`,
    /// one texel per tile), read by `reconstruct`.
    neighbor: CachedTexture,
    /// Full-resolution blurred HDR output written by `reconstruct`
    /// (`rgba16float`), matching the scene-colour format.
    blur_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
    /// Tile-grid extent (`ceil(size / MOTION_BLUR_TILE_SIZE)`), the extent of
    /// the two tile textures.
    pub(crate) tiles: UVec2,
}

impl ViewMotionBlur {
    /// Storage/sampling view of the `TileMax` output. `tile_max` writes it
    /// (storage) and `neighbor_max` reads it (sampled).
    pub(crate) fn tile_max_view(&self) -> &TextureView {
        &self.tile_max.default_view
    }

    /// Storage/sampling view of the `NeighborMax` output. `neighbor_max` writes
    /// it (storage) and `reconstruct` reads it (sampled).
    pub(crate) fn neighbor_view(&self) -> &TextureView {
        &self.neighbor.default_view
    }

    /// Storage view of the full-resolution blurred HDR output written by
    /// `reconstruct`.
    pub(crate) fn blur_out_view(&self) -> &TextureView {
        &self.blur_out.default_view
    }

    /// The blurred-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn blur_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.blur_out.texture
    }
}

/// Tile-grid extent for a framebuffer of `size`: `ceil(dim / tile)` per axis,
/// floored to a single tile so a degenerate zero extent still allocates a valid
/// 1x1 texture.
fn tile_extent(size: UVec2) -> UVec2 {
    UVec2::new(
        size.x.div_ceil(MOTION_BLUR_TILE_SIZE).max(1),
        size.y.div_ceil(MOTION_BLUR_TILE_SIZE).max(1),
    )
}

/// (Re)allocates [`ViewMotionBlur`] for every view that has both a resident
/// visibility buffer and SSR geometry prepass while motion blur is enabled, and
/// removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the motion-vector G-buffer the tiles reduce) and [`ViewSsrTextures`] (the
/// device depth the reconstruction unprojects), and on single-sample views
/// because both backing buffers are themselves single-sample. The textures are
/// re-created whenever the viewport size changes, exactly like the buffers they
/// shadow.
pub(crate) fn prepare_motion_blur_textures(
    mut commands: Commands,
    settings: Res<PrismMotionBlurSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
        Option<&ViewMotionBlur>,
    )>,
) {
    for (entity, camera, msaa, visibility, ssr, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && ssr.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewMotionBlur>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        let tiles = tile_extent(size);

        // TileMax output: one texel per tile, written by `tile_max` (storage)
        // and read by `neighbor_max` (sampled).
        let tile_max = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism motion blur tile max"),
                size: tiles.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MOTION_BLUR_TILE_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // NeighborMax output: same tile extent, written by `neighbor_max`
        // (storage) and read by `reconstruct` (sampled).
        let neighbor = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism motion blur neighbor max"),
                size: tiles.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MOTION_BLUR_TILE_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Blurred output: full-resolution wide HDR, written by `reconstruct`
        // (storage) and sampled downstream by the composite.
        let blur_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism motion blur output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `reconstruct`. COPY_SRC: copied
                // back over `scene_color` after the chain so the downstream
                // composite reads the blurred image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewMotionBlur {
            tile_max,
            neighbor,
            blur_out,
            size,
            tiles,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_extent_rounds_up_and_floors_to_one() {
        // 1920 / 16 = 120 exact; 1080 -> ceil(1080/16) = 68.
        assert_eq!(tile_extent(UVec2::new(1920, 1080)), UVec2::new(120, 68));
        // Partial tiles round up.
        assert_eq!(tile_extent(UVec2::new(17, 1)), UVec2::new(2, 1));
        // A degenerate zero extent still allocates a single tile.
        assert_eq!(tile_extent(UVec2::ZERO), UVec2::new(1, 1));
    }

    #[test]
    fn blur_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
