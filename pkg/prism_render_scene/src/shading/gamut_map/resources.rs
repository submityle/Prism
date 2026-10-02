//! Per-view GPU texture backing the gamut-map pass.
//!
//! The map runs as a single full-screen compute pass over the pre-exposed HDR
//! scene colour: `gamut_map_main` compresses each texel by the golden operator
//! chain and writes the result to a dedicated output that the dispatch then
//! copies back over `scene_color`. This module owns that one cached texture:
//!
//! * `gamut_map_out` — the full-resolution compressed HDR output
//!   (`rgba16float`), copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::color_grade::resources`]. It is gated on
//! the map enable *and* on the presence of the visibility buffer (source of the
//! pre-exposed scene colour the pass reads and copies back over). Like colour
//! grade, the map needs no depth, so it does not couple to the SSR prepass.

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
use super::settings::PrismGamutMapSettings;

/// The per-view gamut-map output texture, present only while the pass is enabled
/// and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewGamutMap {
    /// Full-resolution compressed HDR output written by `gamut_map_main`
    /// (`rgba16float`), copied back over `scene_color`.
    gamut_map_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewGamutMap {
    /// Storage view of the compressed HDR output written by `gamut_map_main`.
    pub(crate) fn gamut_map_out_view(&self) -> &TextureView {
        &self.gamut_map_out.default_view
    }

    /// The compressed-output GPU texture itself, for the
    /// `copy_texture_to_texture` that writes it back over `scene_color`.
    pub(crate) fn gamut_map_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.gamut_map_out.texture
    }
}

/// (Re)allocates [`ViewGamutMap`] for every view that has a resident visibility
/// buffer while the map is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and on single-sample views (the scene colour
/// buffer is itself single-sample). The texture is re-created whenever the
/// viewport size changes, exactly like the buffer it shadows.
pub(crate) fn prepare_gamut_map_textures(
    mut commands: Commands,
    settings: Res<PrismGamutMapSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewGamutMap>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewGamutMap>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Compressed output: full-resolution wide HDR, written by
        // `gamut_map_main` (storage) and copied back over `scene_color`.
        let gamut_map_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism gamut map output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `gamut_map_main`. COPY_SRC: copied
                // back over `scene_color` after the pass so the downstream post
                // chain reads the compressed image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewGamutMap {
            gamut_map_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamut_map_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
