//! Per-view GPU texture backing the ordered-dither pass.
//!
//! The dither runs as a single full-screen compute pass over the pre-exposed
//! HDR scene colour: `ordered_dither_main` posterises each texel by the golden
//! Bayer-threshold operator and writes the result to a dedicated output that the
//! dispatch then copies back over `scene_color`. This module owns that one
//! cached texture:
//!
//! * `ordered_dither_out` — the full-resolution dithered HDR output
//!   (`rgba16float`), copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::gamut_map::resources`]. It is gated on
//! the dither enable *and* on the presence of the visibility buffer (source of
//! the pre-exposed scene colour the pass reads and copies back over). Like the
//! gamut map, the dither needs no depth, so it does not couple to the SSR
//! prepass.

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
use super::settings::PrismOrderedDitherSettings;

/// The per-view ordered-dither output texture, present only while the pass is
/// enabled and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewOrderedDither {
    /// Full-resolution dithered HDR output written by `ordered_dither_main`
    /// (`rgba16float`), copied back over `scene_color`.
    ordered_dither_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewOrderedDither {
    /// Storage view of the dithered HDR output written by
    /// `ordered_dither_main`.
    pub(crate) fn ordered_dither_out_view(&self) -> &TextureView {
        &self.ordered_dither_out.default_view
    }

    /// The dithered-output GPU texture itself, for the
    /// `copy_texture_to_texture` that writes it back over `scene_color`.
    pub(crate) fn ordered_dither_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.ordered_dither_out.texture
    }
}

/// (Re)allocates [`ViewOrderedDither`] for every view that has a resident
/// visibility buffer while the dither is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and on single-sample views (the scene colour
/// buffer is itself single-sample). The texture is re-created whenever the
/// viewport size changes, exactly like the buffer it shadows.
pub(crate) fn prepare_ordered_dither_textures(
    mut commands: Commands,
    settings: Res<PrismOrderedDitherSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewOrderedDither>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewOrderedDither>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Dithered output: full-resolution wide HDR, written by
        // `ordered_dither_main` (storage) and copied back over `scene_color`.
        let ordered_dither_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism ordered dither output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `ordered_dither_main`. COPY_SRC:
                // copied back over `scene_color` after the pass so the
                // downstream post chain reads the dithered image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ViewOrderedDither { ordered_dither_out, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_dither_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
