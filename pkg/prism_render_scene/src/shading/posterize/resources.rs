//! Per-view GPU texture backing the posterize pass.
//!
//! The posterize runs as a single full-screen compute pass over the pre-exposed
//! linear `HDR` scene colour: `posterize_main` bands each texel by the golden
//! operator and writes the result to a dedicated output that the dispatch then
//! copies back over `scene_color`. This module owns that one cached texture:
//!
//! * `posterize_out` — the full-resolution posterized `HDR` output
//!   (`rgba16float`), copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::color_grade::resources`]. It is gated on
//! the posterize enable *and* on the presence of the visibility buffer (source
//! of the scene colour the pass reads and copies back over). Like colour grade,
//! posterize needs no depth, so it does not couple to the `SSR` prepass.

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
use super::settings::PrismPosterizeSettings;

/// The per-view posterize output texture, present only while the pass is
/// enabled and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewPosterize {
    /// Full-resolution posterized `HDR` output written by `posterize_main`
    /// (`rgba16float`), copied back over `scene_color`.
    posterize_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewPosterize {
    /// Storage view of the posterized `HDR` output written by `posterize_main`.
    pub(crate) fn posterize_out_view(&self) -> &TextureView {
        &self.posterize_out.default_view
    }

    /// The posterized-output GPU texture itself, for the
    /// `copy_texture_to_texture` that writes it back over `scene_color`.
    pub(crate) fn posterize_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.posterize_out.texture
    }
}

/// (Re)allocates [`ViewPosterize`] for every view that has a resident
/// visibility buffer while the posterize is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and on single-sample views (the scene colour
/// buffer is itself single-sample). The texture is re-created whenever the
/// viewport size changes, exactly like the buffer it shadows.
pub(crate) fn prepare_posterize_textures(
    mut commands: Commands,
    settings: Res<PrismPosterizeSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewPosterize>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewPosterize>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Posterized output: full-resolution wide `HDR`, written by
        // `posterize_main` (storage) and copied back over `scene_color`.
        let posterize_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism posterize output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `posterize_main`. COPY_SRC: copied
                // back over `scene_color` after the pass so the downstream post
                // chain reads the posterized image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewPosterize {
            posterize_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterize_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
