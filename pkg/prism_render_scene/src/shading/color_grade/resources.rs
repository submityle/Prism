//! Per-view GPU texture backing the colour-grade pass.
//!
//! The grade runs as a single full-screen compute pass over the pre-exposed HDR
//! scene colour: `color_grade_main` grades each texel by the golden operator
//! chain and writes the result to a dedicated output that the dispatch then
//! copies back over `scene_color`. This module owns that one cached texture:
//!
//! * `color_grade_out` — the full-resolution graded HDR output (`rgba16float`),
//!   copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::vignette::resources`]. It is gated on the
//! grade enable *and* on the presence of the visibility buffer (source of the
//! pre-exposed scene colour the pass reads and copies back over). Like vignette,
//! the grade needs no depth, so it does not couple to the SSR prepass.

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
use super::settings::PrismColorGradeSettings;

/// The per-view colour-grade output texture, present only while the pass is
/// enabled and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewColorGrade {
    /// Full-resolution graded HDR output written by `color_grade_main`
    /// (`rgba16float`), copied back over `scene_color`.
    color_grade_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewColorGrade {
    /// Storage view of the graded HDR output written by `color_grade_main`.
    pub(crate) fn color_grade_out_view(&self) -> &TextureView {
        &self.color_grade_out.default_view
    }

    /// The graded-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn color_grade_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.color_grade_out.texture
    }
}

/// (Re)allocates [`ViewColorGrade`] for every view that has a resident
/// visibility buffer while the grade is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and on single-sample views (the scene colour
/// buffer is itself single-sample). The texture is re-created whenever the
/// viewport size changes, exactly like the buffer it shadows.
pub(crate) fn prepare_color_grade_textures(
    mut commands: Commands,
    settings: Res<PrismColorGradeSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewColorGrade>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewColorGrade>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Graded output: full-resolution wide HDR, written by `color_grade_main`
        // (storage) and copied back over `scene_color`.
        let color_grade_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism color grade output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `color_grade_main`. COPY_SRC:
                // copied back over `scene_color` after the pass so the downstream
                // post chain reads the graded image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewColorGrade {
            color_grade_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_grade_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
