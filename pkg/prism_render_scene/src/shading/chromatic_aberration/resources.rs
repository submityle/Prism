//! Per-view GPU texture backing the chromatic-aberration pass.
//!
//! The subsystem runs as a single full-screen compute pass over the resolved,
//! pre-exposed HDR scene colour: `chromatic_aberration_main` fetches the scene
//! colour at three radially split per-channel coordinates and writes the fringed
//! result. This module owns the one cached texture that pass writes:
//!
//! * `chromatic_out` — the full-resolution aberrated HDR output
//!   (`rgba16float`), copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::dof::resources`]. It is gated on the
//! aberration enable *and* on the presence of both the visibility buffer (source
//! of the pre-exposed scene colour the pass reads and copies back over) and the
//! SSR geometry prepass, matching the resident coupling the sibling
//! post-processing passes share, and on single-sample views because the scene
//! colour it reads is itself single-sample.

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
use super::settings::PrismChromaticAberrationSettings;

/// The per-view chromatic-aberration output texture, present only while the
/// subsystem is enabled and the backing visibility + SSR buffers are resident.
#[derive(Component)]
pub(crate) struct ViewChromaticAberration {
    /// Full-resolution aberrated HDR output written by
    /// `chromatic_aberration_main` (`rgba16float`), copied back over
    /// `scene_color`.
    chromatic_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewChromaticAberration {
    /// Storage view of the aberrated HDR output written by the pass.
    pub(crate) fn chromatic_out_view(&self) -> &TextureView {
        &self.chromatic_out.default_view
    }

    /// The aberrated-output GPU texture itself, for the
    /// `copy_texture_to_texture` that writes it back over `scene_color`.
    pub(crate) fn chromatic_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.chromatic_out.texture
    }
}

/// (Re)allocates [`ViewChromaticAberration`] for every view that has both a
/// resident visibility buffer and SSR geometry prepass while the subsystem is
/// enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and [`ViewSsrTextures`], and on single-sample
/// views because the scene colour it reads is single-sample. The texture is
/// re-created whenever the viewport size changes, exactly like the buffers it
/// shadows.
pub(crate) fn prepare_chromatic_aberration_textures(
    mut commands: Commands,
    settings: Res<PrismChromaticAberrationSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
        Option<&ViewChromaticAberration>,
    )>,
) {
    for (entity, camera, msaa, visibility, ssr, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && ssr.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewChromaticAberration>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Aberrated output: full-resolution wide HDR, written by
        // `chromatic_aberration_main` (storage) and copied back over
        // `scene_color`.
        let chromatic_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism chromatic aberration output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by the pass. COPY_SRC: copied back
                // over `scene_color` after the pass so the downstream post chain
                // reads the fringed image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewChromaticAberration {
            chromatic_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromatic_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
