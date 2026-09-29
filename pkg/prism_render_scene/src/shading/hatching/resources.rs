//! Per-view GPU texture backing the cross-hatching pass.
//!
//! The filter runs as a single full-screen compute pass over the resolved HDR
//! scene colour: `hatching_main` inks each texel by the golden coverage ramp and
//! writes the result to a dedicated output that the dispatch then copies back
//! over `scene_color`. This module owns that one cached texture:
//!
//! * `hatching_out` — the full-resolution stylized HDR output (`rgba16float`),
//!   copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::color_grade::resources`]. It is gated on
//! the filter enable *and* on the presence of the visibility buffer (source of
//! the scene colour the pass reads and copies back over). Like colour grade, the
//! filter needs no depth, so it does not couple to the SSR prepass.

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
use super::settings::PrismHatchingSettings;

/// The per-view cross-hatching output texture, present only while the pass is
/// enabled and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewHatching {
    /// Full-resolution stylized HDR output written by `hatching_main`
    /// (`rgba16float`), copied back over `scene_color`.
    hatching_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewHatching {
    /// Storage view of the stylized HDR output written by `hatching_main`.
    pub(crate) fn hatching_out_view(&self) -> &TextureView {
        &self.hatching_out.default_view
    }

    /// The stylized-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn hatching_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.hatching_out.texture
    }
}

/// (Re)allocates [`ViewHatching`] for every view that has a resident visibility
/// buffer while the filter is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the scene colour) and on single-sample views (the scene colour buffer is
/// itself single-sample). The texture is re-created whenever the viewport size
/// changes, exactly like the buffer it shadows.
pub(crate) fn prepare_hatching_textures(
    mut commands: Commands,
    settings: Res<PrismHatchingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewHatching>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewHatching>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Stylized output: full-resolution wide HDR, written by `hatching_main`
        // (storage) and copied back over `scene_color`.
        let hatching_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism hatching output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `hatching_main`. COPY_SRC: copied
                // back over `scene_color` after the pass so the downstream post
                // chain reads the filtered image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ViewHatching { hatching_out, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hatching_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
