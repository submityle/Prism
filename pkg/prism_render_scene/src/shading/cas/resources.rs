//! Per-view GPU texture backing the `CAS` pass.
//!
//! The sharpen runs as a single full-screen compute pass over the scene colour:
//! `cas_main` sharpens each texel by the golden `CAS` operator and writes the
//! result to a dedicated output that the dispatch then copies back over
//! `scene_color`. This module owns that one cached texture:
//!
//! * `cas_out` — the full-resolution sharpened `HDR` output (`rgba16float`),
//!   copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::color_grade::resources`]. It is gated on
//! the sharpen enable *and* on the presence of the visibility buffer (source of
//! the scene colour the pass reads and copies back over). Like colour grade,
//! `CAS` needs no depth, so it does not couple to the `SSR` prepass.

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
use super::settings::PrismCasSettings;

/// The per-view `CAS` output texture, present only while the pass is enabled and
/// the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewCas {
    /// Full-resolution sharpened `HDR` output written by `cas_main`
    /// (`rgba16float`), copied back over `scene_color`.
    cas_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewCas {
    /// Storage view of the sharpened `HDR` output written by `cas_main`.
    pub(crate) fn cas_out_view(&self) -> &TextureView {
        &self.cas_out.default_view
    }

    /// The sharpened-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn cas_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.cas_out.texture
    }
}

/// (Re)allocates [`ViewCas`] for every view that has a resident visibility
/// buffer while the sharpen is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the scene colour) and on single-sample views (the scene colour buffer is
/// itself single-sample). The texture is re-created whenever the viewport size
/// changes, exactly like the buffer it shadows.
pub(crate) fn prepare_cas_textures(
    mut commands: Commands,
    settings: Res<PrismCasSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewCas>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewCas>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Sharpened output: full-resolution wide `HDR`, written by `cas_main`
        // (storage) and copied back over `scene_color`.
        let cas_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism cas output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `cas_main`. COPY_SRC: copied back
                // over `scene_color` after the pass so the downstream post chain
                // reads the sharpened image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewCas { cas_out, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cas_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
