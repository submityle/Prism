//! Per-view GPU texture backing the outline pass.
//!
//! The outline runs as a single full-screen compute pass over the pre-exposed
//! HDR scene colour: `outline_main` reads a four-neighbour cross of the geometry
//! buffer (SSR device depth + view-space normal), raises the golden edge
//! coverage and composites the authored line colour over the scene, writing the
//! result to a dedicated output the dispatch then copies back over
//! `scene_color`. This module owns that one cached texture:
//!
//! * `outline_out` — the full-resolution outlined HDR output (`rgba16float`),
//!   copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::color_grade::resources`]. It is gated on
//! the outline enable *and* on the presence of the visibility buffer (the
//! pre-exposed scene colour) and the SSR textures (the device depth + normal the
//! edge reduction reads). Because it consumes the SSR geometry prepass, the
//! prepare system is ordered `.after(prepare_ssr_textures)` in the plugin wiring.

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
use super::settings::PrismOutlineSettings;

/// The per-view outline output texture, present only while the pass is enabled
/// and the backing visibility buffer and SSR textures are resident.
#[derive(Component)]
pub(crate) struct ViewOutline {
    /// Full-resolution outlined HDR output written by `outline_main`
    /// (`rgba16float`), copied back over `scene_color`.
    outline_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewOutline {
    /// Storage view of the outlined HDR output written by `outline_main`.
    pub(crate) fn outline_out_view(&self) -> &TextureView {
        &self.outline_out.default_view
    }

    /// The outlined-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn outline_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.outline_out.texture
    }
}

/// (Re)allocates [`ViewOutline`] for every view that has a resident visibility
/// buffer and SSR textures while the outline is enabled, and removes it
/// otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and [`ViewSsrTextures`] (the device depth +
/// view-space normal), and on single-sample views (the scene colour buffer is
/// itself single-sample). The texture is re-created whenever the viewport size
/// changes, exactly like the buffers it shadows.
pub(crate) fn prepare_outline_textures(
    mut commands: Commands,
    settings: Res<PrismOutlineSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
        Option<&ViewOutline>,
    )>,
) {
    for (entity, camera, msaa, visibility, ssr, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && ssr.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewOutline>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Outlined output: full-resolution wide HDR, written by `outline_main`
        // (storage) and copied back over `scene_color`.
        let outline_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism outline output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `outline_main`. COPY_SRC: copied
                // back over `scene_color` after the pass so the downstream post
                // chain reads the outlined image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ViewOutline { outline_out, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outline_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
