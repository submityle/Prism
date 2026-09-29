//! Per-view GPU textures backing the depth-of-field passes.
//!
//! `DoF` runs as a three-step compute chain over the pre-exposed HDR scene
//! colour: `dof_coc` writes the per-pixel near/far circle-of-confusion radii,
//! `dof_gather` disk-bokeh blurs the scene colour weighted by that field, and
//! `dof_composite` blends sharp toward blurred by the `CoC`. This module owns the
//! three cached textures those steps read and write:
//!
//! 1. `coc` — the per-pixel near/far `CoC` gather radii (`rg16float`,
//!    r = near px, g = far px);
//! 2. `blurred` — the full-resolution disk-bokeh blurred HDR field
//!    (`rgba16float`); and
//! 3. `dof_out` — the full-resolution composited HDR output (`rgba16float`),
//!    copied back over `scene_color` after the chain.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates them to match the
//! viewport, mirroring [`super::super::motion_blur::resources`]. It is gated on
//! the `DoF` enable *and* on the presence of both the visibility buffer (source of
//! the pre-exposed scene colour the gather blurs and the composite copies back
//! over) and the SSR geometry prepass (source of the reverse-Z device depth the
//! `CoC` prepass unprojects) — the same SSR coupling motion blur relies on.

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
use super::pipeline::DOF_COC_FORMAT;
use super::settings::PrismDofSettings;

/// The three per-view depth-of-field textures, present only while `DoF` is enabled
/// and the backing visibility + SSR buffers are resident.
#[derive(Component)]
pub(crate) struct ViewDof {
    /// Per-pixel near/far `CoC` gather radii written by `dof_coc` (`rg16float`,
    /// r = near px, g = far px), read by `dof_gather` and `dof_composite`.
    coc: CachedTexture,
    /// Full-resolution disk-bokeh blurred HDR field written by `dof_gather`
    /// (`rgba16float`), read by `dof_composite`.
    blurred: CachedTexture,
    /// Full-resolution composited HDR output written by `dof_composite`
    /// (`rgba16float`), copied back over `scene_color`.
    dof_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewDof {
    /// Storage/sampling view of the `CoC` field. `dof_coc` writes it (storage);
    /// `dof_gather` and `dof_composite` read it (sampled).
    pub(crate) fn coc_view(&self) -> &TextureView {
        &self.coc.default_view
    }

    /// Storage/sampling view of the blurred field. `dof_gather` writes it
    /// (storage) and `dof_composite` reads it (sampled).
    pub(crate) fn blurred_view(&self) -> &TextureView {
        &self.blurred.default_view
    }

    /// Storage view of the composited HDR output written by `dof_composite`.
    pub(crate) fn dof_out_view(&self) -> &TextureView {
        &self.dof_out.default_view
    }

    /// The composited-output GPU texture itself, for the
    /// `copy_texture_to_texture` that writes it back over `scene_color`.
    pub(crate) fn dof_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.dof_out.texture
    }
}

/// (Re)allocates [`ViewDof`] for every view that has both a resident visibility
/// buffer and SSR geometry prepass while `DoF` is enabled, and removes it
/// otherwise.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and [`ViewSsrTextures`] (the device depth the
/// `CoC` prepass unprojects), and on single-sample views because both backing
/// buffers are themselves single-sample. The textures are re-created whenever
/// the viewport size changes, exactly like the buffers they shadow.
pub(crate) fn prepare_dof_textures(
    mut commands: Commands,
    settings: Res<PrismDofSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
        Option<&ViewDof>,
    )>,
) {
    for (entity, camera, msaa, visibility, ssr, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && ssr.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewDof>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // CoC field: near/far gather radii in pixels, written by `dof_coc`
        // (storage) and read by `dof_gather` / `dof_composite` (sampled).
        let coc = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism dof coc"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DOF_COC_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Blurred field: full-resolution wide HDR, written by `dof_gather`
        // (storage) and read by `dof_composite` (sampled).
        let blurred = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism dof blurred"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Composited output: full-resolution wide HDR, written by
        // `dof_composite` (storage) and copied back over `scene_color`.
        let dof_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism dof output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `dof_composite`. COPY_SRC: copied
                // back over `scene_color` after the chain so the downstream
                // post chain reads the defocused image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewDof {
            coc,
            blurred,
            dof_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dof_textures_share_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn coc_field_is_two_channel_half_float() {
        assert_eq!(
            DOF_COC_FORMAT,
            bevy_render::render_resource::TextureFormat::Rg16Float
        );
    }
}
