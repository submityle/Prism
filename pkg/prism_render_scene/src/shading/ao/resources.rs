//! Per-view GPU textures backing the GTAO passes.
//!
//! GTAO runs in two compute steps between the visibility raster and the
//! shading resolve:
//!
//! 1. a *geometry prepass* decodes the visibility buffer into a linear
//!    view-depth texture and a view-space normal texture (the two inputs the
//!    [`prism_render_shading::ao`] golden consumes), and
//! 2. the *GTAO kernel* (`shaders/gtao.wesl`) reads those and writes per-pixel
//!    ambient visibility into the AO texture, which the resolve stage then
//!    multiplies into its indirect/ambient term.
//!
//! This module owns the three cached textures and the [`ExtractedCamera`]-driven
//! prepare system that (re)allocates them to match the viewport, mirroring
//! [`super::super::resources::prepare_visibility_buffers`].

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;

/// Linear view-space depth written by the GTAO geometry prepass and sampled by
/// the GTAO kernel. `R32Float` keeps positive metric depth at full precision so
/// the horizon search reconstructs view positions exactly like the golden.
pub(crate) const GTAO_DEPTH_FORMAT: TextureFormat = TextureFormat::R32Float;
/// View-space unit normal written by the prepass and sampled by the kernel.
/// `Rgba16Float` holds a signed unit vector with headroom to spare.
pub(crate) const GTAO_NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Ambient visibility (`1` = unoccluded) written by the GTAO kernel and sampled
/// by the resolve stage. Scalar `R32Float` matches `texture_storage_2d<r32float>`.
pub(crate) const GTAO_AO_FORMAT: TextureFormat = TextureFormat::R32Float;

/// The three per-view GTAO textures, present only while GTAO is enabled and the
/// viewport size is known.
#[derive(Component)]
pub(crate) struct ViewGtaoTextures {
    linear_depth: CachedTexture,
    view_normal: CachedTexture,
    ambient_occlusion: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewGtaoTextures {
    /// Storage/sampling view of the linear view-depth prepass target.
    pub(crate) fn linear_depth_view(&self) -> &TextureView {
        &self.linear_depth.default_view
    }

    /// Storage/sampling view of the view-space normal prepass target.
    pub(crate) fn view_normal_view(&self) -> &TextureView {
        &self.view_normal.default_view
    }

    /// Storage/sampling view of the ambient-visibility target.
    #[expect(dead_code, reason = "read by the GTAO compute bind groups in a following slice")]
    pub(crate) fn ambient_occlusion_view(&self) -> &TextureView {
        &self.ambient_occlusion.default_view
    }
}

/// (Re)allocates [`ViewGtaoTextures`] for every view that has a resident
/// visibility buffer while GTAO is enabled, and removes them otherwise.
///
/// Gated on both `enable_gtao` and `enable_visibility_buffer`: GTAO's geometry
/// prepass decodes the visibility buffer, so it is meaningless without it. The
/// textures are re-created whenever the viewport size changes, exactly like the
/// visibility buffer they shadow.
pub(crate) fn prepare_gtao_textures(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewGtaoTextures>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enable_gtao
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewGtaoTextures>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        let linear_depth = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism GTAO linear depth"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: GTAO_DEPTH_FORMAT,
                // Written by the prepass, sampled by the kernel.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let view_normal = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism GTAO view normal"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: GTAO_NORMAL_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let ambient_occlusion = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism GTAO ambient occlusion"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: GTAO_AO_FORMAT,
                // Written by the GTAO kernel, sampled by the resolve stage.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewGtaoTextures {
            linear_depth,
            view_normal,
            ambient_occlusion,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gtao_texture_formats_match_the_shader_bindings() {
        // linear depth + AO are scalar float storage; normals need three signed
        // channels, so a wide RGBA16F carries them.
        assert_eq!(GTAO_DEPTH_FORMAT, TextureFormat::R32Float);
        assert_eq!(GTAO_AO_FORMAT, TextureFormat::R32Float);
        assert_eq!(GTAO_NORMAL_FORMAT, TextureFormat::Rgba16Float);
    }
}
