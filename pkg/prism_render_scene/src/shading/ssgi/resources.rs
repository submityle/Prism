//! Per-view GPU textures backing the SSGI trace and composite.
//!
//! Screen-space global illumination reuses the reflection subsystem's rebuilt
//! inputs — the reverse-Z Hi-Z pyramid, the packed view-space `normal_roughness`
//! and the current-frame colour pyramid all live in
//! [`super::super::ssr::ViewSsrTextures`] — so this module only owns the two
//! full-resolution targets the diffuse gather itself needs:
//!
//! 1. `ssgi_out` — the trace's raw output (`rgb` = pre-albedo mean indirect
//!    radiance × intensity, `a` = blend confidence), and
//! 2. `gi_base` — a scratch copy of the shaded `scene_color` taken *after* the
//!    reflection composite so the GI fold reads a stable base without a
//!    read/write aliasing hazard on the `scene_color` storage image.
//!
//! Both are viewport-sized and single-mip; the [`ExtractedCamera`]-driven
//! prepare system (re)allocates them to match the framebuffer, gated exactly
//! like [`super::super::ssr::prepare_ssr_textures`] because SSGI consumes SSR's
//! rebuilt inputs and is meaningless without them.

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

/// Diffuse-GI buffer the trace writes (`rgb` = pre-albedo mean indirect radiance
/// × intensity, `a` = blend confidence) and the composite folds over the pure
/// IBL/SH ambient. Wide HDR so the gathered radiance keeps its range up to the
/// composite, matching [`super::super::ssr::resources`]' reflection output.
pub(crate) const SSGI_OUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Scratch copy of the composited `scene_color` the GI fold reads as its base.
/// Shares [`super::super::resources::SCENE_COLOR_FORMAT`] byte for byte so the
/// copy is loss-free. Because `rgba16float` is not read-write storage-capable,
/// the composite copies `scene_color` into this distinct texture first and only
/// the write side touches `scene_color`, keeping the pass free of any storage
/// read/write aliasing hazard.
pub(crate) const SSGI_BASE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The two per-view SSGI targets, present only while SSGI is enabled and the
/// viewport size is known.
#[derive(Component)]
pub(crate) struct ViewSsgiTextures {
    /// Raw diffuse-GI gather output written by the trace and read by the
    /// composite. Single mip, full resolution.
    ssgi_out: CachedTexture,
    /// Scratch base written by the composite's copy pass (a 1:1 lift of the
    /// composited `scene_color`) and read back by the fold pass. Single mip,
    /// full resolution.
    gi_base: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewSsgiTextures {
    /// Storage/sampling view of the trace's diffuse-GI output. The trace writes
    /// it (storage) and the composite reads it (`textureLoad`).
    pub(crate) fn ssgi_out_view(&self) -> &TextureView {
        &self.ssgi_out.default_view
    }

    /// Storage/sampling view of the scratch base copy. The composite copy pass
    /// writes it (storage) and the fold pass reads it (`textureLoad`).
    pub(crate) fn gi_base_view(&self) -> &TextureView {
        &self.gi_base.default_view
    }
}

/// (Re)allocates [`ViewSsgiTextures`] for every view that has a resident
/// visibility buffer while SSGI is enabled, and removes them otherwise.
///
/// Gated on `enable_ssgi`, `enable_ssr` and `enable_visibility_buffer`: the
/// gather reuses SSR's rebuilt Hi-Z, `normal_roughness` and colour pyramid, so
/// it is meaningless without them, and on single-sample views because the
/// visibility buffer those inputs decode is itself single-sample. The textures
/// are re-created whenever the viewport size changes, exactly like the SSR
/// textures they shadow.
pub(crate) fn prepare_ssgi_textures(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsgiTextures>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enable_ssgi
            && settings.enable_ssr
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewSsgiTextures>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        let ssgi_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSGI output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSGI_OUT_FORMAT,
                // Written by the trace, sampled by the composite.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let gi_base = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSGI base copy"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSGI_BASE_FORMAT,
                // Written by the composite copy pass, sampled by the fold pass.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewSsgiTextures {
            ssgi_out,
            gi_base,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssgi_texture_formats_match_the_shader_bindings() {
        // The gather output carries wide-HDR radiance plus a confidence in a,
        // and the scratch base shares the scene-colour encoding for a loss-free
        // copy.
        assert_eq!(SSGI_OUT_FORMAT, TextureFormat::Rgba16Float);
        assert_eq!(SSGI_BASE_FORMAT, TextureFormat::Rgba16Float);
        assert_eq!(SSGI_BASE_FORMAT, super::super::super::resources::SCENE_COLOR_FORMAT);
    }
}
