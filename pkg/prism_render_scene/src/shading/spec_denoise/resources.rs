//! Per-view resident target backing the specular-GI *spatial denoise* pass.
//!
//! The spatial pass is a screen-space, single-dispatch cross-bilateral filter
//! that sits between the `spec_gi` reuse/composite resolve and the composite's
//! energy-conserving fold: it reads the noisy per-pixel specular estimate (the
//! `spec_gi` resolved target), the SSR trace's per-pixel hit distance and the
//! shared geometry prepass (`normal_roughness` + reverse-Z `scene_depth`), and
//! writes an edge-aware denoised specular+confidence estimate the composite
//! consumes in place of the raw resolve.
//!
//! Unlike the `spec_gi` reuse buffers this subsystem owns no ping-pong state —
//! the spatial filter is a pure current-frame gather, so one viewport-sized
//! output target is enough and the only reallocation trigger is a framebuffer
//! size change (exactly like [`super::super::ssgi::resources`]'s targets). The
//! subsystem is gated identically to `spec_gi` — on `enable_spec_gi`,
//! `enable_ssr` and `enable_visibility_buffer`, a resident visibility buffer,
//! and a single-sample view — because it only has work to do when the
//! `spec_gi` resolve it filters exists.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
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

/// Denoised specular estimate written by the spatial kernel: `rgb` = edge-aware
/// filtered specular radiance, `a` = ReSTIR confidence passed through unchanged
/// from the `spec_gi` resolve so the energy-conserving composite reads every GI
/// source uniformly. Wide HDR, matching [`super::super::spec_gi::resources`]'s
/// resolved target and the SSR reflection output.
pub(crate) const SPEC_DENOISE_FILTERED_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Per-view resident spatial-denoise target, present only while the spatial
/// pass is enabled and the viewport size is known.
#[derive(Component)]
pub(crate) struct ViewSpecDenoise {
    /// Filtered specular+confidence target the kernel writes (storage write)
    /// and the composite reads (`textureLoad`). Single mip, full resolution. A
    /// distinct texture from the `spec_gi` resolve it reads because
    /// `rgba16float` is not read-write storage-capable, so the filter cannot
    /// run in place on the input it samples.
    filtered: CachedTexture,
    /// Live viewport extent the target was allocated for; a change retriggers
    /// reallocation.
    pub(crate) size: UVec2,
}

impl ViewSpecDenoise {
    /// Storage/sampling view of the filtered specular estimate. The spatial
    /// kernel writes it (storage) and the composite reads it (sampled).
    pub(crate) fn filtered_view(&self) -> &TextureView {
        &self.filtered.default_view
    }
}

/// `PrepareResources` system: (re)allocates the per-view spatial-denoise target
/// whenever the viewport size changes, and removes it when the pass is disabled.
///
/// Gated identically to the `spec_gi` reuse resources — the spatial filter only
/// has work to do when the resolve it reads exists — so the two allocate and
/// free in lockstep.
pub(crate) fn prepare_spec_denoise_resources(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&mut ViewSpecDenoise>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &mut views {
        let enabled = settings.enable_spec_gi
            && settings.enable_ssr
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewSpecDenoise>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };

        // Steady state: an existing allocation already at the live viewport is
        // reused as-is (the filter is stateless across frames). Only a resize
        // falls through to reallocate the target below.
        if let Some(view) = existing {
            if view.size == size {
                continue;
            }
        }

        let filtered = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism spec_denoise filtered"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SPEC_DENOISE_FILTERED_FORMAT,
                // Written by the spatial kernel (storage), sampled by the
                // energy-conserving composite.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ViewSpecDenoise { filtered, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filtered_format_is_wide_hdr_with_confidence_alpha() {
        // rgb filtered specular + a passthrough confidence; shares the spec_gi
        // resolve / SSR reflection wide-HDR encoding so the composite reads
        // every GI source uniformly.
        assert_eq!(SPEC_DENOISE_FILTERED_FORMAT, TextureFormat::Rgba16Float);
    }
}
