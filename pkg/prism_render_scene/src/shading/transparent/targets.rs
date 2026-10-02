//! Per-view weighted-blended OIT render targets.
//!
//! Transparent geometry cannot live in the visibility buffer (one opaque
//! surface per pixel), so it is drawn in a forward pass that accumulates into
//! two MRT targets with `McGuire` & Bavoil blending (JCGT 2013):
//!
//! * [`OIT_ACCUM_FORMAT`] (`Rgba16Float`) - additive: weighted premultiplied
//!   colour in `rgb`, summed weighted alpha in `a`;
//! * [`OIT_REVEALAGE_FORMAT`] (`R16Float`) - multiplicative: the running
//!   product of `1 - alpha`.
//!
//! Both carry `RENDER_ATTACHMENT` (written by the forward pass, cleared to the
//! WBOIT identity every frame) and `TEXTURE_BINDING` (read by the fullscreen
//! composite). They are sized and gated exactly like [`ViewVisibilityBuffer`]
//! so a view either has the whole visibility+transparency chain or none of it.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureFormat, TextureUsages},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;

/// Accumulation target: weighted premultiplied colour (`rgb`) + summed weighted
/// alpha (`a`). `Rgba16Float` matches the additive MRT contribution in
/// `oit.wesl` and keeps enough range for the depth-weighted sums.
pub(crate) const OIT_ACCUM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Revealage target: the running product of `1 - alpha`. A single `R16Float`
/// channel is enough; it is cleared to `1.0` (nothing occluded) each frame.
pub(crate) const OIT_REVEALAGE_FORMAT: TextureFormat = TextureFormat::R16Float;

/// The two WBOIT targets owned by one view for the current frame.
#[derive(Component)]
pub(crate) struct ViewOitTargets {
    accum: CachedTexture,
    revealage: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewOitTargets {
    /// Colour-attachment views for the transparent forward pass, in MRT order:
    /// `location(0)` = accumulation, `location(1)` = revealage.
    pub(crate) fn attachments(
        &self,
    ) -> (
        &bevy_render::render_resource::TextureView,
        &bevy_render::render_resource::TextureView,
    ) {
        (&self.accum.default_view, &self.revealage.default_view)
    }

    /// Sampling view of the accumulation target for the composite bind group.
    pub(crate) fn accum_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.accum.default_view
    }

    /// Sampling view of the revealage target for the composite bind group.
    pub(crate) fn revealage_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.revealage.default_view
    }
}

/// `RenderSystems::PrepareResources` system allocating the two WBOIT targets for
/// every view that already carries a [`ViewVisibilityBuffer`] this frame, and
/// dropping them for views that sat the visibility path out (feature off or
/// MSAA active) so no target can outlive the pass that writes it.
pub(crate) fn prepare_oit_targets(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewOitTargets>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let single_sample = msaa.is_none_or(|value| value.samples() == 1);
        let live = settings.enable_visibility_buffer && single_sample && visibility.is_some();
        let size = camera.physical_viewport_size;
        if !live || size.is_none() {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewOitTargets>();
            }
            continue;
        }
        let size = size.unwrap();
        if existing.is_some_and(|targets| targets.size == size) {
            continue;
        }
        let accum = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism oit accumulation"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: OIT_ACCUM_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let revealage = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism oit revealage"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: OIT_REVEALAGE_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(ViewOitTargets {
            accum,
            revealage,
            size,
        });
    }
}
