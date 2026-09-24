//! Render-schedule systems that keep the shadow atlas allocation, the shadow
//! storage buffers, and the shadow bind group up to date for the resolve pass.

use bevy_ecs::prelude::*;
use bevy_render::renderer::{RenderDevice, RenderQueue};

use super::{
    bindings::ShadowBindGroup,
    resources::{ExtractedShadows, ShadowAtlas, ShadowAtlasConfig, ShadowGpuBuffers},
};

/// Reallocates the atlas array texture when the requested [`ShadowAtlasConfig`]
/// changes (resolution or layer budget), leaving it untouched otherwise.
///
/// Runs in `PrepareResources` so a freshly sized atlas is available before the
/// bind group is (re)built later in the frame.
pub(crate) fn ensure_shadow_atlas(
    config: Res<ShadowAtlasConfig>,
    mut atlas: ResMut<ShadowAtlas>,
    device: Res<RenderDevice>,
) {
    atlas.ensure(&device, *config);
}

/// Repacks the per-frame [`ExtractedShadows`] into the GPU staging arrays.
///
/// Runs in `PrepareResources` so the packed data is ready before the flush
/// stage writes it to the device.
pub(crate) fn rebuild_shadow_buffers(
    shadows: Res<ExtractedShadows>,
    mut buffers: ResMut<ShadowGpuBuffers>,
) {
    buffers.rebuild(&shadows);
}

/// Flushes the staged shadow arrays to the GPU.
pub(crate) fn write_shadow_buffers(
    mut buffers: ResMut<ShadowGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

/// (Re)builds the shadow bind group once the atlas exists and the buffers have
/// uploaded.
pub(crate) fn prepare_shadow_bind_group(
    atlas: Res<ShadowAtlas>,
    buffers: Res<ShadowGpuBuffers>,
    mut bindings: ResMut<ShadowBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &atlas, &buffers);
}
