//! Render-schedule systems that stage, upload, and bind the light buffers.

use bevy_ecs::prelude::*;
use bevy_render::renderer::{RenderDevice, RenderQueue};

use super::{bindings::LightBindGroup, buffers::LightGpuBuffers, extract::ExtractedLights};

/// Repacks the per-frame [`ExtractedLights`] into the GPU staging arrays.
///
/// Runs in `PrepareResources` so the packed data is ready before the flush
/// stage writes it to the device.
pub(crate) fn rebuild_light_buffers(
    lights: Res<ExtractedLights>,
    mut buffers: ResMut<LightGpuBuffers>,
) {
    buffers.rebuild(&lights);
}

/// Flushes the staged light arrays to the GPU.
pub(crate) fn write_light_buffers(
    mut buffers: ResMut<LightGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

/// (Re)builds the light bind group once the buffers have uploaded.
pub(crate) fn prepare_light_bind_group(
    buffers: Res<LightGpuBuffers>,
    mut bindings: ResMut<LightBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &buffers);
}
