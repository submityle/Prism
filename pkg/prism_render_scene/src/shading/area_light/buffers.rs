//! GPU storage buffer backing the extracted area-light table.
//!
//! The resolve compute pass binds a single storage buffer (group 7, binding 0)
//! holding the per-frame [`GpuAreaLight`] array. It is repacked from
//! [`ExtractedAreaLights`] every frame; an empty table is padded with one
//! degenerate record (`half_width == 0`) so the storage binding is never
//! zero-sized and the shader's extent guard skips it, contributing nothing.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::abi::GpuAreaLight;
use super::extract::ExtractedAreaLights;

/// A single degenerate rectangle (`half_width == half_height == 0`) the shader's
/// extent guard skips. Used to pad an empty table so the storage binding stays
/// valid in a scene with no area lights.
fn degenerate_light() -> GpuAreaLight {
    GpuAreaLight::rect(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.0,
        0.0,
        [0.0; 3],
        0.0,
    )
}

/// The storage buffer mirroring the extracted area lights onto the GPU.
#[derive(Resource)]
pub(crate) struct AreaLightGpuBuffer {
    lights: RawBufferVec<GpuAreaLight>,
    version: u32,
}

impl FromWorld for AreaLightGpuBuffer {
    fn from_world(_: &mut World) -> Self {
        let mut lights = RawBufferVec::new(BufferUsages::STORAGE);
        lights.set_label(Some("prism area lights"));
        Self { lights, version: 1 }
    }
}

impl AreaLightGpuBuffer {
    /// Repacks the extracted area lights into the storage array.
    pub(crate) fn rebuild(&mut self, lights: &ExtractedAreaLights) {
        self.lights.clear();
        self.lights.extend(lights.lights.iter().copied());
        self.version = self.version.wrapping_add(1).max(1);
    }

    /// Streams the packed array to the GPU. An empty table is padded with one
    /// degenerate record so the storage binding is never zero-sized.
    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.lights.is_empty() {
            self.lights.push(degenerate_light());
        }
        self.lights.write_buffer(device, queue);
    }

    /// Monotonic version bumped on every rebuild, for bind-group caching.
    #[expect(
        dead_code,
        reason = "exposed for bind-group cache invalidation wired in a later slice"
    )]
    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    /// The storage buffer once it has been uploaded at least once.
    pub(crate) fn buffer(&self) -> Option<&Buffer> {
        self.lights.buffer()
    }
}

/// Repacks the per-frame [`ExtractedAreaLights`] into the GPU staging array.
///
/// Runs in `PrepareResources` so the packed data is ready before the flush
/// stage writes it to the device.
pub(crate) fn rebuild_area_light_buffers(
    lights: Res<ExtractedAreaLights>,
    mut buffer: ResMut<AreaLightGpuBuffer>,
) {
    buffer.rebuild(&lights);
}

/// Flushes the staged area-light array to the GPU.
pub(crate) fn write_area_light_buffers(
    mut buffer: ResMut<AreaLightGpuBuffer>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffer.upload(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::Color;

    use crate::shading::area_light::component::AreaLight;

    fn one_light() -> ExtractedAreaLights {
        let mut extracted = ExtractedAreaLights::default();
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 2.0);
        let gpu = GpuAreaLight::rect(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            light.half_width,
            light.half_height,
            [1.0, 1.0, 1.0],
            light.intensity,
        );
        extracted.lights.push(gpu);
        extracted
    }

    #[test]
    fn rebuild_packs_the_lights_and_bumps_version() {
        let mut world = World::new();
        let mut buffer = AreaLightGpuBuffer::from_world(&mut world);
        let before = buffer.version();
        buffer.rebuild(&one_light());
        assert_eq!(buffer.lights.values().len(), 1);
        assert_ne!(buffer.version(), before);
    }

    #[test]
    fn rebuild_empty_leaves_no_records_before_upload_pads() {
        let mut world = World::new();
        let mut buffer = AreaLightGpuBuffer::from_world(&mut world);
        buffer.rebuild(&ExtractedAreaLights::default());
        assert_eq!(buffer.lights.values().len(), 0);
    }

    #[test]
    fn degenerate_pad_is_skipped_by_extent() {
        let pad = degenerate_light();
        assert_eq!(pad.half_width, 0.0);
        assert_eq!(pad.half_height, 0.0);
    }
}
