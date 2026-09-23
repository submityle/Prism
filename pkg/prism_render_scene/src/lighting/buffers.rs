//! GPU storage buffers backing the extracted light tables.
//!
//! The resolve compute pass binds three parallel storage buffers: a directional
//! light array, a punctual (point/spot) light array, and a single-element
//! environment record carrying the ambient/probe state and light counts.  All
//! three are packed from [`ExtractedLights`] every frame; empty arrays are
//! padded with one neutral element so the bind group always has a valid,
//! non-null binding even in an unlit scene.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::{
    abi::{GpuDirectionalLight, GpuLightEnvironment, GpuPunctualLight},
    extract::ExtractedLights,
};

/// The three storage buffers mirroring the extracted lights onto the GPU.
#[derive(Resource)]
pub struct LightGpuBuffers {
    directionals: RawBufferVec<GpuDirectionalLight>,
    punctuals: RawBufferVec<GpuPunctualLight>,
    environment: RawBufferVec<GpuLightEnvironment>,
    version: u32,
}

impl FromWorld for LightGpuBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut directionals = RawBufferVec::new(BufferUsages::STORAGE);
        directionals.set_label(Some("prism directional lights"));
        let mut punctuals = RawBufferVec::new(BufferUsages::STORAGE);
        punctuals.set_label(Some("prism punctual lights"));
        let mut environment = RawBufferVec::new(BufferUsages::STORAGE);
        environment.set_label(Some("prism light environment"));
        Self {
            directionals,
            punctuals,
            environment,
            version: 1,
        }
    }
}

impl LightGpuBuffers {
    /// Repacks the extracted lights into the parallel storage arrays.
    ///
    /// The environment always holds exactly one record, so its light counts
    /// stay authoritative even when both light arrays are empty.
    pub fn rebuild(&mut self, lights: &ExtractedLights) {
        self.directionals.clear();
        self.punctuals.clear();
        self.environment.clear();

        self.directionals
            .extend(lights.directionals.iter().copied());
        self.punctuals.extend(lights.punctuals.iter().copied());
        self.environment.push(lights.environment);

        self.version = self.version.wrapping_add(1).max(1);
    }

    /// Streams the packed arrays to the GPU. Empty light arrays are padded with
    /// a single neutral element so the storage bindings are never zero-sized.
    pub fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.directionals.is_empty() {
            self.directionals.push(GpuDirectionalLight::default());
        }
        if self.punctuals.is_empty() {
            self.punctuals.push(GpuPunctualLight::default());
        }
        if self.environment.is_empty() {
            self.environment.push(GpuLightEnvironment::default());
        }
        self.directionals.write_buffer(device, queue);
        self.punctuals.write_buffer(device, queue);
        self.environment.write_buffer(device, queue);
    }

    /// Monotonic version bumped on every rebuild, for bind-group caching.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The three storage buffers once they have been uploaded at least once.
    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.directionals.buffer()?,
            self.punctuals.buffer()?,
            self.environment.buffer()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_packs_both_arrays_and_single_environment() {
        let mut world = World::new();
        let mut buffers = LightGpuBuffers::from_world(&mut world);
        let mut lights = ExtractedLights::default();
        lights.directionals.push(GpuDirectionalLight::default());
        lights.punctuals.push(GpuPunctualLight::default());
        lights.punctuals.push(GpuPunctualLight::default());
        lights.environment.ambient = [0.1, 0.2, 0.3];
        lights.environment.directional_count = 1;
        lights.environment.punctual_count = 2;
        buffers.rebuild(&lights);

        assert_eq!(buffers.directionals.values().len(), 1);
        assert_eq!(buffers.punctuals.values().len(), 2);
        assert_eq!(buffers.environment.values().len(), 1);
        assert_eq!(buffers.environment.values()[0].ambient, [0.1, 0.2, 0.3]);
        assert_eq!(buffers.environment.values()[0].punctual_count, 2);
    }

    #[test]
    fn rebuild_bumps_version_monotonically() {
        let mut world = World::new();
        let mut buffers = LightGpuBuffers::from_world(&mut world);
        let before = buffers.version();
        buffers.rebuild(&ExtractedLights::default());
        assert_ne!(buffers.version(), before);
        // The environment is always present even in an unlit scene.
        assert_eq!(buffers.environment.values().len(), 1);
    }
}
