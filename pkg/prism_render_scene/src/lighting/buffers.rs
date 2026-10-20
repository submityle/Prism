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
    version: u64,
    directional_dirty: Vec<std::ops::Range<usize>>,
    punctual_dirty: Vec<std::ops::Range<usize>>,
    environment_dirty: Vec<std::ops::Range<usize>>,
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
            directional_dirty: Vec::new(),
            punctual_dirty: Vec::new(),
            environment_dirty: Vec::new(),
        }
    }
}

impl LightGpuBuffers {
    /// Repacks the extracted lights into the parallel storage arrays.
    ///
    /// The environment always holds exactly one record, so its light counts
    /// stay authoritative even when both light arrays are empty.
    pub fn rebuild(&mut self, lights: &ExtractedLights) {
        self.directional_dirty = replace_changed_values(
            &mut self.directionals,
            &lights.directionals,
            GpuDirectionalLight::default(),
        );
        self.punctual_dirty = replace_changed_values(
            &mut self.punctuals,
            &lights.punctuals,
            GpuPunctualLight::default(),
        );
        self.environment_dirty = replace_changed_values(
            &mut self.environment,
            std::slice::from_ref(&lights.environment),
            GpuLightEnvironment::default(),
        );
        if !self.directional_dirty.is_empty()
            || !self.punctual_dirty.is_empty()
            || !self.environment_dirty.is_empty()
        {
            self.version = self
                .version
                .checked_add(1)
                .expect("light revision exhausted");
        }
    }

    /// Streams the packed arrays to the GPU. Empty light arrays are padded with
    /// a single neutral element so the storage bindings are never zero-sized.
    pub fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        upload_changed(
            &mut self.directionals,
            &mut self.directional_dirty,
            device,
            queue,
        );
        upload_changed(&mut self.punctuals, &mut self.punctual_dirty, device, queue);
        upload_changed(
            &mut self.environment,
            &mut self.environment_dirty,
            device,
            queue,
        );
    }

    /// CPU content revision. Bind groups are keyed by GPU buffer identity, not this revision.
    pub fn version(&self) -> u64 {
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

fn replace_changed_values<T>(
    buffer: &mut RawBufferVec<T>,
    incoming: &[T],
    neutral: T,
) -> Vec<std::ops::Range<usize>>
where
    T: bytemuck::NoUninit + Copy + PartialEq,
{
    let old = buffer.values();
    let physical_len = incoming.len().max(1);
    let mut dirty = Vec::new();
    let mut run_start = None;
    for index in 0..physical_len {
        let value = incoming.get(index).copied().unwrap_or(neutral);
        let changed = old.get(index).map_or(true, |previous| *previous != value);
        match (run_start, changed) {
            (None, true) => run_start = Some(index),
            (Some(start), false) => {
                dirty.push(start..index);
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        dirty.push(start..physical_len);
    }
    let values = buffer.values_mut();
    values.clear();
    values.extend_from_slice(incoming);
    if values.is_empty() {
        values.push(neutral);
    }
    dirty
}

fn upload_changed<T>(
    buffer: &mut RawBufferVec<T>,
    dirty: &mut Vec<std::ops::Range<usize>>,
    device: &RenderDevice,
    queue: &RenderQueue,
) where
    T: bytemuck::NoUninit + Copy,
{
    if dirty.is_empty() {
        return;
    }
    let old_capacity = buffer.capacity();
    let item_count = buffer.len();
    let missing_buffer = buffer.buffer().is_none();
    buffer.reserve(item_count, device);
    if missing_buffer || old_capacity < item_count {
        buffer.write_buffer(device, queue);
    } else {
        for range in dirty.iter().cloned() {
            buffer
                .write_buffer_range(queue, range)
                .expect("validated light upload range must fit the reserved GPU buffer");
        }
    }
    dirty.clear();
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
    fn rebuild_bumps_version_only_when_contents_change() {
        let mut world = World::new();
        let mut buffers = LightGpuBuffers::from_world(&mut world);
        let before = buffers.version();
        buffers.rebuild(&ExtractedLights::default());
        assert_ne!(buffers.version(), before);
        // The environment is always present even in an unlit scene.
        assert_eq!(buffers.environment.values().len(), 1);
        let after_change = buffers.version();
        buffers.rebuild(&ExtractedLights::default());
        assert_eq!(buffers.version(), after_change);
    }

    #[test]
    fn identical_rebuild_has_no_dirty_ranges_and_sparse_edits_are_coalesced() {
        let mut world = World::new();
        let mut buffers = LightGpuBuffers::from_world(&mut world);
        let mut lights = ExtractedLights::default();
        lights.punctuals = (0..5)
            .map(|i| GpuPunctualLight {
                position: [i as f32; 3],
                ..Default::default()
            })
            .collect();
        buffers.rebuild(&lights);
        assert_eq!(buffers.punctual_dirty, vec![0..5]);
        buffers.rebuild(&lights);
        assert!(buffers.punctual_dirty.is_empty());
        lights.punctuals[1].position[0] = 9.0;
        lights.punctuals[4].position[0] = 8.0;
        buffers.rebuild(&lights);
        assert_eq!(buffers.punctual_dirty, vec![1..2, 4..5]);
    }
}
