//! Per-view GPU resources backing the light-routing cull pass.
//!
//! The pass refines a view's cluster active-light bitset by lighting channel
//! and NPR light layer over three storage buffers this module owns:
//!
//! * `routing_records` — one [`GpuLightRouting`] per light, uploaded from the
//!   [`PrismLightRoutingSettings`] records (read-only on device).
//! * `visible_lights` — the channel-gated visibility mask, one `u32` per
//!   32-light cluster word (written by the cull, read by the resolve / shared
//!   cluster cull).
//! * `layer_lights` — the NPR per-layer contribution masks,
//!   `word_count * MAX_LIGHT_LAYERS` `u32`s (written by the cull, read by the
//!   stylized composite).
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates and re-uploads
//! them whenever the record count changes or the settings resource is edited,
//! and removes the component when the subsystem is disabled — mirroring
//! [`super::super::world_space_gi::resources`]'s buffer plumbing.

use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};

use super::abi::{word_count, GpuLightRouting, MAX_LIGHT_LAYERS};
use super::settings::PrismLightRoutingSettings;

/// The per-view light-routing resources, present only while the cull pass is
/// enabled.
#[derive(Component)]
pub(crate) struct ViewLightRouting {
    /// Per-light routing records (`GpuLightRouting` x `light_count`), read-only
    /// on device. At least one element (a zeroed pad) so the storage binding is
    /// never zero-sized even when no lights are registered.
    routing: Buffer,
    /// Channel-gated visibility mask, one `u32` per cluster word (read-write).
    visible: Buffer,
    /// NPR per-layer contribution masks, `word_count * MAX_LIGHT_LAYERS` `u32`s
    /// (read-write), laid out layer-minor.
    layers: Buffer,
    /// The record count this allocation was sized/uploaded for; re-upload is
    /// triggered when it changes.
    pub(crate) light_count: u32,
    /// Cluster-word count = `word_count(light_count)`, the dispatch extent.
    pub(crate) word_count: u32,
}

impl ViewLightRouting {
    /// The per-light routing records buffer (read-only binding 0).
    pub(crate) fn routing_buffer(&self) -> &Buffer {
        &self.routing
    }

    /// The channel-gated visibility mask buffer (read-write binding 1).
    pub(crate) fn visible_buffer(&self) -> &Buffer {
        &self.visible
    }

    /// The NPR per-layer contribution mask buffer (read-write binding 2).
    pub(crate) fn layer_buffer(&self) -> &Buffer {
        &self.layers
    }
}

/// (Re)allocates [`ViewLightRouting`] for every camera view while the cull is
/// enabled, and removes it otherwise.
///
/// Gated on `settings.enabled`. The routing buffer is re-uploaded (and the
/// export buffers resized) whenever the record count changes or the settings
/// resource is edited; unchanged views keep their existing allocation.
pub(crate) fn prepare_light_routing_buffers(
    mut commands: Commands,
    settings: Res<PrismLightRoutingSettings>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, Option<&ViewLightRouting>)>,
) {
    for (entity, _camera, existing) in &views {
        if !settings.enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewLightRouting>();
            }
            continue;
        }

        let light_count = settings.records.len() as u32;
        // Re-upload only when the record count changed or the resource was
        // edited this frame; otherwise the existing allocation is still valid.
        let up_to_date = existing.is_some_and(|resources| resources.light_count == light_count)
            && !settings.is_changed();
        if up_to_date {
            continue;
        }

        // Flatten the golden records; pad to a single zeroed record so the
        // read-only storage binding is never zero-sized.
        let mut records: Vec<GpuLightRouting> =
            settings.records.iter().copied().map(Into::into).collect();
        if records.is_empty() {
            records.push(GpuLightRouting {
                channels: 0,
                layers: 0,
            });
        }

        let words = word_count(light_count);
        // Export buffers are keyed by cluster word; keep at least one element so
        // the read-write bindings are never zero-sized when no lights exist.
        let visible_elems = words.max(1) as u64;
        let layer_elems = (words.max(1) * MAX_LIGHT_LAYERS) as u64;
        let u32_size = size_of::<u32>() as u64;

        let routing = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism light routing records"),
            contents: bytemuck::cast_slice(&records),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let visible = device.create_buffer(&BufferDescriptor {
            label: Some("prism light routing visible"),
            size: visible_elems * u32_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layers = device.create_buffer(&BufferDescriptor {
            label: Some("prism light routing layers"),
            size: layer_elems * u32_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands.entity(entity).insert(ViewLightRouting {
            routing,
            visible,
            layers,
            light_count,
            word_count: words,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_element_counts_follow_the_word_count() {
        // Layer buffer is `MAX_LIGHT_LAYERS` deep per cluster word, visible is
        // one per word; both clamp to a single element so the bindings are
        // never zero-sized.
        let words = word_count(40);
        assert_eq!(words, 2);
        assert_eq!(words.max(1) * MAX_LIGHT_LAYERS, 8);

        let empty = word_count(0);
        assert_eq!(empty, 0);
        assert_eq!(empty.max(1), 1);
        assert_eq!(empty.max(1) * MAX_LIGHT_LAYERS, MAX_LIGHT_LAYERS);
    }
}
