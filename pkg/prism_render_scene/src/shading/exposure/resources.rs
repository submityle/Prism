//! Persistent per-view GPU buffers backing auto-exposure.
//!
//! Two storage buffers travel with every view that runs the Prism visibility
//! path:
//!
//! 1. `histogram` — a 64-bin log-luminance histogram the build pass fills with
//!    `atomicAdd` and the resolve pass reduces then zeroes for the next frame.
//!    It is transient (rebuilt every frame) so it only has to be zero on
//!    creation, which `wgpu` guarantees.
//! 2. `state` — the [`GpuExposureState`] the resolve pass integrates across
//!    frames (the smoothed adapted luminance + the exposure multiplier the
//!    composite applies). It is **persistent**: created once, never re-created,
//!    so eye adaptation carries between frames. It is seeded to a no-op multiply
//!    with a mid-grey adapted luminance so the very first frame is neither
//!    black-clamped nor over-exposed.
//!
//! Unlike the trace/gather targets, these buffers are **not** gated on
//! `enable_exposure`: the composite always binds the state buffer (multiplying
//! by a stationary `1.0` when exposure is disabled), so it must always exist.
//! Only the build/resolve passes and their bind groups are gated, so a disabled
//! frame leaves the state parked at its last value.

use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{GpuExposureState, EXPOSURE_BIN_COUNT};

/// The two persistent exposure buffers for one view. Present on every view that
/// carries a [`ViewVisibilityBuffer`]; created once and never rebuilt so the
/// eye-adaptation state survives across frames.
#[derive(Component)]
pub(crate) struct ViewExposureBuffers {
    /// 64-bin log-luminance histogram. `read_write` storage: the build pass
    /// atomically accumulates it, the resolve pass reduces then zeroes it.
    histogram: Buffer,
    /// Persistent [`GpuExposureState`]: the resolve pass reads last frame's
    /// adaptation, writes the new multiplier; the composite reads the
    /// multiplier. Never re-created.
    state: Buffer,
}

impl ViewExposureBuffers {
    /// The histogram storage buffer, bound by both exposure passes.
    pub(crate) fn histogram_buffer(&self) -> &Buffer {
        &self.histogram
    }

    /// The persistent exposure-state storage buffer, bound by the resolve pass
    /// (read/write) and the composite (read-only).
    pub(crate) fn state_buffer(&self) -> &Buffer {
        &self.state
    }
}

/// Creates [`ViewExposureBuffers`] once for every view that has a resident
/// [`ViewVisibilityBuffer`], and never touches them again.
///
/// Intentionally ungated on `enable_exposure`: the composite unconditionally
/// binds the state buffer, so it must exist whenever the composite runs (i.e.
/// whenever the visibility buffer does). The histogram is zero-initialised
/// (guaranteed by `wgpu`) so the first build accumulates from empty; the state
/// is seeded to a no-op `1.0` multiply with a mid-grey adapted luminance so the
/// first frame is neither black-clamped nor blown out. Because the state is
/// persistent, existing buffers are left untouched even across a viewport
/// resize — a stale histogram size is impossible since it is fixed at 64 bins.
pub(crate) fn prepare_exposure_buffers(
    mut commands: Commands,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera), (With<ViewVisibilityBuffer>, Without<ViewExposureBuffers>)>,
) {
    for (entity, camera) in &views {
        // A view without a known viewport never dispatched the visibility path,
        // so there is nothing to meter yet; wait until it has an extent.
        if camera.physical_viewport_size.is_none() {
            continue;
        }

        let histogram = device.create_buffer(&BufferDescriptor {
            label: Some("prism exposure histogram"),
            size: u64::from(EXPOSURE_BIN_COUNT) * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism exposure state"),
            contents: bytemuck::bytes_of(&GpuExposureState::default()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        commands
            .entity(entity)
            .insert(ViewExposureBuffers { histogram, state });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buffer_is_one_u32_per_bin() {
        assert_eq!(
            u64::from(EXPOSURE_BIN_COUNT) * size_of::<u32>() as u64,
            256
        );
    }
}
