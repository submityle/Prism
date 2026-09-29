//! GPU->CPU readback bridge exposing the material-classification fault counters.
//!
//! The `classify_count` / `scatter_work` passes
//! ([`super::classification_gpu::dispatch_material_classification`]) atomically
//! accumulate a four-slot `diagnostics` buffer per view every frame
//! (background / stale / unsupported / overflow, see
//! `shaders/material_classification.wesl`). Without a readback that data is
//! write-only on the device and invisible to the CPU; this module copies it
//! back one frame late and folds it into the public
//! [`PrismShadingDiagnostics::gpu_classification`] so callers can observe what
//! the device actually classified (and detect faults) through the existing
//! diagnostics API.
//!
//! It runs the same three-stage `RenderGraph` state machine as the
//! virtual-shadow-map page readback: `request` enqueues the copy after the
//! render graph has recorded (so it captures this frame's counters),
//! `map_submitted` starts the async map once command buffers are submitted, and
//! `collect` consumes the mapped bytes the next frame. State is keyed by
//! [`RetainedViewEntity`] because the render-world `Entity` is rebuilt each
//! frame and cannot be held across the one-frame latency. Because the public
//! resource is global, per-view counters are aggregated (saturating sum) across
//! every live view.

use std::sync::{
    mpsc::{self, Receiver},
    Mutex,
};

use bevy_ecs::prelude::*;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        Buffer, BufferAsyncError, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
        MapMode,
    },
    renderer::{PendingCommandBuffers, RenderDevice},
    view::{ExtractedView, RetainedViewEntity},
};

use super::resources::ViewShadingBuffers;
use super::runtime::{GpuClassificationCounters, PrismShadingDiagnostics, PrismShadingSettings};

/// Number of `u32` counters in the per-view classification diagnostics buffer.
const DIAGNOSTICS_SLOTS: usize = 4;
/// Byte size of the diagnostics buffer copied back each frame.
const DIAGNOSTICS_BYTES: u64 = (DIAGNOSTICS_SLOTS * size_of::<u32>()) as u64;

/// Render-world resource holding each view's in-flight diagnostics readback plus
/// the most recent decoded counters.
#[derive(Resource, Default)]
pub(crate) struct ClassificationDiagnosticsReadback {
    inner: Mutex<ReadbackInner>,
}

#[derive(Default)]
struct ReadbackInner {
    /// Copies queued or mapping, keyed by view. At most one per view in flight.
    states: HashMap<RetainedViewEntity, ReadbackState>,
    /// Last successfully decoded counters per view, aggregated at collect time.
    latest: HashMap<RetainedViewEntity, GpuClassificationCounters>,
}

/// One view's readback lifecycle: a copy queued this frame, or a map in flight
/// awaiting the mapping callback.
enum ReadbackState {
    CopyQueued(Buffer),
    Mapping {
        buffer: Buffer,
        receiver: Receiver<Result<(), BufferAsyncError>>,
    },
}

/// Decodes the four raw counter slots into structured counters, tolerating a
/// short slice by treating missing slots as zero.
fn decode_counters(slots: &[u32]) -> GpuClassificationCounters {
    GpuClassificationCounters {
        background_pixels: slots.first().copied().unwrap_or(0),
        stale_pixels: slots.get(1).copied().unwrap_or(0),
        unsupported_pixels: slots.get(2).copied().unwrap_or(0),
        overflow_pixels: slots.get(3).copied().unwrap_or(0),
    }
}

/// Aggregates per-view counters into the single global counter reported through
/// [`PrismShadingDiagnostics`].
fn aggregate<I>(counters: I) -> GpuClassificationCounters
where
    I: IntoIterator<Item = GpuClassificationCounters>,
{
    counters
        .into_iter()
        .fold(GpuClassificationCounters::default(), |acc, c| {
            acc.saturating_add(c)
        })
}

/// Enqueues a copy of each view's classification diagnostics buffer into a
/// mappable buffer (`RenderGraphSystems::Render`, after `camera_driver`, so the
/// copy observes this frame's classify/scatter writes).
///
/// A view whose previous readback is still in flight is skipped (its frame is
/// dropped) so the one-frame-latency state machine never overlaps two copies
/// for the same view.
pub(crate) fn request_classification_readback(
    settings: Res<PrismShadingSettings>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    readback: Res<ClassificationDiagnosticsReadback>,
    views: Query<(&ExtractedView, &ViewShadingBuffers)>,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let mut inner = readback.inner.lock().unwrap();
    for (view, buffers) in &views {
        let retained = view.retained_view_entity;
        if inner.states.contains_key(&retained) {
            continue;
        }
        let target = device.create_buffer(&BufferDescriptor {
            label: Some("prism classification diagnostics readback"),
            size: DIAGNOSTICS_BYTES,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism classification diagnostics readback"),
        });
        encoder.copy_buffer_to_buffer(
            &buffers.diagnostics,
            0,
            &target,
            0,
            Some(DIAGNOSTICS_BYTES),
        );
        pending.push_encoder(encoder, "prism classification diagnostics readback");
        inner
            .states
            .insert(retained, ReadbackState::CopyQueued(target));
    }
}

/// Starts the async buffer map for every queued copy, once
/// `RenderGraphSystems::Finish` has submitted the command buffers.
pub(crate) fn map_submitted_classification_readback(
    device: Res<RenderDevice>,
    readback: Res<ClassificationDiagnosticsReadback>,
) {
    let mut inner = readback.inner.lock().unwrap();
    let queued: Vec<RetainedViewEntity> = inner
        .states
        .iter()
        .filter_map(|(retained, state)| {
            matches!(state, ReadbackState::CopyQueued(_)).then_some(*retained)
        })
        .collect();
    for retained in queued {
        let Some(ReadbackState::CopyQueued(buffer)) = inner.states.remove(&retained) else {
            continue;
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        device.map_buffer(&buffer.slice(..), MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        inner
            .states
            .insert(retained, ReadbackState::Mapping { buffer, receiver });
    }
}

/// Consumes each mapped diagnostics buffer (`RenderGraphSystems::Begin`, next
/// frame): decodes the counters, drops views that no longer exist and folds the
/// aggregate into [`PrismShadingDiagnostics::gpu_classification`].
///
/// When visibility-buffer shading is disabled every counter is cleared so no
/// stale fault total leaks into a later re-enable.
pub(crate) fn collect_classification_readback(
    settings: Res<PrismShadingSettings>,
    readback: Res<ClassificationDiagnosticsReadback>,
    mut diagnostics: ResMut<PrismShadingDiagnostics>,
    views: Query<&ExtractedView>,
) {
    let mut inner = readback.inner.lock().unwrap();
    if !settings.enable_visibility_buffer {
        inner.states.clear();
        inner.latest.clear();
        diagnostics.gpu_classification = GpuClassificationCounters::default();
        return;
    }

    let retained_keys: Vec<RetainedViewEntity> = inner.states.keys().copied().collect();
    for retained in retained_keys {
        let Some(state) = inner.states.remove(&retained) else {
            continue;
        };
        let ReadbackState::Mapping { buffer, receiver } = state else {
            // Copy queued but not yet mapped: keep it for map_submitted.
            inner.states.insert(retained, state);
            continue;
        };
        match receiver.try_recv() {
            Ok(Ok(())) => {}
            Err(mpsc::TryRecvError::Empty) => {
                // Mapping not ready yet: re-insert and try again next frame.
                inner
                    .states
                    .insert(retained, ReadbackState::Mapping { buffer, receiver });
                continue;
            }
            // Mapping failed or the callback channel dropped: discard so a fresh
            // copy is queued next frame.
            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => continue,
        }

        let counters = {
            let mapped = buffer.slice(..).get_mapped_range().unwrap();
            let counters = decode_counters(bytemuck::cast_slice::<u8, u32>(&mapped));
            drop(mapped);
            counters
        };
        buffer.unmap();
        inner.latest.insert(retained, counters);
    }

    // Drop counters and in-flight copies for views that no longer exist so a
    // removed camera's faults do not linger in the aggregate.
    let live: HashSet<RetainedViewEntity> =
        views.iter().map(|view| view.retained_view_entity).collect();
    inner.latest.retain(|retained, _| live.contains(retained));
    inner.states.retain(|retained, _| live.contains(retained));

    diagnostics.gpu_classification = aggregate(inner.latest.values().copied());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_maps_slots_to_named_counters() {
        let counters = decode_counters(&[7, 3, 2, 5]);
        assert_eq!(counters.background_pixels, 7);
        assert_eq!(counters.stale_pixels, 3);
        assert_eq!(counters.unsupported_pixels, 2);
        assert_eq!(counters.overflow_pixels, 5);
    }

    #[test]
    fn decode_tolerates_short_slices() {
        assert_eq!(decode_counters(&[]), GpuClassificationCounters::default());
        let counters = decode_counters(&[9, 1]);
        assert_eq!(counters.background_pixels, 9);
        assert_eq!(counters.stale_pixels, 1);
        assert_eq!(counters.unsupported_pixels, 0);
        assert_eq!(counters.overflow_pixels, 0);
    }

    #[test]
    fn aggregate_sums_every_view() {
        let a = GpuClassificationCounters {
            background_pixels: 10,
            stale_pixels: 1,
            unsupported_pixels: 0,
            overflow_pixels: 2,
        };
        let b = GpuClassificationCounters {
            background_pixels: 5,
            stale_pixels: 0,
            unsupported_pixels: 4,
            overflow_pixels: 1,
        };
        let total = aggregate([a, b]);
        assert_eq!(total.background_pixels, 15);
        assert_eq!(total.stale_pixels, 1);
        assert_eq!(total.unsupported_pixels, 4);
        assert_eq!(total.overflow_pixels, 3);
    }

    #[test]
    fn aggregate_of_nothing_is_default() {
        assert_eq!(
            aggregate(std::iter::empty()),
            GpuClassificationCounters::default()
        );
    }

    #[test]
    fn faults_ignore_background_but_flag_real_errors() {
        let background_only = GpuClassificationCounters {
            background_pixels: 100,
            ..Default::default()
        };
        assert!(!background_only.has_faults());

        for counters in [
            GpuClassificationCounters {
                stale_pixels: 1,
                ..Default::default()
            },
            GpuClassificationCounters {
                unsupported_pixels: 1,
                ..Default::default()
            },
            GpuClassificationCounters {
                overflow_pixels: 1,
                ..Default::default()
            },
        ] {
            assert!(counters.has_faults());
        }
    }

    #[test]
    fn saturating_add_does_not_wrap() {
        let max = GpuClassificationCounters {
            background_pixels: u32::MAX,
            stale_pixels: u32::MAX,
            unsupported_pixels: u32::MAX,
            overflow_pixels: u32::MAX,
        };
        let one = GpuClassificationCounters {
            background_pixels: 1,
            stale_pixels: 1,
            unsupported_pixels: 1,
            overflow_pixels: 1,
        };
        assert_eq!(max.saturating_add(one), max);
    }
}
