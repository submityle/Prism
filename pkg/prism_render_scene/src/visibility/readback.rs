use std::sync::{
    mpsc::{self, Receiver},
    Mutex,
};

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode},
    renderer::{PendingCommandBuffers, RenderDevice},
};

use super::{
    buffers::UnifiedVisibilityBuffers,
    rows::RenderVisibilityCounter,
    runtime::{PrismVisibilityDiagnostics, UnifiedVisibilitySettings, UnifiedVisibilityState},
};

fn parity_counts(counters: &[RenderVisibilityCounter], cpu: &[u32]) -> (u64, u64) {
    counters
        .iter()
        .zip(cpu)
        .fold((0, 0), |(matching, mismatched), (gpu, cpu)| {
            if gpu.visible_count == *cpu {
                (matching + 1, mismatched)
            } else {
                (matching, mismatched + 1)
            }
        })
}

#[derive(Resource, Default)]
pub(crate) struct VisibilityParityReadback {
    in_flight: Mutex<Option<InFlightReadback>>,
}

struct InFlightReadback {
    buffer: Buffer,
    receiver: Receiver<Result<(), bevy_render::render_resource::BufferAsyncError>>,
    cpu_counts: Vec<u32>,
    view_count: usize,
}

pub(crate) fn request_visibility_parity_readback(
    buffers: Res<UnifiedVisibilityBuffers>,
    state: Res<UnifiedVisibilityState>,
    settings: Res<UnifiedVisibilitySettings>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    readback: Res<VisibilityParityReadback>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
) {
    if !settings.gpu_parity_readback {
        return;
    }
    let mut in_flight = readback.in_flight.lock().unwrap();
    if state.views.is_empty() || in_flight.is_some() {
        diagnostics.parity_dropped_frames += u64::from(in_flight.is_some());
        return;
    }
    let Some(counters) = buffers.parity_counters() else {
        return;
    };
    let size = state.views.len() as u64 * size_of::<RenderVisibilityCounter>() as u64;
    let target = device.create_buffer(&BufferDescriptor {
        label: Some("prism visibility parity readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism visibility parity readback"),
    });
    encoder.copy_buffer_to_buffer(counters, 0, &target, 0, Some(size));
    pending.push_encoder(encoder, "prism visibility parity readback");
    let (sender, receiver) = mpsc::sync_channel(1);
    device.map_buffer(&target.slice(..), MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let cpu_counts = state
        .views
        .iter()
        .map(|view| {
            state
                .frame
                .views
                .get(&view.handle)
                .map_or(0, |output| output.visible_instances.count)
        })
        .collect();
    *in_flight = Some(InFlightReadback {
        buffer: target,
        receiver,
        cpu_counts,
        view_count: state.views.len(),
    });
}

pub(crate) fn collect_visibility_parity_readback(
    readback: Res<VisibilityParityReadback>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
) {
    let mut slot = readback.in_flight.lock().unwrap();
    let Some(in_flight) = slot.as_mut() else {
        return;
    };
    let Ok(result) = in_flight.receiver.try_recv() else {
        return;
    };
    if result.is_err() {
        diagnostics.parity_readback_failures += 1;
        *slot = None;
        return;
    }
    let mapped = in_flight.buffer.slice(..).get_mapped_range().unwrap();
    let counters: &[RenderVisibilityCounter] = bytemuck::cast_slice(&mapped);
    let (matching, mismatched) =
        parity_counts(&counters[..in_flight.view_count], &in_flight.cpu_counts);
    diagnostics.parity_matching_views += matching;
    diagnostics.parity_mismatched_views += mismatched;
    diagnostics.parity_frames += 1;
    drop(mapped);
    in_flight.buffer.unmap();
    *slot = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_parity_classifies_matching_and_mismatching_views() {
        let counters = [
            RenderVisibilityCounter {
                visible_count: 3,
                ..Default::default()
            },
            RenderVisibilityCounter {
                visible_count: 2,
                ..Default::default()
            },
        ];
        let cpu = [3, 4];
        assert_eq!(parity_counts(&counters, &cpu), (1, 1));
    }
}
