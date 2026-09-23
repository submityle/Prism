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

#[derive(Resource, Default)]
pub(crate) struct VisibilityParityReadback {
    in_flight: Mutex<Option<InFlightReadback>>,
}

struct InFlightReadback {
    buffer: Buffer,
    receiver: Receiver<Result<(), bevy_render::render_resource::BufferAsyncError>>,
    cpu_work: Vec<super::rows::RenderVisibilityWorkItem>,
    cpu_ranges: Vec<prism_render_visibility::BufferRange>,
    compare_lod: bool,
    view_count: usize,
    slots_per_view: usize,
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
    let Some((counters, work)) = buffers.parity_readback_buffers() else {
        return;
    };
    let counter_size = state.views.len() as u64 * size_of::<RenderVisibilityCounter>() as u64;
    let work_size = state.views.len() as u64
        * buffers.gpu_slots_per_view() as u64
        * size_of::<super::rows::RenderVisibilityWorkItem>() as u64;
    let size = counter_size + work_size;
    let target = device.create_buffer(&BufferDescriptor {
        label: Some("prism visibility parity readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism visibility parity readback"),
    });
    encoder.copy_buffer_to_buffer(counters, 0, &target, 0, Some(counter_size));
    encoder.copy_buffer_to_buffer(work, 0, &target, counter_size, Some(work_size));
    pending.push_encoder(encoder, "prism visibility parity readback");
    let (sender, receiver) = mpsc::sync_channel(1);
    device.map_buffer(&target.slice(..), MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let cpu_work = state
        .frame
        .work_items
        .iter()
        .copied()
        .map(super::rows::RenderVisibilityWorkItem::from)
        .collect();
    let cpu_ranges = state
        .views
        .iter()
        .map(|view| state.frame.views[&view.handle].visible_instances)
        .collect();
    *in_flight = Some(InFlightReadback {
        buffer: target,
        receiver,
        cpu_work,
        cpu_ranges,
        compare_lod: false,
        view_count: state.views.len(),
        slots_per_view: buffers.gpu_slots_per_view() as usize,
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
    let counter_bytes = in_flight.view_count * size_of::<RenderVisibilityCounter>();
    let counters: &[RenderVisibilityCounter] = bytemuck::cast_slice(&mapped[..counter_bytes]);
    let gpu_work: &[super::rows::RenderVisibilityWorkItem] =
        bytemuck::cast_slice(&mapped[counter_bytes..]);
    let mut matching = 0;
    let mut mismatched = 0;
    for (view_index, counter) in counters.iter().enumerate() {
        let count = counter.visible_count.min(in_flight.slots_per_view as u32) as usize;
        let gpu_start = view_index * in_flight.slots_per_view;
        let gpu_end = gpu_start + count;
        let cpu_range = in_flight.cpu_ranges[view_index];
        let cpu_start = cpu_range.start as usize;
        let cpu_end = cpu_start + cpu_range.count as usize;
        let same_count = counter.visible_count == cpu_range.count;
        let same_work = same_count
            && unordered_work_matches(
                &gpu_work[gpu_start..gpu_end],
                &in_flight.cpu_work[cpu_start..cpu_end],
                in_flight.compare_lod,
            );
        if same_work {
            matching += 1;
        } else {
            mismatched += 1;
        }
    }
    diagnostics.parity_matching_views += matching;
    diagnostics.parity_mismatched_views += mismatched;
    diagnostics.parity_frames += 1;
    drop(mapped);
    in_flight.buffer.unmap();
    *slot = None;
}

fn unordered_work_matches(
    gpu: &[super::rows::RenderVisibilityWorkItem],
    cpu: &[super::rows::RenderVisibilityWorkItem],
    compare_lod: bool,
) -> bool {
    let mut gpu = gpu.to_vec();
    let mut cpu = cpu.to_vec();
    gpu.sort_by_key(|work| work.scene_index);
    cpu.sort_by_key(|work| work.scene_index);
    gpu.iter().zip(cpu).all(|(gpu, cpu)| {
        gpu.scene_index == cpu.scene_index
            && gpu.scene_generation == cpu.scene_generation
            && gpu.geometry_index == cpu.geometry_index
            && gpu.geometry_generation == cpu.geometry_generation
            && gpu.material_index == cpu.material_index
            && gpu.pass_mask == cpu.pass_mask
            && (!compare_lod || gpu.lod_or_cluster == cpu.lod_or_cluster)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unordered_work_parity_ignores_compaction_order() {
        let work = |scene_index| super::super::rows::RenderVisibilityWorkItem {
            scene_index,
            scene_generation: 1,
            geometry_index: scene_index,
            geometry_generation: 1,
            material_index: 0,
            pass_mask: 1,
            ..Default::default()
        };
        assert!(unordered_work_matches(
            &[work(7), work(2)],
            &[work(2), work(7)],
            false,
        ));
        assert!(!unordered_work_matches(&[work(7)], &[work(2)], false));
    }
}
