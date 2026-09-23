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
    rows::{RenderDrawBinHeader, RenderVisibilityCounter},
    runtime::{PrismVisibilityDiagnostics, UnifiedVisibilitySettings, UnifiedVisibilityState},
};

#[derive(Resource, Default)]
pub(crate) struct VisibilityParityReadback {
    state: Mutex<Option<ReadbackState>>,
}

enum ReadbackState {
    CopyQueued(PendingReadback),
    Mapping(InFlightReadback),
}

struct PendingReadback {
    buffer: Buffer,
    cpu_work: Vec<super::rows::RenderVisibilityWorkItem>,
    cpu_ranges: Vec<prism_render_visibility::BufferRange>,
    compare_lod: bool,
    view_count: usize,
    slots_per_view: usize,
    expected_bin_counts: Vec<u32>,
    expected_late_bin_capacity: Vec<u32>,
    view_handles: Vec<(u32, u32)>,
}

struct InFlightReadback {
    pending: PendingReadback,
    receiver: Receiver<Result<(), bevy_render::render_resource::BufferAsyncError>>,
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
    let mut readback_state = readback.state.lock().unwrap();
    if state.views.is_empty() || readback_state.is_some() {
        diagnostics.parity_dropped_frames += u64::from(readback_state.is_some());
        return;
    }
    let Some((counters, work, bin_headers, late_counters, late_bin_headers)) =
        buffers.parity_readback_buffers()
    else {
        return;
    };
    let counter_size = state.views.len() as u64 * size_of::<RenderVisibilityCounter>() as u64;
    let work_size = state.views.len() as u64
        * buffers.gpu_slots_per_view() as u64
        * size_of::<super::rows::RenderVisibilityWorkItem>() as u64;
    let bin_size = state.draw_bins.iter().map(|view| view.bins.len()).sum::<usize>() as u64
        * size_of::<RenderDrawBinHeader>() as u64;
    let size = counter_size.saturating_mul(2) + work_size + bin_size.saturating_mul(2);
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
    encoder.copy_buffer_to_buffer(
        late_counters,
        0,
        &target,
        counter_size + work_size + bin_size,
        Some(counter_size),
    );
    if bin_size != 0 {
        encoder.copy_buffer_to_buffer(
            bin_headers,
            0,
            &target,
            counter_size + work_size,
            Some(bin_size),
        );
        encoder.copy_buffer_to_buffer(
            late_bin_headers,
            0,
            &target,
            counter_size.saturating_mul(2) + work_size + bin_size,
            Some(bin_size),
        );
    }
    pending.push_encoder(encoder, "prism visibility parity readback");
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
    *readback_state = Some(ReadbackState::CopyQueued(PendingReadback {
        buffer: target,
        cpu_work,
        cpu_ranges,
        compare_lod: true,
        view_count: state.views.len(),
        slots_per_view: buffers.gpu_slots_per_view() as usize,
        expected_bin_counts: state
            .draw_bins
            .iter()
            .flat_map(|view| view.bins.iter().map(|bin| bin.command_capacity))
            .collect(),
        expected_late_bin_capacity: state
            .draw_bins
            .iter()
            .flat_map(|view| view.bins.iter().map(|bin| bin.command_capacity))
            .collect(),
        view_handles: state
            .views
            .iter()
            .map(|view| (view.handle.index, view.handle.generation))
            .collect(),
    }));
}

/// Starts mapping only after `RenderGraphSystems::Submit` has submitted the copy.
pub(crate) fn map_submitted_visibility_parity_readback(
    device: Res<RenderDevice>,
    readback: Res<VisibilityParityReadback>,
) {
    let mut state = readback.state.lock().unwrap();
    let Some(ReadbackState::CopyQueued(pending)) = state.take() else {
        return;
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    device.map_buffer(&pending.buffer.slice(..), MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    *state = Some(ReadbackState::Mapping(InFlightReadback {
        pending,
        receiver,
    }));
}

pub(crate) fn collect_visibility_parity_readback(
    readback: Res<VisibilityParityReadback>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
) {
    let mut slot = readback.state.lock().unwrap();
    let Some(ReadbackState::Mapping(in_flight)) = slot.as_mut() else {
        return;
    };
    let result = match in_flight.receiver.try_recv() {
        Ok(result) => result,
        Err(mpsc::TryRecvError::Empty) => return,
        Err(mpsc::TryRecvError::Disconnected) => {
            diagnostics.parity_readback_failures += 1;
            *slot = None;
            return;
        }
    };
    if result.is_err() {
        diagnostics.parity_readback_failures += 1;
        *slot = None;
        return;
    }
    let pending = &in_flight.pending;
    let mapped = pending.buffer.slice(..).get_mapped_range().unwrap();
    let counter_bytes = pending.view_count * size_of::<RenderVisibilityCounter>();
    let counters: &[RenderVisibilityCounter] = bytemuck::cast_slice(&mapped[..counter_bytes]);
    let gpu_work: &[super::rows::RenderVisibilityWorkItem] =
        bytemuck::cast_slice(
            &mapped[counter_bytes
                ..counter_bytes
                    + pending.view_count
                        * pending.slots_per_view
                        * size_of::<super::rows::RenderVisibilityWorkItem>()],
        );
    let bin_bytes = counter_bytes
        + pending.view_count
            * pending.slots_per_view
            * size_of::<super::rows::RenderVisibilityWorkItem>();
    let one_bin_table_bytes = pending.expected_bin_counts.len() * size_of::<RenderDrawBinHeader>();
    let gpu_bins: &[RenderDrawBinHeader] =
        bytemuck::cast_slice(&mapped[bin_bytes..bin_bytes + one_bin_table_bytes]);
    let late_counter_start = bin_bytes + one_bin_table_bytes;
    let late_counter_end = late_counter_start + counter_bytes;
    let late_counters: &[RenderVisibilityCounter] =
        bytemuck::cast_slice(&mapped[late_counter_start..late_counter_end]);
    let gpu_late_bins: &[RenderDrawBinHeader] = bytemuck::cast_slice(&mapped[late_counter_end..]);
    if !bin_counts_match(gpu_bins, &pending.expected_bin_counts) {
        diagnostics.parity_mismatched_bin_counts += 1;
    }
    let late_summary = summarize_late_bins(gpu_late_bins, &pending.expected_late_bin_capacity);
    diagnostics.hzb_late_visible_commands += late_summary.visible_commands;
    diagnostics.hzb_late_overflowed_bins += late_summary.overflowed_bins;
    if !late_summary.valid {
        diagnostics.parity_mismatched_late_bin_counts += 1;
    }
    if !late_counters_match_bins(late_counters, gpu_late_bins, &pending.view_handles) {
        diagnostics.parity_mismatched_late_bin_counts += 1;
    }
    let mut matching = 0;
    let mut mismatched = 0;
    for (view_index, counter) in counters.iter().enumerate() {
        if counter.overflow_count != 0 {
            diagnostics.parity_overflowed_views += 1;
        }
        diagnostics.gpu_indexed_commands += counter.indexed_count as u64;
        diagnostics.gpu_non_indexed_commands += counter.non_indexed_count as u64;
        let bounded_visible = counter.visible_count.min(in_flight.pending.slots_per_view as u32);
        if counter.indexed_count.saturating_add(counter.non_indexed_count) != bounded_visible {
            diagnostics.parity_mismatched_command_counts += 1;
        }
        let count = counter.visible_count.min(pending.slots_per_view as u32) as usize;
        let gpu_start = view_index * pending.slots_per_view;
        let gpu_end = gpu_start + count;
        let cpu_range = pending.cpu_ranges[view_index];
        let cpu_start = cpu_range.start as usize;
        let cpu_end = cpu_start + cpu_range.count as usize;
        let same_count = counter.visible_count == cpu_range.count;
        let same_work = counter.overflow_count == 0
            && same_count
            && unordered_work_matches(
                &gpu_work[gpu_start..gpu_end],
                &pending.cpu_work[cpu_start..cpu_end],
                pending.compare_lod,
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
    pending.buffer.unmap();
    *slot = None;
}

fn bin_counts_match(gpu: &[RenderDrawBinHeader], expected: &[u32]) -> bool {
    gpu.len() == expected.len()
        && gpu
            .iter()
            .zip(expected)
            .all(|(header, expected)| {
                header.command_count == *expected
                    && header.command_count <= header.command_capacity
            })
}

#[derive(Debug, Default, Eq, PartialEq)]
struct LateBinSummary {
    visible_commands: u64,
    overflowed_bins: u64,
    valid: bool,
}

fn summarize_late_bins(gpu: &[RenderDrawBinHeader], capacities: &[u32]) -> LateBinSummary {
    let mut summary = LateBinSummary {
        valid: gpu.len() == capacities.len(),
        ..Default::default()
    };
    for (header, capacity) in gpu.iter().zip(capacities) {
        summary.visible_commands += u64::from(header.command_count.min(*capacity));
        if header.command_count > *capacity || header.command_capacity != *capacity {
            summary.overflowed_bins += 1;
            summary.valid = false;
        }
    }
    summary
}

fn late_counters_match_bins(
    counters: &[RenderVisibilityCounter],
    bins: &[RenderDrawBinHeader],
    view_handles: &[(u32, u32)],
) -> bool {
    counters.len() == view_handles.len()
        && counters.iter().zip(view_handles).all(|(counter, view)| {
            let view_bins = bins
                .iter()
                .filter(|bin| bin.view_index == view.0 && bin.view_generation == view.1);
            let bin_count = view_bins.clone()
                .map(|bin| bin.command_count.min(bin.command_capacity))
                .sum::<u32>();
            let indexed_count = view_bins.clone()
                .filter(|bin| bin.indexed != 0)
                .map(|bin| bin.command_count.min(bin.command_capacity))
                .sum::<u32>();
            let non_indexed_count = view_bins
                .filter(|bin| bin.indexed == 0)
                .map(|bin| bin.command_count.min(bin.command_capacity))
                .sum::<u32>();
            counter.visible_count == bin_count
                && counter.indexed_count == indexed_count
                && counter.non_indexed_count == non_indexed_count
                && counter.overflow_count == 0
        })
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

    #[test]
    fn command_counts_cover_bounded_visible_work() {
        let counter = RenderVisibilityCounter {
            visible_count: 7,
            indexed_count: 4,
            non_indexed_count: 3,
            ..Default::default()
        };
        assert_eq!(counter.indexed_count + counter.non_indexed_count, counter.visible_count);
    }

    #[test]
    fn bin_readback_requires_exact_bounded_counts() {
        let header = |count, capacity| RenderDrawBinHeader {
            command_count: count,
            command_capacity: capacity,
            ..Default::default()
        };
        assert!(bin_counts_match(&[header(2, 2), header(1, 3)], &[2, 1]));
        assert!(!bin_counts_match(&[header(3, 2)], &[3]));
        assert!(!bin_counts_match(&[header(1, 2)], &[2]));
    }

    #[test]
    fn late_bin_readback_accepts_sparse_counts_and_rejects_overflow() {
        let header = |count, capacity| RenderDrawBinHeader {
            command_count: count,
            command_capacity: capacity,
            ..Default::default()
        };
        assert_eq!(
            summarize_late_bins(&[header(0, 2), header(1, 3)], &[2, 3]),
            LateBinSummary {
                visible_commands: 1,
                overflowed_bins: 0,
                valid: true,
            }
        );
        assert_eq!(
            summarize_late_bins(&[header(4, 3)], &[3]),
            LateBinSummary {
                visible_commands: 3,
                overflowed_bins: 1,
                valid: false,
            }
        );
    }

    #[test]
    fn late_counters_cover_their_view_bins() {
        let bins = [
            RenderDrawBinHeader {
                view_index: 0,
                indexed: 1,
                command_count: 1,
                command_capacity: 2,
                ..Default::default()
            },
            RenderDrawBinHeader {
                view_index: 1,
                indexed: 0,
                command_count: 2,
                command_capacity: 2,
                ..Default::default()
            },
        ];
        let counters = [
            RenderVisibilityCounter {
                visible_count: 1,
                indexed_count: 1,
                ..Default::default()
            },
            RenderVisibilityCounter {
                visible_count: 2,
                non_indexed_count: 2,
                ..Default::default()
            },
        ];
        assert!(late_counters_match_bins(
            &counters,
            &bins,
            &[(0, 0), (1, 0)]
        ));
        let invalid = [RenderVisibilityCounter {
            visible_count: 3,
            non_indexed_count: 3,
            ..counters[1]
        }];
        assert!(!late_counters_match_bins(&invalid, &bins, &[(1, 0)]));
    }
}
