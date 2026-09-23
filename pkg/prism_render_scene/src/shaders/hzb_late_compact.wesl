//! Builds an independent late command stream from candidates accepted by the
//! current-frame HZB. Early atomics are never reused.

struct VisibilityIndirect {
    index_count: u32, instance_count: u32, first_index: u32,
    base_vertex: i32, first_instance: u32,
}
struct VisibilityNonIndexedIndirect {
    vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32,
}
struct DrawBinHeader {
    geometry_index: u32, geometry_generation: u32, pipeline_class: u32,
    vertex_buffer_class: u32, index_buffer_class: u32, indexed: u32,
    command_start: u32, command_capacity: u32, command_count: atomic<u32>,
    view_index: u32, view_generation: u32, lod_or_cluster: u32,
    primitive_kind: u32, padding_tail: vec3<u32>,
}
struct LateCompactDispatch {
    candidate_count: u32, stage_start: u32, candidate_bin_start: u32,
    bin_start: u32, command_start: u32, command_end: u32,
    indirect_first_instance: u32, padding: u32,
}

@group(0) @binding(0) var<storage, read> stages: array<u32>;
@group(0) @binding(1) var<storage, read> candidate_bins: array<u32>;
@group(0) @binding(2) var<storage, read_write> late_bins: array<DrawBinHeader>;
@group(0) @binding(3) var<storage, read_write> late_indexed: array<VisibilityIndirect>;
@group(0) @binding(4) var<storage, read_write> late_non_indexed: array<VisibilityNonIndexedIndirect>;
var<immediate> dispatch: LateCompactDispatch;

@compute @workgroup_size(64)
fn compact_late_hzb(@builtin(global_invocation_id) id: vec3<u32>) {
    let candidate = id.x;
    if candidate >= dispatch.candidate_count { return; }
    let stage_index = dispatch.stage_start + candidate;
    if stage_index >= arrayLength(&stages) || (stages[stage_index] & 4u) == 0u { return; }
    let lookup = dispatch.candidate_bin_start + candidate;
    if lookup >= arrayLength(&candidate_bins) { return; }
    let local_bin = candidate_bins[lookup];
    let bin_index = dispatch.bin_start + local_bin;
    if local_bin == 0xffffffffu || bin_index >= arrayLength(&late_bins) { return; }
    let local = atomicAdd(&late_bins[bin_index].command_count, 1u);
    if local >= late_bins[bin_index].command_capacity { return; }
    let command = dispatch.command_start + late_bins[bin_index].command_start + local;
    if command >= dispatch.command_end { return; }
    let instance = select(0u, candidate, dispatch.indirect_first_instance != 0u);
    // Geometry element ranges are filled by the merge/resolve kernel in the
    // next slice; zero instance_count makes an incomplete command harmless.
    late_indexed[command] = VisibilityIndirect(0u, 0u, 0u, 0, instance);
    late_non_indexed[command] = VisibilityNonIndexedIndirect(0u, 0u, 0u, instance);
}
