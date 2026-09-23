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
struct GeometryHeader {
    generation: u32, revision: u32, lod_offset: u32, lod_count: u32,
    primitive_start: u32, primitive_count: u32, flags: u32, padding: u32,
}
struct GeometryLod {
    element_count: u32, first_element: u32, base_vertex: i32, vertex_count: u32,
    vertex_buffer_class: u32, index_buffer_class: u32, flags: u32, level: u32,
    screen_error: f32, padding: vec3<u32>,
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
@group(0) @binding(5) var<storage, read> geometry_headers: array<GeometryHeader>;
@group(0) @binding(6) var<storage, read> geometry_lods: array<GeometryLod>;
var<immediate> dispatch: LateCompactDispatch;

@compute @workgroup_size(64)
fn compact_late_hzb(@builtin(global_invocation_id) id: vec3<u32>) {
    let candidate = id.x;
    if candidate >= dispatch.candidate_count { return; }
    let stage_index = dispatch.stage_start + candidate;
    // Only candidates rejected by previous-HZB and accepted by current-HZB
    // belong to the late stream. EARLY|LATE_VISIBLE items are already owned by
    // the early stream and must not be emitted twice.
    if stage_index >= arrayLength(&stages) || (stages[stage_index] & 6u) != 6u { return; }
    let lookup = dispatch.candidate_bin_start + candidate;
    if lookup >= arrayLength(&candidate_bins) { return; }
    let local_bin = candidate_bins[lookup];
    let bin_index = dispatch.bin_start + local_bin;
    if local_bin == 0xffffffffu || bin_index >= arrayLength(&late_bins) { return; }
    let bin = late_bins[bin_index];
    if bin.geometry_index >= arrayLength(&geometry_headers) { return; }
    let geometry = geometry_headers[bin.geometry_index];
    if geometry.generation != bin.geometry_generation || (geometry.flags & 1u) == 0u { return; }
    var lod_index = 0xffffffffu;
    for (var local_lod = 0u; local_lod < geometry.lod_count; local_lod += 1u) {
        let index = geometry.lod_offset + local_lod;
        if index >= arrayLength(&geometry_lods) { break; }
        let candidate_lod = geometry_lods[index];
        if candidate_lod.level == bin.lod_or_cluster && (candidate_lod.flags & 5u) == 5u {
            lod_index = index;
            break;
        }
    }
    if lod_index == 0xffffffffu { return; }
    let lod = geometry_lods[lod_index];
    let indexed = (lod.flags & 2u) != 0u;
    if indexed != (bin.indexed != 0u) { return; }
    let local = atomicAdd(&late_bins[bin_index].command_count, 1u);
    if local >= bin.command_capacity { return; }
    let command = dispatch.command_start + late_bins[bin_index].command_start + local;
    if command >= dispatch.command_end
        || command >= arrayLength(&late_indexed)
        || command >= arrayLength(&late_non_indexed) { return; }
    let instance = select(0u, candidate, dispatch.indirect_first_instance != 0u);
    late_indexed[command] = VisibilityIndirect(
        select(0u, lod.element_count, indexed),
        1u, lod.first_element, lod.base_vertex, instance,
    );
    late_non_indexed[command] = VisibilityNonIndexedIndirect(
        select(lod.element_count, 0u, indexed),
        1u, lod.first_element, instance,
    );
}
