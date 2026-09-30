// Uniform-grid "find cell start" pass, device kernel.
//
// One invocation per sorted key. After the particles have been stably sorted by
// their cell index, adjacent equal keys form each cell's contiguous run. A
// thread marks the run boundaries it sees: it writes `cell_start[c] = i` when it
// is the first entry of cell `c` (the array start, or its left neighbour is a
// different cell) and `cell_end[c] = i + 1` when it is the last (the array end,
// or its right neighbour differs). The `||` short-circuits, so the neighbour
// load never runs at index 0 or `n - 1` and stays in bounds.
//
// `cell_start`/`cell_end` are pre-filled with the EMPTY sentinel by the host, so
// cells that no thread touches keep the sentinel, matching `cpu_grid_sort`.
//
// Provenance: the sort-then-find-cell-start construction of Green, "Particle
// Simulation using CUDA" (NVIDIA 2008). No Unreal Engine source or derived code.

struct Params {
    // Number of sorted keys (particles).
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Cell indices in ascending sorted order.
@group(0) @binding(1) var<storage, read> sorted_cells: array<u32>;
// Output: per-cell first position into the sorted arrays.
@group(0) @binding(2) var<storage, read_write> cell_start: array<u32>;
// Output: per-cell one-past-last position into the sorted arrays.
@group(0) @binding(3) var<storage, read_write> cell_end: array<u32>;

@compute @workgroup_size(64)
fn cell_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let cell = sorted_cells[i];
    if (i == 0u || sorted_cells[i - 1u] != cell) {
        cell_start[cell] = i;
    }
    if (i == params.n - 1u || sorted_cells[i + 1u] != cell) {
        cell_end[cell] = i + 1u;
    }
}
