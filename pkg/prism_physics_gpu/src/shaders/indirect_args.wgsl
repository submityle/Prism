// Builds a `dispatch_workgroups_indirect` argument block from a device-side
// count, so a GPU-driven pipeline chooses its own launch size with no host
// round-trip. Golden twin: `indirect::args::cpu_dispatch_args`.
//
// `counts[count_offset]` is the work-item count a prior stage wrote. The grid
// is `ceil(count / workgroup_size)` groups along x, tiling the overflow into y
// once the x total would exceed `max_dim`. `args[0..3]` is the indirect launch
// the driver reads; `args[3]` publishes the x-row width for a consumer that
// linearises a tiled launch.
//
// Provenance: standard ceil-division grid sizing for indirect dispatch. No
// Unreal Engine source or derived code.

struct Params {
    workgroup_size: u32,
    count_offset: u32,
    max_dim: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> args: array<u32, 4>;

@compute @workgroup_size(1)
fn build_args() {
    let count = counts[params.count_offset];
    let wg = max(params.workgroup_size, 1u);
    let total = (count + wg - 1u) / wg;
    if (total == 0u) {
        args[0] = 0u;
        args[1] = 1u;
        args[2] = 1u;
        args[3] = 0u;
        return;
    }
    let max_dim = max(params.max_dim, 1u);
    if (total <= max_dim) {
        args[0] = total;
        args[1] = 1u;
        args[2] = 1u;
        args[3] = total;
    } else {
        let x = max_dim;
        let y = (total + max_dim - 1u) / max_dim;
        args[0] = x;
        args[1] = y;
        args[2] = 1u;
        args[3] = x;
    }
}
