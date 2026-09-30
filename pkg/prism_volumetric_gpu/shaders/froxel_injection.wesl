// Froxel injection-weight twin: evaluates on-device the per-depth-slice weight
// used to add fog energy into the shared froxel volume, mirroring the CPU
// golden `froxel_injection_weight` (in
// `prism_render_architecture::volumetric::fog`).
//
// The unified volumetric fog (design section 9f) only *adds* energy to the
// shared froxel volume, never overwriting deeper contributions: near slices
// (small depth_slice) receive full weight and far slices taper linearly to
// zero. The weight is
//   0                                     when slice_count == 0
//   saturate(1 - min(depth_slice, slice_count) / slice_count)   otherwise
// so it always lands in `0..=1` and clamping depth_slice to the slice count
// keeps an out-of-range index from producing a negative weight. The CPU golden
// `froxel_injection_weight` owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The kernel contains no transcendental call - an unsigned min, a
// division and a saturating clamp - so CPU and GPU evaluate the same
// closed-form algebra and the parity test can assert a tight tolerance. The
// inputs are unsigned integers, matching the CPU signature exactly.
//
// It uses only the portable core-WGSL subset - unsigned min, integer-to-float
// conversion, divide and clamp - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard additive froxel-injection weighting; no Unreal Engine
// source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One froxel-injection query: the depth slice index and the total slice count.
// Matches `GpuQuery` in `src/froxel_injection.rs`.
struct Query {
    depth_slice: u32,
    slice_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Injection weight for a froxel depth slice into the shared froxel volume,
// mirroring `froxel_injection_weight`.
fn froxel_injection_weight(depth_slice: u32, slice_count: u32) -> f32 {
    if (slice_count == 0u) {
        return 0.0;
    }
    let slice = min(depth_slice, slice_count);
    return saturate_s(1.0 - f32(slice) / f32(slice_count));
}

@compute @workgroup_size(64)
fn froxel_injection_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = froxel_injection_weight(q.depth_slice, q.slice_count);
}
