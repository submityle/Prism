// Carve-application twin: evaluates on-device the single choke point that
// applies a carve delta to a base density, mirroring the CPU golden
// `apply_carve` (in `prism_render_architecture::volumetric::coupling`).
//
// Two-way scene coupling (design section 9d) carves cavities into the cloud via
// a negative density delta; `apply_carve` is the one place that guarantees the
// carving pipeline never produces a negative density (and therefore never a
// negative optical depth or transmittance): the sum is saturated into `0..=1`,
// so a large negative delta merely drives the density to `0` and an over-dense
// base clamps to `1`. The CPU golden `apply_carve` owns that math; this kernel
// is the on-device twin that runs one thread per query and reproduces the same
// value.
//
// The kernel contains no transcendental call at all - one add and a saturating
// clamp - so CPU and GPU evaluate the same closed-form algebra and the parity
// test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp and add - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard saturating density accumulation; no Unreal Engine source
// or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One carve-application query: the base density and the signed carve delta.
// Matches `GpuQuery` in `src/apply_carve.rs`.
struct Query {
    base_density: f32,
    delta: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Applies a carve delta to a base density, clamped into `0..=1`, mirroring
// `apply_carve`.
fn apply_carve(base_density: f32, delta: f32) -> f32 {
    return saturate_s(base_density + delta);
}

@compute @workgroup_size(64)
fn apply_carve_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = apply_carve(q.base_density, q.delta);
}
