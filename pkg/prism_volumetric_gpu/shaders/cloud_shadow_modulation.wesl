// Cloud-shadow ground-modulation twin: evaluates on-device the lit-ground
// factor from the cloud shadow, mirroring the CPU golden
// `cloud_shadow_modulation` (in
// `prism_render_architecture::volumetric::coupling`).
//
// Two-way scene coupling (design section 9d) closes here: the cloud's
// transmittance controls how much sunlight reaches the ground, and the ground
// albedo controls how much of that light is reflected back to modulate the
// aerial perspective / GI sky-light. The lit-ground factor is the product of
// the two saturated inputs and therefore stays in `0..=1`; a fully opaque cloud
// (`transmittance == 0`) drives the ground contribution to `0`. The CPU golden
// `cloud_shadow_modulation` owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The kernel contains no transcendental call at all: it is two saturating
// clamps and a multiply, so CPU and GPU evaluate the same closed-form algebra
// and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp and multiply - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard cloud-shadow ground coupling; no Unreal Engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One modulation query: the cloud transmittance reaching the ground and the
// ground albedo. Matches `GpuQuery` in `src/cloud_shadow_modulation.rs`.
struct Query {
    cloud_transmittance: f32,
    ground_albedo: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Lit-ground factor from the cloud shadow, mirroring
// `cloud_shadow_modulation`.
fn cloud_shadow_modulation(cloud_transmittance: f32, ground_albedo: f32) -> f32 {
    return saturate_s(saturate_s(ground_albedo) * saturate_s(cloud_transmittance));
}

@compute @workgroup_size(64)
fn cloud_shadow_modulation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = cloud_shadow_modulation(q.cloud_transmittance, q.ground_albedo);
}
