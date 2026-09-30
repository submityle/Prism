// God-ray scattering-mask twin: gates volumetric-beam injection at a fragment,
// mirroring the CPU golden `scattering_mask` (in
// `prism_render_architecture::volumetric::shadow`).
//
// Volumetric beams are visible where light reaches the fragment (high
// `shadow_transmittance`) AND there is medium to scatter off (nonzero
// density). The mask is the product of the two saturated terms, so it is
// always in `[0, 1]` and vanishes in either full shadow or clear air. The CPU
// golden owns that math; this kernel is the on-device twin that runs one
// thread per query and reproduces the same value.
//
// The mask contains NO transcendental call - it is two saturates and a single
// multiply - so CPU and GPU evaluate the identical closed-form algebra with no
// room for fma contraction (a lone multiply cannot be fused), which lets the
// parity test assert the tightest tolerance.
//
// It uses only the portable core-WGSL subset - clamp and multiply - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard screen-space god-ray gating; no Unreal Engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One mask query: the fragment's shadow transmittance and local density.
// Matches `GpuQuery` in `src/mask.rs`.
struct Query {
    shadow_transmittance: f32,
    density: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// God-ray scattering mask, mirroring `scattering_mask`.
fn scattering_mask(shadow_transmittance: f32, density: f32) -> f32 {
    return saturate_s(shadow_transmittance) * saturate_s(density);
}

@compute @workgroup_size(64)
fn mask_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = scattering_mask(q.shadow_transmittance, q.density);
}
