// Early-termination twin: decides whether a raymarch column has become opaque
// enough to stop, mirroring the CPU golden `should_early_terminate` (in
// `prism_render_architecture::volumetric::raymarch`).
//
// As a volumetric ray marches, its accumulated `transmittance` decays
// monotonically toward zero; once it falls below the config cutoff the
// remaining medium contributes negligibly and the walk stops early to save work
// (design section 6). This kernel evaluates the predicate per query:
//   transmittance < transmittance_cutoff
// The CPU golden owns that logic; this kernel is the on-device twin that runs
// one thread per query and reproduces the same decision.
//
// The result is a u32 boolean (0 = continue, 1 = terminate). It uses only the
// portable core-WGSL subset - a single float comparison - so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard transmittance-cutoff early termination; no Unreal Engine
// source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One termination query: the accumulated transmittance and the config cutoff
// below which the walk stops. Matches `GpuQuery` in
// `src/should_early_terminate.rs`.
struct Query {
    transmittance: f32,
    transmittance_cutoff: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// The predicate, mirroring `should_early_terminate`.
fn should_early_terminate(transmittance: f32, transmittance_cutoff: f32) -> bool {
    return transmittance < transmittance_cutoff;
}

@compute @workgroup_size(64)
fn should_early_terminate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = select(0u, 1u, should_early_terminate(q.transmittance, q.transmittance_cutoff));
}
