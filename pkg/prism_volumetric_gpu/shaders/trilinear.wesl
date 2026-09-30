// Trilinear corner-weights twin: builds on-device the eight unit-cube corner
// weights of a trilinear blend, mirroring the CPU golden `trilinear_weights`
// (in `prism_render_architecture::volumetric::multiscatter`).
//
// The multi-scatter LUT and the scatter probe grid both interpolate their
// tabulated energy with a trilinear blend (design section 7b). Corner `k` uses
// bit 2 for `fx`, bit 1 for `fy` and bit 0 for `fz`; each axis contributes
// `frac` or `1 - frac`, so the eight weights sum to exactly one for any
// fractions (a partition of unity), which keeps interpolated energy bounded.
// The CPU golden `trilinear_weights` owns that math; this kernel is the
// on-device twin that runs one thread per query and returns the same eight
// weights.
//
// The blend is only multiply/add on the raw fractions - no transcendental, no
// clamp, no optional device feature - so CPU and GPU evaluate the identical
// closed-form algebra and differ at most by a legal multiply-add contraction of
// a few ULP. It uses only the portable core-WGSL subset, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard trilinear partition-of-unity interpolation; no Unreal
// Engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the three per-axis interpolation fractions. Matches `GpuQuery` in
// `src/trilinear.rs`.
struct Query {
    fx: f32,
    fy: f32,
    fz: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// Eight f32 per query (the unit-cube corner weights), tightly packed.
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(64)
fn trilinear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let base = idx * 8u;
    var k = 0u;
    while (k < 8u) {
        let bx = (k >> 2u) & 1u;
        let by = (k >> 1u) & 1u;
        let bz = k & 1u;
        var wx = 1.0 - q.fx;
        if (bx == 1u) {
            wx = q.fx;
        }
        var wy = 1.0 - q.fy;
        if (by == 1u) {
            wy = q.fy;
        }
        var wz = 1.0 - q.fz;
        if (bz == 1u) {
            wz = q.fz;
        }
        results[base + k] = wx * wy * wz;
        k = k + 1u;
    }
}
