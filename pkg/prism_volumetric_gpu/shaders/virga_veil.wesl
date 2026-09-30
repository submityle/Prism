// Virga-veil twin: evaluates on-device the precipitation-veil density beneath a
// storm base, mirroring the CPU golden `StormState::virga_veil` (in
// `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus precipitation model (design section 9b) trails a virga veil
// below the cloud base whose strength grows as the storm matures. `virga_veil`
// scales the authored `virga_fade` falloff by the state machine's current
// `virga` veil strength at a normalized `veil_fraction`:
//   saturate(virga * virga_fade(veil_fraction))
// with `virga_fade(f) = smoothstep(0, 1, saturate(f))`, so a young storm (no
// veil) yields zero and a mature storm trails a fading precipitation curtain,
// densest at the cloud base (veil_fraction == 1) and evaporating toward the
// trailing tip (veil_fraction == 0). The result is bounded to 0..=1 and
// monotonically non-decreasing in both the veil strength and the height
// fraction. The CPU golden `StormState::virga_veil` owns that math; this kernel
// is the on-device twin that runs one thread per query and reproduces the same
// value.
//
// The `smoothstep` is expanded to the SAME closed form the CPU
// `math::smoothstep` uses (a degenerate-span guard, a saturated ramp, then the
// cubic `t*t*(3-2t)`), and the kernel contains no transcendental call -
// saturate and multiply/add - so CPU and GPU evaluate the same algebra and the
// parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp/saturate and multiply/add -
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard virga precipitation-veil falloff; no Unreal Engine
// source or derived code.

// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One virga-veil query: the state machine's veil strength plus the normalized
// veil height. Matches `GpuQuery` in `src/virga_veil.rs`.
struct Query {
    virga: f32,
    veil_fraction: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Cubic Hermite smoothstep, mirroring `math::smoothstep` bit for bit.
fn smoothstep_approx(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Virga precipitation-veil weight at a normalized height, mirroring
// `virga_fade`.
fn virga_fade(height_fraction: f32) -> f32 {
    return smoothstep_approx(0.0, 1.0, saturate(height_fraction));
}

// Virga-veil density beneath the storm base, mirroring `StormState::virga_veil`.
fn virga_veil(virga: f32, veil_fraction: f32) -> f32 {
    return saturate(virga * virga_fade(veil_fraction));
}

@compute @workgroup_size(64)
fn virga_veil_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = virga_veil(q.virga, q.veil_fraction);
}
