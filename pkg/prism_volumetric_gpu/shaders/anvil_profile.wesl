// Anvil-profile twin: evaluates on-device the cumulonimbus anvil density weight
// at a normalized height, mirroring the CPU golden `anvil_profile` (in
// `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus vertical development model (design section 9b) flares a flat
// anvil out of the storm top. `anvil_profile` returns the anvil density weight
// at a normalized `height_fraction` for a given maturity `spread`:
//   h    = saturate(height_fraction)
//   s    = saturate(spread)
//   base = lerp(0.85, 0.5, s)   // a wider anvil flares from a lower altitude
//   w    = saturate(s * smoothstep(base, 1, h))
// so the anvil concentrates in the upper band and a more mature spread flares it
// lower and stronger, always bounded to 0..=1. The CPU golden `anvil_profile`
// owns that math; this kernel is the on-device twin that runs one thread per
// query and reproduces the same value.
//
// The `lerp` and `smoothstep` are expanded to the SAME closed forms the CPU
// `math::lerp` / `math::smoothstep` use (`a + (b - a) * t`; a degenerate-span
// guard, a saturated ramp, then the cubic `t*t*(3-2t)`), and the kernel contains
// no transcendental call - saturate and multiply/add - so CPU and GPU evaluate
// the same algebra and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp/saturate and multiply/add -
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard cumulonimbus anvil-spread profile; no Unreal Engine
// source or derived code.

// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One anvil-profile query: the normalized height plus the anvil maturity
// spread. Matches `GpuQuery` in `src/anvil_profile.rs`.
struct Query {
    height_fraction: f32,
    spread: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` bit for bit.
fn lerp_approx(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

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

// Anvil density weight at a normalized height for a maturity spread, mirroring
// `anvil_profile`.
fn anvil_profile(height_fraction: f32, spread: f32) -> f32 {
    let h = saturate(height_fraction);
    let s = saturate(spread);
    let base = lerp_approx(0.85, 0.5, s);
    return saturate(s * smoothstep_approx(base, 1.0, h));
}

@compute @workgroup_size(64)
fn anvil_profile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = anvil_profile(q.height_fraction, q.spread);
}
