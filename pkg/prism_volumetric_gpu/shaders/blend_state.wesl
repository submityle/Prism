// Sky-state blend twin: smoothly interpolates the target cloud coverage between
// two sky-states on-device, mirroring the CPU golden `blend_state` (in
// `prism_render_architecture::volumetric::weather`).
//
// The sky-state machine (Clear -> Fair -> Overcast -> Storm) drives a target
// coverage per state; when the weather transitions between two states the
// target coverage is interpolated with a Hermite smoothstep so it starts and
// ends with zero slope - no visible jump at either endpoint (design section 9).
// For a transition parameter `t` running `0..=1`:
//   a = target_coverage(from)
//   b = target_coverage(to)
//   w = smoothstep(0, 1, t)   (t saturated, then cubic Hermite)
//   result = saturate(a + (b - a) * w)
// The blend is monotone in `t`, the endpoints are exact (t=0 -> a, t=1 -> b),
// and the result stays within the coverage interval spanned by the two states,
// hence within 0..=1. The CPU golden owns that logic; this kernel is the
// on-device twin that runs one thread per query.
//
// The state target coverages are the same constants as the CPU module
// (CLEAR/FAIR/OVERCAST/STORM_COVERAGE). `from` and `to` are uploaded as u32
// ordinals (0 = Clear, 1 = Fair, 2 = Overcast, 3 = Storm). The smoothstep
// mirrors `math::smoothstep` bit for bit, so the on-device result matches
// bit-close. Result is one f32 per query. It uses only the portable core-WGSL
// subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard smoothstep coverage blend / weather state machine; no
// Unreal Engine source or derived code.

// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;

// Sky-state target coverages, matching the CPU `weather` module constants.
const CLEAR_COVERAGE: f32 = 0.05;
const FAIR_COVERAGE: f32 = 0.30;
const OVERCAST_COVERAGE: f32 = 0.75;
const STORM_COVERAGE: f32 = 0.95;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One blend query: the source and destination sky-state ordinals and the
// transition parameter. Matches `GpuQuery` in `src/blend_state.rs`.
struct Query {
    from_state: u32,
    to_state: u32,
    t: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
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
    let t = saturate_s((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Target coverage for a sky-state ordinal, mirroring `SkyState::target_coverage`.
fn target_coverage(state: u32) -> f32 {
    if (state == 0u) {
        return CLEAR_COVERAGE;
    }
    if (state == 1u) {
        return FAIR_COVERAGE;
    }
    if (state == 2u) {
        return OVERCAST_COVERAGE;
    }
    return STORM_COVERAGE;
}

// Smooth coverage blend between two states, mirroring `blend_state`.
fn blend_state(from_state: u32, to_state: u32, t: f32) -> f32 {
    let a = target_coverage(from_state);
    let b = target_coverage(to_state);
    let w = smoothstep_approx(0.0, 1.0, t);
    return saturate_s(a + (b - a) * w);
}

@compute @workgroup_size(64)
fn blend_state_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = blend_state(q.from_state, q.to_state, q.t);
}
