// AVSM transmittance-sampling twin: evaluates the read-only piecewise-linear
// transmittance lookup of an adaptive volumetric shadow-map curve for many
// query depths in parallel, mirroring the CPU golden
// `AvsmCurve::transmittance_at` (in `prism_render_architecture::volumetric::avsm`).
//
// The curve is a set of `(depth, transmittance)` control points kept sorted by
// ascending depth with a monotone non-increasing transmittance sequence. The
// insert/compress state machine that builds the curve is inherently sequential
// and stays on the CPU; this kernel twins only the pure sampling query, which
// is what a shadowing raymarch evaluates per froxel/sample. Given the shared
// node array, each invocation samples one query depth: before the first node
// (or on an empty curve) transmittance is 1; at or beyond the last node it is
// the last node's transmittance; in between the two bracketing nodes are
// linearly interpolated. Every arithmetic step mirrors the CPU exactly, so the
// two agree to a few ULP. It uses only the portable core-WGSL subset, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard piecewise-linear shadow-curve lookup (Salvi-style AVSM
// sampling); no Unreal Engine source or derived code.

const EPS: f32 = 1e-6;

struct Params {
    query_count: u32,
    node_count: u32,
    pad0: u32,
    pad1: u32,
}

// One curve control point. Matches `GpuNode` in `src/avsm_transmittance.rs`.
struct Node {
    depth: f32,
    transmittance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> queries: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// `a + (b - a) * t`, matching `math::lerp` (t is not clamped).
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Piecewise-linear transmittance lookup, mirroring `AvsmCurve::transmittance_at`.
fn transmittance_at(depth: f32) -> f32 {
    let count = params.node_count;
    if (count == 0u) {
        return 1.0;
    }
    let first = nodes[0];
    if (depth < first.depth) {
        return 1.0;
    }
    let last = nodes[count - 1u];
    if (depth >= last.depth) {
        return saturate_s(last.transmittance);
    }
    var i = 0u;
    loop {
        if (i + 1u >= count) {
            break;
        }
        let lo = nodes[i];
        let hi = nodes[i + 1u];
        if (depth <= hi.depth) {
            let span = hi.depth - lo.depth;
            var t = 0.0;
            if (span > EPS) {
                t = (depth - lo.depth) / span;
            }
            return saturate_s(lerp(lo.transmittance, hi.transmittance, t));
        }
        i = i + 1u;
    }
    return saturate_s(last.transmittance);
}

@compute @workgroup_size(64)
fn avsm_transmittance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    results[idx] = transmittance_at(queries[idx]);
}
