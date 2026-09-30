// Imposter-fade twin: evaluates on-device the volumetric-to-imposter cross-fade
// weight, mirroring the CPU golden `imposter_fade` (in
// `prism_render_architecture::volumetric::cloud_lod`).
//
// The cloud LOD chain (design section 11) forbids hard switching between the
// live volumetric raymarch and the pre-rendered imposter billboard: instead the
// weight is
//   imposter_fade(distance, start, end) = smoothstep(start, end, distance)
// which is 0 (show the raymarch in full) up to `start`, 1 (show the imposter in
// full) past `end`, and a soft smoothstep cross-fade between, monotonically
// non-decreasing in distance. When `start == end` the smoothstep degrades to a
// hard step without dividing by zero. The CPU golden `imposter_fade` owns that
// math; this kernel is the on-device twin that runs one thread per query.
//
// The `smoothstep` is expanded to the SAME closed form the CPU `math::smoothstep`
// uses (a degenerate-span guard, a saturated ramp, then the cubic `t*t*(3-2t)`),
// so CPU and GPU evaluate the same algebra and the parity test can assert a
// tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp/saturate and multiply/add
// - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard smoothstep LOD cross-fade; no Unreal Engine source or
// derived code.

// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One imposter-fade query: the view distance and the fade start/end distances.
// Matches `GpuQuery` in `src/imposter_fade.rs`.
struct Query {
    distance: f32,
    start: f32,
    end: f32,
    pad0: f32,
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

// The imposter cross-fade weight, mirroring `imposter_fade`.
fn imposter_fade(distance: f32, start: f32, end: f32) -> f32 {
    return smoothstep_approx(start, end, distance);
}

@compute @workgroup_size(64)
fn imposter_fade_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = imposter_fade(q.distance, q.start, q.end);
}
