// Sunset-reddening twin: evaluates on-device the horizon reddening amount,
// mirroring the CPU golden `sunset_reddening` (in
// `prism_render_architecture::volumetric::spectral`).
//
// The spectral sky model (design section 8) reddens the light near the horizon
// as the sun sinks: short wavelengths scatter away along the long grazing path
// so the surviving light shifts red. The amount is
//   saturate(1 - smoothstep(0, REDDEN_MAX_ALTITUDE, sun_altitude))
// where `sun_altitude` is the sun's angular altitude in radians (zero at the
// horizon, positive above). It is one at and below the horizon (full
// reddening), decreases monotonically as the sun climbs, and is zero once the
// sun rises above REDDEN_MAX_ALTITUDE, so it stays in 0..=1 and is monotone
// non-increasing in altitude. The CPU golden `sunset_reddening` owns that math;
// this kernel is the on-device twin that runs one thread per query.
//
// The `smoothstep` is expanded to the SAME closed form the CPU `math::smoothstep`
// uses (a degenerate-span guard, a saturated ramp, then the cubic `t*t*(3-2t)`),
// and the kernel contains no transcendental call, so CPU and GPU evaluate the
// same algebra and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - clamp/saturate and multiply/add
// - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard horizon-reddening falloff; no Unreal Engine source or
// derived code.

// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;
// Sun altitude (radians) above which there is no reddening, matching
// `REDDEN_MAX_ALTITUDE` in the CPU reference.
const REDDEN_MAX_ALTITUDE: f32 = 0.35;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One sunset-reddening query: the sun's angular altitude in radians. Matches
// `GpuQuery` in `src/sunset_reddening.rs`.
struct Query {
    sun_altitude: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
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

// The horizon reddening amount, mirroring `sunset_reddening`.
fn sunset_reddening(sun_altitude: f32) -> f32 {
    return saturate(1.0 - smoothstep_approx(0.0, REDDEN_MAX_ALTITUDE, sun_altitude));
}

@compute @workgroup_size(64)
fn sunset_reddening_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = sunset_reddening(q.sun_altitude);
}
