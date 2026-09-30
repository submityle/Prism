// Single-scatter reference twin: evaluates the free-flight importance-sampled
// Monte-Carlo single-scatter radiance estimator for a HOMOGENEOUS medium with
// a CONSTANT phase value on-device, mirroring the CPU golden
// `single_scatter_reference` (in
// `prism_render_architecture::volumetric::reference`) when its phase closure is
// the constant `phase`.
//
// Each of `samples` walks re-seeds a deterministic Weyl+hash RNG from
// `sample_seed(seed, i)`, draws one free-flight scatter distance
// `t = -ln(1 - u) / sigma_t` from the extinction pdf, and — when `t` lands
// inside the segment — contributes `sigma_s * phase * light_radiance / sigma_t`
// (the transmittance and pdf cancel). The mean over the walks converges to the
// analytic single-scatter radiance for a constant phase.
//
// The RNG, `ln_approx` and arithmetic mirror the CPU golden exactly (WGSL u32
// wraps like `wrapping_*`; `ln_approx` is the same bit-reconstruction + atanh
// series). Floating-point division is spec-allowed to differ by a few ULP
// across devices, so the parity test uses a tolerance rather than bit equality.
// The kernel uses only the portable core-WGSL subset, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard free-flight importance-sampled single-scatter estimator;
// no Unreal Engine source or derived code.

const LN2: f32 = 0.6931471805599453;
const SQRT_2: f32 = 1.4142135623730951;
const EPS: f32 = 1e-6;
const GOLDEN_GAMMA: u32 = 0x9E3779B9u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One single-scatter query. Matches `GpuQuery` in
// `src/single_scatter_reference.rs`.
struct Query {
    sigma_t: f32,
    sigma_s: f32,
    light_radiance: f32,
    phase: f32,
    distance: f32,
    seed: u32,
    samples: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn hash_u32(x: u32) -> u32 {
    var h = (x ^ 61u) * 0x27d4eb2du;
    h ^= x >> 16u;
    h = h + (h << 3u);
    h ^= h >> 4u;
    h = h * 0x27d4eb2du;
    h ^= h >> 15u;
    return h;
}

fn unit_from_u32(bits: u32) -> f32 {
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

fn sample_seed(seed: u32, index: u32) -> u32 {
    return seed ^ (index * GOLDEN_GAMMA);
}

fn ln_approx(x_in: f32) -> f32 {
    var x = x_in;
    // f32::MIN_POSITIVE, the smallest positive normal.
    if (x < 1.1754944e-38) {
        x = 1.1754944e-38;
    }
    let bits = bitcast<u32>(x);
    let exp_field = i32((bits >> 23u) & 0xFFu);
    var e = exp_field - 127;
    let mantissa_bits = (bits & 0x007FFFFFu) | (127u << 23u);
    var m = bitcast<f32>(mantissa_bits);
    if (m > SQRT_2) {
        m = m * 0.5;
        e = e + 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let series = t * (1.0 + t2 * (0.33333334 + t2 * (0.2 + t2 * (0.14285715 + t2 * 0.11111111))));
    return f32(e) * LN2 + 2.0 * series;
}

fn free_flight_step(u: f32, maj: f32) -> f32 {
    return -ln_approx(1.0 - u) / maj;
}

// `single_scatter_reference` for a homogeneous medium and constant phase.
fn single_scatter(
    sigma_t: f32,
    sigma_s: f32,
    light_radiance: f32,
    phase: f32,
    distance: f32,
    seed: u32,
    samples: u32,
) -> f32 {
    let st = max(sigma_t, EPS);
    let ss = max(sigma_s, 0.0);
    let lr = max(light_radiance, 0.0);
    let ph = max(phase, 0.0);
    let dist = max(distance, 0.0);
    if (dist < EPS || samples == 0u) {
        return 0.0;
    }
    var acc = 0.0;
    var i = 0u;
    loop {
        if (i >= samples) {
            break;
        }
        var state = hash_u32(sample_seed(seed, i) ^ GOLDEN_GAMMA);
        state = state + GOLDEN_GAMMA;
        let u0 = unit_from_u32(hash_u32(state));
        let t = free_flight_step(u0, st);
        if (t < dist) {
            acc = acc + ss * ph * lr / st;
        }
        i = i + 1u;
    }
    return acc / f32(samples);
}

@compute @workgroup_size(64)
fn single_scatter_reference_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = single_scatter(
        q.sigma_t,
        q.sigma_s,
        q.light_radiance,
        q.phase,
        q.distance,
        q.seed,
        q.samples,
    );
}
