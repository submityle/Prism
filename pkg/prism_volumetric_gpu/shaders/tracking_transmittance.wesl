// Tracking-transmittance twin: evaluates the delta-tracking and ratio-tracking
// Monte-Carlo transmittance estimators for a HOMOGENEOUS medium on-device,
// mirroring the CPU goldens `delta_tracking_transmittance` and
// `ratio_tracking_transmittance` (in
// `prism_render_architecture::volumetric::reference`) when their extinction
// closure is the constant `sigma_t`.
//
// These are the unbiased reference oracles the analytic transmittance of design
// section 6/9c must converge to. Each of `samples` walks re-seeds a
// deterministic Weyl+hash RNG from `sample_seed(seed, i)`, then integrates
// free-flight steps `t += -ln(1 - u) / majorant`:
//   * delta tracking counts the fraction of walks that reach `distance` before
//     a real collision (accepted with probability saturate(sigma_t / majorant)
//     at each tentative event); it draws TWO unit floats per interior event.
//   * ratio tracking accumulates the residual-ratio product
//     prod (1 - saturate(sigma_t / majorant)) over the null-collision events; it
//     draws ONE unit float per event.
// Both return 1.0 for a zero-length segment or an empty budget and clamp into
// [0, 1].
//
// The RNG, `ln_approx` and arithmetic mirror the CPU goldens exactly (WGSL u32
// wraps like `wrapping_*`; `ln_approx` is the same bit-reconstruction + atanh
// series). Floating-point division is spec-allowed to differ by a few ULP
// across devices, so the parity test uses a tolerance rather than bit equality.
// The kernel uses only the portable core-WGSL subset, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard delta / ratio tracking estimators (Novak et al.); no
// Unreal Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
const LN2: f32 = 0.6931471805599453;
const SQRT_2: f32 = 1.4142135623730951;
const EPS: f32 = 1e-6;
const GOLDEN_GAMMA: u32 = 0x9E3779B9u;
const MAX_EVENTS: u32 = 1000000u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One tracking query. Matches `GpuQuery` in `src/tracking_transmittance.rs`.
struct Query {
    sigma_t: f32,
    majorant: f32,
    distance: f32,
    seed: u32,
    samples: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One tracking result: the delta and ratio estimates. Matches `GpuResult` in
// `src/tracking_transmittance.rs`.
struct Res {
    delta: f32,
    ratio: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

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

// `delta_tracking_transmittance` for a homogeneous medium.
fn delta_tracking(sigma_t: f32, majorant: f32, distance: f32, seed: u32, samples: u32) -> f32 {
    let dist = max(distance, 0.0);
    let maj = max(majorant, EPS);
    if (dist < EPS || samples == 0u) {
        return 1.0;
    }
    let accept = clamp(sigma_t / maj, 0.0, 1.0);
    var survived = 0u;
    var i = 0u;
    loop {
        if (i >= samples) {
            break;
        }
        var state = hash_u32(sample_seed(seed, i) ^ GOLDEN_GAMMA);
        var t = 0.0;
        var alive = true;
        var events = 0u;
        loop {
            state = state + GOLDEN_GAMMA;
            let u0 = unit_from_u32(hash_u32(state));
            t = t + free_flight_step(u0, maj);
            if (t >= dist) {
                break;
            }
            state = state + GOLDEN_GAMMA;
            let u1 = unit_from_u32(hash_u32(state));
            if (u1 < accept) {
                alive = false;
                break;
            }
            events = events + 1u;
            if (events >= MAX_EVENTS) {
                break;
            }
        }
        if (alive) {
            survived = survived + 1u;
        }
        i = i + 1u;
    }
    return clamp(f32(survived) / f32(samples), 0.0, 1.0);
}

// `ratio_tracking_transmittance` for a homogeneous medium.
fn ratio_tracking(sigma_t: f32, majorant: f32, distance: f32, seed: u32, samples: u32) -> f32 {
    let dist = max(distance, 0.0);
    let maj = max(majorant, EPS);
    if (dist < EPS || samples == 0u) {
        return 1.0;
    }
    let residual = 1.0 - clamp(sigma_t / maj, 0.0, 1.0);
    var sum = 0.0;
    var i = 0u;
    loop {
        if (i >= samples) {
            break;
        }
        var state = hash_u32(sample_seed(seed, i) ^ GOLDEN_GAMMA);
        var t = 0.0;
        var weight = 1.0;
        var events = 0u;
        loop {
            state = state + GOLDEN_GAMMA;
            let u0 = unit_from_u32(hash_u32(state));
            t = t + free_flight_step(u0, maj);
            if (t >= dist) {
                break;
            }
            weight = weight * residual;
            events = events + 1u;
            if (events >= MAX_EVENTS) {
                break;
            }
        }
        sum = sum + weight;
        i = i + 1u;
    }
    return clamp(sum / f32(samples), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn tracking_transmittance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let d = delta_tracking(q.sigma_t, q.majorant, q.distance, q.seed, q.samples);
    let r = ratio_tracking(q.sigma_t, q.majorant, q.distance, q.seed, q.samples);
    results[idx] = Res(d, r);
}
