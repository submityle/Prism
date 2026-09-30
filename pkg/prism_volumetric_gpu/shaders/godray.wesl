// God-ray sample-weight twin: evaluates the geometric decay of a radial god-ray
// sample, mirroring the CPU golden `god_ray_weight` (in
// `prism_render_architecture::volumetric::shadow`).
//
// The weight is `weight * decay^index`, a geometric decay away from the light,
// with `decay` and `weight` saturated into `[0, 1]` and the product saturated
// again, so the result is in `[0, 1]`, monotone non-increasing in the sample
// index, and its partial sums are bounded by `weight / (1 - decay)` (design
// section 12). The CPU golden owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The power `decay^index` uses the SAME hand-rolled `pow_approx` the reference
// uses - `exp_approx(index * ln_approx(decay))` with both the base-two exp and
// the exponent-extraction/atanh-series ln mirrored bit-for-bit - NOT the
// device-native `pow`, so the twin stays close to the reference and the parity
// test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard radial god-ray geometric decay; no Unreal Engine source
// or derived code.

const LOG2_E: f32 = 1.4426950408889634;
const LN2: f32 = 0.6931471805599453;
const SQRT_2: f32 = 1.4142135623730951;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One god-ray query: the radial sample index plus the config fields. Matches
// `GpuQuery` in `src/godray.rs`. `sample_count` and `exposure` are carried for
// a faithful mirror of the CPU config even though `god_ray_weight` reads only
// `decay` and `weight`.
struct Query {
    sample_index: u32,
    sample_count: u32,
    decay: f32,
    weight: f32,
    exposure: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return bitcast<f32>(0x7f800000u);
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
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

fn pow_approx(base: f32, exponent: f32) -> f32 {
    if (base <= 0.0) {
        return 0.0;
    }
    return exp_approx(exponent * ln_approx(base));
}

// God-ray sample weight, mirroring `god_ray_weight`.
fn god_ray_weight(sample_index: u32, decay: f32, weight: f32) -> f32 {
    let d = saturate_s(decay);
    let w = saturate_s(weight);
    return saturate_s(w * pow_approx(d, f32(sample_index)));
}

@compute @workgroup_size(64)
fn godray_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = god_ray_weight(q.sample_index, q.decay, q.weight);
}
