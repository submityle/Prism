// Transcendental approximation twin: evaluates the hand-rolled `exp_approx`,
// `ln_approx` and `pow_approx` per query, mirroring the CPU goldens
// `math::exp_approx` / `math::ln_approx` / `math::pow_approx` (in
// `prism_render_architecture::volumetric::math`).
//
// The determinism policy forbids the hardware transcendentals, so the CPU
// builds `exp(x)` as `2^(x*log2e)` with an integer power assembled from the
// `f32` exponent field plus a fractional polynomial, and `ln(x)` from exponent
// extraction plus the `atanh` series on the centred mantissa; `pow(b, e)` is
// `exp(e * ln(b))` with non-positive bases mapped to `0`. Several volume
// kernels embed `exp_approx` / `ln_approx` inline, but only over the narrow
// ranges those kernels produce. This kernel isolates all three so a wide-domain
// sweep exercises the saturation branches (`+inf` for large positive `exp`, the
// `0` flush for large negative, the non-positive `ln` clamp, the `SQRT_2`
// mantissa-centring branch, and the negative-base `pow` guard) in one place.
//
// The blocks are copied verbatim from the already-verified `exp_approx`
// (`shaders/analytic_transmittance.wesl`) and `ln_approx`
// (`shaders/single_scatter_reference.wesl`) so CPU and GPU share identical
// closed-form algebra, differing at most by a legal multiply-add contraction of
// a few ULP. It uses only the portable core-WGSL subset, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard base-two range-reduced exp/ln; no Unreal Engine source
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

// One query: the `exp` argument, the `ln` argument and the `pow` base/exponent.
// Matches `GpuQuery` in `src/transcendental_approx.rs`.
struct Query {
    exp_x: f32,
    ln_x: f32,
    pow_base: f32,
    pow_exp: f32,
}

// Result: `exp_approx`, `ln_approx` and `pow_approx`. Matches `GpuResult` in
// `src/transcendental_approx.rs`.
struct Res {
    exp: f32,
    ln: f32,
    pow: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// `2^k` for an integer `k`, assembled from the `f32` exponent field. Underflows
// to `0` for `k < -126` and saturates to `+inf` for `k > 127`.
fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

// `2^f` for `f` in `[0, 1]` via the truncated `exp(f*ln2)` series.
fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

// Hand-rolled `exp`, mirroring `math::exp_approx` bit for bit.
fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Hand-rolled `ln`, mirroring `math::ln_approx` bit for bit.
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

// Hand-rolled `pow(base, exponent)` as `exp(exponent * ln(base))`, mirroring
// `math::pow_approx`; non-positive bases map to `0`.
fn pow_approx(base: f32, exponent: f32) -> f32 {
    if (base <= 0.0) {
        return 0.0;
    }
    return exp_approx(exponent * ln_approx(base));
}

@compute @workgroup_size(64)
fn transcendental_approx_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx].exp = exp_approx(q.exp_x);
    results[idx].ln = ln_approx(q.ln_x);
    results[idx].pow = pow_approx(q.pow_base, q.pow_exp);
    results[idx].pad = 0.0;
}
