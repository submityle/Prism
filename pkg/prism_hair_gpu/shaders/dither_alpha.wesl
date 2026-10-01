// Compute twin of Prism's blue-noise dither-alpha kernel
// (prism_render_architecture::hair::reactive_mask::dither_alpha_map). One
// thread per pixel derives a deterministic blue-noise dither threshold in
// [0, 1) from the pixel's (x, y, frame) via an integer xorshift / multiply
// finaliser, then makes a hard sub-pixel draw decision: alpha = 1 when the
// sanitised sub-pixel coverage is at or above the threshold, else 0. This is
// the stochastic fallback for the thinnest fibres where the analytic
// line_coverage ramp is not applied.
//
// Unlike every prior twin this kernel is BIT-EXACT, not tolerance-checked: the
// threshold is pure 32-bit integer arithmetic (wrapping multiply, xor-shift)
// finalised to an f32 by a single u32->f32 conversion and an exact divide by
// 2^32, and the alpha is a hard 0/1 decision, so the device output must match
// the golden to the bit (compared via the raw bit pattern) on any conforming
// implementation. WGSL unsigned integer arithmetic wraps on overflow, exactly
// mirroring Rust's wrapping_mul / wrapping_add.

struct Params {
    base_x: u32,
    base_y: u32,
    frame: u32,
    element_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-pixel sub-pixel coverage in [0, 1] (sanitised in shader).
@group(0) @binding(1) var<storage, read> coverages: array<f32>;
// One hard draw decision per pixel: 1.0 draws, 0.0 skips.
@group(0) @binding(2) var<storage, read_write> alpha: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;
// 2^32, used to normalise the 32-bit hash into [0, 1) by exact exponent shift
// (never u32::MAX, so the result is strictly below 1).
const TWO_POW_32: f32 = 4294967296.0;

// Mirror the golden `is_finite()` test without a float equality on a literal:
// `x == x` rejects NaN, the finite-magnitude bound rejects +/-inf.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Golden `dither_threshold`: mix the three coordinates with large odd constants
// and pass through an integer xorshift / multiply finaliser, then normalise by
// 2^32. `frame` rotates the sequence in time. Pure integer arithmetic, so it is
// bit-exact and never panics. WGSL u32 multiply/add wrap exactly like Rust's
// wrapping_mul / wrapping_add.
fn dither_threshold(x: u32, y: u32, frame: u32) -> f32 {
    var h: u32 = x * 0x9E3779B1u;
    h = h ^ (y * 0x85EBCA77u);
    h = h ^ (frame * 0xC2B2AE3Du);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h) / TWO_POW_32;
}

// Golden `dither_alpha`: sanitise coverage (finite -> clamp [0,1], else 0) and
// threshold (finite -> clamp [0,1], else 1 so a corrupt threshold suppresses the
// fibre), then draw (1.0) when coverage is at or above threshold, else 0.0.
fn dither_alpha(coverage: f32, threshold: f32) -> f32 {
    var c: f32;
    if is_finite_f32(coverage) {
        c = clamp(coverage, 0.0, 1.0);
    } else {
        c = 0.0;
    }
    var t: f32;
    if is_finite_f32(threshold) {
        t = clamp(threshold, 0.0, 1.0);
    } else {
        t = 1.0;
    }
    if c >= t {
        return 1.0;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    // base_x + idx wraps on overflow, mirroring the golden's wrapping_add over
    // the pixel row.
    let x = params.base_x + idx;
    let threshold = dither_threshold(x, params.base_y, params.frame);
    alpha[idx] = dither_alpha(coverages[idx], threshold);
}
