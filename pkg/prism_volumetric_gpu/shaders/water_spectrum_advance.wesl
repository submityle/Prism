// Spectral amplitude-advance twin: evaluates the Tessendorf ocean time
// evolution of one Fourier amplitude per query, mirroring the CPU golden
// `dispersion` + `advance_amplitude` (in
// `prism_render_architecture::water::spectrum`).
//
// For a wave vector of magnitude `k`, the deep-water dispersion relation gives
// the angular frequency `omega = sqrt(g*k)` (and `0` for `k <= 0`). The complex
// amplitude at time `t` is the Hermitian-paired phasor advance
//   h(k,t) = h0(+k) e^{+i omega t} + conj(h0(-k)) e^{-i omega t},
// which keeps the inverse-FFT height field real-valued. This kernel reproduces
// that exact sequence: it computes `omega` with the same `k <= 0` guard, forms
// `theta = omega * t`, advances `h0` by `e^{+i theta}` and `conj(h0_neg)` by
// `e^{-i theta}`, and sums them.
//
// The transcendental `sin`/`cos` are forbidden by the determinism policy, so
// the phasor uses the hand-rolled `sin_approx`/`cos_approx` mirrored bit-for-bit
// from the CPU `water::mod` math (range reduction `wrap_pi` with round-half-away
// ties, fold into [-PI/2, PI/2], seventh-order Taylor). WGSL's native `round`
// rounds ties to even, so `round_ties_away` reproduces Rust's `f32::round`.
// Everything else is multiply/add plus one `sqrt`, all in the portable
// core-WGSL subset, so the kernel runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 `prism_render_architecture::water::spectrum`；standard
// Tessendorf ocean spectral evolution；no Unreal Engine source or derived code.

// Angle constants matching `PI`, `TWO_PI`, `FRAC_PI_2` in the CPU math module
// (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;

// Deep-water gravitational acceleration, matching `water::GRAVITY = 9.81`.
const GRAVITY: f32 = 9.81;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the +k and -k initial amplitudes, the wave-number magnitude and
// the time. Matches `GpuQuery` in `src/water_spectrum_advance.rs` (32 bytes).
struct Query {
    h0_re: f32,
    h0_im: f32,
    h0_neg_re: f32,
    h0_neg_im: f32,
    k: f32,
    t: f32,
    pad0: f32,
    pad1: f32,
}

// Result: the real and imaginary parts of the advanced amplitude. Matches
// `GpuResult` in `src/water_spectrum_advance.rs` (8 bytes).
struct Res {
    re: f32,
    im: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Round halves away from zero, matching Rust's `f32::round`. WGSL's native
// `round` rounds ties to even, so it cannot be used here.
fn round_ties_away(v: f32) -> f32 {
    let r = floor(abs(v) + 0.5);
    return select(r, -r, v < 0.0);
}

// Reduce an angle to [-PI, PI], mirroring `math::wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let kk = round_ties_away(x / TWO_PI);
    return x - kk * TWO_PI;
}

// Hand-rolled sine, mirroring `water::sin_approx` bit for bit.
fn sin_approx(x: f32) -> f32 {
    var r = wrap_pi(x);
    if (r > FRAC_PI_2) {
        r = PI - r;
    } else if (r < -FRAC_PI_2) {
        r = -PI - r;
    }
    let x2 = r * r;
    return r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))));
}

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `water::cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Complex product `(a.x + i a.y)(b.x + i b.y)`, mirroring `Complex::mul`.
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Multiply by the unit phasor `e^{i theta} = cos(theta) + i sin(theta)`,
// mirroring `Complex::mul_phasor`.
fn mul_phasor(c: vec2<f32>, theta: f32) -> vec2<f32> {
    return cmul(c, vec2<f32>(cos_approx(theta), sin_approx(theta)));
}

// Deep-water dispersion `omega(k) = sqrt(g*k)`, `0` for `k <= 0`, mirroring
// `spectrum::dispersion`.
fn dispersion(k: f32) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    return sqrt(GRAVITY * k);
}

@compute @workgroup_size(64)
fn advance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let omega = dispersion(q.k);
    let theta = omega * q.t;
    let h0 = vec2<f32>(q.h0_re, q.h0_im);
    let h0_neg_conj = vec2<f32>(q.h0_neg_re, -q.h0_neg_im);
    let forward = mul_phasor(h0, theta);
    let backward = mul_phasor(h0_neg_conj, -theta);
    results[idx].re = forward.x + backward.x;
    results[idx].im = forward.y + backward.y;
}
