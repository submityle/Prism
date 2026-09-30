// Gravity-wave twin: evaluates on-device the stable-layer ripple value above a
// mature storm, mirroring the CPU golden `gravity_wave` (in
// `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus vertical-development model (design section 9b) rides a
// bounded sinusoid on the stable layer above the storm. `gravity_wave` returns
// the ripple value for a wave `phase` (a unit cycle) at a horizontal phase
// offset `x` in radians:
//   gravity_wave(phase, x) = sin_approx(TWO_PI * phase + x)
// so the result stays within roughly [-1, 1] and is exactly periodic in phase.
// The CPU golden `gravity_wave` owns that math; this kernel is the on-device
// twin that runs one thread per query and reproduces the same value.
//
// The sine uses the SAME hand-rolled `sin_approx` the reference uses: reduce the
// angle to [-PI, PI] via `wrap_pi` (subtract the nearest whole multiple of
// TWO_PI, rounding halves away from zero exactly like Rust's `f32::round`), fold
// into [-PI/2, PI/2] via `sin(PI - x)`, then the seventh-order Taylor
// polynomial. It does NOT use the device-native `sin`, so the twin stays
// bit-close to the reference and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, abs, select and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard stable-layer gravity-wave ripple; no Unreal Engine
// source or derived code.

// Angle constants matching `PI`, `TWO_PI` and `FRAC_PI_2` in the CPU `math`
// module (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One gravity-wave query: the wave phase (a unit cycle) plus the horizontal
// phase offset in radians. Matches `GpuQuery` in `src/gravity_wave.rs`.
struct Query {
    phase: f32,
    x: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Round halves away from zero, matching Rust's `f32::round`. WGSL's native
// `round` rounds halves to even, so it must NOT be used here.
fn round_ties_away(v: f32) -> f32 {
    let r = floor(abs(v) + 0.5);
    return select(r, -r, v < 0.0);
}

// Reduces an angle to [-PI, PI], mirroring `math::wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let k = round_ties_away(x / TWO_PI);
    return x - k * TWO_PI;
}

// Hand-rolled sine, mirroring `math::sin_approx` bit for bit.
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

// Gravity-wave ripple value, mirroring `gravity_wave`.
fn gravity_wave(phase: f32, x: f32) -> f32 {
    return sin_approx(TWO_PI * phase + x);
}

@compute @workgroup_size(64)
fn gravity_wave_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = gravity_wave(q.phase, q.x);
}
