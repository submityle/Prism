// Trigonometric approximation twin: evaluates the hand-rolled `sin_approx` and
// `cos_approx` per query, mirroring the CPU golden `math::sin_approx` /
// `math::cos_approx` (in `prism_render_architecture::volumetric::math`).
//
// The determinism policy forbids the hardware transcendental `sin`/`cos`, so
// the CPU reduces the angle to [-PI, PI] via `wrap_pi` (round-half-away), folds
// it into [-PI/2, PI/2] through `sin(PI - x)`, and evaluates a seventh-order
// Taylor polynomial; cosine is `sin(x + PI/2)`. This kernel mirrors that exact
// sequence so CPU and GPU agree bit-for-bit on the reduction and share the same
// polynomial, differing at most by a legal multiply-add contraction of a few
// ULP. WGSL's native `round` rounds ties to even, so `round_ties_away`
// reproduces Rust's `f32::round` explicitly. It uses only the portable
// core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard range-reduced Taylor sine; no Unreal Engine source or
// derived code.

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

// One query: the angle in radians. Matches `GpuQuery` in `src/trig_approx.rs`.
struct Query {
    angle: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// Result: the sine and cosine of the angle. Matches `GpuResult` in
// `src/trig_approx.rs`.
struct Res {
    sin: f32,
    cos: f32,
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

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `math::cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

@compute @workgroup_size(64)
fn trig_approx_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let angle = queries[idx].angle;
    results[idx].sin = sin_approx(angle);
    results[idx].cos = cos_approx(angle);
}
