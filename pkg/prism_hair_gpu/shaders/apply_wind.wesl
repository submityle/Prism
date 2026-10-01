// Per-particle wind pre-pass twin: applies on-device the same deterministic
// external wind displacement the CPU golden `apply_wind` writes into every free
// guide particle before the XPBD constraint solve, one thread per particle.
//
// Ambient wind is injected as an external pre-pass (design 6.3): each FREE
// particle gains the wind displacement `wind_acceleration(field,pos,t) * dt^2`
// - the same semi-implicit `dt^2` scaling the dynamics integrator uses for
// gravity, so the next solve reads it back as injected velocity. PINNED
// particles (inverse_mass <= 0, the skinned root) are never moved: the kernel
// copies their position through unchanged, bit-for-bit. The host short-circuits
// a non-positive or non-finite `dt` (a no-op) before dispatch, so this kernel
// always runs with a positive finite `dt`.
//
// The acceleration field itself is `wind_acceleration` (the CPU golden in
// `prism_render_architecture::hair::wind`), mirrored here exactly - steady push
// + along-wind gust pulse + cross-wind turbulent flutter - including the
// hand-written range-reduced 9th-order Taylor sine (`sin_turns`) the reference
// uses in place of a hardware `sin` for bit-reproducibility. Unlike the sister
// `wind` twin (which outputs raw accelerations for a batch of point/time/field
// triples and never touches particles), this twin reproduces the FULL pre-pass:
// the pinned skip, the `dt^2` integration scaling and the in-place position
// update over one shared wind field.
//
// It uses only the portable core-WGSL subset (sqrt/min/max/dot/round/fma/mul/
// add) - no exp, no pow, no hardware sin, no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard steady+gust+turbulence wind pre-pass with a hand-written
// Taylor sine; no Unreal Engine source or derived code.

// Matches `core::f32::consts`: nearest-f32 of pi, tau and pi/2.
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;

// Matches `wind::EPS_LEN_SQ` (via `Vec3::normalize_or_zero`): vectors shorter
// than the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;

// Fixed swirl axis mixing the three position components into one gust phase,
// matching `wind::GUST_SWIRL`.
const GUST_SWIRL: vec3<f32> = vec3<f32>(0.7, 1.3, 0.5);

// Shared parameters for one wind pre-pass dispatch. `64`-byte layout matching
// the `bytemuck` upload struct `Params`: the particle count (and uniform pad),
// then the shared wind field, then the sample time and the frame `dt`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    speed: f32,
    gust_amplitude: f32,
    gust_frequency: f32,
    turbulence: f32,
    time: f32,
    dt: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

// One guide particle. `16`-byte stride matching the `bytemuck` upload struct
// `WindParticle`: the world-space position then the reciprocal mass
// (`inv_mass <= 0` is pinned).
struct Particle {
    px: f32,
    py: f32,
    pz: f32,
    inv_mass: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero,
// mirroring `Vec3::normalize_or_zero` term order (`v * (1 / sqrt(len_sq))`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Deterministic sine of an angle in radians, mirroring `wind::sin_turns`:
// range-reduces to `[-PI/2, PI/2]` and evaluates a 9th-order Taylor polynomial
// in Horner form. `fma` reproduces the reference's `f32::mul_add` fusion.
fn sin_turns(x: f32) -> f32 {
    var a = x - round(x / TAU) * TAU;
    if (a > FRAC_PI_2) {
        a = PI - a;
    } else if (a < -FRAC_PI_2) {
        a = -PI - a;
    }
    let x2 = a * a;
    let p0 = fma(x2, 1.0 / 362880.0, -1.0 / 5040.0);
    let p1 = fma(p0, x2, 1.0 / 120.0);
    let p2 = fma(p1, x2, -1.0 / 6.0);
    let poly = fma(p2, x2, 1.0);
    return a * poly;
}

// Evaluates the shared wind acceleration at world-space `pos`, mirroring
// `wind_acceleration` with the field carried by `params`.
fn wind_acceleration(pos: vec3<f32>) -> vec3<f32> {
    let dir = normalize_or_zero(vec3<f32>(params.dx, params.dy, params.dz));
    let freq = max(params.gust_frequency, 0.0);

    let steady = dir * params.speed;

    let gust_phase = freq * (dot(pos, GUST_SWIRL) + params.time);
    let gust_scalar = sin_turns(gust_phase * TAU);
    let gust = dir * (params.gust_amplitude * gust_scalar);

    let base = freq * params.time;
    let flutter = vec3<f32>(
        sin_turns((pos.x + base) * TAU),
        sin_turns((pos.y * 1.3 + base + 0.37) * TAU),
        sin_turns((pos.z * 0.7 + base + 0.71) * TAU),
    ) * params.turbulence;

    return steady + gust + flutter;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let p = particles[i];
    let pos = vec3<f32>(p.px, p.py, p.pz);
    let base = 3u * i;

    // Pinned particle (inverse_mass <= 0): never moved, copied through exactly.
    if (p.inv_mass <= 0.0) {
        values[base] = pos.x;
        values[base + 1u] = pos.y;
        values[base + 2u] = pos.z;
        return;
    }

    let dt_sq = params.dt * params.dt;
    let moved = pos + wind_acceleration(pos) * dt_sq;
    values[base] = moved.x;
    values[base + 1u] = moved.y;
    values[base + 2u] = moved.z;
}
