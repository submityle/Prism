//! GPU wind-field pre-pass kernel for guide-strand dynamics.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::wind::apply_wind`. The sandbox has no GPU,
//! so this shader is validated by naga compilation only (see `shader_tests`);
//! it is the external-force stage that runs *before* the persistent GPU-driven
//! XPBD sim ([`hair_sim`], design §8) each frame, nudging every free particle
//! by the wind displacement so the constraint solve reads it back as injected
//! velocity — exactly the semi-implicit ordering the CPU dynamics integrator
//! uses for gravity. This is what sells a groom as *alive* rather than merely
//! falling (design §6.3, shared wind data with cloth/particles), mirroring the
//! `UE5` Groom / `AMD` `TressFX` ambient-wind coupling at the algorithm level
//! without reusing their code.
//!
//! One invocation owns one particle. Pinned particles (the skinned root,
//! `inverse_mass <= 0`) are never moved and a dead-calm field adds zero, so the
//! pass is safe to dispatch unconditionally. The turbulence is a hand-written
//! deterministic sine of position and time — never a hardware `sin`, whose
//! result is not bit-reproducible across platforms — so the same groom in the
//! same wind at the same time always deforms identically and stays a faithful
//! twin of the deterministic CPU golden.
//!
//! Buffer layout: `positions` is the shared particle pool, one `vec4<f32>` per
//! particle whose `xyz` is the world position and whose `w` is the inverse mass
//! (`0` = pinned), identical to [`hair_sim`] so the two stages share the same
//! buffer. The kernel updates positions in place and preserves `w`.

/// Squared-length threshold below which the wind direction is treated as zero,
/// mirroring the CPU `Vec3::normalize_or_zero` guard (`EPS_LEN_SQ`).
const HAIR_WIND_EPS_LEN_SQ: f32 = 1.0e-24;

/// Mathematical constants at f32 precision (baseline WGSL has no `PI` builtin);
/// identical values to the CPU `core::f32::consts`.
const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const FRAC_PI_2: f32 = 1.5707963267949;

/// Fixed swirl axis mixing the three position components into one gust phase,
/// so the pulse varies across the groom instead of moving it rigidly. Mirrors
/// the CPU `GUST_SWIRL`.
const GUST_SWIRL: vec3<f32> = vec3<f32>(0.7, 1.3, 0.5);

/// Wind field plus the frame's time / step and the particle count that bounds
/// the dispatch. Mirrors the CPU `WindField` (`direction` normalized in-kernel,
/// so any non-zero length is fine) plus the `apply_wind` arguments.
struct HairWindParams {
    // Wind heading; normalized in-kernel.
    direction: vec3<f32>,
    // Steady acceleration magnitude along `direction`.
    speed: f32,
    // Peak extra along-wind acceleration of the gust pulse.
    gust_amplitude: f32,
    // Gust cycles per (world unit + second); clamped non-negative in-kernel.
    gust_frequency: f32,
    // Cross-wind flutter magnitude.
    turbulence: f32,
    // Simulation time in seconds.
    time: f32,
    // Frame timestep; a non-positive or non-finite value is a no-op.
    dt: f32,
    // Number of particles, bounding the dispatch.
    particle_count: u32,
}

// Particle state: `xyz` = position, `w` = inverse mass (0 = pinned). Shared
// with `hair_sim`.
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;

var<immediate> params: HairWindParams;

/// Round half away from zero, matching Rust `f32::round` (WGSL `round` is
/// ties-to-even, so it is not used here).
fn round_away(v: f32) -> f32 {
    return sign(v) * floor(abs(v) + 0.5);
}

/// Deterministic sine of an angle in radians, hand-written to avoid a hardware
/// `sin` (banned for its non-reproducible cross-platform result). Range-reduces
/// to `[-PI/2, PI/2]` and evaluates a 9th-order Taylor polynomial in Horner
/// form; the `fma` chain mirrors the CPU `f32::mul_add` fused multiply-adds so
/// the GPU stays a faithful twin. Bit-mirrors the CPU `wind::sin_turns`.
fn sin_turns(x: f32) -> f32 {
    var a = x - round_away(x / TAU) * TAU;
    if a > FRAC_PI_2 {
        a = PI - a;
    } else if a < -FRAC_PI_2 {
        a = -PI - a;
    }
    let x2 = a * a;
    let poly = fma(fma(fma(fma(x2, 1.0 / 362880.0, -1.0 / 5040.0), x2, 1.0 / 120.0), x2, -1.0 / 6.0), x2, 1.0);
    return a * poly;
}

/// Unit vector along `v`, or zero when `v` is (numerically) the zero vector.
/// Bit-mirrors the CPU `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > HAIR_WIND_EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Wind acceleration at `position` for the current `params.time`. The steady
/// term is `direction * speed`; the gust term pulses that along-wind push by
/// `gust_amplitude * sin(phase)` where the phase mixes position and time; the
/// turbulence term adds a small phase-shifted per-axis cross-wind flutter.
/// Mirrors the CPU `wind_acceleration` value-for-value, including the
/// left-associative `steady + gust + flutter` accumulation order.
fn wind_acceleration(position: vec3<f32>) -> vec3<f32> {
    let dir = normalize_or_zero(params.direction);
    let freq = max(params.gust_frequency, 0.0);

    let steady = dir * params.speed;

    // One scalar gust phase (in turns) shared by the whole along-wind pulse.
    let gust_phase = freq * (dot(position, GUST_SWIRL) + params.time);
    let gust_scalar = sin_turns(gust_phase * TAU);
    let gust = dir * (params.gust_amplitude * gust_scalar);

    // Per-axis flutter: phase-shifted sines of position and time give a small
    // divergent cross-wind component without needing a real noise field.
    let base = freq * params.time;
    let flutter = vec3<f32>(
        sin_turns((position.x + base) * TAU),
        sin_turns((position.y * 1.3 + base + 0.37) * TAU),
        sin_turns((position.z * 0.7 + base + 0.71) * TAU),
    ) * params.turbulence;

    return steady + gust + flutter;
}

@compute @workgroup_size(64)
fn hair_wind(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }

    // Non-positive dt is a no-op (CPU `dt <= 0.0`). Baseline WGSL has no
    // `isFinite`; `dt - dt` is exactly `0.0` only for a finite `dt`, so this
    // also mirrors the CPU `!dt.is_finite()` guard (Inf/NaN skip).
    if params.dt <= 0.0 {
        return;
    }
    if params.dt - params.dt != 0.0 {
        return;
    }

    let inv_mass = positions[i].w;
    // Pinned root rides the skinned scalp; never moved (CPU `is_pinned`).
    if inv_mass <= 0.0 {
        return;
    }

    // Each free particle gains the wind displacement `acceleration * dt^2`,
    // matching the semi-implicit scaling the dynamics integrator uses for
    // gravity, so the next solve reads it back as injected velocity.
    let dt_sq = params.dt * params.dt;
    let pos = positions[i].xyz;
    let accel = wind_acceleration(pos);
    let moved = pos + accel * dt_sq;
    positions[i] = vec4<f32>(moved, inv_mass);
}
