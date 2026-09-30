// Wind-field coupling twin: evaluates on-device the same deterministic external
// acceleration field the CPU golden `wind_acceleration` computes, one thread
// per query.
//
// Ambient wind sells hair as alive: a steady breeze plus turbulent gusts push
// each free guide particle before the XPBD constraint solve (design 6.3). The
// CPU golden is `wind_acceleration` (in `prism_render_architecture::hair::wind`);
// this kernel mirrors it exactly:
//
//   steady   = normalize_or_zero(direction) * speed
//   gust     = dir * (gust_amplitude * sin_turns(freq * (dot(pos, SWIRL) + t) * TAU))
//   flutter  = (sin_turns((x + base)*TAU),
//               sin_turns((y*1.3 + base + 0.37)*TAU),
//               sin_turns((z*0.7 + base + 0.71)*TAU)) * turbulence
//   result   = steady + gust + flutter        with base = freq * t, freq = max(gust_frequency, 0)
//
// The turbulence is a hand-written deterministic `sin_turns` (a range-reduced
// 9th-order Taylor polynomial), never a hardware `sin`, so the field is
// bit-reproducible across platforms; this kernel reproduces that polynomial and
// its range reduction rather than calling the WGSL `sin` builtin, so the twin
// diverges from the golden only through legal fused-multiply-add contraction.
// It uses only the portable core-WGSL subset (sqrt/min/max/dot/round/fma/mul/add)
// - no exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard steady+gust+turbulence wind acceleration with a
// hand-written Taylor sine; no Unreal Engine source or derived code.

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

struct Params {
    // Number of valid entries in `queries` / points in `values`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One wind-evaluation query. `48`-byte stride matching the `bytemuck` upload
// struct `WindQuery`: the sample point and time, then the wind field.
struct Query {
    px: f32,
    py: f32,
    pz: f32,
    time: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    speed: f32,
    gust_amplitude: f32,
    gust_frequency: f32,
    turbulence: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
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

// Evaluates the wind acceleration the query `q` encodes, mirroring
// `wind_acceleration`.
fn wind_acceleration(q: Query) -> vec3<f32> {
    let dir = normalize_or_zero(vec3<f32>(q.dx, q.dy, q.dz));
    let freq = max(q.gust_frequency, 0.0);
    let pos = vec3<f32>(q.px, q.py, q.pz);

    let steady = dir * q.speed;

    let gust_phase = freq * (dot(pos, GUST_SWIRL) + q.time);
    let gust_scalar = sin_turns(gust_phase * TAU);
    let gust = dir * (q.gust_amplitude * gust_scalar);

    let base = freq * q.time;
    let flutter = vec3<f32>(
        sin_turns((pos.x + base) * TAU),
        sin_turns((pos.y * 1.3 + base + 0.37) * TAU),
        sin_turns((pos.z * 0.7 + base + 0.71) * TAU),
    ) * q.turbulence;

    return steady + gust + flutter;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let out = wind_acceleration(queries[i]);
    let base = 3u * i;
    values[base] = out.x;
    values[base + 1u] = out.y;
    values[base + 2u] = out.z;
}
