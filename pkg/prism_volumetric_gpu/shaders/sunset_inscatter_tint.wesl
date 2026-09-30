// Sunset in-scatter tint twin: collapses on-device the sun altitude into a
// warm-to-neutral airlight tint, mirroring the CPU golden
// `sunset_inscatter_tint` (in `prism_render_architecture::volumetric::atmosphere`).
//
// The atmosphere hookup (design section 8) warms the sampled airlight as the
// sun sinks. The reddening amount `sunset_reddening(sun_altitude)` drives a
// linear interpolation from the neutral tint (1, 1, 1) (sun high) toward a warm
// twilight tint: a fixed short-to-long twilight spectral distribution is
// normalised to a partition of unity, collapsed to linear RGB via the same
// `spectral_to_rgb` math, then rescaled so its brightest channel is exactly
// one. Because every channel of the warm tint is in 0..=1 and the neutral tint
// is one, the interpolated tint is in 0..=1 on every channel, so multiplying
// airlight by it can only attenuate (never amplify) a channel. The CPU golden
// owns that math; this kernel is the on-device twin that runs one thread per
// query and reproduces the value.
//
// The reddening `smoothstep` and per-band Gaussian response use the SAME closed
// forms and the SAME hand-rolled `exp_approx` the reference uses, NOT the
// device-native `exp`, and the `lerp` is expanded to the SAME closed form, so
// the twin stays bit-close to the reference and the parity test can assert a
// tight tolerance. Results are `vec4` (xyz = rgb tint, w unused) so the storage
// buffer stays 16-byte aligned.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops,
// clamp/saturate and multiply/add - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard horizon-reddening falloff plus CIE-flavoured
// spectral-to-RGB collapse; no Unreal Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;
// CIE-flavoured response centre wavelengths (nm), matching the CPU constants.
const R_CENTER_NM: f32 = 610.0;
const G_CENTER_NM: f32 = 550.0;
const B_CENTER_NM: f32 = 465.0;
// Gaussian half-width (nm) shared by the three RGB response lobes.
const RGB_WIDTH_NM: f32 = 50.0;
// Visible-span bounds (nm) mapped to buckets.
const VIS_MIN_NM: f32 = 380.0;
const VIS_MAX_NM: f32 = 700.0;
// Solar altitude (radians) above which there is no twilight reddening.
const REDDEN_MAX_ALTITUDE: f32 = 0.35;
// Fixed short-to-long twilight spectrum (blue, green, red) matching the CPU
// constants; normalised at runtime to a partition of unity.
const TWILIGHT_BLUE_WEIGHT: f32 = 0.2;
const TWILIGHT_GREEN_WEIGHT: f32 = 0.5;
const TWILIGHT_RED_WEIGHT: f32 = 1.0;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the sun's angular altitude in radians. Matches `GpuQuery` in
// `src/sunset_inscatter_tint.rs`.
struct Query {
    sun_altitude: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;

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

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` bit for bit.
fn lerp_approx(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Cubic smoothstep with a degenerate-span guard, mirroring `math::smoothstep`.
fn smoothstep_approx(e0: f32, e1: f32, x: f32) -> f32 {
    let span = e1 - e0;
    if (abs(span) < EPS) {
        if (x < e0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate((x - e0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Horizon reddening amount in 0..=1, mirroring `sunset_reddening`.
fn sunset_reddening(sun_altitude: f32) -> f32 {
    return saturate(1.0 - smoothstep_approx(0.0, REDDEN_MAX_ALTITUDE, sun_altitude));
}

// A non-negative Gaussian lobe, mirroring `gaussian`.
fn gaussian(x: f32, center: f32, width: f32, peak: f32) -> f32 {
    let w = max(width, 1e-3);
    let d = (x - center) / w;
    return max(peak, 0.0) * exp_approx(-d * d);
}

// L1-normalised CIE-flavoured RGB response, mirroring `cie_rgb_response`.
fn cie_rgb_response(nm: f32) -> vec3<f32> {
    let r = gaussian(nm, R_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let g = gaussian(nm, G_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let b = gaussian(nm, B_CENTER_NM, RGB_WIDTH_NM, 1.0);
    let sum = r + g + b;
    if (sum <= EPS) {
        return vec3<f32>(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    return vec3<f32>(r / sum, g / sum, b / sum);
}

// The fixed twilight spectrum collapsed to a peak-normalised warm tint,
// mirroring `sunset_inscatter_tint`'s use of the spectral helpers. The three
// raw weights are normalised to a partition of unity (SpectralBands::new), then
// `spectral_to_rgb` accumulates the per-bucket responses, then the result is
// rescaled so its brightest channel is one.
fn warm_twilight_tint() -> vec3<f32> {
    // Mirror SpectralBands::new: floor negatives (none here), sum, renormalise
    // (uniform when the total collapses).
    let w0 = max(TWILIGHT_BLUE_WEIGHT, 0.0);
    let w1 = max(TWILIGHT_GREEN_WEIGHT, 0.0);
    let w2 = max(TWILIGHT_RED_WEIGHT, 0.0);
    let total = w0 + w1 + w2;
    var n0 = 1.0 / 3.0;
    var n1 = 1.0 / 3.0;
    var n2 = 1.0 / 3.0;
    if (total > EPS) {
        let inv = 1.0 / total;
        n0 = w0 * inv;
        n1 = w1 * inv;
        n2 = w2 * inv;
    }
    // Mirror spectral_to_rgb over three buckets (frac = (i + 0.5) / 3).
    let n = 3.0;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let nm0 = lerp_approx(VIS_MIN_NM, VIS_MAX_NM, 0.5 / n);
    let nm1 = lerp_approx(VIS_MIN_NM, VIS_MAX_NM, 1.5 / n);
    let nm2 = lerp_approx(VIS_MIN_NM, VIS_MAX_NM, 2.5 / n);
    acc = acc + cie_rgb_response(nm0) * n0;
    acc = acc + cie_rgb_response(nm1) * n1;
    acc = acc + cie_rgb_response(nm2) * n2;
    // Peak-normalise so the brightest channel is one.
    let peak = max(max(max(acc.x, acc.y), acc.z), EPS);
    return acc * (1.0 / peak);
}

@compute @workgroup_size(64)
fn sunset_inscatter_tint_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let reddening = sunset_reddening(q.sun_altitude);
    let warm = warm_twilight_tint();
    let neutral = vec3<f32>(1.0, 1.0, 1.0);
    let tint = vec3<f32>(
        lerp_approx(neutral.x, warm.x, reddening),
        lerp_approx(neutral.y, warm.y, reddening),
        lerp_approx(neutral.z, warm.z, reddening),
    );
    results[idx] = vec4<f32>(tint, 0.0);
}
