// Spectral-to-RGB twin: collapses on-device a variable-length spectral
// band distribution into a linear RGB vector, mirroring the CPU golden
// `spectral_to_rgb` (in `prism_render_architecture::volumetric::spectral`).
//
// The spectral night-sky model (design section 8b) represents a spectral power
// distribution as a partition-of-unity set of wavelength-bucket weights.
// `spectral_to_rgb` assigns each bucket a representative wavelength at its
// centre within the visible span (VIS_MIN_NM..=VIS_MAX_NM), converts it to a
// non-negative, L1-normalised CIE-flavoured RGB response, and accumulates it
// weighted by the bucket weight:
//   frac_i    = (i + 0.5) / n
//   nm_i      = lerp(VIS_MIN_NM, VIS_MAX_NM, frac_i)
//   rgb       = sum_i weights[i] * cie_rgb_response(nm_i)
// Because each per-band response sums to one across its channels, the total of
// the RGB channels equals the sum of the band weights (one for a normalised,
// non-empty band set), so the mapping conserves weight and never produces a
// negative channel. An empty band set maps to the zero vector. The CPU golden
// `spectral_to_rgb` owns that math; this kernel is the on-device twin that runs
// one thread per query (one distribution per thread) and reproduces the value.
//
// The per-band Gaussian response uses the SAME hand-rolled `exp_approx` the
// reference uses (base-two range reduction: a fractional seven-term polynomial
// times an integer power assembled from the float exponent field), NOT the
// device-native `exp`, and the `lerp` is expanded to the SAME closed form the
// CPU `math` module uses, so the twin stays bit-close to the reference and the
// parity test can assert a tight tolerance.
//
// Each query indexes a shared, flattened weights buffer via an `offset`/`len`
// pair, so distributions of any bucket count coexist in one dispatch. Results
// are `vec4` (xyz = rgb, w unused) so the storage buffer stays 16-byte aligned.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops,
// clamp/saturate and multiply/add - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard CIE-flavoured spectral-to-RGB collapse; no Unreal Engine
// source or derived code.

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

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One spectral-to-RGB query: an `offset`/`len` window into the shared flattened
// weights buffer. Matches `GpuQuery` in `src/spectral_to_rgb.rs`.
struct Query {
    offset: u32,
    len: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4<f32>>;

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
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

@compute @workgroup_size(64)
fn spectral_to_rgb_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let n = q.len;
    var i = 0u;
    loop {
        if (i >= n) {
            break;
        }
        let frac = (f32(i) + 0.5) / f32(n);
        let nm = lerp_approx(VIS_MIN_NM, VIS_MAX_NM, frac);
        let response = cie_rgb_response(nm);
        acc = acc + response * weights[q.offset + i];
        i = i + 1u;
    }
    results[idx] = vec4<f32>(acc, 0.0);
}
