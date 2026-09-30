// Ozone-absorption twin: evaluates on-device the non-negative ozone absorption
// coefficient at a wavelength, mirroring the CPU golden `ozone_absorption` (in
// `prism_render_architecture::volumetric::spectral`).
//
// The spectral sky model (design section 8) tints the daytime zenith and the
// twilight sky with ozone, which absorbs in two bands: the broad visible
// Chappuis band and the near-UV Huggins band. Each band is a non-negative
// Gaussian lobe:
//   gaussian(x, c, w, p) = max(p, 0) * e^{-((x - c) / max(w, 1e-3))^2}
//   ozone = gaussian(x, 602, 90, 0.06) + gaussian(x, 320, 40, 0.01)
// Because both lobes are >= 0, the coefficient is non-negative for any
// wavelength, and it decays to zero far from either band. The CPU golden
// `ozone_absorption` owns that math; this kernel is the on-device twin that
// runs one thread per query and reproduces the same value.
//
// The exponential uses the SAME hand-rolled `exp_approx` the reference uses
// (base-two range reduction: a fractional seven-term polynomial times an
// integer power assembled from the float exponent field), NOT the device-native
// `exp`, so the twin stays bit-close to the reference and the parity test can
// assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard two-band Gaussian ozone model; no Unreal Engine source
// or derived code.

const LOG2_E: f32 = 1.4426950408889634;

// Chappuis (broad visible) ozone band, matching the CPU constants.
const CHAPPUIS_CENTER_NM: f32 = 602.0;
const CHAPPUIS_WIDTH_NM: f32 = 90.0;
const CHAPPUIS_PEAK: f32 = 0.06;
// Huggins (near-UV) ozone band, matching the CPU constants.
const HUGGINS_CENTER_NM: f32 = 320.0;
const HUGGINS_WIDTH_NM: f32 = 40.0;
const HUGGINS_PEAK: f32 = 0.01;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One ozone-absorption query: the wavelength in nanometres. Matches `GpuQuery`
// in `src/ozone_absorption.rs`.
struct Query {
    wavelength_nm: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

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

// A single non-negative Gaussian lobe, mirroring the CPU `gaussian` helper.
fn gaussian(x: f32, center: f32, width: f32, peak: f32) -> f32 {
    let w = max(width, 1e-3);
    let d = (x - center) / w;
    return max(peak, 0.0) * exp_approx(-d * d);
}

// Non-negative ozone absorption coefficient at a wavelength, mirroring
// `ozone_absorption`.
fn ozone_absorption(wavelength_nm: f32) -> f32 {
    let chappuis = gaussian(wavelength_nm, CHAPPUIS_CENTER_NM, CHAPPUIS_WIDTH_NM, CHAPPUIS_PEAK);
    let huggins = gaussian(wavelength_nm, HUGGINS_CENTER_NM, HUGGINS_WIDTH_NM, HUGGINS_PEAK);
    return chappuis + huggins;
}

@compute @workgroup_size(64)
fn ozone_absorption_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = ozone_absorption(q.wavelength_nm);
}
