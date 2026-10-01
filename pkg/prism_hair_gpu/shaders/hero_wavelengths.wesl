// `Hero-wavelength` stratified spectral-sampling twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::spectral_absorption::hero_wavelengths`
// plus `hero_sigma_a`. A spectral hair `BSDF` that follows `Wilkie` 2014
// ("Hero Wavelength Spectral Sampling") carries, per stochastic sample, four
// stratified wavelengths spaced a quarter of the visible band apart and wrapped
// cyclically through it; the hero (index `0`) is the importance-sampled primary
// and the other three ride along for free, decorrelating colour noise and
// giving physically faithful dispersion without per-channel passes. One thread
// owns one hero sample: it rotates the stratified unit coordinate, lays out the
// four wavelengths, then folds each with the fibre's two pigment concentrations
// into `sigma_a` via the same 15-point `LUT` interpolation the sibling
// `spectrum_sample_map` twin uses.
//
// The two 15-point pigment spectra arrive in a storage buffer from the host
// (`[0..15]` eumelanin, `[15..30]` pheomelanin) so the shader never duplicates
// (and never drifts from) the golden's canonical table constants; only the
// structural band layout (`380`nm .. `730`nm, 15 samples, quarter-band stride)
// lives in the shader, exactly like the fixed `64`-wide dispatch shape.
//
// `wrap_unit`, `clamp_concentration` and `sanitize_wl` are bit-faithful mirrors
// of the golden, so non-finite / out-of-range stratified coordinates fold into
// `[0, 1)`, negative / non-finite concentrations collapse to `0`, and every
// emitted wavelength lands in `[380, 730)`nm with a finite, non-negative
// `sigma_a`.
//
// Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
// hero-wavelength spectral sampling and `wgpu` compute dispatch; no Unreal
// Engine source or derived code.

struct Params {
    sample_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// 30 entries: [0..15] eumelanin spectrum, [15..30] pheomelanin spectrum.
@group(0) @binding(1) var<storage, read> spectra: array<f32>;
// Per sample, 4 f32: stratified unit coord `u`, per-sample `rotate`, then the
// fibre's eumelanin and pheomelanin concentrations.
@group(0) @binding(2) var<storage, read> inputs: array<f32>;
// Per sample, 8 f32: the four wavelengths [0..4] then their four `sigma_a`
// [4..8], in hero-index order.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// Structural band layout, mirroring the golden's SPECTRUM_* constants. These
// describe the fixed table geometry (not the pigment data), analogous to the
// fixed workgroup width, so they are structural shader constants, not duplicated
// magic absorption values.
const SPECTRUM_SAMPLES: u32 = 15u;
const SPECTRUM_MIN_NM: f32 = 380.0;
const SPECTRUM_MAX_NM: f32 = 730.0;

// Largest finite f32; `v <= MAX_FINITE_F32` rejects +inf (and `v == v` rejects
// NaN), so the guards reproduce `f32::is_finite()` exactly.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Mirrors the golden's `wrap_unit`: non-finite -> `0`; otherwise the fractional
// part `v - floor(v)`, itself mapped to `0` if it escapes `[0, 1)` (guarding the
// `floor` rounding corner the golden guards with `(0.0..1.0).contains`).
fn wrap_unit(v: f32) -> f32 {
    if !(v == v && v >= -MAX_FINITE_F32 && v <= MAX_FINITE_F32) {
        return 0.0;
    }
    let fractional = v - floor(v);
    if fractional >= 0.0 && fractional < 1.0 {
        return fractional;
    }
    return 0.0;
}

// Mirrors the golden's `clamp_concentration`: a finite, strictly positive value
// passes through unchanged; everything else (negative, zero, NaN, +/-inf)
// collapses to `0.0`, with no floating-point equality test on a literal.
fn clamp_concentration(v: f32) -> f32 {
    if v == v && v > 0.0 && v <= MAX_FINITE_F32 {
        return v;
    }
    return 0.0;
}

// Mirrors the golden's `sanitized_wavelength`: finite wavelengths clamp into the
// sampled band; non-finite inputs map to the short-wavelength endpoint.
fn sanitize_wl(w: f32) -> f32 {
    if w == w && w >= -MAX_FINITE_F32 && w <= MAX_FINITE_F32 {
        return clamp(w, SPECTRUM_MIN_NM, SPECTRUM_MAX_NM);
    }
    return SPECTRUM_MIN_NM;
}

// Linearly interpolates one 15-sample spectrum (based at `offset` in `spectra`)
// at wavelength `w`, clamping to the endpoints outside the sampled range.
// Mirrors the golden's `sample_spectrum` floor-index + lerp exactly.
fn sample_spectrum(offset: u32, w: f32) -> f32 {
    let clamped = sanitize_wl(w);
    let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
    let step = span / (f32(SPECTRUM_SAMPLES) - 1.0);
    let position = (clamped - SPECTRUM_MIN_NM) / step;
    let lower_f = floor(position);
    let lower = u32(lower_f);
    if lower >= SPECTRUM_SAMPLES - 1u {
        return spectra[offset + SPECTRUM_SAMPLES - 1u];
    }
    let frac = position - lower_f;
    return spectra[offset + lower] * (1.0 - frac) + spectra[offset + lower + 1u] * frac;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.sample_count {
        return;
    }

    let in_base = t * 4u;
    let u = inputs[in_base + 0u];
    let rotate = inputs[in_base + 1u];
    let eu = clamp_concentration(inputs[in_base + 2u]);
    let pheo = clamp_concentration(inputs[in_base + 3u]);

    let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
    let base = wrap_unit(wrap_unit(u) + wrap_unit(rotate));

    let out_base = t * 8u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let offset = base + 0.25 * f32(j);
        let lambda = SPECTRUM_MIN_NM + wrap_unit(offset) * span;
        let eu_sample = sample_spectrum(0u, lambda);
        let pheo_sample = sample_spectrum(SPECTRUM_SAMPLES, lambda);
        out[out_base + j] = lambda;
        out[out_base + 4u + j] = eu * eu_sample + pheo * pheo_sample;
    }
}
