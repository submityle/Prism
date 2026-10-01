// Per-wavelength spectral melanin-absorption `LUT` sampling twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::spectral_absorption::spectrum_sample_map`.
// A physically based spectral hair `BSDF` needs the full `lambda -> sigma_a`
// curve (not just the three RGB primaries the sibling `melanin` twin folds), so
// that dispersion, fluorescent dye and narrow-band studio lights read correctly
// and pigmented hair avoids the RGB "hue-shift under saturated light" error.
// One thread per queried wavelength samples the two per-unit pigment spectra —
// `floor`-indexed linear interpolation across the uniformly sampled visible band
// `[380, 730]`nm — and folds them with the fibre's two non-negative pigment
// concentrations: `sigma_a = eu * eu_sample + pheo * pheo_sample`.
//
// The two 15-point pigment spectra are passed in a storage buffer from the host
// (`[0..15]` eumelanin, `[15..30]` pheomelanin) so the shader never duplicates
// (and never drifts from) the golden's canonical table constants; only the
// structural band layout (`380`nm .. `730`nm, 15 samples) lives in the shader,
// exactly like the fixed `64`-wide dispatch shape. Concentrations are sanitized
// bit-faithfully to the golden's `clamp_concentration` and wavelengths to its
// `sanitized_wavelength`, so negative / `NaN` / `+/-inf` inputs collapse to the
// same bounded, finite, non-negative `sigma_a` the reference emits.
//
// Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
// spectral sampling and `wgpu` compute dispatch; no Unreal Engine source or
// derived code.

struct Params {
    eumelanin: f32,
    pheomelanin: f32,
    wavelength_count: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// 30 entries: [0..15] eumelanin spectrum, [15..30] pheomelanin spectrum.
@group(0) @binding(1) var<storage, read> spectra: array<f32>;
// Per query: the sampled wavelength in nanometres.
@group(0) @binding(2) var<storage, read> wavelengths: array<f32>;
// Per query: the folded spectral absorption coefficient sigma_a.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// Structural band layout, mirroring the golden's SPECTRUM_* constants. These
// describe the fixed table geometry (not the pigment data), analogous to the
// fixed workgroup width, so they are structural shader constants, not duplicated
// magic absorption values.
const SPECTRUM_SAMPLES: u32 = 15u;
const SPECTRUM_MIN_NM: f32 = 380.0;
const SPECTRUM_MAX_NM: f32 = 730.0;

// Largest finite f32; `v <= MAX_FINITE_F32` rejects +inf (and `v == v` rejects
// NaN), so the guard reproduces `f32::is_finite() && v > 0.0` exactly.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

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
// sampled band; non-finite inputs map to the short-wavelength endpoint. Keeps
// the interpolation in-bounds and panic-free (the table is clamped, not
// extrapolated, outside its range).
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
    if t >= params.wavelength_count {
        return;
    }
    let w = wavelengths[t];
    let eu = clamp_concentration(params.eumelanin);
    let pheo = clamp_concentration(params.pheomelanin);
    let eu_sample = sample_spectrum(0u, w);
    let pheo_sample = sample_spectrum(SPECTRUM_SAMPLES, w);
    out[t] = eu * eu_sample + pheo * pheo_sample;
}
