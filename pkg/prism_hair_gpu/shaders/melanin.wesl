// Per-fibre melanin-pigment -> RGB absorption twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::melanin::melanin_absorption`
// (`melanin_absorption_map` for the batch form). One thread per fibre folds the
// two non-negative pigment concentrations (eumelanin, pheomelanin) into the RGB
// absorption coefficient `sigma_a` the shading closure feeds to Beer-Lambert
// transmittance: `sigma_a = eu * EUMELANIN_SIGMA_A + pheo * PHEOMELANIN_SIGMA_A`
// per channel. This is a pure non-negative linear combination with no
// transcendental step, exactly as the Chiang 2016 / pbrt pigment model
// specifies.
//
// The two per-unit pigment spectra are passed in `Params` from the host so the
// shader never duplicates (and never drifts from) the golden's canonical
// constants. Concentrations are sanitized bit-faithfully to the golden's
// `clamp_concentration`: negative, NaN and +/-inf inputs collapse to `0` so the
// absorption stays finite and non-negative for every input.
//
// Provenance: Chiang 2016 / pbrt melanin pigment parameterisation plus `wgpu`
// compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // xyz = per-unit eumelanin absorption at the RGB primaries; w unused.
    eu_sigma: vec4<f32>,
    // xyz = per-unit pheomelanin absorption at the RGB primaries; w unused.
    pheo_sigma: vec4<f32>,
    profile_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per fibre: x = eumelanin concentration, y = pheomelanin concentration.
@group(0) @binding(1) var<storage, read> profiles: array<vec2<f32>>;
// Per fibre: xyz = RGB sigma_a, w = 0 padding.
@group(0) @binding(2) var<storage, read_write> absorption: array<vec4<f32>>;

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

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.profile_count {
        return;
    }
    let raw = profiles[idx];
    let eu = clamp_concentration(raw.x);
    let pheo = clamp_concentration(raw.y);
    let sigma = eu * params.eu_sigma.xyz + pheo * params.pheo_sigma.xyz;
    absorption[idx] = vec4<f32>(sigma, 0.0);
}
