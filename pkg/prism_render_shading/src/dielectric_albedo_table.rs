//! Baked directional albedo `E_dielectric(µ, r)` of the **dielectric** specular
//! lobe — the quantity `OpenPBR`'s `glossy_diffuse_albedo_constraint` weights the
//! diffuse slab by (`1 − E_dielectric(ωo)`).
//!
//! Baked with VNDF importance sampling of the *exact-Fresnel* lobe at the
//! specification's default dielectric (`η = 1.5`, `F0 = 0.04`). Uniform
//! hemisphere quadrature cannot resolve a near-mirror GGX lobe, and a
//! Schlick-split DFG fit is inconsistent with the exact Fresnel this renderer
//! now uses (its `bias` term is non-zero at `F0 == 0`, where the exact lobe
//! reflects nothing).
//!
//! The shape is baked once at the reference F0 and scaled linearly by the
//! authored `dielectric_f0 / 0.04`; that is exact at `F0 == 0` (no interface) and
//! at the default, and monotone in between. Regenerate with
//! `cargo test -p prism_render_shading --lib print_dielectric_albedo_table --
//! --ignored --nocapture`.
//!
//! Layout: `DIELECTRIC_ALBEDO[j * DIEL_N_MU + i]` is `E(µ_i, r_j)` with
//! `µ_i = i/16`, `r_j = j/8`.

/// Number of `µ = n·v` samples per row.
pub const DIEL_N_MU: usize = 17;
/// Number of roughness rows.
pub const DIEL_N_R: usize = 9;

/// `E_dielectric(µ, r)` at the reference `F0 = 0.04`, row-major over roughness.
pub const DIELECTRIC_ALBEDO: [f32; DIEL_N_MU * DIEL_N_R] = [
    0.991738, 0.701614, 0.500000, 0.361128, 0.264190, 0.195931, 0.147630, 0.113399, 0.089187,
    0.072157, 0.060302, 0.052187, 0.046781, 0.043333, 0.041301, 0.040287, 0.040000, 0.764850,
    0.644223, 0.484574, 0.355432, 0.261916, 0.195040, 0.147497, 0.113488, 0.089340, 0.072315,
    0.060440, 0.052298, 0.046862, 0.043389, 0.041335, 0.040302, 0.040001, 0.509576, 0.428823,
    0.365235, 0.296800, 0.233476, 0.181438, 0.141125, 0.110840, 0.088530, 0.072355, 0.060806,
    0.052732, 0.047223, 0.043644, 0.041473, 0.040305, 0.039882, 0.331714, 0.281623, 0.239760,
    0.203679, 0.171336, 0.142631, 0.118013, 0.097659, 0.081365, 0.068684, 0.059069, 0.051971,
    0.046890, 0.043397, 0.041137, 0.039818, 0.039220, 0.223957, 0.189272, 0.161310, 0.138173,
    0.118762, 0.102329, 0.088397, 0.076662, 0.066899, 0.058912, 0.052510, 0.047500, 0.043691,
    0.040902, 0.038963, 0.037724, 0.037051, 0.161297, 0.133988, 0.113894, 0.097904, 0.084905,
    0.074226, 0.065410, 0.058119, 0.052102, 0.047164, 0.043150, 0.039931, 0.037400, 0.035462,
    0.034037, 0.033049, 0.032436, 0.125001, 0.100881, 0.084903, 0.072798, 0.063266, 0.055614,
    0.049403, 0.044328, 0.040168, 0.036757, 0.033965, 0.031692, 0.029856, 0.028393, 0.027246,
    0.026373, 0.025733, 0.103650, 0.080339, 0.066429, 0.056394, 0.048741, 0.042727, 0.037914,
    0.034007, 0.030806, 0.028165, 0.025972, 0.024146, 0.022621, 0.021346, 0.020281, 0.019392,
    0.018652, 0.090763, 0.066837, 0.053940, 0.045021, 0.038412, 0.033320, 0.029293, 0.026045,
    0.023387, 0.021183, 0.019337, 0.017776, 0.016445, 0.015302, 0.014314, 0.013455, 0.012703,
];

/// Bilinear `E_dielectric(µ, r)`; arguments clamp to `[0, 1]`. Mirrors
/// `brdf.wesl::dielectric_albedo_lookup`.
#[must_use]
pub fn dielectric_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = n_dot_v.clamp(0.0, 1.0) * (DIEL_N_MU - 1) as f32;
    let y = roughness.clamp(0.0, 1.0) * (DIEL_N_R - 1) as f32;
    let i0 = (x as usize).min(DIEL_N_MU - 2);
    let j0 = (y as usize).min(DIEL_N_R - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = DIELECTRIC_ALBEDO[j0 * DIEL_N_MU + i0];
    let b = DIELECTRIC_ALBEDO[j0 * DIEL_N_MU + i0 + 1];
    let c = DIELECTRIC_ALBEDO[(j0 + 1) * DIEL_N_MU + i0];
    let d = DIELECTRIC_ALBEDO[(j0 + 1) * DIEL_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}
