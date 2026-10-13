//! Baked Schlick-*bias* albedo `B(µ, r)` of the GGX lobe: the lobe's albedo with
//! the Schlick bias weight `(1 − v·h)^5`. With the white-lobe albedo `E_white`
//! it gives the **exact** Schlick-Fresnel lobe albedo `F0·E_white + (1−F0)·B`,
//! which the coat energy closure (and any other Schlick-Fresnel albedo question)
//! needs. Regenerate with `cargo test -p prism_render_shading --lib
//! print_schlick_bias_table -- --ignored --nocapture`; `brdf.wesl` mirrors it.
//!
//! Layout: `SCHLICK_BIAS[j * BIAS_N_MU + i]` is `B(µ_i, r_j)`, `µ_i = i/16`,
//! `r_j = j/8` (column 0 is grazing, column 16 is normal).

/// Number of `µ = n·v` samples per row.
pub const BIAS_N_MU: usize = 17;
/// Number of roughness rows.
pub const BIAS_N_R: usize = 9;

/// `B(µ, r)` on the uniform grid, row-major over roughness.
pub const SCHLICK_BIAS: [f32; BIAS_N_MU * BIAS_N_R] = [
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.774244,
    0.663815, 0.495332, 0.347318, 0.234679, 0.152748, 0.095450, 0.056640, 0.031595, 0.016307,
    0.007606, 0.003092, 0.001032, 0.000253, 0.000036, 0.000002, 0.000000, 0.508795, 0.431615,
    0.362063, 0.281038, 0.203548, 0.139449, 0.090800, 0.056082, 0.032642, 0.017708, 0.008802,
    0.003916, 0.001500, 0.000479, 0.000124, 0.000025, 0.000003, 0.313340, 0.263798, 0.218489,
    0.176234, 0.136619, 0.101114, 0.071250, 0.047687, 0.030196, 0.017975, 0.009964, 0.005073,
    0.002328, 0.000937, 0.000319, 0.000085, 0.000012, 0.192295, 0.157866, 0.128437, 0.102946,
    0.080828, 0.061808, 0.045781, 0.032675, 0.022347, 0.014553, 0.008951, 0.005144, 0.002719,
    0.001290, 0.000527, 0.000168, 0.000027, 0.122265, 0.096436, 0.076552, 0.060369, 0.047053,
    0.036096, 0.027142, 0.019915, 0.014187, 0.009753, 0.006421, 0.004007, 0.002336, 0.001243,
    0.000579, 0.000213, 0.000042, 0.082617, 0.061668, 0.047356, 0.036463, 0.027948, 0.021230,
    0.015922, 0.011744, 0.008483, 0.005970, 0.004066, 0.002656, 0.001641, 0.000940, 0.000479,
    0.000199, 0.000048, 0.060034, 0.041666, 0.030684, 0.022905, 0.017144, 0.012793, 0.009478,
    0.006945, 0.005013, 0.003548, 0.002448, 0.001633, 0.001041, 0.000622, 0.000336, 0.000152,
    0.000043, 0.046881, 0.029659, 0.020799, 0.014960, 0.010876, 0.007932, 0.005773, 0.004175,
    0.002989, 0.002108, 0.001457, 0.000979, 0.000633, 0.000387, 0.000217, 0.000104, 0.000034,
];

/// Bilinear `B(µ, r)`; arguments clamp to `[0, 1]`. Mirrors
/// `brdf.wesl::schlick_bias_lookup`.
#[must_use]
pub fn schlick_bias_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = n_dot_v.clamp(0.0, 1.0) * (BIAS_N_MU - 1) as f32;
    let y = roughness.clamp(0.0, 1.0) * (BIAS_N_R - 1) as f32;
    let i0 = (x as usize).min(BIAS_N_MU - 2);
    let j0 = (y as usize).min(BIAS_N_R - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = SCHLICK_BIAS[j0 * BIAS_N_MU + i0];
    let b = SCHLICK_BIAS[j0 * BIAS_N_MU + i0 + 1];
    let c = SCHLICK_BIAS[(j0 + 1) * BIAS_N_MU + i0];
    let d = SCHLICK_BIAS[(j0 + 1) * BIAS_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}
