//! Baked directional albedo `E_fuzz(µ, r)` of the Charlie (fuzz/sheen) lobe —
//! the `E_fuzz` term of the specification's `fuzz-layering-approx`
//! (`layer = F·f_fuzz + (1 − F·E_fuzz)·f_base`).
//!
//! The basic Cloth class used to *add* the sheen on top of a full-strength
//! Lambert base, which over-reflected by 14.7% in the white furnace; the
//! substrate must lose exactly the fuzz layer's albedo. Values are clamped to
//! `[0, 1]` (Charlie over-reflects at grazing for near-zero roughness).
//! Regenerate with `cargo test -p prism_render_shading --lib
//! print_charlie_albedo_table -- --ignored --nocapture`.
//!
//! Layout: `FUZZ_ALBEDO[j * FUZZ_N_MU + i]` is `E_fuzz(µ_i, r_j)`, `µ_i = i/16`,
//! `r_j = j/8`.

/// Number of `µ = n·v` samples per row.
pub const FUZZ_N_MU: usize = 17;
/// Number of roughness rows.
pub const FUZZ_N_R: usize = 9;

/// `E_fuzz(µ, r)` on the uniform grid, row-major over roughness.
pub const FUZZ_ALBEDO: [f32; FUZZ_N_MU * FUZZ_N_R] = [
    1.000000, 1.000000, 0.649052, 0.282295, 0.116762, 0.044961, 0.015821, 0.004987, 0.001375,
    0.000322, 0.000061, 0.000009, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000,
    0.868676, 0.677266, 0.538730, 0.431285, 0.345282, 0.275282, 0.217819, 0.170482, 0.131497,
    0.099501, 0.073411, 0.052347, 0.035572, 0.022467, 0.012500, 0.005209, 0.928916, 0.716657,
    0.595808, 0.504953, 0.431375, 0.369553, 0.316454, 0.270209, 0.229555, 0.193579, 0.161614,
    0.133140, 0.107750, 0.085115, 0.064965, 0.047073, 0.031251, 0.810533, 0.646225, 0.552173,
    0.480724, 0.422060, 0.371965, 0.328132, 0.289150, 0.254069, 0.222217, 0.193104, 0.166352,
    0.141669, 0.118817, 0.097613, 0.077898, 0.059546, 0.744316, 0.604668, 0.524836, 0.463977,
    0.413764, 0.370588, 0.332490, 0.298281, 0.267155, 0.238550, 0.212042, 0.187323, 0.164145,
    0.142310, 0.121656, 0.102055, 0.083397, 0.701662, 0.577044, 0.506022, 0.451871, 0.407107,
    0.368500, 0.334302, 0.303441, 0.275201, 0.249079, 0.224705, 0.201788, 0.180111, 0.159499,
    0.139803, 0.120900, 0.102697, 0.671750, 0.557241, 0.492234, 0.442722, 0.401779, 0.366425,
    0.335052, 0.306665, 0.280621, 0.256435, 0.233778, 0.212381, 0.192041, 0.172592, 0.153896,
    0.135837, 0.118325, 0.649554, 0.542347, 0.481705, 0.435588, 0.397465, 0.364541, 0.335312,
    0.308831, 0.284496, 0.261864, 0.240608, 0.220482, 0.201294, 0.182884, 0.165127, 0.147906,
    0.131134, 0.632425, 0.530705, 0.473381, 0.429855, 0.393906, 0.362883, 0.335331, 0.310368,
    0.287407, 0.266029, 0.245935, 0.226876, 0.208675, 0.191184, 0.174267, 0.157825, 0.141770,
];

/// Bilinear `E_fuzz(µ, r)`; arguments clamp to `[0, 1]`. Mirrors
/// `cloth.wesl::fuzz_albedo_lookup`.
#[must_use]
pub fn fuzz_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = n_dot_v.clamp(0.0, 1.0) * (FUZZ_N_MU - 1) as f32;
    let y = roughness.clamp(0.0, 1.0) * (FUZZ_N_R - 1) as f32;
    let i0 = (x as usize).min(FUZZ_N_MU - 2);
    let j0 = (y as usize).min(FUZZ_N_R - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = FUZZ_ALBEDO[j0 * FUZZ_N_MU + i0];
    let b = FUZZ_ALBEDO[j0 * FUZZ_N_MU + i0 + 1];
    let c = FUZZ_ALBEDO[(j0 + 1) * FUZZ_N_MU + i0];
    let d = FUZZ_ALBEDO[(j0 + 1) * FUZZ_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}
