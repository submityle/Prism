//! Baked GGX single-scatter albedo tables used by the multiscatter energy
//! compensation (`gi::env_brdf::multiscatter::multiscatter_direct`).
//!
//! Why a table instead of the analytic Karis fit: `env_brdf_approx`'s
//! `scale + bias` is **view-independent** (it collapses to a roughness-only
//! polynomial), so plugging it into Kulla-Conty produced a compensation of the
//! right *average* magnitude but the wrong *shape*. Measured against the lobe
//! the shader actually evaluates, the fit under-states the albedo by up to
//! ~0.19, which made a white metal reflect ≈1.11 of the incident energy instead
//! of 1.0 (caught by
//! `openpbr_reference::conformance::white_furnace_conserves_energy_for_the_specification_configurations`).
//!
//! These values are the true `E(µ, r)` of that lobe, taken from this
//! repository's own GGX **VNDF importance-sampled** bake
//! (`gi::env_brdf::dfg_lut::integrate_dfg` via
//! `multiscatter::directional_albedo`) — no invented fit. Regenerate with
//! `cargo test -p prism_render_shading --lib print_ggx_albedo_table -- --ignored
//! --nocapture`; the mirror row is pinned to 1 and the grazing column onto its
//! first resolved sample (the VNDF estimator is degenerate exactly at `µ = 0`).
//!
//! Layout: `GGX_ALBEDO[j * GGX_N_MU + i]` is `E(µ_i, r_j)` with `µ_i = i/16`,
//! `r_j = j/8`. `GGX_AVERAGE[j]` is `2∫E(µ, r_j)·µ dµ`.

/// Number of `µ = n·v` samples per row.
pub const GGX_N_MU: usize = 17;
/// Number of roughness rows.
pub const GGX_N_R: usize = 9;

/// `E(µ, r)` on the uniform grid, row-major over roughness.
pub const GGX_ALBEDO: [f32; GGX_N_MU * GGX_N_R] = [
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    0.964381, 0.964381, 0.990634, 0.996070, 0.997740, 0.998607, 0.999081, 0.999299, 0.999487, 0.999554, 0.999628, 0.999718, 0.999756, 0.999778, 0.999793, 0.999830, 0.999844,
    0.888715, 0.888715, 0.910110, 0.941431, 0.961581, 0.973538, 0.980757, 0.985280, 0.988275, 0.990371, 0.991976, 0.993108, 0.993935, 0.994528, 0.995015, 0.995406, 0.995791,
    0.911760, 0.911760, 0.877837, 0.876056, 0.888036, 0.903585, 0.918350, 0.930953, 0.941175, 0.949317, 0.955872, 0.961098, 0.965282, 0.968609, 0.971358, 0.973665, 0.975570,
    0.921149, 0.921149, 0.876500, 0.851384, 0.840079, 0.838098, 0.841831, 0.848806, 0.857358, 0.866410, 0.875333, 0.883814, 0.891623, 0.898722, 0.905101, 0.910799, 0.915882,
    0.911364, 0.911364, 0.860054, 0.824380, 0.799449, 0.782489, 0.771666, 0.765553, 0.763024, 0.763164, 0.765273, 0.768800, 0.773327, 0.778502, 0.784061, 0.789830, 0.795653,
    0.888982, 0.888982, 0.825800, 0.779841, 0.744536, 0.716752, 0.694690, 0.677148, 0.663252, 0.652352, 0.643936, 0.637587, 0.632968, 0.629821, 0.627890, 0.626996, 0.626978,
    0.858492, 0.858492, 0.779168, 0.720858, 0.674986, 0.637531, 0.606256, 0.579731, 0.556964, 0.537255, 0.520047, 0.504953, 0.491672, 0.479948, 0.469558, 0.460338, 0.452170,
    0.823009, 0.823009, 0.725394, 0.653967, 0.597712, 0.551612, 0.512862, 0.479666, 0.450817, 0.425447, 0.402928, 0.382790, 0.364667, 0.348231, 0.333275, 0.319598, 0.307007,
];

/// `E_avg(r)` on the same roughness grid.
pub const GGX_AVERAGE: [f32; GGX_N_R] = [0.999995, 0.998962, 0.987304, 0.951403, 0.882399, 0.780349, 0.656290, 0.527348, 0.409282];

/// Bilinear `E(µ, r)`; both arguments are clamped to `[0, 1]`. Mirrors
/// `brdf.wesl::ggx_albedo_lookup`.
#[must_use]
pub fn ggx_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {
    let x = n_dot_v.clamp(0.0, 1.0) * (GGX_N_MU - 1) as f32;
    let y = roughness.clamp(0.0, 1.0) * (GGX_N_R - 1) as f32;
    let i0 = (x as usize).min(GGX_N_MU - 2);
    let j0 = (y as usize).min(GGX_N_R - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = GGX_ALBEDO[j0 * GGX_N_MU + i0];
    let b = GGX_ALBEDO[j0 * GGX_N_MU + i0 + 1];
    let c = GGX_ALBEDO[(j0 + 1) * GGX_N_MU + i0];
    let d = GGX_ALBEDO[(j0 + 1) * GGX_N_MU + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}

/// Linear `E_avg(r)`; clamped to `[0, 1]`.
#[must_use]
pub fn ggx_average_lookup(roughness: f32) -> f32 {
    let y = roughness.clamp(0.0, 1.0) * (GGX_N_R - 1) as f32;
    let j0 = (y as usize).min(GGX_N_R - 2);
    let fy = y - j0 as f32;
    let a = GGX_AVERAGE[j0];
    let b = GGX_AVERAGE[j0 + 1];
    (a + (b - a) * fy).clamp(0.0, 1.0)
}
