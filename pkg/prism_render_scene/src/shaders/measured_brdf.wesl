//! Shared measured / tabulated BRDF (Rusinkiewicz parameterisation) helpers;
//! no bindings or entry points.
//!
//! GPU twin of `prism_render_shading/src/measured_brdf.rs`. An isotropic
//! acquired BRDF (MERL, Dupuy & Jakob 2018) is stored in the half/difference
//! basis of Rusinkiewicz 1998: the half-vector elevation `theta_h` (square-root
//! warped towards the specular peak), the difference elevation `theta_d`, and
//! the reciprocity-folded difference azimuth `phi_d` in `[0, pi)`. Isotropy
//! drops `phi_h`, collapsing the 4D BRDF to a compact 3D table.
//!
//! This module mirrors the CPU golden's coordinate transform, square-root axis
//! warp, continuous grid mapping, trilinear interpolation and analytic
//! synthesiser (Cook-Torrance GGX + Lambert) byte-for-byte: identical operation
//! order, clamps, floors and degenerate-input fallbacks. Because WESL cannot
//! take an arbitrarily sized table as an argument, the trilinear helper accepts
//! the eight bracketing RGB cells plus the three `[0, 1]` fractions a caller has
//! already fetched; the index/warp helpers produce exactly those fractions.
//!
//! The local shading frame is `+Z` = surface normal, `+X` = tangent. All
//! directions are normalised with a safe fallback, all cosines and fractions are
//! clamped, and no transcendental is evaluated outside its stable domain.

/// `core::f32::consts::PI` rounded to f32, byte-identical to the CPU golden.
const MB_PI: f32 = 3.1415927;
/// `pi / 2`, the upper bound of both tabulated elevation axes.
const MB_FRAC_PI_2: f32 = 1.5707964;
/// `2 * pi`, the full outgoing azimuth span used by the sampler.
const MB_TAU: f32 = 6.2831855;
/// Smallest cosine / denominator evaluated; keeps grazing angles off the
/// microfacet and normalisation singularities.
const MB_MIN_COS: f32 = 1.0e-4;
/// Smallest squared length treated as a non-degenerate direction.
const MB_MIN_LEN_SQ: f32 = 1.0e-12;
/// Microfacet roughness floor shared with the analytic synthesiser.
const MB_MIN_ROUGHNESS: f32 = 0.045;
/// Smallest positive normal f32, matching the CPU `f32::MIN_POSITIVE` guard.
const MB_MIN_POSITIVE: f32 = 1.1754944e-38;

/// Scalar finite predicate, matching the shared isfinite idiom.
fn mb_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

/// Clamps to `[0, 1]`.
fn mb_clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

/// `x^5` via three multiplies.
fn mb_pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

/// Linear interpolation of two RGB triples (`a + (b - a) * t`).
fn mb_lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

/// Normalises `v`, returning `fallback` for degenerate / non-finite inputs.
fn mb_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > MB_MIN_LEN_SQ && mb_is_finite(len_sq)) {
        return v * (1.0 / sqrt(len_sq));
    }
    return fallback;
}

/// Normalises `v`, returning the zero vector for degenerate inputs.
fn mb_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    return mb_normalize_or(v, vec3<f32>(0.0, 0.0, 0.0));
}

/// Half vector of an incident/outgoing pair, falling back to the normal when
/// the sum is degenerate (back-to-back directions).
fn mb_half_vector(wi: vec3<f32>, wo: vec3<f32>) -> vec3<f32> {
    return mb_normalize_or(wi + wo, vec3<f32>(0.0, 0.0, 1.0));
}

/// Rotation of `v` by `angle` about `+Z` (right-handed).
fn mb_rotate_about_z(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}

/// Rotation of `v` by `angle` about `+Y` (right-handed).
fn mb_rotate_about_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

/// Wraps an angle into `[0, 2*pi)` (`rem_euclid(angle, TAU)` semantics).
fn mb_wrap_tau(angle: f32) -> f32 {
    var a = angle - MB_TAU * floor(angle / MB_TAU);
    if (a < 0.0) {
        a = a + MB_TAU;
    }
    return a;
}

/// Folds a signed difference azimuth into `[0, pi)` by reciprocity.
fn mb_fold_phi_d(phi_d: f32) -> f32 {
    var p = mb_wrap_tau(phi_d);
    if (p >= MB_PI) {
        p = p - MB_PI;
    }
    return clamp(p, 0.0, MB_PI);
}

/// Square-root warp of the half elevation into `[0, 1]` (MERL convention):
/// `u = sqrt(theta_h / (pi/2))`.
fn mb_theta_h_to_unit(theta_h: f32) -> f32 {
    return mb_clamp01(sqrt(mb_clamp01(theta_h / MB_FRAC_PI_2)));
}

/// Inverse of [`mb_theta_h_to_unit`]: `theta_h = u^2 * (pi/2)`.
fn mb_unit_to_theta_h(u: f32) -> f32 {
    let uu = mb_clamp01(u);
    return uu * uu * MB_FRAC_PI_2;
}

/// Converts a local-frame `(wi, wo)` pair to Rusinkiewicz
/// `(theta_h, theta_d, phi_d)` coordinates (packed as a `vec3`), folding `phi_d`
/// to `[0, pi)` by reciprocity. Mirrors `directions_to_rusinkiewicz`.
fn mb_directions_to_rusinkiewicz(wi_in: vec3<f32>, wo_in: vec3<f32>) -> vec3<f32> {
    let wi = mb_normalize_or_zero(wi_in);
    let wo = mb_normalize_or_zero(wo_in);
    let h = mb_half_vector(wi, wo);

    let theta_h = acos(clamp(h.z, -1.0, 1.0));
    let phi_h = atan2(h.y, h.x);

    let tmp = mb_rotate_about_z(wi, -phi_h);
    let diff = mb_rotate_about_y(tmp, -theta_h);

    let theta_d = acos(clamp(diff.z, -1.0, 1.0));
    let phi_d = mb_fold_phi_d(atan2(diff.y, diff.x));

    return vec3<f32>(theta_h, theta_d, phi_d);
}

/// Reconstructs the representative incident direction from Rusinkiewicz
/// coordinates (half-vector azimuth pinned to zero). Mirrors the `wi` branch of
/// `rusinkiewicz_to_directions`.
fn mb_rusinkiewicz_wi(coords: vec3<f32>) -> vec3<f32> {
    let theta_h = coords.x;
    let theta_d = coords.y;
    let phi_d = coords.z;
    let sin_d = sin(theta_d);
    let cos_d = cos(theta_d);
    let diff = vec3<f32>(sin_d * cos(phi_d), sin_d * sin(phi_d), cos_d);
    return mb_normalize_or_zero(mb_rotate_about_y(diff, theta_h));
}

/// Reconstructs the representative outgoing direction (incident mirrored about
/// the half vector). Mirrors the `wo` branch of `rusinkiewicz_to_directions`.
fn mb_rusinkiewicz_wo(coords: vec3<f32>) -> vec3<f32> {
    let theta_h = coords.x;
    let theta_d = coords.y;
    let phi_d = coords.z;
    let sin_d = sin(theta_d);
    let cos_d = cos(theta_d);
    let diff = vec3<f32>(sin_d * cos(phi_d), sin_d * sin(phi_d), cos_d);
    let wi = mb_rotate_about_y(diff, theta_h);

    let sin_h = sin(theta_h);
    let cos_h = cos(theta_h);
    let h = vec3<f32>(sin_h, 0.0, cos_h);
    let d = 2.0 * dot(wi, h);
    let wo = d * h - wi;
    return mb_normalize_or_zero(wo);
}

/// Maps a `[0, 1]` unit coordinate to a continuous grid coordinate whose cell
/// centres sit at integer indices, clamped to the valid sampling range.
/// `n` is the axis resolution as a float. Mirrors `unit_to_grid`.
fn mb_unit_to_grid(u: f32, n: f32) -> f32 {
    let g = mb_clamp01(u) * n - 0.5;
    return clamp(g, 0.0, n - 1.0);
}

/// Lower integer cell (as a float) of a continuous grid coordinate.
/// Mirrors the `i0` of `split_grid`.
fn mb_grid_lower(grid: f32, n: f32) -> f32 {
    let g = clamp(grid, 0.0, n - 1.0);
    let base = floor(g);
    return min(base, n - 1.0);
}

/// `[0, 1]` fraction of a continuous grid coordinate towards the next cell.
/// Mirrors the `fi` of `split_grid`.
fn mb_grid_frac(grid: f32, n: f32) -> f32 {
    let g = clamp(grid, 0.0, n - 1.0);
    let base = floor(g);
    return clamp(g - base, 0.0, 1.0);
}

/// Full continuous grid coordinate for the half-elevation axis: applies the
/// square-root warp then the cell-centre remap. Mirrors `grid_coords[0]`.
fn mb_grid_coord_theta_h(theta_h: f32, n: f32) -> f32 {
    return mb_unit_to_grid(mb_theta_h_to_unit(theta_h), n);
}

/// Full continuous grid coordinate for the (linear) difference-elevation axis.
/// Mirrors `grid_coords[1]`.
fn mb_grid_coord_theta_d(theta_d: f32, n: f32) -> f32 {
    return mb_unit_to_grid(mb_clamp01(theta_d / MB_FRAC_PI_2), n);
}

/// Full continuous grid coordinate for the reciprocity-folded azimuth axis.
/// Mirrors `grid_coords[2]`.
fn mb_grid_coord_phi_d(phi_d: f32, n: f32) -> f32 {
    return mb_unit_to_grid(mb_clamp01(mb_fold_phi_d(phi_d) / MB_PI), n);
}

/// Trilinear interpolation of the eight bracketing RGB cells, with `phi_d`
/// (fraction `fk`) blended fastest, then `theta_d` (`fj`), then `theta_h`
/// (`fi`) - the exact reduction order of `sample_grid`.
fn mb_trilinear(
    c000: vec3<f32>,
    c001: vec3<f32>,
    c010: vec3<f32>,
    c011: vec3<f32>,
    c100: vec3<f32>,
    c101: vec3<f32>,
    c110: vec3<f32>,
    c111: vec3<f32>,
    fi: f32,
    fj: f32,
    fk: f32,
) -> vec3<f32> {
    let c00 = mb_lerp3(c000, c001, fk);
    let c01 = mb_lerp3(c010, c011, fk);
    let c10 = mb_lerp3(c100, c101, fk);
    let c11 = mb_lerp3(c110, c111, fk);
    let c0 = mb_lerp3(c00, c01, fj);
    let c1 = mb_lerp3(c10, c11, fj);
    return mb_lerp3(c0, c1, fi);
}

/// Scalar reduction of an RGB BRDF value used to drive importance sampling.
fn mb_scalar_brdf(rgb: vec3<f32>) -> f32 {
    return max((rgb.x + rgb.y + rgb.z) / 3.0, 0.0);
}

/// Analytic Cook-Torrance (GGX) + Lambert BRDF used to synthesise a table.
/// Mirrors `analytic_brdf`: GGX NDF, height-correlated Smith visibility and
/// Schlick Fresnel over a Lambertian base, returned as RGB radiance per
/// irradiance. Below-horizon pairs return zero.
fn mb_analytic_brdf(
    wi: vec3<f32>,
    wo: vec3<f32>,
    base_color: vec3<f32>,
    roughness_in: f32,
    metallic_in: f32,
    reflectance_in: f32,
) -> vec3<f32> {
    let n = vec3<f32>(0.0, 0.0, 1.0);
    let n_dot_l0 = dot(n, wi);
    let n_dot_v0 = dot(n, wo);
    if (n_dot_l0 <= 0.0 || n_dot_v0 <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let h = mb_half_vector(wi, wo);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    let n_dot_l = max(n_dot_l0, MB_MIN_COS);
    let n_dot_v = max(n_dot_v0, MB_MIN_COS);

    let metallic = clamp(metallic_in, 0.0, 1.0);
    let reflectance = clamp(reflectance_in, 0.0, 1.0);
    let roughness = clamp(roughness_in, MB_MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let base = max(base_color, vec3<f32>(0.0, 0.0, 0.0));

    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = vec3<f32>(f0_dielectric, f0_dielectric, f0_dielectric) * (1.0 - metallic) + base * metallic;

    // GGX NDF.
    let d_denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0;
    let d = alpha * alpha / max(MB_PI * d_denom * d_denom, MB_MIN_POSITIVE);

    // Height-correlated Smith visibility (includes the 1/(4 nl nv) term).
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    let vis = 0.5 / max(lambda_v + lambda_l, MB_MIN_POSITIVE);

    // Schlick Fresnel.
    let fc = mb_pow5(1.0 - v_dot_h);
    let fresnel = f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * fc;

    let diffuse_scale = (1.0 - metallic) / MB_PI;
    let specular = d * vis * fresnel;
    let out = base * diffuse_scale + specular;
    return max(out, vec3<f32>(0.0, 0.0, 0.0));
}
