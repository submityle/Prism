//! Shared thin-film iridescence (Belcour-Barla Airy reflectance) helpers; no
//! bindings or entry points.
//!
//! GPU twin of `gi/material/thin_film.rs`. A thin dielectric film over a
//! substrate reflects light from its outer (medium->film) and inner
//! (film->substrate) interfaces; the two reflected trains interfere with an
//! optical phase `phi = 4*pi*n_film*d*cos(theta_film)/lambda`, tinting soap
//! bubbles, oil slicks, and anodised metals.
//!
//! For lossless-dielectric Fresnel amplitudes `r1`, `r2` the per-polarization
//! intensity reflectance of the single film has the closed Airy form
//! `R = (r1^2 + r2^2 + 2*r1*r2*cos phi) / (1 + r1^2*r2^2 + 2*r1*r2*cos phi)`,
//! averaged over s/p for unpolarised light and reconstructed on the three fixed
//! primaries 630/532/465 nm.
//!
//! Keep arithmetic order, clamps, floors and fallback branches aligned with the
//! CPU golden byte-for-byte. All indices of refraction are clamped `>= 1`, all
//! cosines to `[1e-4, 1]`; a zero thickness collapses the Airy formula EXACTLY
//! to the direct outer->substrate Fresnel reflectance (no tint). Rust's
//! `Option<cos_t>` total-internal-reflection result is encoded here as a
//! negative sentinel (`< 0`), which callers branch on; the surviving arithmetic
//! order is unchanged.

/// Representative R/G/B sampling wavelengths in nanometres (classic primaries).
const TF_WAVELENGTH_R_NM: f32 = 630.0;
const TF_WAVELENGTH_G_NM: f32 = 532.0;
const TF_WAVELENGTH_B_NM: f32 = 465.0;
/// Smallest incidence cosine evaluated; keeps grazing angles off the Fresnel
/// denominator singularity.
const TF_MIN_COS: f32 = 1.0e-4;
/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const TF_MIN_IOR: f32 = 1.0;
/// Smallest NDF denominator / Airy denominator floor.
const TF_MIN_ALPHA: f32 = 1.0e-4;
/// Smallest positive normal f32, matching the CPU `f32::MIN_POSITIVE` guard.
const TF_MIN_POSITIVE: f32 = 1.1754944e-38;
/// `core::f32::consts::PI` rounded to f32, byte-identical to the CPU golden.
const TF_PI: f32 = 3.1415927;

/// Scalar finite predicate, matching the shared isfinite idiom.
fn tf_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
fn tf_clamp_ior(n: f32) -> f32 {
    if (tf_is_finite(n)) {
        return max(n, TF_MIN_IOR);
    }
    return TF_MIN_IOR;
}

/// Clamps a cosine of an incidence angle to `[MIN_COS, 1]`.
fn tf_clamp_cos(c: f32) -> f32 {
    if (tf_is_finite(c)) {
        return clamp(c, TF_MIN_COS, 1.0);
    }
    return TF_MIN_COS;
}

/// Cosine of the transmitted angle via Snell's law. Returns a negative sentinel
/// (total internal reflection has no real transmitted direction); otherwise a
/// non-negative transmitted cosine. Mirrors `transmitted_cos`'s input clamps.
fn tf_transmitted_cos(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let ci = tf_clamp_cos(cos_i);
    let ni = tf_clamp_ior(n_i);
    let nt = tf_clamp_ior(n_t);
    let sin_i2 = max(1.0 - ci * ci, 0.0);
    let eta = ni / nt;
    let sin_t2 = eta * eta * sin_i2;
    if (sin_t2 >= 1.0) {
        return -1.0;
    }
    return sqrt(max(1.0 - sin_t2, 0.0));
}

/// Real Fresnel amplitude coefficients `(r_s, r_p)` for one dielectric
/// interface, returned as a `vec2` carrying the half-wave sign flip.
fn tf_fresnel_amplitudes(cos_i: f32, cos_t: f32, n_i: f32, n_t: f32) -> vec2<f32> {
    let a = n_i * cos_i;
    let b = n_t * cos_t;
    let c = n_t * cos_i;
    let d = n_i * cos_t;
    let r_s = (a - b) / (a + b);
    let r_p = (c - d) / (c + d);
    return vec2<f32>(r_s, r_p);
}

/// Unpolarised intensity reflectance of one dielectric interface; `1` beyond
/// the critical angle. This is the reference the Airy stack collapses to at
/// zero thickness.
fn tf_fresnel_dielectric_unpolarized(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let ci = tf_clamp_cos(cos_i);
    let ni = tf_clamp_ior(n_i);
    let nt = tf_clamp_ior(n_t);
    let cos_t = tf_transmitted_cos(ci, ni, nt);
    if (cos_t < 0.0) {
        return 1.0;
    }
    let r = tf_fresnel_amplitudes(ci, cos_t, ni, nt);
    return clamp(0.5 * (r.x * r.x + r.y * r.y), 0.0, 1.0);
}

/// Accumulated optical phase `phi = 4*pi*n_film*d*cos(theta_film)/lambda`,
/// returned unwrapped. `cos_film` is the cosine inside the film.
fn tf_optical_phase(film_ior: f32, cos_film: f32, thickness_nm: f32, wavelength_nm: f32) -> f32 {
    let n1 = tf_clamp_ior(film_ior);
    let cf = clamp(cos_film, 0.0, 1.0);
    let d = max(thickness_nm, 0.0);
    let lambda = max(wavelength_nm, 1.0);
    return 4.0 * TF_PI * n1 * d * cf / lambda;
}

/// One polarization of the single-film Airy reflectance given both interface
/// amplitudes and the shared phase cosine. The denominator floor matches the
/// CPU golden's `MIN_ALPHA` guard.
fn tf_airy_one(r1: f32, r2: f32, cos_phi: f32) -> f32 {
    let num = r1 * r1 + r2 * r2 + 2.0 * r1 * r2 * cos_phi;
    let den = 1.0 + r1 * r1 * r2 * r2 + 2.0 * r1 * r2 * cos_phi;
    if (abs(den) > TF_MIN_ALPHA) {
        return clamp(num / den, 0.0, 1.0);
    }
    return 1.0;
}

/// Single-film Airy intensity reflectance at one wavelength for the stack
/// outer(`outer_ior`) -> film(`film_ior`, `thickness_nm`) -> substrate
/// (`base_ior`) at incident cosine `cos_outer`. Total internal reflection at
/// either interface returns `1`; zero thickness returns the direct
/// outer->substrate Fresnel reflectance.
fn tf_airy_reflectance(
    outer_ior: f32,
    film_ior: f32,
    base_ior: f32,
    cos_outer: f32,
    thickness_nm: f32,
    wavelength_nm: f32,
) -> f32 {
    let n0 = tf_clamp_ior(outer_ior);
    let n1 = tf_clamp_ior(film_ior);
    let n2 = tf_clamp_ior(base_ior);
    let cos0 = tf_clamp_cos(cos_outer);

    // Angle inside the film (outer -> film). TIR here is a perfect mirror.
    let cos1 = tf_transmitted_cos(cos0, n0, n1);
    if (cos1 < 0.0) {
        return 1.0;
    }
    // Angle inside the substrate (film -> base). TIR here is a perfect mirror.
    let cos2 = tf_transmitted_cos(cos1, n1, n2);
    if (cos2 < 0.0) {
        return 1.0;
    }

    let r01 = tf_fresnel_amplitudes(cos0, cos1, n0, n1);
    let r12 = tf_fresnel_amplitudes(cos1, cos2, n1, n2);

    let phi = tf_optical_phase(n1, cos1, thickness_nm, wavelength_nm);
    let cos_phi = cos(phi);

    return 0.5 * (tf_airy_one(r01.x, r12.x, cos_phi) + tf_airy_one(r01.y, r12.y, cos_phi));
}

/// Iridescent RGB reflectance sampled on the three fixed primaries. Each
/// channel is in `[0, 1]`; zero thickness yields the achromatic base Fresnel
/// reflectance (no tint).
fn tf_iridescent_reflectance_rgb(
    outer_ior: f32,
    film_ior: f32,
    base_ior: f32,
    cos_outer: f32,
    thickness_nm: f32,
) -> vec3<f32> {
    return vec3<f32>(
        tf_airy_reflectance(outer_ior, film_ior, base_ior, cos_outer, thickness_nm, TF_WAVELENGTH_R_NM),
        tf_airy_reflectance(outer_ior, film_ior, base_ior, cos_outer, thickness_nm, TF_WAVELENGTH_G_NM),
        tf_airy_reflectance(outer_ior, film_ior, base_ior, cos_outer, thickness_nm, TF_WAVELENGTH_B_NM),
    );
}

// Substrate IOR implied by a Schlick `f0`, inverting `f0 = ((n-1)/(n+1))^2`
// for an air outer medium. Mirrors `base_ior_from_f0` in `thin_film.rs`.
fn tf_base_ior_from_f0(f0: f32) -> f32 {
    let s = sqrt(clamp(f0, 0.0, 1.0 - TF_MIN_ALPHA));
    return (1.0 + s) / (1.0 - s);
}

// Per-channel iridescent Fresnel for a substrate whose RGB normal-incidence
// reflectance is `f0`. Recovers a distinct base IOR per channel from `f0` and
// evaluates the Airy stack on each primary (glTF `KHR_materials_iridescence`).
// Mirrors `iridescent_fresnel_rgb` in `thin_film.rs`.
fn tf_iridescent_fresnel_rgb(
    outer_ior: f32,
    film_ior: f32,
    f0: vec3<f32>,
    cos_outer: f32,
    thickness_nm: f32,
) -> vec3<f32> {
    return vec3<f32>(
        tf_airy_reflectance(outer_ior, film_ior, tf_base_ior_from_f0(f0.x), cos_outer, thickness_nm, TF_WAVELENGTH_R_NM),
        tf_airy_reflectance(outer_ior, film_ior, tf_base_ior_from_f0(f0.y), cos_outer, thickness_nm, TF_WAVELENGTH_G_NM),
        tf_airy_reflectance(outer_ior, film_ior, tf_base_ior_from_f0(f0.z), cos_outer, thickness_nm, TF_WAVELENGTH_B_NM),
    );
}


/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
fn tf_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > TF_MIN_POSITIVE) {
        return v * (1.0 / sqrt(len_sq));
    }
    return fallback;
}

/// Disney anisotropic roughness remap to GGX `(alpha_t, alpha_b)` as a `vec2`,
/// both floored at `MIN_ALPHA`.
fn tf_anisotropic_alphas(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let r = clamp(roughness, 0.0, 1.0);
    let a = clamp(anisotropy, -1.0, 1.0);
    let alpha = r * r;
    let aspect = sqrt(max(1.0 - 0.9 * a, TF_MIN_ALPHA));
    let alpha_t = max(alpha / aspect, TF_MIN_ALPHA);
    let alpha_b = max(alpha * aspect, TF_MIN_ALPHA);
    return vec2<f32>(alpha_t, alpha_b);
}

/// Anisotropic Trowbridge-Reitz (GGX) NDF in the `(tangent, bitangent, normal)`
/// frame; `0` for half-vectors in or below the tangent plane.
fn tf_ggx_aniso_ndf(
    half: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let n = tf_normalize_or(normal, vec3<f32>(0.0, 0.0, 1.0));
    let t = tf_normalize_or(tangent, vec3<f32>(1.0, 0.0, 0.0));
    let b = tf_normalize_or(bitangent, vec3<f32>(0.0, 1.0, 0.0));
    let h = tf_normalize_or(half, n);
    let at = max(alpha_t, TF_MIN_ALPHA);
    let ab = max(alpha_b, TF_MIN_ALPHA);

    let hn = dot(h, n);
    if (hn <= 0.0) {
        return 0.0;
    }
    let ht = dot(h, t);
    let hb = dot(h, b);

    let a = ht / at;
    let c = hb / ab;
    let inner = a * a + c * c + hn * hn;
    let denom = TF_PI * at * ab * inner * inner;
    if (denom > TF_MIN_POSITIVE) {
        return max(1.0 / denom, 0.0);
    }
    return 0.0;
}

/// Smith `G1` masking-shadowing term for anisotropic GGX via Heitz's Lambda;
/// in `(0, 1]`, and `0` for directions in or below the tangent plane.
fn tf_smith_g1_aniso(
    v: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let n = tf_normalize_or(normal, vec3<f32>(0.0, 0.0, 1.0));
    let t = tf_normalize_or(tangent, vec3<f32>(1.0, 0.0, 0.0));
    let b = tf_normalize_or(bitangent, vec3<f32>(0.0, 1.0, 0.0));
    let dir = tf_normalize_or(v, n);
    let at = max(alpha_t, TF_MIN_ALPHA);
    let ab = max(alpha_b, TF_MIN_ALPHA);

    let vn = dot(dir, n);
    if (vn <= 0.0) {
        return 0.0;
    }
    let vt = dot(dir, t);
    let vb = dot(dir, b);
    let ratio = (at * at * vt * vt + ab * ab * vb * vb) / (vn * vn);
    let lambda = 0.5 * (-1.0 + sqrt(max(1.0 + ratio, 0.0)));
    return clamp(1.0 / (1.0 + lambda), 0.0, 1.0);
}
