//! Two-sided foliage forward scattering — the CPU golden reference for light
//! transmitted through thin double-sided surfaces such as leaves, grass blades,
//! paper, and cloth.
//!
//! Thin vegetation is lit from both faces: a viewer looking roughly *towards*
//! the light through a backlit leaf sees a bright, forward-peaked glow as light
//! refracts and scatters through the blade.  This module models that glow with
//! a Henyey-Greenstein-style *forward lobe* on the view/light geometry, a
//! normal-independent *two-sided wrap* diffuse term (so both faces respond), and
//! Beer-Lambert attenuation over the leaf thickness:
//!
//! ```text
//!   cos_fwd = dot(V, -L)                             (view toward the light)
//!   p(cos)  = HG(cos_fwd, g),  g = g(roughness)      (forward lobe)
//!   wrap    = max(0, (|N·L| + w)/(1 + w))            (two-sided, normal-free)
//!   L_trans = light · leaf · translucency · p · exp(-sigma_t·|d|) · wrap
//! ```
//!
//! Smoother leaves (`roughness → 0`) produce a tight, specular-like forward
//! lobe; rougher leaves (`roughness → 1`) scatter the transmitted light almost
//! isotropically.  The wrap term uses `|N·L|` so the effect is symmetric across
//! the leaf's two sides and independent of which way the authored normal faces.
//!
//! # Conventions
//! * `view` (`V`) points from the surface toward the camera; `light` (`L`)
//!   points from the surface toward the light.  Both are normalized defensively;
//!   a degenerate (near-zero) direction falls back to a neutral cosine of `0`.
//! * `roughness` is in `[0, 1]`; it maps to the forward anisotropy
//!   `g = (1 - roughness)` clamped just inside `[0, 1)`.
//! * `translucency` is a non-negative transmission gain (typically `[0, 1]`).
//! * `thickness` is a leaf path length in world units; attenuation uses its
//!   magnitude.  `sigma_t` is a non-negative linear-RGB extinction.
//! * `wrap` is a non-negative wrap factor; `n_dot_l` is clamped to `[-1, 1]`.
//! * All colours are non-negative linear-RGB; every result is finite and
//!   non-negative per channel (never `NaN`).
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Reciprocal of `4π`, the isotropic phase-function value.
const INV_4PI: f32 = 1.0 / (4.0 * PI);

/// Largest magnitude the forward anisotropy `g` may take; keeps the HG
/// denominator `1 + g^2 - 2 g c > 0`.
const MAX_G: f32 = 1.0 - 1.0e-4;

/// Largest optical depth evaluated before the exponential is treated as zero.
const MAX_OPTICAL_DEPTH: f32 = 80.0;

/// Squared length below which a direction is treated as degenerate.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// Henyey-Greenstein-style forward lobe `p(cos_theta, g)`.
///
/// Returns `(1 - g^2) / (4π (1 + g^2 - 2 g · cos_theta)^{3/2})`, the normalised
/// angular scattering distribution (unit integral over the sphere).  For
/// foliage, `cos_theta` is the forward cosine `dot(V, -L)`, so positive `g`
/// concentrates transmitted light along the view-toward-light direction.  `g`
/// is clamped to `(-1, 1)` and `cos_theta` to `[-1, 1]`; as `g → 0` the lobe
/// collapses to the isotropic `1 / 4π`.
#[inline]
pub fn forward_phase(cos_theta: f32, g: f32) -> f32 {
    let cos_theta = clamp_finite(cos_theta, -1.0, 1.0);
    let g = clamp_finite(g, -MAX_G, MAX_G);
    let denom = (1.0 + g * g - 2.0 * g * cos_theta).max(1.0e-12);
    let value = (1.0 - g * g) * INV_4PI / ops::powf(denom, 1.5);
    if value.is_finite() {
        value.max(0.0)
    } else {
        INV_4PI
    }
}

/// Maps a leaf `roughness ∈ [0, 1]` to the forward-lobe anisotropy `g`.
///
/// Uses `g = 1 - roughness`, clamped just inside `[0, 1)`: a smooth leaf
/// (`roughness = 0`) gives a tight forward lobe (`g → 1`), while a rough leaf
/// (`roughness = 1`) gives isotropic transmission (`g = 0`).  `roughness` is
/// clamped to `[0, 1]`.
#[inline]
pub fn forward_g_from_roughness(roughness: f32) -> f32 {
    let r = clamp_finite(roughness, 0.0, 1.0);
    (1.0 - r).clamp(0.0, MAX_G)
}

/// Two-sided (normal-independent) wrap-around diffuse response.
///
/// Equal to `max(0, (|N·L| + wrap)/(1 + wrap))`: taking the magnitude of the
/// cosine makes both leaf faces respond identically, and the `wrap` shift
/// softens the terminator.  `n_dot_l` is clamped to `[-1, 1]`, `wrap` to be
/// non-negative, and the result lies in `[0, 1]`.
#[inline]
pub fn two_sided_wrap(n_dot_l: f32, wrap: f32) -> f32 {
    let x = clamp_finite(n_dot_l, -1.0, 1.0).abs();
    let w = clamp_non_negative(wrap);
    let value = (x + w) / (1.0 + w);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Transmitted radiance through a two-sided foliage element.
///
/// Combines the forward lobe on the view/light geometry, the normal-independent
/// two-sided wrap diffuse, Beer-Lambert attenuation over the leaf `thickness`,
/// and the `translucency` gain:
///
/// ```text
///   L = light_color · leaf_color · translucency
///       · forward_phase(dot(V,-L), g(roughness))
///       · exp(-sigma_t·|thickness|)
///       · two_sided_wrap(N·L, wrap).
/// ```
///
/// `view` and `light` are normalized defensively; a degenerate direction uses a
/// neutral forward cosine of `0`.  All colours are clamped non-negative and the
/// result is finite and non-negative per channel.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn foliage_transmission(
    view: Vec3,
    light: Vec3,
    normal: Vec3,
    roughness: f32,
    translucency: f32,
    thickness: f32,
    sigma_t: Vec3,
    light_color: Vec3,
    leaf_color: Vec3,
    wrap: f32,
) -> Vec3 {
    // Forward cosine dot(V, -L): the view looking back toward the light.
    let cos_fwd = match (safe_normalize(view), safe_normalize(light)) {
        (Some(v), Some(l)) => clamp_finite(-v.dot(l), -1.0, 1.0),
        _ => 0.0,
    };
    let g = forward_g_from_roughness(roughness);
    let phase = forward_phase(cos_fwd, g);

    let n_dot_l = match (safe_normalize(normal), safe_normalize(light)) {
        (Some(n), Some(l)) => clamp_finite(n.dot(l), -1.0, 1.0),
        _ => 0.0,
    };
    let wrap_term = two_sided_wrap(n_dot_l, wrap);

    let attenuation = beer_lambert_rgb(sigma_t, thickness);
    let translucency = clamp_non_negative(translucency);

    let out = sanitize_rgb(light_color)
        * sanitize_rgb(leaf_color)
        * attenuation
        * (translucency * phase * wrap_term);
    sanitize_rgb(out)
}

/// Spectral Beer-Lambert transmittance `exp(-sigma_t·|thickness|)` per channel.
///
/// A thin convenience mirroring the transmission module so foliage stays
/// self-contained; each channel of `sigma_t` is clamped non-negative and the
/// result lies in `[0, 1]`.
#[inline]
pub fn beer_lambert_rgb(sigma_t: Vec3, thickness: f32) -> Vec3 {
    let thickness = clamp_non_negative(abs_finite(thickness));
    Vec3::new(
        beer_lambert(sigma_t.x, thickness),
        beer_lambert(sigma_t.y, thickness),
        beer_lambert(sigma_t.z, thickness),
    )
}

/// Scalar Beer-Lambert transmittance `exp(-sigma_t · thickness)`.
#[inline]
fn beer_lambert(sigma_t: f32, thickness: f32) -> f32 {
    let sigma_t = clamp_non_negative(sigma_t);
    let tau = (sigma_t * thickness).min(MAX_OPTICAL_DEPTH);
    let t = ops::exp(-tau);
    t.clamp(0.0, 1.0)
}

/// Normalizes `v`, returning `None` for a near-zero (degenerate) direction.
#[inline]
fn safe_normalize(v: Vec3) -> Option<Vec3> {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_LEN_SQ {
        let inv = 1.0 / len_sq.sqrt();
        let n = v * inv;
        if n.is_finite() {
            return Some(n);
        }
    }
    None
}

/// Absolute value with non-finite inputs mapped to `0`.
#[inline]
fn abs_finite(value: f32) -> f32 {
    if value.is_finite() {
        value.abs()
    } else {
        0.0
    }
}

/// Clamps `value` into `[lo, hi]`, mapping non-finite inputs to `lo`.
#[inline]
fn clamp_finite(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        lo
    }
}

/// Clamps `value` to be non-negative and finite.
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Replaces any non-finite channel with `0` and clamps every channel
/// non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Midpoint quadrature over the sphere: `∫ f dω = 2π ∫_{-1}^{1} f dμ`.
    fn sphere_integral(f: impl Fn(f32) -> f32) -> f32 {
        let steps = 20_000;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let mu = -1.0 + 2.0 * (i as f32 + 0.5) / steps as f32;
            acc += f(mu) as f64;
        }
        let dmu = 2.0 / steps as f32;
        (2.0 * PI as f64 * acc * dmu as f64) as f32
    }

    #[test]
    fn forward_phase_integrates_to_unity() {
        for g in [0.0f32, 0.3, 0.6, 0.9] {
            let integral = sphere_integral(|mu| forward_phase(mu, g));
            assert!((integral - 1.0).abs() < 2e-3, "g={g} integral={integral}");
        }
    }

    #[test]
    fn forward_phase_peaks_forward() {
        let g = 0.7;
        let fwd = forward_phase(1.0, g);
        let bwd = forward_phase(-1.0, g);
        assert!(fwd > bwd, "fwd={fwd} bwd={bwd}");
        // g = 0 is isotropic.
        for c in [-1.0f32, 0.0, 1.0] {
            assert!((forward_phase(c, 0.0) - INV_4PI).abs() < 1e-6);
        }
    }

    #[test]
    fn forward_phase_clamps_extreme_anisotropy() {
        for g in [-5.0f32, -1.0, 1.0, 5.0, f32::NAN] {
            for c in [-1.0f32, 0.0, 1.0] {
                let p = forward_phase(c, g);
                assert!(p.is_finite() && p >= 0.0, "g={g} c={c} p={p}");
            }
        }
    }

    #[test]
    fn g_from_roughness_is_monotonic_and_bounded() {
        let mut prev = 1.0;
        for i in 0..=10 {
            let r = i as f32 / 10.0;
            let g = forward_g_from_roughness(r);
            assert!((0.0..1.0).contains(&g), "r={r} g={g}");
            assert!(g <= prev + 1e-6, "not decreasing at r={r}");
            prev = g;
        }
        assert!(forward_g_from_roughness(0.0) > 0.9);
        assert_eq!(forward_g_from_roughness(1.0), 0.0);
    }

    #[test]
    fn two_sided_wrap_is_symmetric_and_bounded() {
        let w = 0.4;
        for x in [-1.0f32, -0.6, -0.2, 0.2, 0.6, 1.0] {
            let a = two_sided_wrap(x, w);
            let b = two_sided_wrap(-x, w);
            assert!((a - b).abs() < 1e-6, "x={x} a={a} b={b}");
            assert!((0.0..=1.0).contains(&a));
        }
    }

    #[test]
    fn transmission_peaks_viewing_toward_light() {
        // Light coming from +Z toward surface => L points to the light (+Z).
        let light = Vec3::Z;
        let normal = Vec3::Y;
        let sigma = Vec3::splat(0.5);
        let lc = Vec3::splat(2.0);
        let leaf = Vec3::new(0.3, 0.6, 0.2);
        // Viewing from behind the leaf toward the light: V = -Z, so dot(V,-L)=1.
        let aligned = foliage_transmission(
            -Vec3::Z, light, normal, 0.1, 1.0, 1.0, sigma, lc, leaf, 0.5,
        );
        // Viewing from the same side as the light: V = +Z, dot(V,-L) = -1.
        let anti = foliage_transmission(
            Vec3::Z, light, normal, 0.1, 1.0, 1.0, sigma, lc, leaf, 0.5,
        );
        assert!(aligned.x > anti.x, "aligned={aligned} anti={anti}");
    }

    #[test]
    fn transmission_vanishes_without_translucency() {
        let out = foliage_transmission(
            -Vec3::Z, Vec3::Z, Vec3::Y, 0.2, 0.0, 1.0, Vec3::splat(0.5), Vec3::ONE, Vec3::ONE, 0.5,
        );
        assert!(out.max_element() < 1e-6, "out={out}");
    }

    #[test]
    fn transmission_attenuates_with_thickness() {
        let thin = foliage_transmission(
            -Vec3::Z, Vec3::Z, Vec3::Y, 0.3, 1.0, 0.5, Vec3::splat(1.0), Vec3::ONE, Vec3::ONE, 0.5,
        );
        let thick = foliage_transmission(
            -Vec3::Z, Vec3::Z, Vec3::Y, 0.3, 1.0, 4.0, Vec3::splat(1.0), Vec3::ONE, Vec3::ONE, 0.5,
        );
        assert!(thin.x > thick.x, "thin={thin} thick={thick}");
    }

    #[test]
    fn degenerate_direction_uses_neutral_cosine() {
        // Zero view vector must not NaN; falls back to cos = 0 (isotropic use).
        let out = foliage_transmission(
            Vec3::ZERO, Vec3::Z, Vec3::Y, 0.5, 1.0, 1.0, Vec3::splat(0.5), Vec3::ONE, Vec3::ONE, 0.5,
        );
        assert!(out.is_finite());
    }

    #[test]
    fn is_deterministic() {
        let args = || {
            foliage_transmission(
                -Vec3::Z, Vec3::Z, Vec3::Y, 0.25, 0.8, 1.2, Vec3::splat(0.6), Vec3::ONE,
                Vec3::splat(0.5), 0.3,
            )
        };
        assert_eq!(args(), args());
        assert_eq!(forward_phase(0.4, 0.5), forward_phase(0.4, 0.5));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        let out = foliage_transmission(
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            f32::NAN,
            f32::NAN,
            f32::NAN,
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            f32::NAN,
        );
        assert!(out.is_finite());
        assert!(forward_phase(f32::NAN, f32::NAN).is_finite());
        assert!(two_sided_wrap(f32::NAN, f32::NAN).is_finite());
        assert!(beer_lambert_rgb(Vec3::splat(f32::NAN), f32::NAN).is_finite());
    }
}
