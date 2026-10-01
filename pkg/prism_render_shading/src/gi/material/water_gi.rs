//! Water-surface and wet-surface GI blending — CPU golden.
//!
//! Water is a smooth dielectric (`n ~= 1.33`) and its global-illumination
//! response is governed almost entirely by the Fresnel split between a
//! reflected and a refracted ray.  This module is the backend-neutral reference
//! for that split and for the "wet surface" look that a thin water layer gives
//! an otherwise diffuse material.
//!
//! It provides:
//!
//! * Two interchangeable Fresnel evaluators — the fast Schlick approximation
//!   and the exact unpolarised dielectric reflectance — so the GPU twin can be
//!   validated against the physical ground truth.
//! * Mirror [`reflect`] and Snell [`refract`] directions, with total internal
//!   reflection (TIR) reported as `None`.
//! * An energy-conserving [`reflect_refract_split`] that returns reflected and
//!   transmitted weights summing to one.
//! * A classic wet-surface modulation ([`wet_modulate`]) that lowers the
//!   effective roughness (water smooths micro-geometry) and darkens the albedo
//!   (water fills surface pores), both driven by a `wetness` parameter.
//!
//! # Conventions
//! * Direction vectors use the common real-time convention: the incident
//!   vector `incident` *points toward the surface* (from the eye/light into the
//!   surface), and `normal` points *out of* the surface toward the incident
//!   medium.  Reflected and refracted vectors are returned as unit vectors.
//! * `eta` passed to [`refract`] is the relative index `n_i / n_t` of the
//!   incident over the transmitted medium.
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`;
//!   square roots via the `f32::sqrt` method (never `f32::exp`).
//! * All inputs are defensively clamped (indices to `>= 1`, cosines to
//!   `[0, 1]`, `wetness` to `[0, 1]`); every result is finite — no `NaN`, no
//!   division by zero.
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::{ops, Vec3};

/// Index of refraction of liquid water at visible wavelengths.
pub const WATER_IOR: f32 = 1.33;

/// Index of refraction of air / vacuum (the usual incident medium).
pub const AIR_IOR: f32 = 1.0;

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Fraction a fully wet surface's roughness is scaled *toward*: at
/// `wetness == 1` the effective roughness is `WET_ROUGHNESS_FLOOR` of the dry
/// value, modelling the mirror-smooth water film over the micro-geometry.
const WET_ROUGHNESS_FLOOR: f32 = 0.2;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Clamps a cosine of an incidence angle to `[0, 1]`.
#[inline]
fn clamp_cos(cos: f32) -> f32 {
    if cos.is_finite() {
        cos.clamp(0.0, 1.0)
    } else {
         0.0
    }
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Normal-incidence reflectance `F0` for a dielectric interface.
///
/// Returns `((n_t - n_i) / (n_t + n_i))^2`, the fraction of power reflected at
/// perpendicular incidence.  For an air → water interface
/// (`n_i = 1`, `n_t = 1.33`) this is `~= 0.02`, the familiar value used to seed
/// the Schlick approximation.
#[inline]
pub fn f0_dielectric(n_i: f32, n_t: f32) -> f32 {
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    let r = (n_t - n_i) / (n_t + n_i);
    (r * r).clamp(0.0, 1.0)
}

/// Schlick's approximation of the Fresnel reflectance.
///
/// `F(cos) = F0 + (1 - F0) * (1 - cos)^5`, where `cos` is the cosine of the
/// angle between the incident direction and the surface normal and `f0` is the
/// normal-incidence reflectance (see [`f0_dielectric`]).  Fast and widely used
/// for GI, exact at `cos = 1`, and increasingly divergent from the true
/// dielectric curve only near grazing angles.
#[inline]
pub fn fresnel_schlick(cos: f32, f0: f32) -> f32 {
    let cos = clamp_cos(cos);
    let f0 = f0.clamp(0.0, 1.0);
    let m = 1.0 - cos;
    let m2 = m * m;
    let m5 = m2 * m2 * m;
    (f0 + (1.0 - f0) * m5).clamp(0.0, 1.0)
}

/// Exact unpolarised Fresnel reflectance of a dielectric interface.
///
/// Averages the s- and p-polarised power reflectances for light crossing from
/// medium `n_i` into medium `n_t` at incident cosine `cos_i`.  Beyond the
/// critical angle (total internal reflection, only possible when
/// `n_i > n_t`) the surface reflects everything and the result is `1`.
///
/// This is the physical ground truth [`fresnel_schlick`] approximates; the two
/// agree closely away from grazing angles (verified in the tests).
#[inline]
pub fn fresnel_dielectric(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let cos_i = clamp_cos(cos_i);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);

    let sin_i2 = (1.0 - cos_i * cos_i).max(0.0);
    let eta = n_i / n_t;
    let sin_t2 = eta * eta * sin_i2;
    if sin_t2 >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t2).max(0.0).sqrt();

    let r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    let r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t);
    (0.5 * (r_s * r_s + r_p * r_p)).clamp(0.0, 1.0)
}

/// Mirror reflection of an incident direction about a surface normal.
///
/// With `incident` pointing *into* the surface and `normal` pointing *out*,
/// returns the unit reflected direction `I - 2*(N·I)*N`.  Both inputs are
/// normalised defensively; a degenerate normal falls back to `+Y`.
#[inline]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    let i = normalize_or(incident, Vec3::NEG_Y);
    let n = normalize_or(normal, Vec3::Y);
    let r = i - 2.0 * n.dot(i) * n;
    normalize_or(r, i)
}

/// Snell refraction of an incident direction through a dielectric interface.
///
/// `incident` points into the surface, `normal` points out toward the incident
/// medium, and `eta = n_i / n_t` is the relative index.  Returns the unit
/// refracted direction, or `None` on total internal reflection (when the
/// radicand `k = 1 - eta^2 * (1 - (N·I)^2)` is negative).
///
/// Uses the standard closed form `eta*I + (eta*c - sqrt(k))*N` with
/// `c = -(N·I)`.
#[inline]
pub fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let i = normalize_or(incident, Vec3::NEG_Y);
    let n = normalize_or(normal, Vec3::Y);
    let eta = if eta.is_finite() { eta.max(0.0) } else { 1.0 };

    let cos_i = -n.dot(i);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        let t = eta * i + (eta * cos_i - k.sqrt()) * n;
        Some(normalize_or(t, i))
    }
}

/// Reflected and transmitted energy weights for a dielectric interface.
///
/// Returns `(reflect_weight, transmit_weight)` where `reflect_weight` is the
/// exact Fresnel reflectance at incident cosine `cos_i` for the `n_i -> n_t`
/// interface and `transmit_weight = 1 - reflect_weight`.  The two always sum to
/// one, so no energy is created or lost; at total internal reflection the pair
/// is `(1, 0)`.
#[inline]
pub fn reflect_refract_split(cos_i: f32, n_i: f32, n_t: f32) -> (f32, f32) {
    let r = fresnel_dielectric(cos_i, n_i, n_t);
    (r, (1.0 - r).clamp(0.0, 1.0))
}

/// The result of wetting a dry surface: a modulated roughness and albedo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetModulation {
    /// Effective perceptual roughness after wetting; `<=` the dry roughness.
    pub roughness: f32,
    /// Darkened RGB albedo after wetting; each channel `<=` the dry channel.
    pub albedo: Vec3,
}

/// Classic wet-surface modulation of a dry material's roughness and albedo.
///
/// A water film does two things to the apparent BRDF, both parameterised by
/// `wetness` in `[0, 1]`:
///
/// * **Smoother specular.** The water layer flattens micro-geometry, so the
///   effective roughness is lerped toward [`WET_ROUGHNESS_FLOOR`] of its dry
///   value: `r' = r * mix(1, WET_ROUGHNESS_FLOOR, wetness)`.
/// * **Darker diffuse.** Water seeping into surface pores increases internal
///   light absorption, so the albedo is deepened via
///   `albedo' = albedo^(1 + wetness)` per channel.  Because albedo is in
///   `[0, 1]`, raising it to a power `>= 1` only ever darkens it, and
///   `wetness = 0` leaves it unchanged.
///
/// Both outputs are clamped to their valid ranges; `wetness` is clamped to
/// `[0, 1]`.
#[inline]
pub fn wet_modulate(dry_roughness: f32, dry_albedo: Vec3, wetness: f32) -> WetModulation {
    let w = if wetness.is_finite() {
        wetness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let r = dry_roughness.clamp(0.0, 1.0);

    // Smoother: scale roughness toward WET_ROUGHNESS_FLOOR * r.
    let scale = 1.0 + (WET_ROUGHNESS_FLOOR - 1.0) * w;
    let roughness = (r * scale).clamp(0.0, 1.0);

    // Darker: albedo^(1 + wetness), per channel, with albedo clamped to [0,1].
    let exponent = 1.0 + w;
    let a = dry_albedo
        .max(Vec3::ZERO)
        .min(Vec3::ONE);
    let albedo = Vec3::new(
        ops::powf(a.x, exponent).clamp(0.0, 1.0),
        ops::powf(a.y, exponent).clamp(0.0, 1.0),
        ops::powf(a.z, exponent).clamp(0.0, 1.0),
    );

    WetModulation { roughness, albedo }
}

/// Critical angle cosine for total internal reflection across `n_i -> n_t`.
///
/// Returns `Some(cos_c)` — the smallest incident cosine that still transmits —
/// when `n_i > n_t` (TIR is possible), or `None` when it is not (the ray always
/// transmits).  `cos_c = sqrt(1 - (n_t / n_i)^2)`.
#[inline]
pub fn critical_angle_cos(n_i: f32, n_t: f32) -> Option<f32> {
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    if n_i <= n_t {
        None
    } else {
        let s = n_t / n_i;
        Some((1.0 - s * s).max(0.0).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_normal_incidence_is_two_percent() {
        // Air -> water head-on reflects ~2% of the light.
        let f0 = f0_dielectric(AIR_IOR, WATER_IOR);
        assert!((f0 - 0.02).abs() < 0.002, "f0={f0}");
        let exact = fresnel_dielectric(1.0, AIR_IOR, WATER_IOR);
        assert!((exact - f0).abs() < 1.0e-5, "exact={exact} f0={f0}");
        let schlick = fresnel_schlick(1.0, f0);
        assert!((schlick - f0).abs() < 1.0e-6, "schlick={schlick}");
    }

    #[test]
    fn total_internal_reflection_above_critical_angle() {
        // Water -> air. Critical cos ~ sqrt(1 - (1/1.33)^2) ~ 0.659.
        let cos_c = critical_angle_cos(WATER_IOR, AIR_IOR).unwrap();
        assert!((cos_c - 0.659).abs() < 0.01, "cos_c={cos_c}");

        // Shallower than critical (smaller cos) => full reflection, no refract.
        let super_critical = cos_c * 0.5;
        let r = fresnel_dielectric(super_critical, WATER_IOR, AIR_IOR);
        assert_eq!(r, 1.0);

        // Build an incident direction at that super-critical angle and confirm
        // refract() reports TIR.
        let sin = (1.0 - super_critical * super_critical).max(0.0).sqrt();
        let incident = Vec3::new(sin, -super_critical, 0.0);
        let eta = WATER_IOR / AIR_IOR;
        assert!(refract(incident, Vec3::Y, eta).is_none());
    }

    #[test]
    fn below_critical_angle_still_refracts() {
        let cos_c = critical_angle_cos(WATER_IOR, AIR_IOR).unwrap();
        let steep = (cos_c + 1.0) * 0.5; // between critical and normal
        let r = fresnel_dielectric(steep, WATER_IOR, AIR_IOR);
        assert!(r < 1.0, "expected transmission, r={r}");
        let sin = (1.0 - steep * steep).max(0.0).sqrt();
        let incident = Vec3::new(sin, -steep, 0.0);
        assert!(refract(incident, Vec3::Y, WATER_IOR / AIR_IOR).is_some());
    }

    #[test]
    fn air_to_water_never_tir() {
        // Going into the denser medium can never TIR.
        assert!(critical_angle_cos(AIR_IOR, WATER_IOR).is_none());
        for ci in 0..=10 {
            let cos = ci as f32 / 10.0;
            let sin = (1.0 - cos * cos).max(0.0).sqrt();
            let incident = Vec3::new(sin, -cos.max(1.0e-3), 0.0);
            assert!(
                refract(incident, Vec3::Y, AIR_IOR / WATER_IOR).is_some(),
                "cos={cos}"
            );
        }
    }

    #[test]
    fn reflect_and_transmit_sum_to_one() {
        for ci in 0..=10 {
            let cos = ci as f32 / 10.0;
            let (r, t) = reflect_refract_split(cos, AIR_IOR, WATER_IOR);
            assert!((r + t - 1.0).abs() < 1.0e-6, "cos={cos} r={r} t={t}");
            assert!((0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&t));
        }
        // In the TIR regime the split is all-reflection.
        let (r, t) = reflect_refract_split(0.1, WATER_IOR, AIR_IOR);
        assert_eq!(r, 1.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn schlick_matches_exact_at_moderate_angles() {
        // Air -> water, cos in [0.6, 0.9]: Schlick and exact agree closely.
        let f0 = f0_dielectric(AIR_IOR, WATER_IOR);
        for ci in 6..=9 {
            let cos = ci as f32 / 10.0;
            let s = fresnel_schlick(cos, f0);
            let e = fresnel_dielectric(cos, AIR_IOR, WATER_IOR);
            assert!((s - e).abs() < 0.015, "cos={cos} schlick={s} exact={e}");
        }
    }

    #[test]
    fn reflect_normal_incidence_bounces_back() {
        let r = reflect(Vec3::NEG_Y, Vec3::Y);
        assert!((r - Vec3::Y).length() < 1.0e-6, "r={r:?}");
    }

    #[test]
    fn reflect_preserves_tangential_flips_normal() {
        // 45 deg incidence: tangential component kept, normal component flipped.
        let incident = normalize_or(Vec3::new(1.0, -1.0, 0.0), Vec3::NEG_Y);
        let r = reflect(incident, Vec3::Y);
        let expected = normalize_or(Vec3::new(1.0, 1.0, 0.0), Vec3::Y);
        assert!((r - expected).length() < 1.0e-6, "r={r:?}");
        assert!((r.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn refract_normal_incidence_passes_straight() {
        let t = refract(Vec3::NEG_Y, Vec3::Y, AIR_IOR / WATER_IOR).unwrap();
        assert!((t - Vec3::NEG_Y).length() < 1.0e-6, "t={t:?}");
    }

    #[test]
    fn refract_bends_toward_normal_into_denser_medium() {
        // Entering water, the ray bends toward the normal: its angle from the
        // surface normal shrinks (|cos| grows).
        let incident = normalize_or(Vec3::new(1.0, -1.0, 0.0), Vec3::NEG_Y);
        let cos_in = (-Vec3::Y.dot(incident)).abs();
        let t = refract(incident, Vec3::Y, AIR_IOR / WATER_IOR).unwrap();
        let cos_out = (-Vec3::Y.dot(t)).abs();
        assert!(cos_out > cos_in, "cos_in={cos_in} cos_out={cos_out}");
        assert!((t.length() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn wet_zero_is_identity() {
        let albedo = Vec3::new(0.6, 0.4, 0.2);
        let wm = wet_modulate(0.5, albedo, 0.0);
        assert!((wm.roughness - 0.5).abs() < 1.0e-6);
        assert!((wm.albedo - albedo).length() < 1.0e-6);
    }

    #[test]
    fn wet_full_darkens_and_smooths() {
        let albedo = Vec3::new(0.6, 0.4, 0.2);
        let wm = wet_modulate(0.5, albedo, 1.0);
        // Roughness scaled to the floor fraction.
        assert!((wm.roughness - 0.5 * WET_ROUGHNESS_FLOOR).abs() < 1.0e-6);
        // Every channel darkened (albedo^2 for wetness=1).
        assert!(wm.albedo.x < albedo.x);
        assert!(wm.albedo.y < albedo.y);
        assert!(wm.albedo.z < albedo.z);
        assert!((wm.albedo.x - albedo.x * albedo.x).abs() < 1.0e-5);
    }

    #[test]
    fn wet_is_monotonic() {
        let albedo = Vec3::new(0.7, 0.5, 0.3);
        let mut prev_rough = f32::INFINITY;
        let mut prev_lum = f32::INFINITY;
        for wi in 0..=10 {
            let w = wi as f32 / 10.0;
            let wm = wet_modulate(0.6, albedo, w);
            let lum = wm.albedo.x + wm.albedo.y + wm.albedo.z;
            assert!(wm.roughness <= prev_rough + 1.0e-6, "rough up at w={w}");
            assert!(lum <= prev_lum + 1.0e-6, "albedo brightened at w={w}");
            prev_rough = wm.roughness;
            prev_lum = lum;
        }
    }

    #[test]
    fn wet_clamps_out_of_range_inputs() {
        let albedo = Vec3::new(1.5, -0.2, 0.5); // out of [0,1]
        let wm = wet_modulate(2.0, albedo, 3.0); // wetness/roughness over-range
        assert!((0.0..=1.0).contains(&wm.roughness));
        for c in [wm.albedo.x, wm.albedo.y, wm.albedo.z] {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c), "c={c}");
        }
        // Negative wetness is treated as fully dry (clamped to 0).
        let dry = wet_modulate(0.5, Vec3::splat(0.5), -1.0);
        assert!((dry.roughness - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn fresnel_is_monotonic_toward_grazing() {
        // Exact dielectric reflectance rises as the angle grazes (cos -> 0).
        let mut prev = -1.0;
        for ci in (0..=10).rev() {
            let cos = ci as f32 / 10.0;
            let r = fresnel_dielectric(cos, AIR_IOR, WATER_IOR);
            assert!(r.is_finite() && (0.0..=1.0).contains(&r));
            assert!(r >= prev - 1.0e-6, "non-monotonic: {prev} -> {r}");
            prev = r;
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            fresnel_dielectric(0.37, 1.0, 1.33),
            fresnel_dielectric(0.37, 1.0, 1.33)
        );
        let a = wet_modulate(0.4, Vec3::new(0.3, 0.6, 0.9), 0.55);
        let b = wet_modulate(0.4, Vec3::new(0.3, 0.6, 0.9), 0.55);
        assert_eq!(a, b);
        assert_eq!(
            refract(Vec3::new(0.3, -0.9, 0.1), Vec3::Y, 0.75),
            refract(Vec3::new(0.3, -0.9, 0.1), Vec3::Y, 0.75)
        );
    }

    #[test]
    fn degenerate_directions_do_not_nan() {
        assert!(reflect(Vec3::ZERO, Vec3::ZERO).is_finite());
        let t = refract(Vec3::ZERO, Vec3::ZERO, 0.75);
        if let Some(v) = t {
            assert!(v.is_finite());
        }
    }
}
