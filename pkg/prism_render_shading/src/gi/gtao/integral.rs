//! Cosine-weighted visibility integral for ground-truth ambient occlusion.
//!
//! Once [`super::horizon`] has found the two horizon angles of a slice, GTAO
//! (Jimenez et al. 2016) evaluates a closed-form *arc integral* that measures
//! how much of the cosine-weighted hemisphere, projected into the slice plane,
//! lies between the horizons.  This module is the backend-neutral CPU reference
//! for that integral and for the multi-slice accumulation that turns a set of
//! slices into a scalar ambient-occlusion value plus a bent normal.
//!
//! The slice is parameterised in the plane spanned by the view direction `V`
//! and an in-plane tangent `T` (perpendicular to `V`).  The surface normal is
//! projected into that plane; `gamma` is the signed angle of the projected
//! normal from `V` and `proj_len` is its length (the slice's importance — how
//! strongly the cosine lobe leans into this plane).  The arc integral is
//!
//! ```text
//! arc(h, gamma) = -cos(2*h - gamma) + cos(gamma) + 2*h*sin(gamma)
//!               = integral_0^h 4*sin(t)*cos(t - gamma) dt
//! ```
//!
//! and the slice's unnormalised visibility is
//! `0.25 * proj_len * (arc(h1, gamma) + arc(h2, gamma))` with the clamped,
//! signed horizons `h1 <= 0 <= h2`.
//!
//! # Normalisation
//! The raw arc integral still carries the cosine lobe, so for an *open* slice
//! it evaluates to `proj_len * (N . V)` rather than `1`.  To produce an
//! ambient-occlusion value that is `1` under no occlusion for any normal, each
//! slice's visibility is divided by the same integral evaluated at the fully
//! open horizons (magnitude `PI/2` on both sides), giving a per-slice *visible
//! fraction* in `[0, 1]`.  The slices are then combined as a `proj_len`-weighted
//! average, which recovers `AO = 1` when every fraction is `1` and `AO = 0`
//! when every fraction is `0`.
//!
//! # Conventions
//! * `V`, `T`, and the normal are view-space `Vec3`s; `V` and the normal are
//!   renormalised and `T` is re-orthogonalised against `V` defensively.
//! * Horizon inputs are magnitudes in `[0, PI/2]` as produced by
//!   [`super::horizon::search_horizons`]; this module applies the signs.
//! * `gamma` is clamped to `[-PI/2, PI/2]`.
//! * AO is a *visibility* term in `[0, 1]` (`1` fully unoccluded).  The bent
//!   normal is unit length; degenerate inputs fall back to the geometric normal
//!   (then the view direction, then `+Y`).
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent
//!   method.  No RNG, no I/O, no GPU, no allocation; never emits `NaN`.

use bevy_math::{ops, Vec3};
use core::f32::consts::FRAC_PI_2;

/// Smallest squared length treated as a usable direction / projection.
const MIN_LEN_SQ: f32 = 1.0e-12;

/// One azimuthal GTAO slice: its in-plane tangent and the two horizon
/// magnitudes found along it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    /// In-plane tangent direction, perpendicular to the view vector.
    ///
    /// Re-orthogonalised against the view direction before use; the
    /// positive-tangent side carries horizon [`h2`](Self::h2).
    pub tangent: Vec3,
    /// Horizon magnitude on the negative-tangent side, in `[0, PI/2]`.
    pub h1: f32,
    /// Horizon magnitude on the positive-tangent side, in `[0, PI/2]`.
    pub h2: f32,
}

/// Result of accumulating the visibility integral over a set of slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityIntegral {
    /// Scalar ambient-occlusion / visibility term in `[0, 1]` (`1` unoccluded).
    pub ao: f32,
    /// Unit-length bent normal: the mean unoccluded direction across slices.
    pub bent_normal: Vec3,
}

/// The GTAO slice arc integral `integral_0^h 4*sin(t)*cos(t - gamma) dt`.
///
/// In closed form `-cos(2*h - gamma) + cos(gamma) + 2*h*sin(gamma)`.  `h` is a
/// signed horizon angle (radians) measured from the view direction and `gamma`
/// is the signed projected-normal angle.  `arc(0, gamma) == 0` for every
/// `gamma`.
#[inline]
pub fn arc_integral(h: f32, gamma: f32) -> f32 {
    let value = -ops::cos(2.0 * h - gamma) + ops::cos(gamma) + 2.0 * h * ops::sin(gamma);
    if value.is_finite() { value } else { 0.0 }
}

/// Clamps the signed horizons into the hemisphere around the projected normal.
///
/// Takes horizon *magnitudes* `h1, h2 >= 0`, applies the negative sign to the
/// first side, and clamps each so it stays within `[gamma - PI/2, gamma + PI/2]`
/// — the hemisphere centred on the projected normal.  Returns the signed,
/// clamped pair `(h1c, h2c)` with `h1c <= h2c`.
#[inline]
fn clamp_horizons(gamma: f32, h1: f32, h2: f32) -> (f32, f32) {
    let h1 = h1.clamp(0.0, FRAC_PI_2);
    let h2 = h2.clamp(0.0, FRAC_PI_2);
    let h1c = gamma + (-h1 - gamma).max(-FRAC_PI_2);
    let h2c = gamma + (h2 - gamma).min(FRAC_PI_2);
    (h1c, h2c)
}

/// Projects the normal into a slice plane, returning `(proj_len, gamma)`.
///
/// `view` and `tangent` are assumed orthonormal.  `proj_len` is the length of
/// the normal's component in the plane and `gamma` is the signed angle of that
/// component from `view`, clamped to `[-PI/2, PI/2]`.  A projection that
/// vanishes (normal perpendicular to the plane) yields `(0, 0)`.
#[inline]
fn project_normal(view: Vec3, tangent: Vec3, normal: Vec3) -> (f32, f32) {
    let comp_v = normal.dot(view);
    let comp_t = normal.dot(tangent);
    let proj_len = ops::hypot(comp_v, comp_t);
    if !proj_len.is_finite() || proj_len <= 0.0 {
        return (0.0, 0.0);
    }
    let gamma = ops::atan2(comp_t, comp_v).clamp(-FRAC_PI_2, FRAC_PI_2);
    (proj_len, gamma)
}

/// Re-orthogonalises `tangent` against `view`, returning `None` if the result
/// is degenerate (parallel to the view direction).
#[inline]
fn orthonormal_tangent(view: Vec3, tangent: Vec3) -> Option<Vec3> {
    let projected = tangent - view * view.dot(tangent);
    let len_sq = projected.length_squared();
    if len_sq.is_finite() && len_sq > MIN_LEN_SQ {
        Some(projected * len_sq.sqrt().recip())
    } else {
        None
    }
}

/// Per-slice evaluation: the visible fraction, slice importance, and the
/// slice's bent-normal direction.
///
/// `view`, `normal` must be unit; `tangent` must be orthonormal to `view`.
/// Returns `(fraction, proj_len, bent_dir)` where `fraction` is the clamped
/// visible fraction in `[0, 1]`, `proj_len` the projected-normal length, and
/// `bent_dir` a unit direction at the midpoint of the open arc (the geometric
/// normal when the slice is degenerate).
fn slice_fraction(view: Vec3, normal: Vec3, tangent: Vec3, h1: f32, h2: f32) -> (f32, f32, Vec3) {
    let (proj_len, gamma) = project_normal(view, tangent, normal);
    if proj_len <= 0.0 {
        return (1.0, 0.0, normal);
    }
    let (h1c, h2c) = clamp_horizons(gamma, h1, h2);
    let occluded = arc_integral(h1c, gamma) + arc_integral(h2c, gamma);
    // Fully open reference (horizons at PI/2 on both sides) for normalisation.
    let (o1, o2) = clamp_horizons(gamma, FRAC_PI_2, FRAC_PI_2);
    let open = arc_integral(o1, gamma) + arc_integral(o2, gamma);
    let fraction = if open > MIN_LEN_SQ {
        (occluded / open).clamp(0.0, 1.0)
    } else {
        // Degenerate open integral (grazing normal): treat as unoccluded.
        1.0
    };
    // Bent direction: midpoint of the open arc, expressed in the slice plane.
    let bent_angle = 0.5 * (h1c + h2c);
    let (sin_b, cos_b) = ops::sin_cos(bent_angle);
    let bent_dir = view * cos_b + tangent * sin_b;
    let bent_dir = if bent_dir.length_squared() > MIN_LEN_SQ {
        bent_dir
    } else {
        normal
    };
    (fraction, proj_len, bent_dir)
}

/// Normalises `v`, returning `fallback` for a degenerate input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_LEN_SQ {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Accumulates the visibility integral and bent normal over `slices`.
///
/// `view` and `normal` are view-space directions (renormalised defensively).
/// Each slice contributes its visible fraction weighted by its projected-normal
/// length (slice importance); the AO is that weighted average and the bent
/// normal is the fraction/importance-weighted mean of the per-slice directions.
/// With no slices, or when every slice is degenerate, the result is fully
/// unoccluded with the bent normal set to the geometric normal.
pub fn integrate(view: Vec3, normal: Vec3, slices: &[Slice]) -> VisibilityIntegral {
    let normal = normalize_or(normal, Vec3::Y);
    let view = normalize_or(view, normal);
    if slices.is_empty() {
        return VisibilityIntegral {
            ao: 1.0,
            bent_normal: normal,
        };
    }

    let mut weighted_fraction = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    let mut fraction_sum = 0.0_f32;
    let mut valid_slices = 0_u32;
    let mut bent_acc = Vec3::ZERO;

    for slice in slices {
        let Some(tangent) = orthonormal_tangent(view, slice.tangent) else {
            continue;
        };
        let (fraction, proj_len, bent_dir) =
            slice_fraction(view, normal, tangent, slice.h1, slice.h2);
        valid_slices += 1;
        fraction_sum += fraction;
        weighted_fraction += fraction * proj_len;
        weight_sum += proj_len;
        bent_acc += bent_dir * (fraction * proj_len);
    }

    if valid_slices == 0 {
        return VisibilityIntegral {
            ao: 1.0,
            bent_normal: normal,
        };
    }

    let ao = if weight_sum > MIN_LEN_SQ {
        (weighted_fraction / weight_sum).clamp(0.0, 1.0)
    } else {
        // All projections vanished: fall back to the unweighted mean fraction.
        (fraction_sum / valid_slices as f32).clamp(0.0, 1.0)
    };

    let bent_normal = normalize_or(bent_acc, normal);

    VisibilityIntegral { ao, bent_normal }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Numerical Riemann sum of the arc integrand `4*sin(t)*cos(t - gamma)`.
    fn arc_numeric(h: f32, gamma: f32) -> f32 {
        let steps = 20_000;
        let dt = h / steps as f32;
        let mut acc = 0.0_f32;
        for i in 0..steps {
            // Midpoint rule.
            let t = (i as f32 + 0.5) * dt;
            acc += 4.0 * ops::sin(t) * ops::cos(t - gamma) * dt;
        }
        acc
    }

    #[test]
    fn arc_closed_form_matches_riemann_sum() {
        let gammas = [-1.0, -0.4, 0.0, 0.3, 0.9];
        let horizons = [-1.2, -0.5, 0.0, 0.7, 1.3];
        for &g in &gammas {
            for &h in &horizons {
                let closed = arc_integral(h, g);
                let numeric = arc_numeric(h, g);
                assert!(
                    approx(closed, numeric, 2.0e-3),
                    "mismatch h={h} g={g}: closed={closed} numeric={numeric}"
                );
            }
        }
    }

    #[test]
    fn arc_is_zero_at_zero_horizon() {
        for g in [-1.0, 0.0, 0.5, 1.2] {
            assert!(approx(arc_integral(0.0, g), 0.0, 1.0e-6));
        }
    }

    fn uniform_slices(h1: f32, h2: f32, count: usize) -> alloc::vec::Vec<Slice> {
        use core::f32::consts::PI;
        (0..count)
            .map(|i| {
                let phi = PI * i as f32 / count as f32;
                let (s, c) = ops::sin_cos(phi);
                Slice {
                    tangent: Vec3::new(c, s, 0.0),
                    h1,
                    h2,
                }
            })
            .collect()
    }

    #[test]
    fn no_occlusion_gives_unit_ao_facing_normal() {
        // Normal along the view direction (+Z); open horizons -> AO = 1.
        let slices = uniform_slices(FRAC_PI_2, FRAC_PI_2, 8);
        let r = integrate(Vec3::Z, Vec3::Z, &slices);
        assert!(approx(r.ao, 1.0, 1.0e-4), "ao = {}", r.ao);
    }

    #[test]
    fn no_occlusion_gives_unit_ao_tilted_normal() {
        // A tilted normal must still read as fully unoccluded with open
        // horizons, thanks to the open-integral normalisation.
        let slices = uniform_slices(FRAC_PI_2, FRAC_PI_2, 16);
        let normal = Vec3::new(0.4, 0.2, 1.0).normalize();
        let r = integrate(Vec3::Z, normal, &slices);
        assert!(approx(r.ao, 1.0, 1.0e-3), "ao = {}", r.ao);
    }

    #[test]
    fn full_occlusion_gives_zero_ao() {
        let slices = uniform_slices(0.0, 0.0, 8);
        let r = integrate(Vec3::Z, Vec3::Z, &slices);
        assert!(approx(r.ao, 0.0, 1.0e-5), "ao = {}", r.ao);
    }

    #[test]
    fn ao_is_monotonic_in_horizon() {
        let mut prev = -1.0_f32;
        for k in 0..=10 {
            let h = FRAC_PI_2 * k as f32 / 10.0;
            let slices = uniform_slices(h, h, 12);
            let r = integrate(Vec3::Z, Vec3::Z, &slices);
            assert!(r.ao >= prev - 1.0e-5, "not monotonic at k={k}: {} < {prev}", r.ao);
            prev = r.ao;
        }
    }

    #[test]
    fn multi_slice_count_converges() {
        // The AO for a fixed horizon should stabilise as the slice count grows.
        let h = 0.6_f32;
        let coarse = integrate(Vec3::Z, Vec3::Z, &uniform_slices(h, h, 4));
        let fine = integrate(Vec3::Z, Vec3::Z, &uniform_slices(h, h, 64));
        assert!(approx(coarse.ao, fine.ao, 5.0e-2), "{} vs {}", coarse.ao, fine.ao);
    }

    #[test]
    fn partial_occlusion_is_between_zero_and_one() {
        let slices = uniform_slices(0.5, 0.5, 12);
        let r = integrate(Vec3::Z, Vec3::Z, &slices);
        assert!(r.ao > 0.0 && r.ao < 1.0, "ao = {}", r.ao);
    }

    #[test]
    fn bent_normal_is_unit_length() {
        let slices = uniform_slices(0.4, 0.9, 12);
        let r = integrate(Vec3::Z, Vec3::Z, &slices);
        assert!(approx(r.bent_normal.length(), 1.0, 1.0e-4));
    }

    #[test]
    fn bent_normal_leans_toward_open_side() {
        // One slice: the negative side is closed (h1 = 0) and the positive
        // side open (h2 = PI/2).  The bent normal should tilt toward +tangent.
        let slice = Slice {
            tangent: Vec3::X,
            h1: 0.0,
            h2: FRAC_PI_2,
        };
        let r = integrate(Vec3::Z, Vec3::Z, &[slice]);
        assert!(r.bent_normal.x > 0.0, "bent normal = {:?}", r.bent_normal);
        assert!(approx(r.bent_normal.length(), 1.0, 1.0e-4));
    }

    #[test]
    fn empty_slices_are_unoccluded() {
        let r = integrate(Vec3::Z, Vec3::Z, &[]);
        assert_eq!(r.ao, 1.0);
        assert!(approx(r.bent_normal.length(), 1.0, 1.0e-5));
    }

    #[test]
    fn degenerate_inputs_stay_finite() {
        let slices = [
            Slice {
                tangent: Vec3::ZERO, // skipped (degenerate tangent)
                h1: 0.3,
                h2: 0.3,
            },
            Slice {
                tangent: Vec3::Z, // parallel to view -> skipped
                h1: 0.3,
                h2: 0.3,
            },
            Slice {
                tangent: Vec3::X,
                h1: f32::NAN, // sanitised by clamp
                h2: 0.5,
            },
        ];
        let r = integrate(Vec3::ZERO, Vec3::ZERO, &slices);
        assert!(r.ao.is_finite() && (0.0..=1.0).contains(&r.ao));
        assert!(r.bent_normal.is_finite());
        assert!(approx(r.bent_normal.length(), 1.0, 1.0e-4));
    }
}
