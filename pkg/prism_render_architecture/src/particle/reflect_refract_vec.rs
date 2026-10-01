//! Specular reflection and `Snell`-law refraction for 3D direction vectors.
//!
//! This module is the *optical response* half of the particle-vs-surface
//! interaction math. When a particle (or a traced sample ray) meets a surface
//! it either bounces specularly, bends as it crosses an index-of-refraction
//! boundary, or — past the critical angle — undergoes total internal
//! reflection (`TIR`). This file provides the closed-form answers to all three
//! questions plus the `Schlick` approximation of the `Fresnel` reflectance that
//! blends between them, and a kinematic bounce helper for solid collisions.
//!
//! Design boundaries:
//!
//! * [`crate::particle::collision`] owns *collision response*: it turns an
//!   already-detected penetration into a corrected position and velocity. This
//!   module is pure direction math — it never touches positions or time steps,
//!   it only answers "which way does the ray/velocity go after the surface?".
//! * This module is deliberately dependency-free. Every vector operation is a
//!   hand-rolled free function over `[f32; 3]`; nothing implements an inherent
//!   arithmetic operator on the array type, so there is no `should_implement_trait`
//!   surprise and the layout matches a future `GPU` kernel bit for bit.
//!
//! Numerical contract:
//!
//! * The only floating-point primitives used are `f32::sqrt`, `f32::abs`,
//!   `f32::min`, `f32::max`, and `f32::clamp`. No transcendental function
//!   (`sin`/`cos`/`exp`/`pow`/…) appears anywhere; squares and the `Schlick`
//!   fifth power are written as explicit multiplications.
//! * No `==` / `!=` comparison on an `f32` ever appears. Normalisation guards
//!   against division by zero with an epsilon length test, and the tests assert
//!   with epsilon tolerances.
//!
//! Vector conventions follow the usual optics sign convention: `incident`
//! points *into* the surface (toward it, along the ray of travel) and `normal`
//! points *out of* the surface, back toward where the ray came from. Both the
//! `incident` and `normal` arguments to [`reflect`], [`refract`],
//! [`is_total_internal_reflection`] are assumed to be unit length; the caller is
//! responsible for normalising them (use [`v_normalize`] semantics).

/// Dot product of two 3D vectors.
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b`.
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Uniform scale of a vector by a scalar.
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean length of a vector, via `f32::sqrt`.
fn v_len(a: [f32; 3]) -> f32 {
    v_dot(a, a).sqrt()
}

/// Returns the unit vector in the direction of `a`.
///
/// If the length is below the epsilon guard the input is returned unchanged
/// (which for a zero vector is the zero vector itself), so this never divides
/// by zero.
fn v_normalize(a: [f32; 3]) -> [f32; 3] {
    let len = v_len(a);
    if len > 1e-12 {
        v_scale(a, 1.0 / len)
    } else {
        a
    }
}

/// Mirror-reflects `incident` about `normal`: `i - 2 (i·n) n`.
///
/// `normal` is assumed to be unit length. The result has the same magnitude as
/// `incident`.
pub fn reflect(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let d = v_dot(incident, normal);
    v_sub(incident, v_scale(normal, 2.0 * d))
}

/// Refracts `incident` through a surface using the vector form of `Snell`'s law.
///
/// `eta` is the ratio of indices of refraction `n1 / n2` (incoming over
/// outgoing). With `cos_i = -(i·n)` the discriminant is
/// `k = 1 - eta² (1 - cos_i²)`, where `eta²` and `cos_i²` are formed with plain
/// multiplication. When `k < 0` the geometry is past the critical angle — total
/// internal reflection (`TIR`) — and `None` is returned. Otherwise the
/// transmitted direction is `eta·i + (eta·cos_i - sqrt(k))·n`.
///
/// Both `incident` and `normal` are assumed to be unit length.
pub fn refract(incident: [f32; 3], normal: [f32; 3], eta: f32) -> Option<[f32; 3]> {
    let cos_i = -v_dot(incident, normal);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        let bent_i = v_scale(incident, eta);
        let bent_n = v_scale(normal, eta * cos_i - k.sqrt());
        Some(v_add(bent_i, bent_n))
    }
}

/// Returns `true` when the configuration undergoes total internal reflection.
///
/// This is exactly the `k < 0` test of [`refract`]: with `cos_i = -(i·n)` the
/// discriminant `k = 1 - eta² (1 - cos_i²)` going negative means no transmitted
/// ray exists. Inputs are assumed unit length.
pub fn is_total_internal_reflection(incident: [f32; 3], normal: [f32; 3], eta: f32) -> bool {
    let cos_i = -v_dot(incident, normal);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    k < 0.0
}

/// `Schlick`'s approximation of the `Fresnel` reflectance.
///
/// Computes `r0 + (1 - r0) (1 - cos)^5`, where `cos` is the cosine of the angle
/// between the view/incident direction and the normal, clamped to `[0, 1]`. The
/// fifth power is written as an explicit chain of multiplications rather than a
/// `powf`/`powi` call.
pub fn fresnel_schlick_reflectance(cos_theta: f32, r0: f32) -> f32 {
    let cos = cos_theta.clamp(0.0, 1.0);
    let m = 1.0 - cos;
    let m5 = m * m * m * m * m;
    r0 + (1.0 - r0) * m5
}

/// Normal-incidence reflectance `r0` from a pair of refractive indices.
///
/// Computes `((n1 - n2) / (n1 + n2))²` with an explicit square. For an
/// air→glass boundary (`n1 = 1.0`, `n2 = 1.5`) this yields ≈ `0.04`.
pub fn fresnel_schlick_r0(n1: f32, n2: f32) -> f32 {
    let r = (n1 - n2) / (n1 + n2);
    r * r
}

/// Reflects a particle `velocity` off a surface with a coefficient of
/// restitution.
///
/// The velocity is split into a component along `normal` and a tangential
/// component. The normal component is negated and scaled by `restitution`
/// (`1.0` = perfectly elastic, `0.0` = the particle slides with no normal
/// rebound) while the tangential component is preserved. `normal` need not be
/// unit length — it is normalised internally via the epsilon-guarded helper.
pub fn reflect_with_restitution(
    velocity: [f32; 3],
    normal: [f32; 3],
    restitution: f32,
) -> [f32; 3] {
    let n = v_normalize(normal);
    let vn = v_dot(velocity, n);
    let normal_component = v_scale(n, vn);
    let tangential = v_sub(velocity, normal_component);
    let rebound = v_scale(normal_component, -restitution);
    v_add(tangential, rebound)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vclose(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        close(a[0], b[0], eps) && close(a[1], b[1], eps) && close(a[2], b[2], eps)
    }

    // --- vector helpers --------------------------------------------------

    #[test]
    fn v_dot_basic() {
        assert!(close(v_dot([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]), 12.0, EPS));
    }

    #[test]
    fn v_dot_orthogonal() {
        assert!(close(v_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0, EPS));
    }

    #[test]
    fn v_add_basic() {
        assert!(vclose(
            v_add([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]),
            [1.5, 1.0, 5.0],
            EPS
        ));
    }

    #[test]
    fn v_sub_basic() {
        assert!(vclose(
            v_sub([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]),
            [0.5, 3.0, 1.0],
            EPS
        ));
    }

    #[test]
    fn v_scale_basic() {
        assert!(vclose(
            v_scale([1.0, -2.0, 3.0], 2.0),
            [2.0, -4.0, 6.0],
            EPS
        ));
    }

    #[test]
    fn v_len_345() {
        assert!(close(v_len([3.0, 4.0, 0.0]), 5.0, EPS));
    }

    #[test]
    fn v_len_zero() {
        assert!(close(v_len([0.0, 0.0, 0.0]), 0.0, EPS));
    }

    #[test]
    fn v_normalize_unit() {
        assert!(vclose(v_normalize([0.0, 5.0, 0.0]), [0.0, 1.0, 0.0], EPS));
    }

    #[test]
    fn v_normalize_preserves_direction_length_one() {
        let n = v_normalize([1.0, 2.0, 2.0]);
        assert!(close(v_len(n), 1.0, EPS));
    }

    #[test]
    fn v_normalize_zero_vector_no_div_by_zero() {
        // Below the epsilon guard: must return the (zero) input unchanged.
        assert!(vclose(v_normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], EPS));
    }

    // --- reflect ---------------------------------------------------------

    #[test]
    fn reflect_normal_incidence() {
        let r = reflect([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(vclose(r, [0.0, 1.0, 0.0], EPS));
    }

    #[test]
    fn reflect_forty_five_degrees() {
        // i = (1, -1, 0) off an up-facing normal → (1, 1, 0).
        let r = reflect([1.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(vclose(r, [1.0, 1.0, 0.0], EPS));
    }

    #[test]
    fn reflect_x_axis_wall() {
        let r = reflect([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(vclose(r, [1.0, 0.0, 0.0], EPS));
    }

    #[test]
    fn reflect_preserves_length() {
        let i = v_normalize([0.3, -0.7, 0.5]);
        let n = [0.0, 1.0, 0.0];
        let r = reflect(i, n);
        assert!(close(v_len(r), v_len(i), EPS));
    }

    #[test]
    fn reflect_tangential_component_unchanged() {
        let i = [0.6, -0.8, 0.0];
        let n = [0.0, 1.0, 0.0];
        let r = reflect(i, n);
        // X (tangential) component is preserved; Y (normal) is flipped.
        assert!(close(r[0], i[0], EPS));
        assert!(close(r[1], -i[1], EPS));
    }

    // --- refract ---------------------------------------------------------

    #[test]
    fn refract_straight_through_eta_one() {
        let r = refract([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 1.0).unwrap();
        assert!(vclose(r, [0.0, -1.0, 0.0], EPS));
    }

    #[test]
    fn refract_perpendicular_direction_unchanged_eta_half() {
        // Perpendicular incidence: direction is independent of eta, length one.
        let r = refract([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.5).unwrap();
        assert!(vclose(r, [0.0, -1.0, 0.0], EPS));
    }

    #[test]
    fn refract_eta_one_is_identity() {
        let i = v_normalize([1.0, -1.0, 0.0]);
        let r = refract(i, [0.0, 1.0, 0.0], 1.0).unwrap();
        assert!(vclose(r, i, EPS));
    }

    #[test]
    fn refract_returns_some_below_critical_angle() {
        let i = v_normalize([1.0, -1.0, 0.0]);
        assert!(refract(i, [0.0, 1.0, 0.0], 0.9).is_some());
    }

    #[test]
    fn refract_snell_tangential_ratio_equals_eta() {
        // The transmitted tangential component scales by exactly eta.
        let eta = 0.9;
        let i = v_normalize([1.0, -1.0, 0.0]);
        let n = [0.0, 1.0, 0.0];
        let r = refract(i, n, eta).unwrap();

        let t_in = v_sub(i, v_scale(n, v_dot(i, n)));
        let t_out = v_sub(r, v_scale(n, v_dot(r, n)));
        let ratio = v_len(t_out) / v_len(t_in);
        assert!(close(ratio, eta, 1e-4));
    }

    #[test]
    fn refract_snell_tangential_ratio_other_angle() {
        let eta = 0.75;
        let i = v_normalize([2.0, -1.0, 0.0]);
        let n = [0.0, 1.0, 0.0];
        let r = refract(i, n, eta).unwrap();

        let t_in = v_sub(i, v_scale(n, v_dot(i, n)));
        let t_out = v_sub(r, v_scale(n, v_dot(r, n)));
        let ratio = v_len(t_out) / v_len(t_in);
        assert!(close(ratio, eta, 1e-4));
    }

    #[test]
    fn refract_result_is_unit_when_inputs_unit_and_eta_one() {
        let i = v_normalize([0.3, -0.9, 0.2]);
        let r = refract(i, [0.0, 1.0, 0.0], 1.0).unwrap();
        assert!(close(v_len(r), 1.0, EPS));
    }

    // --- total internal reflection --------------------------------------

    #[test]
    fn refract_tir_returns_none() {
        // Dense → rare medium (eta = 1.5), 60° incidence → past critical angle.
        let i = v_normalize([0.866_025, -0.5, 0.0]);
        let n = [0.0, 1.0, 0.0];
        assert!(refract(i, n, 1.5).is_none());
    }

    #[test]
    fn tir_flag_true_at_grazing_angle() {
        let i = v_normalize([0.866_025, -0.5, 0.0]);
        let n = [0.0, 1.0, 0.0];
        assert!(is_total_internal_reflection(i, n, 1.5));
    }

    #[test]
    fn tir_flag_false_at_normal_incidence() {
        let i = [0.0, -1.0, 0.0];
        let n = [0.0, 1.0, 0.0];
        assert!(!is_total_internal_reflection(i, n, 1.5));
    }

    #[test]
    fn tir_flag_false_when_entering_denser_medium() {
        // eta < 1 (rare → dense) never produces TIR.
        let i = v_normalize([0.9, -0.1, 0.0]);
        let n = [0.0, 1.0, 0.0];
        assert!(!is_total_internal_reflection(i, n, 0.666_667));
    }

    // --- Fresnel (Schlick) ----------------------------------------------

    #[test]
    fn fresnel_cos_one_returns_r0() {
        assert!(close(fresnel_schlick_reflectance(1.0, 0.04), 0.04, EPS));
    }

    #[test]
    fn fresnel_cos_zero_returns_one() {
        assert!(close(fresnel_schlick_reflectance(0.0, 0.04), 1.0, EPS));
    }

    #[test]
    fn fresnel_clamps_negative_cos_to_zero() {
        assert!(close(fresnel_schlick_reflectance(-0.5, 0.04), 1.0, EPS));
    }

    #[test]
    fn fresnel_clamps_cos_above_one() {
        assert!(close(fresnel_schlick_reflectance(2.0, 0.04), 0.04, EPS));
    }

    #[test]
    fn fresnel_midrange_increases_above_r0() {
        let r = fresnel_schlick_reflectance(0.5, 0.04);
        assert!(r > 0.04);
        assert!(r < 1.0);
    }

    #[test]
    fn fresnel_r0_air_to_glass() {
        assert!(close(fresnel_schlick_r0(1.0, 1.5), 0.04, 0.001));
    }

    #[test]
    fn fresnel_r0_symmetric() {
        let a = fresnel_schlick_r0(1.0, 1.5);
        let b = fresnel_schlick_r0(1.5, 1.0);
        assert!(close(a, b, EPS));
    }

    #[test]
    fn fresnel_r0_equal_indices_is_zero() {
        assert!(close(fresnel_schlick_r0(1.5, 1.5), 0.0, EPS));
    }

    // --- restitution bounce ---------------------------------------------

    #[test]
    fn restitution_full_elastic_bounce() {
        let r = reflect_with_restitution([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 1.0);
        assert!(vclose(r, [0.0, 1.0, 0.0], EPS));
    }

    #[test]
    fn restitution_zero_no_normal_rebound() {
        let r = reflect_with_restitution([1.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert!(vclose(r, [1.0, 0.0, 0.0], EPS));
    }

    #[test]
    fn restitution_half_bounce() {
        let r = reflect_with_restitution([2.0, -3.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        assert!(vclose(r, [2.0, 1.5, 0.0], EPS));
    }

    #[test]
    fn restitution_preserves_tangential_component() {
        let v = [2.0, -3.0, 1.0];
        let n = [0.0, 1.0, 0.0];
        let r = reflect_with_restitution(v, n, 0.5);
        // Tangential (X, Z) components survive the bounce untouched.
        assert!(close(r[0], v[0], EPS));
        assert!(close(r[2], v[2], EPS));
    }

    #[test]
    fn restitution_handles_unnormalized_normal() {
        // A non-unit normal must be normalised internally.
        let r = reflect_with_restitution([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 1.0);
        assert!(vclose(r, [0.0, 2.0, 0.0], EPS));
    }
}
