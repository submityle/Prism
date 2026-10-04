//! Nonlinear Hertzian normal contact for colliding spheres.
//!
//! The soft-sphere law in
//! [`bonded_particle_contact`](super::bonded_particle_contact) uses a *linear*
//! penalty `F_n = kₙ·δ`, which is cheap and robust but does not reproduce the
//! stiffening of a real elastic contact as it is pressed harder. This module
//! supplies the higher-fidelity alternative used throughout contact mechanics
//! and high-end granular solvers: the **Hertz** normal response, in which the
//! contact stiffness grows with penetration so the force scales as `δ^{3/2}`.
//!
//! # Normal response
//!
//! For two elastic spheres of radii `R_a`, `R_b` the effective contact radius
//! is the reduced radius
//!
//! ```text
//!   R* = (R_a · R_b) / (R_a + R_b)
//! ```
//!
//! and, for two bodies of the same material with Young's modulus `E` and
//! Poisson ratio `ν`, the effective modulus is
//!
//! ```text
//!   E* = E / (2 · (1 − ν²)).
//! ```
//!
//! With overlap `δ > 0` the Hertz elastic normal force is
//!
//! ```text
//!   F_elastic = (4/3) · E* · √R* · δ^{3/2}.
//! ```
//!
//! A linear viscous term `−γₙ·v_n` damps the approach (`v_n = v·n̂`, positive
//! while separating), and the total normal force is clamped at zero so the
//! contact never pulls the spheres together as they rebound:
//!
//! ```text
//!   F_n = max(0, F_elastic − γₙ·v_n).
//! ```
//!
//! # Tangential response
//!
//! The tangential response matches the soft-sphere law: a viscous friction
//! `γ_t·‖v_t‖` opposing slip, capped at the Coulomb limit `μ·F_n`. The force on
//! `b` is the sum of the normal and tangential parts; `a` receives its
//! negation, so a resolved contact conserves linear momentum exactly.
//!
//! The `δ^{3/2}` elastic term is evaluated in `f64` and narrowed to `f32` so the
//! fractional power stays off the `f32` fast-math path, matching the numeric
//! discipline used elsewhere in the collider for nonlinear constitutive laws.

use crate::collider::bonded_particle_contact::ContactForce;
use glam::Vec3;

/// Material parameters of the Hertzian contact law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzModel {
    effective_modulus: f32,
    normal_damping: f32,
    tangential_damping: f32,
    friction: f32,
}

impl HertzModel {
    /// Builds a Hertz model from the material Young's modulus `young_modulus`
    /// `E`, Poisson ratio `poisson_ratio` `ν`, normal damping `γₙ`, tangential
    /// (sliding) damping `γ_t`, and Coulomb friction coefficient `μ`, assuming
    /// both contacting bodies share this material.
    ///
    /// The effective modulus is `E* = E / (2·(1 − ν²))`. Returns `None` unless
    /// every value is finite, `E` is strictly positive, `ν ∈ [0, 0.5)`, and
    /// `γₙ`, `γ_t`, `μ` are all non-negative.
    #[must_use]
    pub fn new(
        young_modulus: f32,
        poisson_ratio: f32,
        normal_damping: f32,
        tangential_damping: f32,
        friction: f32,
    ) -> Option<Self> {
        let all_finite = young_modulus.is_finite()
            && poisson_ratio.is_finite()
            && normal_damping.is_finite()
            && tangential_damping.is_finite()
            && friction.is_finite();
        if !all_finite {
            return None;
        }
        if young_modulus <= 0.0
            || !(0.0..0.5).contains(&poisson_ratio)
            || normal_damping < 0.0
            || tangential_damping < 0.0
            || friction < 0.0
        {
            return None;
        }
        let effective_modulus = young_modulus / (2.0 * (1.0 - poisson_ratio * poisson_ratio));
        if !(effective_modulus.is_finite() && effective_modulus > 0.0) {
            return None;
        }
        Some(Self {
            effective_modulus,
            normal_damping,
            tangential_damping,
            friction,
        })
    }

    /// Effective contact modulus `E* = E / (2·(1 − ν²))`.
    #[must_use]
    pub fn effective_modulus(&self) -> f32 {
        self.effective_modulus
    }

    /// Normal viscous damping `γₙ`.
    #[must_use]
    pub fn normal_damping(&self) -> f32 {
        self.normal_damping
    }

    /// Tangential viscous damping `γ_t`.
    #[must_use]
    pub fn tangential_damping(&self) -> f32 {
        self.tangential_damping
    }

    /// Coulomb friction coefficient `μ`.
    #[must_use]
    pub fn friction(&self) -> f32 {
        self.friction
    }
}

/// Hertz elastic normal force magnitude `(4/3)·E*·√R*·δ^{3/2}`, evaluated in
/// `f64` to keep the fractional power off the `f32` fast-math path.
fn hertz_elastic_force(effective_modulus: f32, effective_radius: f32, overlap: f32) -> f32 {
    let e_star = f64::from(effective_modulus);
    let r_eff = f64::from(effective_radius);
    let delta = f64::from(overlap);
    let force = (4.0 / 3.0) * e_star * r_eff.sqrt() * delta.powf(1.5);
    force as f32
}

/// Evaluates the Hertz contact force for a pair whose unit contact axis `axis`
/// points from `a` to `b`, with positive penetration `overlap`, reduced contact
/// radius `effective_radius` `R*`, and relative velocity `rel_vel = v_b − v_a`.
///
/// `axis` is normalised defensively; a degenerate (near-zero) axis, a
/// non-positive `overlap`, or a non-positive `effective_radius` yields a zero
/// force.
#[must_use]
pub fn evaluate_hertz_contact(
    model: &HertzModel,
    axis: Vec3,
    overlap: f32,
    effective_radius: f32,
    rel_vel: Vec3,
) -> ContactForce {
    let axis_len = axis.length();
    let degenerate = axis_len <= f32::EPSILON
        || !(overlap.is_finite() && overlap > 0.0)
        || !(effective_radius.is_finite() && effective_radius > 0.0);
    if degenerate {
        return ContactForce {
            force_on_b: Vec3::ZERO,
            overlap: overlap.max(0.0),
            normal_magnitude: 0.0,
            tangential_magnitude: 0.0,
            sliding: false,
        };
    }
    let n = axis / axis_len;

    // Normal: Hertz elastic penalty minus viscous approach damping, never
    // attractive.
    let v_n = rel_vel.dot(n);
    let elastic = hertz_elastic_force(model.effective_modulus, effective_radius, overlap);
    let normal_magnitude = (elastic - model.normal_damping * v_n).max(0.0);
    let normal_force = normal_magnitude * n;

    // Tangential: viscous friction opposing slip, capped at the Coulomb limit.
    let v_t = rel_vel - v_n * n;
    let speed_t = v_t.length();
    let (tangential_force, tangential_magnitude, sliding) = if speed_t > f32::EPSILON {
        let viscous = model.tangential_damping * speed_t;
        let coulomb = model.friction * normal_magnitude;
        let sliding = viscous >= coulomb;
        let magnitude = viscous.min(coulomb);
        let t_hat = v_t / speed_t;
        (-magnitude * t_hat, magnitude, sliding)
    } else {
        (Vec3::ZERO, 0.0, false)
    };

    ContactForce {
        force_on_b: normal_force + tangential_force,
        overlap,
        normal_magnitude,
        tangential_magnitude,
        sliding,
    }
}

/// Evaluates the Hertz contact between two spheres given their centres, radii,
/// and velocities, returning `None` when the spheres do not overlap (or their
/// centres coincide, leaving the contact axis undefined).
#[must_use]
pub fn hertz_contact_between(
    model: &HertzModel,
    pos_a: Vec3,
    pos_b: Vec3,
    radius_a: f32,
    radius_b: f32,
    vel_a: Vec3,
    vel_b: Vec3,
) -> Option<ContactForce> {
    let delta = pos_b - pos_a;
    let dist = delta.length();
    if dist <= f32::EPSILON {
        return None;
    }
    let overlap = (radius_a + radius_b) - dist;
    if overlap <= 0.0 {
        return None;
    }
    let effective_radius = (radius_a * radius_b) / (radius_a + radius_b);
    let axis = delta / dist;
    let rel_vel = vel_b - vel_a;
    Some(evaluate_hertz_contact(
        model,
        axis,
        overlap,
        effective_radius,
        rel_vel,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    fn model() -> HertzModel {
        HertzModel::new(1.0e7, 0.3, 10.0, 10.0, 0.5).unwrap()
    }

    /// Elastic-only model (no damping) for clean nonlinear-scaling checks.
    fn elastic_model() -> HertzModel {
        HertzModel::new(1.0e7, 0.3, 0.0, 0.0, 0.5).unwrap()
    }

    #[test]
    fn new_rejects_bad_parameters() {
        assert!(HertzModel::new(0.0, 0.3, 1.0, 1.0, 0.5).is_none());
        assert!(HertzModel::new(1.0e7, -0.1, 1.0, 1.0, 0.5).is_none());
        assert!(HertzModel::new(1.0e7, 0.5, 1.0, 1.0, 0.5).is_none());
        assert!(HertzModel::new(1.0e7, 0.3, -1.0, 1.0, 0.5).is_none());
        assert!(HertzModel::new(1.0e7, 0.3, 1.0, -1.0, 0.5).is_none());
        assert!(HertzModel::new(1.0e7, 0.3, 1.0, 1.0, -0.1).is_none());
        assert!(HertzModel::new(f32::NAN, 0.3, 1.0, 1.0, 0.5).is_none());
    }

    #[test]
    fn effective_modulus_matches_formula() {
        let m = model();
        let expected = 1.0e7 / (2.0 * (1.0 - 0.3 * 0.3));
        assert!((m.effective_modulus() - expected).abs() < 1.0);
    }

    #[test]
    fn non_overlapping_spheres_have_no_contact() {
        assert!(hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 3.0,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .is_none());
    }

    #[test]
    fn coincident_centres_have_no_contact() {
        assert!(hertz_contact_between(
            &model(),
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .is_none());
    }

    #[test]
    fn overlap_produces_repulsive_normal_force() {
        // Centres 1.99 apart, radii 1 + 1 = 2 → overlap 0.01.
        let c = hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.99,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        assert!(c.normal_magnitude > 0.0);
        assert!(c.force_on_b.x > 0.0, "b is pushed in +x, away from a");
    }

    #[test]
    fn normal_force_scales_as_delta_to_the_three_halves() {
        // Doubling the overlap must scale the elastic force by 2^{3/2}.
        let near = hertz_contact_between(
            &elastic_model(),
            Vec3::ZERO,
            X * 1.99, // δ = 0.01
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        let far = hertz_contact_between(
            &elastic_model(),
            Vec3::ZERO,
            X * 1.98, // δ = 0.02
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        let ratio = far.normal_magnitude / near.normal_magnitude;
        let expected = 2.0_f32.sqrt() * 2.0; // 2^{3/2}
        assert!(
            (ratio - expected).abs() < 0.05,
            "ratio {ratio} should be ~2^1.5 = {expected}"
        );
    }

    #[test]
    fn contact_force_is_equal_and_opposite() {
        let c = hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.99,
            1.0,
            1.0,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.5, 0.0),
        )
        .expect("overlap");
        // `force_on_b` is returned; `a` receives its negation by construction,
        // so the pair conserves momentum. Here we assert the normal part is
        // repulsive along +x despite the relative velocity.
        assert!(c.force_on_b.x > 0.0);
    }

    #[test]
    fn approach_raises_normal_force_above_the_static_value() {
        let v = Vec3::new(-1.0e2, 0.0, 0.0); // b approaching a along −x
        let moving = hertz_contact_between(&model(), Vec3::ZERO, X * 1.99, 1.0, 1.0, Vec3::ZERO, v)
            .expect("overlap");
        let still = hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.99,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        assert!(moving.normal_magnitude > still.normal_magnitude);
    }

    #[test]
    fn rebound_force_never_sticks() {
        // b separating fast along +x: damping would make the force attractive,
        // but it is clamped to zero.
        let c = hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.99,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::new(1.0e6, 0.0, 0.0),
        )
        .expect("overlap");
        assert_eq!(c.normal_magnitude, 0.0);
        assert_eq!(c.force_on_b, Vec3::ZERO);
    }

    #[test]
    fn tangential_slip_is_capped_by_coulomb_friction() {
        // Heavy slip in +y saturates the friction at μ·F_n and opposes the slip.
        let c = hertz_contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.99,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0e6, 0.0),
        )
        .expect("overlap");
        assert!(c.sliding);
        let coulomb = model().friction() * c.normal_magnitude;
        assert!((c.tangential_magnitude - coulomb).abs() < 1.0);
        assert!(c.force_on_b.y < 0.0, "friction opposes +y slip");
    }
}
