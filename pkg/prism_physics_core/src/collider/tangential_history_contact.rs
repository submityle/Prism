//! Cundall–Strack tangential-history contact law (static friction).
//!
//! The soft-sphere law in
//! [`bonded_particle_contact`](super::bonded_particle_contact) and the Hertz law
//! in [`hertz_contact`](super::hertz_contact) both model friction with a purely
//! *viscous* tangential term `γ_t·‖v_t‖`, capped at the Coulomb limit `μ·F_n`.
//! That is enough to dissipate sliding energy, but it has no memory: when the
//! tangential velocity drops to zero the friction force vanishes, so a grain
//! can never *rest* on a slope — it always creeps. Real granular media sustain
//! a static friction force at zero velocity, which is what lets a sand pile
//! stand at an angle of repose.
//!
//! This module supplies the discrete-element answer to that: the
//! **Cundall–Strack** tangential-history spring. Each contact carries a
//! persistent tangential displacement `ξ` that accumulates the relative
//! tangential motion for as long as the contact lives. The friction force is an
//! elastic spring `−k_t·ξ` (plus a small viscous term), clamped to the Coulomb
//! cone `‖F_t‖ ≤ μ·F_n`. While the spring is under the cap the contact
//! *sticks*; once the cap is reached the contact *slips* and the spring is
//! rescaled so the stored force sits exactly on the cone.
//!
//! # Normal response
//!
//! With the unit contact axis `n̂` from `a` to `b`, overlap `δ > 0`, and
//! relative velocity `v = v_b − v_a`, the normal rate is `v_n = v·n̂` (positive
//! while separating) and
//!
//! ```text
//!   F_n = max(0, kₙ·δ − γₙ·v_n).
//! ```
//!
//! # Tangential response
//!
//! The tangential velocity is `v_t = v − v_n·n̂`. The stored spring is first
//! re-projected onto the current tangent plane (`ξ ← ξ − (ξ·n̂)·n̂`) so it
//! follows the rotating contact frame, then advanced by `ξ += v_t·dt`. The
//! trial force on `b` is
//!
//! ```text
//!   F_t = −k_t·ξ − γ_t·v_t,
//! ```
//!
//! opposing both the accumulated displacement and the instantaneous slip. If
//! `‖F_t‖` exceeds `μ·F_n` the contact slips: the force is rescaled onto the
//! Coulomb cone and the spring is reset to `ξ = −F_t/k_t` so it stores exactly
//! the capped elastic force. The force on `b` is `F_n·n̂ + F_t`; `a` receives
//! its negation, so a resolved contact conserves linear momentum exactly.

use crate::collider::bonded_particle_contact::ContactForce;
use glam::Vec3;

/// Material parameters of the Cundall–Strack tangential-history contact law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CundallStrackModel {
    normal_stiffness: f32,
    normal_damping: f32,
    tangential_stiffness: f32,
    tangential_damping: f32,
    friction: f32,
}

impl CundallStrackModel {
    /// Builds a model from the normal penalty stiffness `kₙ`, normal damping
    /// `γₙ`, tangential spring stiffness `k_t`, tangential damping `γ_t`, and
    /// Coulomb friction coefficient `μ`.
    ///
    /// Returns `None` unless every value is finite, `kₙ` and `k_t` are strictly
    /// positive, and `γₙ`, `γ_t`, `μ` are all non-negative.
    #[must_use]
    pub fn new(
        normal_stiffness: f32,
        normal_damping: f32,
        tangential_stiffness: f32,
        tangential_damping: f32,
        friction: f32,
    ) -> Option<Self> {
        let all_finite = normal_stiffness.is_finite()
            && normal_damping.is_finite()
            && tangential_stiffness.is_finite()
            && tangential_damping.is_finite()
            && friction.is_finite();
        if !all_finite {
            return None;
        }
        if normal_stiffness <= 0.0
            || tangential_stiffness <= 0.0
            || normal_damping < 0.0
            || tangential_damping < 0.0
            || friction < 0.0
        {
            return None;
        }
        Some(Self {
            normal_stiffness,
            normal_damping,
            tangential_stiffness,
            tangential_damping,
            friction,
        })
    }

    /// Normal penalty stiffness `kₙ`.
    #[must_use]
    pub fn normal_stiffness(&self) -> f32 {
        self.normal_stiffness
    }

    /// Normal viscous damping `γₙ`.
    #[must_use]
    pub fn normal_damping(&self) -> f32 {
        self.normal_damping
    }

    /// Tangential spring stiffness `k_t`.
    #[must_use]
    pub fn tangential_stiffness(&self) -> f32 {
        self.tangential_stiffness
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

/// Advances the Cundall–Strack contact force for a contact whose unit axis is
/// `normal` (pointing from `a` to `b`), overlap is `overlap > 0`, and whose
/// relative velocity is `rel_vel = v_b − v_a`.
///
/// `tangential` is the persistent per-contact spring displacement; it is
/// re-projected onto the tangent plane, advanced by `v_t·dt`, and — on slip —
/// rescaled onto the Coulomb cone, so the caller must store it between steps for
/// the same contact pair and reset it to zero when the contact is lost.
///
/// Returns the [`ContactForce`] on `b`. The caller is responsible for `δ > 0`,
/// a unit `normal`, and a strictly positive finite `dt`.
#[must_use]
pub fn evaluate_tangential_history(
    model: &CundallStrackModel,
    normal: Vec3,
    overlap: f32,
    rel_vel: Vec3,
    tangential: &mut Vec3,
    dt: f32,
) -> ContactForce {
    // Normal response (identical to the soft-sphere law).
    let v_n = rel_vel.dot(normal);
    let normal_force = (model.normal_stiffness * overlap - model.normal_damping * v_n).max(0.0);

    // Tangential relative velocity in the contact plane.
    let v_t = rel_vel - v_n * normal;

    // Re-project the stored spring onto the current tangent plane, then advance
    // it by the tangential slip over this step.
    let mut spring = *tangential - tangential.dot(normal) * normal;
    spring += v_t * dt;

    // Trial friction force opposing both the stored displacement and the slip.
    let mut tangential_force =
        -model.tangential_stiffness * spring - model.tangential_damping * v_t;
    let mut tangential_magnitude = tangential_force.length();

    // Coulomb cone: cap the force and rescale the spring onto the cone on slip.
    let max_friction = model.friction * normal_force;
    let mut sliding = false;
    if tangential_magnitude > max_friction {
        if tangential_magnitude > 0.0 {
            let direction = tangential_force / tangential_magnitude;
            tangential_force = direction * max_friction;
            // Reset the elastic spring so it stores exactly the capped force.
            spring = -tangential_force / model.tangential_stiffness;
        } else {
            tangential_force = Vec3::ZERO;
            spring = Vec3::ZERO;
        }
        tangential_magnitude = max_friction;
        sliding = max_friction > 0.0;
    }

    *tangential = spring;

    let force_on_b = normal_force * normal + tangential_force;
    ContactForce {
        force_on_b,
        overlap,
        normal_magnitude: normal_force,
        tangential_magnitude,
        sliding,
    }
}

/// Convenience wrapper that computes the contact geometry for two spheres and
/// advances the Cundall–Strack force, returning `None` when the spheres do not
/// overlap or their centres coincide.
///
/// `positions`, `radii`, and `velocities` are the `(a, b)` pairs of sphere
/// centres, radii, and velocities. `tangential` is the persistent per-contact
/// spring displacement (see [`evaluate_tangential_history`]); it is left
/// untouched when `None` is returned.
#[must_use]
pub fn tangential_history_between(
    model: &CundallStrackModel,
    positions: (Vec3, Vec3),
    radii: (f32, f32),
    velocities: (Vec3, Vec3),
    tangential: &mut Vec3,
    dt: f32,
) -> Option<ContactForce> {
    let (pos_a, pos_b) = positions;
    let (r_a, r_b) = radii;
    let (vel_a, vel_b) = velocities;
    let delta = pos_b - pos_a;
    let distance = delta.length();
    if distance <= 0.0 {
        return None;
    }
    let overlap = (r_a + r_b) - distance;
    if overlap <= 0.0 {
        return None;
    }
    let normal = delta / distance;
    let rel_vel = vel_b - vel_a;
    Some(evaluate_tangential_history(
        model, normal, overlap, rel_vel, tangential, dt,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ.
        CundallStrackModel::new(1.0e5, 0.0, 1.0e5, 0.0, 0.5).unwrap()
    }

    #[test]
    fn new_rejects_bad_parameters() {
        assert!(CundallStrackModel::new(0.0, 0.0, 1.0, 0.0, 0.5).is_none());
        assert!(CundallStrackModel::new(1.0, 0.0, 0.0, 0.0, 0.5).is_none());
        assert!(CundallStrackModel::new(1.0, -1.0, 1.0, 0.0, 0.5).is_none());
        assert!(CundallStrackModel::new(1.0, 0.0, 1.0, -1.0, 0.5).is_none());
        assert!(CundallStrackModel::new(1.0, 0.0, 1.0, 0.0, -0.5).is_none());
        assert!(CundallStrackModel::new(f32::NAN, 0.0, 1.0, 0.0, 0.5).is_none());
        assert!(CundallStrackModel::new(1.0, 0.0, 1.0, 0.0, 0.5).is_some());
    }

    #[test]
    fn non_overlapping_spheres_have_no_contact() {
        let mut spring = Vec3::ZERO;
        let out = tangential_history_between(
            &model(),
            (Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            &mut spring,
            1.0e-3,
        );
        assert!(out.is_none());
        assert_eq!(spring, Vec3::ZERO, "spring untouched when no contact");
    }

    #[test]
    fn coincident_centres_have_no_contact() {
        let mut spring = Vec3::ZERO;
        let out = tangential_history_between(
            &model(),
            (Vec3::ZERO, Vec3::ZERO),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            &mut spring,
            1.0e-3,
        );
        assert!(out.is_none());
    }

    #[test]
    fn pure_normal_overlap_has_no_tangential_force() {
        let mut spring = Vec3::ZERO;
        // Overlap 0.5, no relative velocity → repulsive normal, zero friction.
        let out = tangential_history_between(
            &model(),
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            &mut spring,
            1.0e-3,
        )
        .expect("overlap");
        assert!(out.normal_magnitude > 0.0);
        assert!((out.normal_magnitude - 1.0e5 * 0.5).abs() < 1.0);
        assert_eq!(out.tangential_magnitude, 0.0);
        assert!(!out.sliding);
        assert_eq!(spring, Vec3::ZERO);
    }

    #[test]
    fn static_friction_grows_with_accumulated_displacement() {
        // b slides tangentially (+y) relative to a at constant velocity under
        // the Coulomb cap; the spring force must grow each step as k_t·ξ and
        // the contact must stick (not slide).
        let m = model();
        let mut spring = Vec3::ZERO;
        let normal = Vec3::X;
        let overlap = 0.5; // F_n = 5e4 → cap = μ·F_n = 2.5e4.
        let v_t = Vec3::new(0.0, 0.01, 0.0); // slow slip, stays under the cap.
        let dt = 1.0e-4;

        let mut last = 0.0_f32;
        for step in 1..=5 {
            let out = evaluate_tangential_history(&m, normal, overlap, v_t, &mut spring, dt);
            assert!(!out.sliding, "slow slip must stick, step {step}");
            // Spring stays in the tangent plane (perpendicular to n̂).
            assert!(spring.dot(normal).abs() < 1e-6);
            // Expected elastic magnitude k_t·|ξ| = k_t·v·(step·dt).
            let expected = 1.0e5 * 0.01 * (step as f32 * dt);
            assert!((out.tangential_magnitude - expected).abs() < expected * 1e-3 + 1e-3);
            assert!(
                out.tangential_magnitude > last,
                "friction grows while sticking"
            );
            last = out.tangential_magnitude;
            // Friction on b opposes its +y slip.
            assert!(out.force_on_b.y < 0.0);
        }
    }

    #[test]
    fn fast_slip_saturates_at_the_coulomb_cap() {
        let m = model();
        let mut spring = Vec3::ZERO;
        let normal = Vec3::X;
        let overlap = 0.5; // F_n = 5e4, cap = 2.5e4.
                           // Spring force after one step is k_t·v·dt; it must exceed the cap of
                           // 2.5e4, i.e. v·dt > 0.25, so pick v = 1000 (ξ = 1.0 → 1e5 ≫ cap).
        let v_t = Vec3::new(0.0, 1000.0, 0.0); // huge slip → immediate cap.
        let out = evaluate_tangential_history(&m, normal, overlap, v_t, &mut spring, 1.0e-3);
        assert!(out.sliding);
        let cap = 0.5 * 5.0e4;
        assert!((out.tangential_magnitude - cap).abs() < cap * 1e-4);
        // On slip the spring is parked on the cone: k_t·|ξ| == cap.
        assert!((1.0e5 * spring.length() - cap).abs() < cap * 1e-3);
    }

    #[test]
    fn spring_is_reprojected_when_the_normal_rotates() {
        // Load the spring along +y with the normal along +x, then evaluate with
        // the normal rotated to +y: the stored displacement that is now along
        // the normal must be projected out.
        let m = model();
        let mut spring = Vec3::new(0.0, 1.0e-3, 0.0);
        // Normal now points along +y, so the +y spring component is normal and
        // must be removed; with zero slip the remaining tangential spring is 0.
        let out = evaluate_tangential_history(&m, Vec3::Y, 0.5, Vec3::ZERO, &mut spring, 1.0e-3);
        assert!(spring.dot(Vec3::Y).abs() < 1e-6, "normal component removed");
        assert_eq!(out.tangential_magnitude, 0.0);
    }

    #[test]
    fn loading_then_reversing_unwinds_the_spring() {
        // Slip +y for a few steps, then −y: the accumulated spring must shrink
        // back toward zero rather than keep growing.
        let m = model();
        let mut spring = Vec3::ZERO;
        let normal = Vec3::X;
        let overlap = 0.5;
        let dt = 1.0e-4;
        for _ in 0..3 {
            evaluate_tangential_history(
                &m,
                normal,
                overlap,
                Vec3::new(0.0, 0.01, 0.0),
                &mut spring,
                dt,
            );
        }
        let loaded = spring.length();
        for _ in 0..2 {
            evaluate_tangential_history(
                &m,
                normal,
                overlap,
                Vec3::new(0.0, -0.01, 0.0),
                &mut spring,
                dt,
            );
        }
        assert!(spring.length() < loaded, "reversal must unwind the spring");
    }

    #[test]
    fn force_normal_component_is_repulsive_and_damped() {
        // With normal damping, an approaching pair feels extra repulsion; a
        // separating pair feels less, but never attraction.
        let damped = CundallStrackModel::new(1.0e5, 50.0, 1.0e5, 0.0, 0.5).unwrap();
        let mut spring = Vec3::ZERO;
        // Approaching: v_b − v_a has negative component along n̂ (closing).
        let approach = evaluate_tangential_history(
            &damped,
            Vec3::X,
            0.5,
            Vec3::new(-1.0, 0.0, 0.0),
            &mut spring,
            1.0e-3,
        );
        let mut spring2 = Vec3::ZERO;
        let separate = evaluate_tangential_history(
            &damped,
            Vec3::X,
            0.5,
            Vec3::new(1.0, 0.0, 0.0),
            &mut spring2,
            1.0e-3,
        );
        assert!(approach.normal_magnitude > separate.normal_magnitude);
        assert!(separate.normal_magnitude >= 0.0, "never attractive");
    }
}
