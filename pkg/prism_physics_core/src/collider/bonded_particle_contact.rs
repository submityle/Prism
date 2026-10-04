//! Pairwise discrete-element contact law for colliding spherical particles.
//!
//! The parallel-bond pipeline ([`bonded_particle`](super::bonded_particle),
//! [`bonded_particle_assembly`](super::bonded_particle_assembly),
//! [`bonded_particle_integrator`](super::bonded_particle_integrator)) models the
//! *cohesive* interaction between bonded neighbours, but once a bond breaks the
//! two fragments must still be prevented from passing through one another. This
//! module supplies the complementary *repulsive* contact: a linear
//! spring–dashpot normal response with a Coulomb-limited tangential (friction)
//! response, the standard soft-sphere discrete-element contact model.
//!
//! # Normal response
//!
//! For two spheres of radii `R_a`, `R_b` whose centres are a distance `d`
//! apart, the overlap is `δ = (R_a + R_b) − d`. A contact exists only while
//! `δ > 0`. With the unit contact axis `n̂` pointing from `a` to `b` and the
//! relative velocity `v = v_b − v_a`, the normal approach rate is `v_n = v·n̂`
//! (positive while separating). The normal force magnitude is
//!
//! ```text
//!   F_n = max(0, kₙ·δ − γₙ·v_n)
//! ```
//!
//! a Hookean penalty `kₙ·δ` plus viscous dissipation `−γₙ·v_n` that damps the
//! approach; it is clamped at zero so the contact can never *pull* the spheres
//! together as they rebound (soft spheres do not stick).
//!
//! # Tangential response
//!
//! The tangential relative velocity is `v_t = v − v_n·n̂`. The friction force
//! opposes sliding with magnitude `min(γ_t·‖v_t‖, μ·F_n)` — viscous below the
//! Coulomb limit and capped at `μ·F_n` once sliding. The force the contact
//! applies to `b` is the sum of the normal and tangential parts; the equal and
//! opposite force acts on `a`, so a resolved contact conserves linear momentum
//! exactly.

use glam::Vec3;

/// Material parameters of the soft-sphere contact law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactModel {
    normal_stiffness: f32,
    normal_damping: f32,
    tangential_damping: f32,
    friction: f32,
}

impl ContactModel {
    /// Builds a contact model from the normal penalty stiffness `kₙ`, normal
    /// damping `γₙ`, tangential (sliding) damping `γ_t`, and Coulomb friction
    /// coefficient `μ`.
    ///
    /// Returns `None` unless every value is finite, `kₙ` is strictly positive,
    /// and `γₙ`, `γ_t`, `μ` are all non-negative.
    #[must_use]
    pub fn new(
        normal_stiffness: f32,
        normal_damping: f32,
        tangential_damping: f32,
        friction: f32,
    ) -> Option<Self> {
        if !(normal_stiffness.is_finite()
            && normal_damping.is_finite()
            && tangential_damping.is_finite()
            && friction.is_finite())
        {
            return None;
        }
        if normal_stiffness <= 0.0
            || normal_damping < 0.0
            || tangential_damping < 0.0
            || friction < 0.0
        {
            return None;
        }
        Some(Self {
            normal_stiffness,
            normal_damping,
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

/// Outcome of a single pairwise contact evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactForce {
    /// Force the contact applies to particle `b`; `a` receives its negation.
    pub force_on_b: Vec3,
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential (friction) force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit `μ·F_n` (sliding).
    pub sliding: bool,
}

/// Evaluates the contact force for a pair whose unit contact axis `axis` points
/// from `a` to `b`, with positive penetration `overlap` and relative velocity
/// `rel_vel = v_b − v_a`.
///
/// `axis` is normalised defensively; a degenerate (near-zero) axis or a
/// non-positive `overlap` yields a zero force. The returned `force_on_b` points
/// away from `a` in its normal part and opposes `b`'s tangential slip in its
/// friction part.
#[must_use]
pub fn evaluate_contact(
    model: &ContactModel,
    axis: Vec3,
    overlap: f32,
    rel_vel: Vec3,
) -> ContactForce {
    let axis_len = axis.length();
    if axis_len <= f32::EPSILON || !(overlap.is_finite() && overlap > 0.0) {
        return ContactForce {
            force_on_b: Vec3::ZERO,
            overlap: overlap.max(0.0),
            normal_magnitude: 0.0,
            tangential_magnitude: 0.0,
            sliding: false,
        };
    }
    let n = axis / axis_len;

    // Normal: Hookean penalty minus viscous approach damping, never attractive.
    let v_n = rel_vel.dot(n);
    let normal_magnitude = (model.normal_stiffness * overlap - model.normal_damping * v_n).max(0.0);
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

/// Evaluates the contact between two spheres given their centres, radii, and
/// velocities, returning `None` when the spheres do not overlap (or their
/// centres coincide, leaving the contact axis undefined).
#[must_use]
pub fn contact_between(
    model: &ContactModel,
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
    let axis = delta / dist;
    let rel_vel = vel_b - vel_a;
    Some(evaluate_contact(model, axis, overlap, rel_vel))
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    fn model() -> ContactModel {
        ContactModel::new(1.0e6, 10.0, 10.0, 0.5).unwrap()
    }

    #[test]
    fn new_rejects_bad_parameters() {
        assert!(ContactModel::new(0.0, 1.0, 1.0, 0.5).is_none());
        assert!(ContactModel::new(1.0, -1.0, 1.0, 0.5).is_none());
        assert!(ContactModel::new(1.0, 1.0, -1.0, 0.5).is_none());
        assert!(ContactModel::new(1.0, 1.0, 1.0, -0.1).is_none());
        assert!(ContactModel::new(f32::NAN, 1.0, 1.0, 0.5).is_none());
    }

    #[test]
    fn non_overlapping_spheres_have_no_contact() {
        // Centres 3 apart, radii 1 + 1 = 2 < 3 → separated.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 3.0,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert!(c.is_none());
    }

    #[test]
    fn coincident_centres_have_no_contact() {
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert!(c.is_none());
    }

    #[test]
    fn overlap_produces_repulsive_normal_force() {
        // Centres 1.5 apart, radii 1 + 1 = 2 → overlap 0.5 along +x.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        assert!((c.overlap - 0.5).abs() < 1e-6);
        // F_n = kₙ·δ = 1e6 · 0.5 = 5e5, pushing b in +x.
        assert!((c.normal_magnitude - 5.0e5).abs() < 1.0);
        assert!(c.force_on_b.x > 0.0);
        assert!(c.force_on_b.y.abs() < 1e-3 && c.force_on_b.z.abs() < 1e-3);
        assert!(!c.sliding);
    }

    #[test]
    fn contact_force_is_equal_and_opposite() {
        // The force on a (−force_on_b) exactly balances the force on b.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            X * 0.2,
            Vec3::new(-0.3, 0.4, 0.0),
        )
        .expect("overlap");
        let on_a = -c.force_on_b;
        assert!((on_a + c.force_on_b).length() < 1e-3);
    }

    #[test]
    fn approaching_spheres_dissipate_through_normal_damping() {
        // b moving toward a (−x) raises the normal force above the pure penalty.
        let approaching = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            Vec3::ZERO,
            X * -1.0,
        )
        .expect("overlap");
        let still = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
        )
        .expect("overlap");
        assert!(approaching.normal_magnitude > still.normal_magnitude);
    }

    #[test]
    fn rebound_force_never_sticks() {
        // Fast separation would make kₙ·δ − γₙ·v_n negative; it clamps at zero.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.999_9,
            1.0,
            1.0,
            Vec3::ZERO,
            X * 1.0e6,
        )
        .expect("tiny overlap");
        assert_eq!(c.normal_magnitude, 0.0);
        assert_eq!(c.force_on_b, Vec3::ZERO);
    }

    #[test]
    fn tangential_slip_is_capped_by_coulomb_friction() {
        // Large tangential slip in +y with heavy overlap → friction saturates at
        // μ·F_n and opposes the slip (−y on b). With γ_t = 10 and F_n = 5e5 the
        // Coulomb cap is μ·F_n = 2.5e5, so the slip must exceed 2.5e4 to saturate.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0e5, 0.0),
        )
        .expect("overlap");
        assert!(c.sliding);
        let coulomb = model().friction() * c.normal_magnitude;
        assert!((c.tangential_magnitude - coulomb).abs() < 1.0);
        assert!(c.force_on_b.y < 0.0, "friction opposes +y slip");
    }

    #[test]
    fn small_tangential_slip_stays_viscous() {
        // A gentle slip stays below the Coulomb limit → viscous, not sliding.
        let c = contact_between(
            &model(),
            Vec3::ZERO,
            X * 1.5,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0e-3, 0.0),
        )
        .expect("overlap");
        assert!(!c.sliding);
        let expected = model().tangential_damping() * 1.0e-3;
        assert!((c.tangential_magnitude - expected).abs() < 1e-3);
    }
}
