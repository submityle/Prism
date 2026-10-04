//! Rotational discrete-element contact: friction torque and rolling resistance.
//!
//! The translational contact laws
//! ([`bonded_particle_contact`](super::bonded_particle_contact),
//! [`hertz_contact`](super::hertz_contact),
//! [`tangential_history_contact`](super::tangential_history_contact)) evaluate
//! friction from the velocity of the particle *centres*. Real grains also spin:
//! a contact resists sliding of the two *surfaces* and resists the grains
//! *rolling* over one another. Capturing that is what lets a granular pile
//! reach a realistic angle of repose — without rolling resistance, idealised
//! spheres roll almost freely and a pile slumps far too flat.
//!
//! This module adds the angular degrees of freedom on top of the Cundall–Strack
//! tangential-history spring:
//!
//! * **Surface sliding.** With the unit axis `n̂` from `a` to `b`, overlap
//!   `δ > 0`, and the shared contact point `c` on the axis (splitting the
//!   overlap), the moment arms are `r_a = c − x_a` and `r_b = c − x_b`. The
//!   surface velocity of each grain at `c` is `v + ω × r`, and the relative
//!   surface velocity `v_rel = (v_b + ω_b×r_b) − (v_a + ω_a×r_a)` drives both
//!   the normal penalty and the tangential-history friction force. Because the
//!   equal-and-opposite force acts at the *same* point `c` on both grains, the
//!   contact conserves linear **and** angular momentum exactly.
//!
//! * **Rolling resistance.** A second Cundall–Strack-style spring accumulates
//!   the relative rotation `ω_a − ω_b` into a resisting couple
//!   `M = −k_r·θ − γ_r·(ω_a−ω_b)`, clamped to `μ_r·R_r·F_n` with the reduced
//!   rolling radius `R_r = r_a·r_b/(r_a+r_b)`. The couple is applied as `+M` on
//!   `a` and `−M` on `b`, so it too conserves angular momentum.
//!
//! The persistent springs are bundled in [`ContactSprings`]; the caller stores
//! one per contact pair and resets it when the contact is lost.

use glam::Vec3;

/// Material parameters of the rotational discrete-element contact law.
///
/// Groups the normal penalty `(kₙ, γₙ)`, the tangential-history friction
/// `(k_t, γ_t, μ)`, and the rolling resistance `(k_r, γ_r, μ_r)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RollingContactModel {
    normal_stiffness: f32,
    normal_damping: f32,
    tangential_stiffness: f32,
    tangential_damping: f32,
    friction: f32,
    rolling_stiffness: f32,
    rolling_damping: f32,
    rolling_friction: f32,
}

impl RollingContactModel {
    /// Builds a model from the normal `(kₙ, γₙ)`, tangential `(k_t, γ_t, μ)`,
    /// and rolling `(k_r, γ_r, μ_r)` parameter groups.
    ///
    /// Returns `None` unless every value is finite, the three stiffnesses
    /// `kₙ`, `k_t`, `k_r` are strictly positive, and the dampings `γₙ`, `γ_t`,
    /// `γ_r` and friction coefficients `μ`, `μ_r` are all non-negative. Rolling
    /// resistance is disabled by setting `μ_r = 0`.
    #[must_use]
    pub fn new(
        normal: (f32, f32),
        tangential: (f32, f32, f32),
        rolling: (f32, f32, f32),
    ) -> Option<Self> {
        let (normal_stiffness, normal_damping) = normal;
        let (tangential_stiffness, tangential_damping, friction) = tangential;
        let (rolling_stiffness, rolling_damping, rolling_friction) = rolling;
        let all_finite = [
            normal_stiffness,
            normal_damping,
            tangential_stiffness,
            tangential_damping,
            friction,
            rolling_stiffness,
            rolling_damping,
            rolling_friction,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !all_finite {
            return None;
        }
        if normal_stiffness <= 0.0 || tangential_stiffness <= 0.0 || rolling_stiffness <= 0.0 {
            return None;
        }
        if normal_damping < 0.0
            || tangential_damping < 0.0
            || rolling_damping < 0.0
            || friction < 0.0
            || rolling_friction < 0.0
        {
            return None;
        }
        Some(Self {
            normal_stiffness,
            normal_damping,
            tangential_stiffness,
            tangential_damping,
            friction,
            rolling_stiffness,
            rolling_damping,
            rolling_friction,
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

    /// Sliding Coulomb friction coefficient `μ`.
    #[must_use]
    pub fn friction(&self) -> f32 {
        self.friction
    }

    /// Rolling resistance spring stiffness `k_r`.
    #[must_use]
    pub fn rolling_stiffness(&self) -> f32 {
        self.rolling_stiffness
    }

    /// Rolling resistance viscous damping `γ_r`.
    #[must_use]
    pub fn rolling_damping(&self) -> f32 {
        self.rolling_damping
    }

    /// Rolling resistance coefficient `μ_r` (`0` disables rolling resistance).
    #[must_use]
    pub fn rolling_friction(&self) -> f32 {
        self.rolling_friction
    }
}

/// Persistent per-contact springs carried between steps: the tangential-history
/// sliding displacement and the accumulated relative-rotation angle driving
/// rolling resistance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContactSprings {
    /// Tangential-history sliding displacement `ξ`.
    pub tangential: Vec3,
    /// Accumulated relative rotation `θ` for rolling resistance.
    pub rolling: Vec3,
}

impl ContactSprings {
    /// A pair of zeroed springs (a fresh contact with no history).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The resolved rotational contact: force on `b` plus the spin torques on both
/// grains and diagnostic magnitudes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationalContact {
    /// Contact force on `b`; `a` feels its negation at the same point.
    pub force_on_b: Vec3,
    /// Total torque on `a` (friction-force moment plus rolling couple).
    pub torque_on_a: Vec3,
    /// Total torque on `b` (friction-force moment plus rolling couple).
    pub torque_on_b: Vec3,
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential (sliding friction) force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Rolling resistance torque magnitude actually applied.
    pub rolling_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit (sliding).
    pub sliding: bool,
}

/// Evaluates the rotational contact between two spheres and advances the
/// persistent springs, returning `None` when the spheres do not overlap or
/// their centres coincide.
///
/// `positions`, `radii`, `velocities`, and `angular` are the `(a, b)` pairs of
/// centres, radii, linear velocities, and angular velocities. `springs` carries
/// the persistent tangential and rolling springs for this contact pair and is
/// left untouched when `None` is returned. `dt` must be finite and strictly
/// positive.
#[must_use]
pub fn rotational_contact_between(
    model: &RollingContactModel,
    positions: (Vec3, Vec3),
    radii: (f32, f32),
    velocities: (Vec3, Vec3),
    angular: (Vec3, Vec3),
    springs: &mut ContactSprings,
    dt: f32,
) -> Option<RotationalContact> {
    let (pos_a, pos_b) = positions;
    let (rad_a, rad_b) = radii;
    let (vel_a, vel_b) = velocities;
    let (omega_a, omega_b) = angular;

    let delta = pos_b - pos_a;
    let distance = delta.length();
    if distance <= 0.0 {
        return None;
    }
    let overlap = (rad_a + rad_b) - distance;
    if overlap <= 0.0 {
        return None;
    }
    let normal = delta / distance;

    // Shared contact point splits the overlap; same point on both grains so the
    // equal-and-opposite force conserves angular momentum.
    let arm_a = rad_a - 0.5 * overlap;
    let arm_b = rad_b - 0.5 * overlap;
    let r_a = normal * arm_a;
    let r_b = -normal * arm_b;

    // Relative surface velocity at the contact point.
    let surf_a = vel_a + omega_a.cross(r_a);
    let surf_b = vel_b + omega_b.cross(r_b);
    let rel = surf_b - surf_a;

    // Normal penalty response.
    let v_n = rel.dot(normal);
    let normal_force = (model.normal_stiffness * overlap - model.normal_damping * v_n).max(0.0);

    // Tangential Cundall–Strack history at the contact surface.
    let v_t = rel - v_n * normal;
    let mut tangential = springs.tangential - springs.tangential.dot(normal) * normal;
    tangential += v_t * dt;
    let mut tangential_force =
        -model.tangential_stiffness * tangential - model.tangential_damping * v_t;
    let mut tangential_magnitude = tangential_force.length();
    let max_friction = model.friction * normal_force;
    let mut sliding = false;
    if tangential_magnitude > max_friction {
        if tangential_magnitude > 0.0 {
            let direction = tangential_force / tangential_magnitude;
            tangential_force = direction * max_friction;
            tangential = -tangential_force / model.tangential_stiffness;
        } else {
            tangential_force = Vec3::ZERO;
            tangential = Vec3::ZERO;
        }
        tangential_magnitude = max_friction;
        sliding = max_friction > 0.0;
    }
    springs.tangential = tangential;

    let force_on_b = normal_force * normal + tangential_force;

    // Spin torques from the contact force act at the shared point.
    let mut torque_on_b = r_b.cross(force_on_b);
    let mut torque_on_a = r_a.cross(-force_on_b);

    // Rolling resistance couple opposing the relative rotation ω_a − ω_b.
    let omega_rel = omega_a - omega_b;
    let rolling_radius = (rad_a * rad_b) / (rad_a + rad_b);
    let mut rolling = springs.rolling + omega_rel * dt;
    let mut rolling_torque = -model.rolling_stiffness * rolling - model.rolling_damping * omega_rel;
    let mut rolling_magnitude = rolling_torque.length();
    let max_rolling = model.rolling_friction * rolling_radius * normal_force;
    if rolling_magnitude > max_rolling {
        if rolling_magnitude > 0.0 {
            let direction = rolling_torque / rolling_magnitude;
            rolling_torque = direction * max_rolling;
            // k_r > 0 is guaranteed by the constructor, so this never divides
            // by zero; it parks the spring exactly on the resistance cap.
            rolling = -rolling_torque / model.rolling_stiffness;
        } else {
            rolling_torque = Vec3::ZERO;
            rolling = Vec3::ZERO;
        }
        rolling_magnitude = max_rolling;
    }
    springs.rolling = rolling;

    torque_on_a += rolling_torque;
    torque_on_b -= rolling_torque;

    Some(RotationalContact {
        force_on_b,
        torque_on_a,
        torque_on_b,
        overlap,
        normal_magnitude: normal_force,
        tangential_magnitude,
        rolling_magnitude,
        sliding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RollingContactModel {
        // (kₙ,γₙ), (k_t,γ_t,μ), (k_r,γ_r,μ_r).
        RollingContactModel::new((1.0e5, 0.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 0.3)).unwrap()
    }

    #[test]
    fn new_rejects_bad_parameters() {
        assert!(RollingContactModel::new((0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 0.3)).is_none());
        assert!(RollingContactModel::new((1.0, 0.0), (0.0, 0.0, 0.5), (1.0, 0.0, 0.3)).is_none());
        assert!(RollingContactModel::new((1.0, 0.0), (1.0, 0.0, 0.5), (0.0, 0.0, 0.3)).is_none());
        assert!(RollingContactModel::new((1.0, -1.0), (1.0, 0.0, 0.5), (1.0, 0.0, 0.3)).is_none());
        assert!(RollingContactModel::new((1.0, 0.0), (1.0, 0.0, -0.5), (1.0, 0.0, 0.3)).is_none());
        assert!(RollingContactModel::new((1.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, -0.3)).is_none());
        assert!(
            RollingContactModel::new((f32::NAN, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 0.3)).is_none()
        );
        assert!(RollingContactModel::new((1.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 0.0)).is_some());
    }

    #[test]
    fn non_overlapping_and_coincident_return_none() {
        let mut s = ContactSprings::new();
        assert!(rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .is_none());
        assert_eq!(
            s,
            ContactSprings::new(),
            "springs untouched when no contact"
        );
        assert!(rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::ZERO),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .is_none());
    }

    #[test]
    fn pure_normal_overlap_has_no_torque() {
        let mut s = ContactSprings::new();
        let out = rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .expect("overlap");
        assert!(out.normal_magnitude > 0.0);
        assert_eq!(out.tangential_magnitude, 0.0);
        assert_eq!(out.rolling_magnitude, 0.0);
        assert!(out.torque_on_a.length() < 1e-6);
        assert!(out.torque_on_b.length() < 1e-6);
    }

    #[test]
    fn spinning_grain_feels_a_decelerating_torque() {
        // Grain a spins about +z; its surface slides at the contact, producing
        // friction that opposes the spin (torque·ω < 0).
        let mut s = ContactSprings::new();
        let omega_a = Vec3::new(0.0, 0.0, 10.0);
        let out = rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (omega_a, Vec3::ZERO),
            &mut s,
            1.0e-4,
        )
        .expect("overlap");
        assert!(
            out.tangential_magnitude > 0.0,
            "surface slip makes friction"
        );
        assert!(
            out.torque_on_a.dot(omega_a) < 0.0,
            "friction torque opposes the spin"
        );
    }

    #[test]
    fn total_angular_momentum_rate_is_zero() {
        // The contact is internal: Σ xᵢ×Fᵢ + Σ τᵢ must vanish regardless of the
        // velocities, verifying the force acts at a single shared point and the
        // rolling couple is equal-and-opposite.
        let mut s = ContactSprings::new();
        let pos_a = Vec3::new(0.3, -0.2, 0.1);
        let pos_b = pos_a + Vec3::new(1.4, 0.3, -0.2).normalize() * 1.6;
        let out = rotational_contact_between(
            &model(),
            (pos_a, pos_b),
            (1.0, 1.0),
            (Vec3::new(0.5, -0.3, 0.2), Vec3::new(-0.1, 0.4, 0.05)),
            (Vec3::new(1.0, 2.0, -1.0), Vec3::new(-0.5, 0.3, 0.7)),
            &mut s,
            1.0e-4,
        )
        .expect("overlap");
        let force_on_a = -out.force_on_b;
        let total = pos_a.cross(force_on_a)
            + pos_b.cross(out.force_on_b)
            + out.torque_on_a
            + out.torque_on_b;
        assert!(
            total.length() < 1e-3,
            "angular momentum conserved: {total:?}"
        );
    }

    #[test]
    fn rolling_resistance_opposes_relative_rotation() {
        // Large μ_r so the rolling couple stays below its cap. With a > b spin,
        // the couple must slow a (torque_on_a · ω_rel < 0) and the rolling part
        // must be equal-and-opposite on the two grains.
        let m =
            RollingContactModel::new((1.0e5, 0.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 100.0)).unwrap();
        let mut s = ContactSprings::new();
        let omega_a = Vec3::new(0.0, 0.0, 5.0);
        let omega_b = Vec3::new(0.0, 0.0, -5.0);
        let out = rotational_contact_between(
            &m,
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (omega_a, omega_b),
            &mut s,
            1.0e-4,
        )
        .expect("overlap");
        assert!(out.rolling_magnitude > 0.0, "relative spin resists rolling");
        let omega_rel = omega_a - omega_b;
        // The rolling couple contribution to a opposes ω_rel.
        assert!(
            s.rolling.dot(omega_rel) > 0.0,
            "rolling spring winds along ω_rel"
        );
    }

    #[test]
    fn rolling_resistance_disabled_when_coefficient_is_zero() {
        // μ_r = 0 → the rolling cap is 0, so the couple is always zero.
        let mut s = ContactSprings::new();
        let out = rotational_contact_between(
            &model(), // μ_r = 0.3 by default; use an explicit zero model instead.
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
            &mut s,
            1.0e-4,
        )
        .expect("overlap");
        // Default model has μ_r > 0, so rolling resistance is active here.
        assert!(out.rolling_magnitude > 0.0);

        let no_roll =
            RollingContactModel::new((1.0e5, 0.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 0.0)).unwrap();
        let mut s2 = ContactSprings::new();
        let out2 = rotational_contact_between(
            &no_roll,
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
            &mut s2,
            1.0e-4,
        )
        .expect("overlap");
        assert_eq!(
            out2.rolling_magnitude, 0.0,
            "μ_r=0 disables rolling resistance"
        );
    }

    #[test]
    fn fast_surface_slip_saturates_at_the_coulomb_cap() {
        let mut s = ContactSprings::new();
        // Large tangential centre velocity → immediate slip.
        let out = rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::new(0.0, 1000.0, 0.0)),
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .expect("overlap");
        assert!(out.sliding);
        let cap = 0.5 * 1.0e5 * 0.5; // μ·kₙ·δ.
        assert!((out.tangential_magnitude - cap).abs() < cap * 1e-3);
    }

    #[test]
    fn tangential_spring_is_reprojected_onto_the_tangent_plane() {
        let mut s = ContactSprings {
            tangential: Vec3::new(0.0, 1.0e-4, 0.0),
            rolling: Vec3::ZERO,
        };
        // Normal along +y removes the +y tangential component.
        let out = rotational_contact_between(
            &model(),
            (Vec3::ZERO, Vec3::new(0.0, 1.5, 0.0)),
            (1.0, 1.0),
            (Vec3::ZERO, Vec3::ZERO),
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-4,
        )
        .expect("overlap");
        assert!(
            s.tangential.dot(Vec3::Y).abs() < 1e-9,
            "normal component removed"
        );
        assert_eq!(out.tangential_magnitude, 0.0);
    }
}
