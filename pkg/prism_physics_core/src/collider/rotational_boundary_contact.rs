//! Rotational discrete-element contact against a static half-space boundary.
//!
//! [`rotational_contact`](super::rotational_contact) resolves the spin-aware
//! contact between two *mobile* grains. A real granular scene also needs the
//! grains to rest on something: a floor, the walls of a hopper, the drum of a
//! tumbler. Those boundaries are immovable, so the contact is one-sided — the
//! grain feels a force and a torque, the wall feels nothing. Modelling the
//! boundary analytically (instead of tiling it with frozen grains) keeps a
//! resting pile exact and cheap.
//!
//! A boundary is the half-space [`HalfSpace`] with support point `p` and unit
//! outward normal `n̂` pointing into the free region where grains live. A grain
//! of radius `r` centred at `x` has signed distance `d = (x − p)·n̂` to the
//! plane and overlaps it when `δ = r − d > 0`. The response reuses the grain
//! contact law of [`rotational_contact`](super::rotational_contact):
//!
//! * **Normal penalty.** The contact point `c = x − (r − δ/2)·n̂` lies at the
//!   mid-depth of the overlap, with moment arm `ρ = c − x = −(r − δ/2)·n̂`. The
//!   grain surface velocity there is `v + ω × ρ`; the wall is static so this is
//!   also the relative velocity. The normal force `F_n = max(0, kₙ·δ − γₙ·v_n)`
//!   pushes the grain back out along `+n̂`.
//!
//! * **Surface sliding.** A Cundall–Strack history spring integrates the
//!   tangential surface velocity into `ξ`, giving `F_t = −k_t·ξ − γ_t·v_t`
//!   clamped to the Coulomb cone `μ·F_n`. The friction force applies a spin
//!   torque `ρ × F` about the grain centre.
//!
//! * **Rolling resistance.** With the wall at rest the relative rotation is the
//!   grain spin `ω`, so a second history spring accumulates `θ` into a couple
//!   `M = −k_r·θ − γ_r·ω` capped at `μ_r·R_r·F_n`. The reduced rolling radius
//!   `R_r = r·r_wall/(r + r_wall)` tends to `r` as the wall radius `r_wall → ∞`,
//!   so a flat boundary uses `R_r = r`. This couple is what lets a pile hold a
//!   finite angle of repose on a solid floor rather than slumping flat.
//!
//! The persistent springs reuse [`ContactSprings`]; the caller keeps one per
//! (grain, boundary) pair and clears it when the contact is lost.

use glam::Vec3;

use super::rotational_contact::{ContactSprings, RollingContactModel};

/// A static planar boundary: the half-space on the `−n̂` side of the plane
/// through `point` with unit outward normal `normal` pointing into the free
/// region occupied by grains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSpace {
    point: Vec3,
    normal: Vec3,
}

impl HalfSpace {
    /// Builds a half-space from a support `point` on the plane and an outward
    /// `normal`, normalising the normal.
    ///
    /// Returns `None` unless both inputs are finite and the normal has a
    /// strictly positive length (so it can be normalised).
    #[must_use]
    pub fn new(point: Vec3, normal: Vec3) -> Option<Self> {
        if !point.is_finite() || !normal.is_finite() {
            return None;
        }
        let length = normal.length();
        if length <= 0.0 {
            return None;
        }
        Some(Self {
            point,
            normal: normal / length,
        })
    }

    /// Support point `p` lying on the boundary plane.
    #[must_use]
    pub fn point(&self) -> Vec3 {
        self.point
    }

    /// Unit outward normal `n̂` pointing into the free region.
    #[must_use]
    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    /// Signed distance `d = (q − p)·n̂` of a point `q` from the plane. Positive
    /// in the free region, negative inside the solid boundary.
    #[must_use]
    pub fn signed_distance(&self, query: Vec3) -> f32 {
        (query - self.point).dot(self.normal)
    }
}

/// The resolved grain-versus-boundary contact. All quantities act on the grain;
/// the static wall receives no reaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryContact {
    /// Contact force on the grain.
    pub force: Vec3,
    /// Contact torque on the grain (friction-force moment plus rolling couple).
    pub torque: Vec3,
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

/// Evaluates the rotational contact between a grain and a static half-space and
/// advances the persistent springs, returning `None` when the grain does not
/// overlap the boundary.
///
/// `center` and `radius` describe the grain; `motion` is its `(velocity,
/// angular velocity)` pair. `springs` carries the persistent tangential and
/// rolling springs for this (grain, boundary) pair and is left untouched when
/// `None` is returned. `dt` must be finite and strictly positive.
#[must_use]
pub fn grain_boundary_contact(
    model: &RollingContactModel,
    plane: &HalfSpace,
    center: Vec3,
    radius: f32,
    motion: (Vec3, Vec3),
    springs: &mut ContactSprings,
    dt: f32,
) -> Option<BoundaryContact> {
    if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    if !dt.is_finite() || dt <= 0.0 {
        return None;
    }
    let (velocity, angular) = motion;
    if !velocity.is_finite() || !angular.is_finite() {
        return None;
    }

    let distance = plane.signed_distance(center);
    let overlap = radius - distance;
    if overlap <= 0.0 {
        return None;
    }
    let normal = plane.normal();

    // Contact point at the mid-depth of the overlap; arm from the grain centre
    // points toward the wall along `−n̂`.
    let arm = radius - 0.5 * overlap;
    let r = -normal * arm;

    // Grain surface velocity at the contact point; the wall is static.
    let surface = velocity + angular.cross(r);

    // Normal penalty response.
    let v_n = surface.dot(normal);
    let normal_force = (model.normal_stiffness() * overlap - model.normal_damping() * v_n).max(0.0);

    // Tangential Cundall–Strack history at the contact surface.
    let v_t = surface - v_n * normal;
    let mut tangential = springs.tangential - springs.tangential.dot(normal) * normal;
    tangential += v_t * dt;
    let mut tangential_force =
        -model.tangential_stiffness() * tangential - model.tangential_damping() * v_t;
    let mut tangential_magnitude = tangential_force.length();
    let max_friction = model.friction() * normal_force;
    let mut sliding = false;
    if tangential_magnitude > max_friction {
        if tangential_magnitude > 0.0 {
            let direction = tangential_force / tangential_magnitude;
            tangential_force = direction * max_friction;
            tangential = -tangential_force / model.tangential_stiffness();
        } else {
            tangential_force = Vec3::ZERO;
            tangential = Vec3::ZERO;
        }
        tangential_magnitude = max_friction;
        sliding = max_friction > 0.0;
    }
    springs.tangential = tangential;

    let force = normal_force * normal + tangential_force;
    let mut torque = r.cross(force);

    // Rolling resistance couple opposing the grain spin (wall spin is zero).
    let rolling_radius = radius;
    let mut rolling = springs.rolling + angular * dt;
    let mut rolling_torque =
        -model.rolling_stiffness() * rolling - model.rolling_damping() * angular;
    let mut rolling_magnitude = rolling_torque.length();
    let max_rolling = model.rolling_friction() * rolling_radius * normal_force;
    if rolling_magnitude > max_rolling {
        if rolling_magnitude > 0.0 {
            let direction = rolling_torque / rolling_magnitude;
            rolling_torque = direction * max_rolling;
            // k_r > 0 is guaranteed by the constructor, so this never divides by
            // zero; it parks the spring exactly on the resistance cap.
            rolling = -rolling_torque / model.rolling_stiffness();
        } else {
            rolling_torque = Vec3::ZERO;
            rolling = Vec3::ZERO;
        }
        rolling_magnitude = max_rolling;
    }
    springs.rolling = rolling;

    torque += rolling_torque;

    Some(BoundaryContact {
        force,
        torque,
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

    fn floor() -> HalfSpace {
        // Ground plane at z = 0 with the free region above it.
        HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap()
    }

    #[test]
    fn half_space_rejects_bad_normals() {
        assert!(HalfSpace::new(Vec3::ZERO, Vec3::ZERO).is_none());
        assert!(HalfSpace::new(Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 1.0)).is_none());
        assert!(HalfSpace::new(Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::Z).is_none());
    }

    #[test]
    fn half_space_normalises_and_measures_distance() {
        let plane = HalfSpace::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 5.0)).unwrap();
        assert!((plane.normal() - Vec3::Z).length() < 1.0e-6);
        // A point 3 units above the plane has signed distance +3.
        assert!((plane.signed_distance(Vec3::new(1.0, -4.0, 5.0)) - 3.0).abs() < 1.0e-6);
        // A point below the plane is negative.
        assert!((plane.signed_distance(Vec3::new(0.0, 0.0, 0.5)) + 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn separated_grain_returns_none() {
        let mut s = ContactSprings::new();
        // Centre well above the floor relative to its radius.
        let out = grain_boundary_contact(
            &model(),
            &floor(),
            Vec3::new(0.0, 0.0, 2.0),
            1.0,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        );
        assert!(out.is_none());
        // Springs untouched.
        assert_eq!(s, ContactSprings::new());
    }

    #[test]
    fn invalid_inputs_return_none() {
        let mut s = ContactSprings::new();
        let m = model();
        let p = floor();
        assert!(grain_boundary_contact(
            &m,
            &p,
            Vec3::ZERO,
            -1.0,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1e-3
        )
        .is_none());
        assert!(grain_boundary_contact(
            &m,
            &p,
            Vec3::ZERO,
            1.0,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            0.0
        )
        .is_none());
        assert!(grain_boundary_contact(
            &m,
            &p,
            Vec3::new(0.0, 0.0, f32::NAN),
            1.0,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1e-3,
        )
        .is_none());
    }

    #[test]
    fn resting_normal_force_equals_depth_times_stiffness() {
        let mut s = ContactSprings::new();
        let radius = 1.0;
        let depth = 0.01;
        // Grain centre just below z = radius so it overlaps by `depth`.
        let center = Vec3::new(0.0, 0.0, radius - depth);
        let c = grain_boundary_contact(
            &model(),
            &floor(),
            center,
            radius,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        assert!((c.overlap - depth).abs() < 1.0e-6);
        // No velocity, no spin: pure normal penalty straight up.
        assert!((c.normal_magnitude - 1.0e5 * depth).abs() < 1.0e-1);
        assert!(c.force.x.abs() < 1.0e-4 && c.force.y.abs() < 1.0e-4);
        assert!(c.force.z > 0.0);
        assert!((c.force.z - 1.0e5 * depth).abs() < 1.0e-1);
        assert!(c.torque.length() < 1.0e-4);
        assert!(!c.sliding);
    }

    #[test]
    fn approaching_grain_feels_extra_damping() {
        // A damped normal model: closing velocity adds to the normal force.
        let m =
            RollingContactModel::new((1.0e5, 50.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 0.3)).unwrap();
        let radius = 1.0;
        let depth = 0.01;
        let center = Vec3::new(0.0, 0.0, radius - depth);
        let mut s = ContactSprings::new();
        let still = grain_boundary_contact(
            &m,
            &floor(),
            center,
            radius,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        let mut s2 = ContactSprings::new();
        let closing = grain_boundary_contact(
            &m,
            &floor(),
            center,
            radius,
            (Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO),
            &mut s2,
            1.0e-3,
        )
        .unwrap();
        assert!(closing.normal_magnitude > still.normal_magnitude);
    }

    #[test]
    fn sliding_grain_hits_coulomb_cone() {
        let mut s = ContactSprings::new();
        let radius = 1.0;
        let depth = 0.01;
        let center = Vec3::new(0.0, 0.0, radius - depth);
        // Fast horizontal slide saturates the friction cone.
        let c = grain_boundary_contact(
            &model(),
            &floor(),
            center,
            radius,
            (Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        assert!(c.sliding);
        let expected = 0.5 * c.normal_magnitude;
        assert!((c.tangential_magnitude - expected).abs() < 1.0e-3);
        // Friction opposes the slide, so the tangential force points in `−x`.
        assert!(c.force.x < 0.0);
    }

    #[test]
    fn spinning_grain_feels_decelerating_friction_torque() {
        let mut s = ContactSprings::new();
        let radius = 1.0;
        let depth = 0.01;
        let center = Vec3::new(0.0, 0.0, radius - depth);
        // Spin about +x: the bottom contact point sweeps in +y, so friction
        // acts in −y and the moment about +x opposes the spin (torque.x < 0).
        let c = grain_boundary_contact(
            &model(),
            &floor(),
            center,
            radius,
            (Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        assert!(c.torque.x < 0.0);
    }

    #[test]
    fn rolling_resistance_opposes_spin_and_caps() {
        let mut s = ContactSprings::new();
        let radius = 1.0;
        let depth = 0.01;
        let center = Vec3::new(0.0, 0.0, radius - depth);
        // Large spin about +z saturates the rolling-resistance cap.
        let c = grain_boundary_contact(
            &model(),
            &floor(),
            center,
            radius,
            (Vec3::ZERO, Vec3::new(0.0, 0.0, 100.0)),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        let cap = 0.3 * radius * c.normal_magnitude;
        assert!((c.rolling_magnitude - cap).abs() < 1.0e-3);
        // The couple opposes the +z spin.
        assert!(c.torque.z < 0.0);
    }

    #[test]
    fn tangential_spring_accumulates_across_steps() {
        let mut s = ContactSprings::new();
        let radius = 1.0;
        let depth = 0.01;
        let center = Vec3::new(0.0, 0.0, radius - depth);
        let m = model();
        let p = floor();
        // A slow slide below the Coulomb limit grows the stored spring.
        let slow = (Vec3::new(1.0e-4, 0.0, 0.0), Vec3::ZERO);
        let first = grain_boundary_contact(&m, &p, center, radius, slow, &mut s, 1.0e-3).unwrap();
        let after_first = s.tangential.length();
        let second = grain_boundary_contact(&m, &p, center, radius, slow, &mut s, 1.0e-3).unwrap();
        assert!(!first.sliding && !second.sliding);
        assert!(s.tangential.length() > after_first);
        assert!(second.tangential_magnitude > first.tangential_magnitude);
    }

    #[test]
    fn tilted_boundary_pushes_along_its_normal() {
        // A 45° ramp: normal in the x–z plane.
        let normal = Vec3::new(1.0, 0.0, 1.0);
        let ramp = HalfSpace::new(Vec3::ZERO, normal).unwrap();
        let unit = normal.normalize();
        let radius = 1.0;
        let depth = 0.02;
        // Centre at signed distance `radius − depth` from the ramp.
        let center = unit * (radius - depth);
        let mut s = ContactSprings::new();
        let c = grain_boundary_contact(
            &model(),
            &ramp,
            center,
            radius,
            (Vec3::ZERO, Vec3::ZERO),
            &mut s,
            1.0e-3,
        )
        .unwrap();
        assert!((c.overlap - depth).abs() < 1.0e-5);
        // Pure normal response is parallel to the ramp normal.
        let along = c.force.dot(unit);
        assert!((c.force - along * unit).length() < 1.0e-3);
        assert!(along > 0.0);
    }
}
