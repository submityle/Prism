//! Long-range attachment (LRA) leash from a particle to a fixed anchor.
//!
//! A [`LongRangeConstraint`] is a *one-sided* distance constraint: the particle
//! may move freely as long as it stays within `max_distance` of its anchor, but
//! once it drifts farther it is pulled back onto the leash sphere. This is the
//! classic cloth "long-range attachment" (Kim et al. 2012; used by `NvCloth` and
//! production cloth): seeding every particle with a geodesic leash to a nearby
//! kinematic attachment removes the long-wavelength stretch that pure local
//! distance constraints need many iterations to resolve, so a garment stops
//! sagging like rubber under fast motion without an expensive global solve.
//!
//! The constraint function, active only when overstretched, is
//!
//! ```text
//! C = |p - anchor| - max_distance   (projected only while C > 0)
//! ```
//!
//! with unit gradient `n = (p - anchor) / |p - anchor|`, so the XPBD
//! denominator is `w + alpha_tilde`. Because the anchor is a fixed (infinite
//! mass) point, the particle takes the whole correction. A [`compliance`] of
//! `0` makes the leash perfectly rigid (a hard geodesic tether); a positive
//! value lets it stretch slightly.
//!
//! [`compliance`]: LongRangeConstraint::compliance
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! one-sided long-range-attachment leash is a published position-based-dynamics
//! technique (Kim et al., "Long Range Attachments").

use glam::Vec3;

use crate::math::scalar::{Real, EPSILON};
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A compliant one-sided XPBD leash keeping one particle within a maximum
/// distance of a fixed anchor.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LongRangeConstraint {
    /// The leashed particle.
    pub particle: ParticleHandle,
    /// The fixed world-space anchor the leash is measured from (typically a
    /// kinematic attachment or the particle's skinned rest position).
    pub anchor: Vec3,
    /// Maximum allowed distance from `anchor`; the leash only acts when the
    /// particle is farther than this.
    pub max_distance: Real,
    /// Compliance (inverse stiffness); `0` is a perfectly rigid leash.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl LongRangeConstraint {
    /// Creates a one-sided leash tying `particle` to `anchor` with the given
    /// `max_distance` (clamped to be non-negative) and `compliance` (`0` for a
    /// rigid leash).
    #[must_use]
    pub fn new(
        particle: ParticleHandle,
        anchor: Vec3,
        max_distance: Real,
        compliance: Real,
    ) -> Self {
        LongRangeConstraint {
            particle,
            anchor,
            max_distance: max_distance.max(0.0),
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Moves the anchor to a new world position. Use this each frame to drag a
    /// leash along a kinematic attachment path.
    pub fn set_anchor(&mut self, anchor: Vec3) {
        self.anchor = anchor;
    }

    /// Returns the current accumulated Lagrange multiplier (for diagnostics and
    /// tests).
    #[must_use]
    pub fn lambda(&self) -> Real {
        self.lambda
    }
}

impl ParticleConstraint for LongRangeConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::LongRange
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        // Delegate the arithmetic to the raw-index [`project_long_range`] so the
        // sequential golden and its parallel `GPU` twin stay bit-for-bit the
        // same function with no risk of drift.
        self.lambda = project_long_range(
            positions,
            inverse_masses,
            self.particle.raw(),
            self.anchor,
            self.max_distance,
            self.compliance,
            self.lambda,
            dt,
        );
    }
}

/// Projects a single one-sided long-range-attachment leash in place over a raw
/// particle index so a parallel `GPU` twin can share the exact arithmetic of
/// the sequential [`LongRangeConstraint`] golden.
///
/// Reads and writes `positions[particle]` in place and returns the updated
/// accumulated Lagrange multiplier; the input `lambda` is returned unchanged
/// when the projection is inert (an out-of-range or pinned particle, a particle
/// coincident with its anchor, or one still inside the leash sphere). The
/// gradient is the unit vector from the anchor to the particle, so the XPBD
/// denominator is `w + alpha_tilde` and the anchor (infinite mass) takes none
/// of the correction.
///
/// `max_distance` and `compliance` are used as given (the
/// [`LongRangeConstraint`] constructor already clamps them non-negative).
///
/// # Provenance
///
/// The one-sided long-range-attachment leash is a published position-based
/// dynamics technique (Kim et al., "Long Range Attachments"). No Unreal Engine
/// source or derived code.
#[must_use]
pub fn project_long_range(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    particle: u32,
    anchor: Vec3,
    max_distance: Real,
    compliance: Real,
    lambda: Real,
    dt: Real,
) -> Real {
    let i = particle as usize;
    let (Some(&w), Some(&position)) = (inverse_masses.get(i), positions.get(i)) else {
        return lambda;
    };
    if w <= 0.0 {
        return lambda;
    }
    let delta = position - anchor;
    let length = delta.length();
    if length < EPSILON {
        return lambda;
    }
    // One-sided: slack inside the leash sphere does nothing.
    let c = length - max_distance;
    if c <= 0.0 {
        return lambda;
    }
    let normal = delta / length;
    let alpha_tilde = compliance / (dt * dt);
    let delta_lambda = (-c - alpha_tilde * lambda) / (w + alpha_tilde);
    positions[i] += normal * (delta_lambda * w);
    lambda + delta_lambda
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn rigid_leash_pulls_overstretched_particle_onto_sphere() {
        // Particle 3 units out, leash of 1 unit, rigid: snaps back to radius 1.
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [1.0];
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert!((positions[0].length() - 1.0).abs() < 1e-5, "pos {:?}", positions[0]);
    }

    #[test]
    fn slack_particle_inside_leash_is_inert() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inv = [1.0];
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn pinned_particle_is_not_moved() {
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [0.0];
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn coincident_with_anchor_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn out_of_range_handle_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = LongRangeConstraint::new(h(5), Vec3::ONE, 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn compliant_leash_only_partially_corrects() {
        // A positive compliance should leave the particle beyond the leash
        // radius after a single projection (softer than the rigid snap).
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [1.0];
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 1.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert!(positions[0].length() > 1.0, "pos {:?}", positions[0]);
        assert!(positions[0].length() < 3.0, "pos {:?}", positions[0]);
    }

    #[test]
    fn set_anchor_moves_leash_origin() {
        let mut c = LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0);
        c.set_anchor(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(c.anchor, Vec3::new(1.0, 2.0, 3.0));
    }
}
