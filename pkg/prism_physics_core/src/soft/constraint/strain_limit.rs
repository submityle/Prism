//! Hard strain-limiting (biphasic length clamp) for a stretch edge.
//!
//! A [`StrainLimitConstraint`] is a *geometric* post-projection clamp rather
//! than a compliant force. After the compliant distance/bending sweeps have run
//! (which approach the rest length but can overshoot under fast motion), the
//! strain limiter guarantees that no structural edge exceeds `rest_length *
//! max_scale` (and, optionally, never compresses below `rest_length *
//! min_scale`). This keeps cloth from stretching "like rubber" when yanked, a
//! defect pure XPBD distance constraints only hide with many iterations.
//!
//! The clamp is mass-weighted and never moves a pinned particle: when a
//! structural edge is overstretched to length `d > max_len`, the excess
//! `d - max_len` is removed along the edge direction, split between the two
//! particles in proportion to their inverse masses. Compression below `min_len`
//! is handled symmetrically with the same signed formula. Because this is a
//! pure projection with no Lagrange multiplier, [`reset`](StrainLimitConstraint)
//! is a no-op and [`compliance`](StrainLimitConstraint) is always `0`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Biphasic
//! strain limiting is a standard, publicly documented cloth technique
//! (Provot 1995; Thomaszewski et al. 2009).

use glam::Vec3;

use crate::math::scalar::{Real, EPSILON};
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A hard, mass-weighted biphasic length clamp over a single stretch edge.
///
/// The edge length is kept within `[rest_length * min_scale, rest_length *
/// max_scale]`. Setting `min_scale` to `0` disables the lower (compression)
/// clamp, matching the common "max-stretch-only" strain limiter.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StrainLimitConstraint {
    /// First coupled particle.
    pub a: ParticleHandle,
    /// Second coupled particle.
    pub b: ParticleHandle,
    /// Rest (reference) length of the edge, in metres.
    pub rest_length: Real,
    /// Maximum allowed length as a multiple of `rest_length` (`>= 1` to allow
    /// any stretch). Clamped to be at least `1`.
    pub max_scale: Real,
    /// Minimum allowed length as a multiple of `rest_length`. A value of `0`
    /// disables the compression clamp; otherwise it is clamped into `[0, 1]`.
    pub min_scale: Real,
}

impl StrainLimitConstraint {
    /// Creates a strain limiter over edge `a`-`b` with the given `rest_length`.
    ///
    /// `max_scale` is clamped to be at least `1` (never shorter than the rest
    /// length as an upper bound). `min_scale` is clamped into `[0, 1]`; pass
    /// `0` to disable the compression clamp.
    #[must_use]
    pub fn new(
        a: ParticleHandle,
        b: ParticleHandle,
        rest_length: Real,
        max_scale: Real,
        min_scale: Real,
    ) -> Self {
        StrainLimitConstraint {
            a,
            b,
            rest_length: rest_length.max(0.0),
            max_scale: max_scale.max(1.0),
            min_scale: min_scale.clamp(0.0, 1.0),
        }
    }

    /// Convenience constructor for a max-stretch-only limiter expressed as a
    /// fractional strain `limit` (e.g. `0.1` for a 10% stretch cap), matching
    /// the render-side authoring convention `max_scale = 1 + limit`.
    #[must_use]
    pub fn from_stretch_limit(
        a: ParticleHandle,
        b: ParticleHandle,
        rest_length: Real,
        limit: Real,
    ) -> Self {
        StrainLimitConstraint::new(a, b, rest_length, 1.0 + limit.max(0.0), 0.0)
    }

    /// Returns the maximum allowed edge length in metres.
    #[must_use]
    pub fn max_length(&self) -> Real {
        self.rest_length * self.max_scale
    }

    /// Returns the minimum allowed edge length in metres, or `0` when the
    /// compression clamp is disabled.
    #[must_use]
    pub fn min_length(&self) -> Real {
        self.rest_length * self.min_scale
    }
}

impl ParticleConstraint for StrainLimitConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::StrainLimit
    }

    fn compliance(&self) -> Real {
        // A hard geometric clamp has no compliance.
        0.0
    }

    fn reset(&mut self) {
        // Stateless: no accumulated multiplier to clear.
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], _dt: Real) {
        let ia = self.a.index();
        let ib = self.b.index();
        if ia == ib {
            return;
        }
        let (Some(&wa), Some(&wb)) = (inverse_masses.get(ia), inverse_masses.get(ib)) else {
            return;
        };
        let w_sum = wa + wb;
        if w_sum <= 0.0 {
            return;
        }
        let delta = positions[ia] - positions[ib];
        let length = delta.length();
        if length < EPSILON {
            return;
        }
        let max_len = self.rest_length * self.max_scale;
        let min_len = self.rest_length * self.min_scale;
        // Signed length error outside the allowed band; positive means
        // overstretched, negative means over-compressed. Zero inside the band.
        let error = if length > max_len {
            length - max_len
        } else if self.min_scale > 0.0 && length < min_len {
            length - min_len
        } else {
            return;
        };
        let direction = delta / length;
        let correction = direction * error;
        // Mass-weighted split; moving `a` toward `b` for overstretch (error > 0)
        // and apart for over-compression (error < 0) via the shared sign.
        positions[ia] -= correction * (wa / w_sum);
        positions[ib] += correction * (wb / w_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn overstretched_edge_is_clamped_to_max_length() {
        // Rest 1, max_scale 1.1 => max length 1.1. Edge stretched to 2.0.
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        let length = (positions[0] - positions[1]).length();
        assert!((length - 1.1).abs() < 1e-5, "length was {length}");
        // Symmetric masses => symmetric clamp about the midpoint (1.0, 0, 0).
        assert!((positions[0] - Vec3::new(0.45, 0.0, 0.0)).length() < 1e-5);
        assert!((positions[1] - Vec3::new(1.55, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn edge_within_band_is_inert() {
        // Length 1.05 is inside [0, 1.1]; nothing moves.
        let mut positions = [Vec3::ZERO, Vec3::new(1.05, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
        assert_eq!(positions[1], Vec3::new(1.05, 0.0, 0.0));
    }

    #[test]
    fn pinned_partner_absorbs_all_clamp() {
        // a pinned => only b is pulled back to the max length.
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [0.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
        assert!((positions[1] - Vec3::new(1.1, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn compression_clamp_pushes_edge_out_to_min_length() {
        // Rest 1, min_scale 0.5 => min length 0.5. Edge compressed to 0.2.
        let mut positions = [Vec3::ZERO, Vec3::new(0.2, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.5);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        let length = (positions[0] - positions[1]).length();
        assert!((length - 0.5).abs() < 1e-5, "length was {length}");
    }

    #[test]
    fn compression_clamp_disabled_by_zero_min_scale() {
        // min_scale 0 => compression is allowed; a short edge is inert.
        let mut positions = [Vec3::ZERO, Vec3::new(0.2, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[1], Vec3::new(0.2, 0.0, 0.0));
    }

    #[test]
    fn out_of_range_handle_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(9), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn degenerate_coincident_particles_are_inert() {
        let mut positions = [Vec3::ZERO, Vec3::ZERO];
        let inv = [1.0, 1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.5);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
        assert_eq!(positions[1], Vec3::ZERO);
    }

    #[test]
    fn same_particle_is_inert() {
        let mut positions = [Vec3::new(5.0, 0.0, 0.0)];
        let inv = [1.0];
        let mut c = StrainLimitConstraint::new(h(0), h(0), 1.0, 1.1, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn from_stretch_limit_sets_max_scale() {
        let c = StrainLimitConstraint::from_stretch_limit(h(0), h(1), 2.0, 0.25);
        assert!((c.max_scale - 1.25).abs() < 1e-6);
        assert!((c.max_length() - 2.5).abs() < 1e-6);
        assert_eq!(c.min_scale, 0.0);
    }

    #[test]
    fn constructor_clamps_scales() {
        // max_scale below 1 is clamped up to 1; min_scale above 1 clamped to 1.
        let c = StrainLimitConstraint::new(h(0), h(1), 1.0, 0.5, 2.0);
        assert_eq!(c.max_scale, 1.0);
        assert_eq!(c.min_scale, 1.0);
    }
}
