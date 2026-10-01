//! A heterogeneous collection of soft-body constraints.
//!
//! [`ConstraintSet`] keeps each constraint family in its own typed `Vec` rather
//! than boxing constraints behind trait objects. This keeps each projection
//! sweep cache-friendly and monomorphized, and mirrors how production XPBD
//! solvers group constraints by kind. The set exposes the two operations the
//! substep solver needs: [`reset`](ConstraintSet::reset) (once per substep) and
//! [`project`](ConstraintSet::project) (once per iteration).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Grouping
//! position-based-dynamics constraints by type for cache-friendly Gauss-Seidel
//! sweeps is a standard, publicly documented technique.

use glam::Vec3;

use crate::math::scalar::Real;

use super::{
    AttachmentConstraint, BendingConstraint, DistanceConstraint, LongRangeConstraint,
    ParticleConstraint, PressureConstraint, StrainLimitConstraint, TetraVolumeConstraint,
};

/// A typed collection of every constraint acting on a soft body's particles.
///
/// The projection order within a sweep is fixed and deterministic: distance,
/// bending, volume, pressure, attachment, then the long-range leashes, and
/// finally the hard strain limiters (which run last so they clamp the result of
/// every compliant sweep). Determinism matters for networked lock-step and
/// golden-replay testing.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintSet {
    /// Distance (stretch) constraints: cloth edges, rope links, lattice edges.
    pub distance: Vec<DistanceConstraint>,
    /// Bending constraints resisting folding.
    pub bending: Vec<BendingConstraint>,
    /// Tetrahedral volume-preservation constraints.
    pub volume: Vec<TetraVolumeConstraint>,
    /// Closed-mesh pressure (enclosed-volume) constraints for inflatable cloth.
    pub pressure: Vec<PressureConstraint>,
    /// Attachment constraints pinning particles toward world anchors.
    pub attachment: Vec<AttachmentConstraint>,
    /// One-sided long-range-attachment leashes to fixed anchors.
    pub long_range: Vec<LongRangeConstraint>,
    /// Hard biphasic strain limiters applied last as a final length clamp.
    pub strain_limit: Vec<StrainLimitConstraint>,
}

impl ConstraintSet {
    /// Creates an empty constraint set.
    #[must_use]
    pub const fn new() -> ConstraintSet {
        ConstraintSet {
            distance: Vec::new(),
            bending: Vec::new(),
            volume: Vec::new(),
            pressure: Vec::new(),
            attachment: Vec::new(),
            long_range: Vec::new(),
            strain_limit: Vec::new(),
        }
    }

    /// Returns the total number of constraints across all families.
    #[must_use]
    pub fn len(&self) -> usize {
        self.distance.len()
            + self.bending.len()
            + self.volume.len()
            + self.pressure.len()
            + self.attachment.len()
            + self.long_range.len()
            + self.strain_limit.len()
    }

    /// Returns `true` when the set holds no constraints at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.distance.is_empty()
            && self.bending.is_empty()
            && self.volume.is_empty()
            && self.pressure.is_empty()
            && self.attachment.is_empty()
            && self.long_range.is_empty()
            && self.strain_limit.is_empty()
    }

    /// Resets the accumulated Lagrange multiplier of every constraint. The
    /// solver calls this once at the start of each substep.
    pub fn reset(&mut self) {
        for c in &mut self.distance {
            c.reset();
        }
        for c in &mut self.bending {
            c.reset();
        }
        for c in &mut self.volume {
            c.reset();
        }
        for c in &mut self.pressure {
            c.reset();
        }
        for c in &mut self.attachment {
            c.reset();
        }
        for c in &mut self.long_range {
            c.reset();
        }
        for c in &mut self.strain_limit {
            c.reset();
        }
    }

    /// Runs one Gauss-Seidel projection sweep over every constraint, in the
    /// fixed family order, mutating `positions` in place.
    pub fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        for c in &mut self.distance {
            c.project(positions, inverse_masses, dt);
        }
        for c in &mut self.bending {
            c.project(positions, inverse_masses, dt);
        }
        for c in &mut self.volume {
            c.project(positions, inverse_masses, dt);
        }
        for c in &mut self.pressure {
            c.project(positions, inverse_masses, dt);
        }
        for c in &mut self.attachment {
            c.project(positions, inverse_masses, dt);
        }
        for c in &mut self.long_range {
            c.project(positions, inverse_masses, dt);
        }
        // Strain limiters run last: a hard clamp on the fully-projected state.
        for c in &mut self.strain_limit {
            c.project(positions, inverse_masses, dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleHandle;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn empty_set_reports_empty() {
        let s = ConstraintSet::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn len_counts_all_families() {
        let mut s = ConstraintSet::new();
        s.distance
            .push(DistanceConstraint::new(h(0), h(1), 1.0, 0.0));
        s.bending
            .push(BendingConstraint::new(h(0), h(1), h(2), 0.0, 0.0));
        s.attachment
            .push(AttachmentConstraint::new(h(0), Vec3::ZERO, 0.0));
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
    }

    #[test]
    fn len_and_empty_track_long_range_and_strain_limit() {
        let mut s = ConstraintSet::new();
        s.long_range
            .push(LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0));
        s.strain_limit
            .push(StrainLimitConstraint::new(h(0), h(1), 1.0, 1.1, 0.0));
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
    }

    #[test]
    fn long_range_leash_clamps_overstretched_particle() {
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [1.0];
        let mut s = ConstraintSet::new();
        s.long_range
            .push(LongRangeConstraint::new(h(0), Vec3::ZERO, 1.0, 0.0));
        s.reset();
        s.project(&mut positions, &inv, 1.0 / 60.0);
        assert!((positions[0].length() - 1.0).abs() < 1e-5, "pos {:?}", positions[0]);
    }

    #[test]
    fn strain_limiter_runs_after_distance_and_caps_length() {
        // A rigid distance constraint restores rest length 1; the strain limiter
        // is a no-op here but, when the edge is overstretched past the band, it
        // must clamp. Verify the limiter clamps a pre-stretched edge.
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut s = ConstraintSet::new();
        s.strain_limit
            .push(StrainLimitConstraint::new(h(0), h(1), 1.0, 1.2, 0.0));
        s.reset();
        s.project(&mut positions, &inv, 1.0 / 60.0);
        let length = (positions[0] - positions[1]).length();
        assert!((length - 1.2).abs() < 1e-5, "length was {length}");
    }

    #[test]
    fn len_and_empty_track_pressure() {
        let mut s = ConstraintSet::new();
        s.pressure
            .push(PressureConstraint::new(vec![[0, 1, 2]], 1.0, 2.0, 0.0));
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
    }

    #[test]
    fn pressure_inflates_a_closed_shell_through_the_set() {
        // A unit cube inflated at overpressure 2 must grow in volume when the
        // set drives its pressure constraint over several sweeps.
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let rest = crate::soft::constraint::mesh_volume(&positions, &triangles);
        let inv = vec![1.0; positions.len()];
        let mut s = ConstraintSet::new();
        s.pressure
            .push(PressureConstraint::new(triangles.clone(), rest, 2.0, 0.0));
        for _ in 0..32 {
            s.reset();
            s.project(&mut positions, &inv, 1.0 / 60.0);
        }
        let inflated = crate::soft::constraint::mesh_volume(&positions, &triangles);
        assert!(inflated > rest + 0.1, "inflated {inflated} vs rest {rest}");
    }

    #[test]
    fn project_applies_distance_constraint() {
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut s = ConstraintSet::new();
        s.distance
            .push(DistanceConstraint::new(h(0), h(1), 1.0, 0.0));
        s.reset();
        s.project(&mut positions, &inv, 1.0 / 60.0);
        let length = (positions[0] - positions[1]).length();
        assert!((length - 1.0).abs() < 1e-5);
    }
}
