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
    AttachmentConstraint, BendingConstraint, DistanceConstraint, ParticleConstraint,
    TetraVolumeConstraint,
};

/// A typed collection of every constraint acting on a soft body's particles.
///
/// The projection order within a sweep is fixed and deterministic: distance,
/// then bending, then volume, then attachment constraints. Determinism matters
/// for networked lock-step and golden-replay testing.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintSet {
    /// Distance (stretch) constraints: cloth edges, rope links, lattice edges.
    pub distance: Vec<DistanceConstraint>,
    /// Bending constraints resisting folding.
    pub bending: Vec<BendingConstraint>,
    /// Tetrahedral volume-preservation constraints.
    pub volume: Vec<TetraVolumeConstraint>,
    /// Attachment constraints pinning particles toward world anchors.
    pub attachment: Vec<AttachmentConstraint>,
}

impl ConstraintSet {
    /// Creates an empty constraint set.
    #[must_use]
    pub const fn new() -> ConstraintSet {
        ConstraintSet {
            distance: Vec::new(),
            bending: Vec::new(),
            volume: Vec::new(),
            attachment: Vec::new(),
        }
    }

    /// Returns the total number of constraints across all families.
    #[must_use]
    pub fn len(&self) -> usize {
        self.distance.len() + self.bending.len() + self.volume.len() + self.attachment.len()
    }

    /// Returns `true` when the set holds no constraints at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.distance.is_empty()
            && self.bending.is_empty()
            && self.volume.is_empty()
            && self.attachment.is_empty()
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
        for c in &mut self.attachment {
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
        for c in &mut self.attachment {
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
