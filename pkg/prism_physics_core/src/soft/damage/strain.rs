//! Shared tensile-strain primitive for the permanent-damage models.
//!
//! Tearing and plasticity both key off the same scalar: the tensile strain of a
//! distance edge, `(len - rest) / rest`. Centralising it here keeps the two
//! models numerically identical (the same floor, the same divide guard) so a
//! torn edge and a plastically creeping edge always agree on what "overstretched"
//! means.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Tensile
//! strain `(len - rest) / rest` is an elementary continuum-mechanics quantity.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::constraint::DistanceConstraint;

/// Numerical floor below which a rest length is treated as degenerate and the
/// edge is skipped, avoiding a divide-by-zero in the strain ratio.
pub(crate) const EPS_REST: Real = 1e-9;

/// Returns the tensile strain `(len - rest) / rest` of a distance `edge` under
/// `positions`, or `None` when the rest length is degenerate or either endpoint
/// index is out of range.
///
/// A positive value is tension (stretched past rest), a negative value is
/// compression. Distance constraints are always two-sided fabric edges, so
/// unlike one-sided leashes they are all eligible for tearing and plastic creep.
#[must_use]
pub(crate) fn edge_strain(edge: &DistanceConstraint, positions: &[Vec3]) -> Option<Real> {
    if edge.rest_length <= EPS_REST {
        return None;
    }
    let a = edge.a.index();
    let b = edge.b.index();
    let pa = positions.get(a)?;
    let pb = positions.get(b)?;
    let len = (*pa - *pb).length();
    Some((len - edge.rest_length) / edge.rest_length)
}

/// Returns the current separation `|p_a - p_b|` of a distance `edge`, or `None`
/// when either endpoint index is out of range.
#[must_use]
pub(crate) fn edge_length(edge: &DistanceConstraint, positions: &[Vec3]) -> Option<Real> {
    let pa = positions.get(edge.a.index())?;
    let pb = positions.get(edge.b.index())?;
    Some((*pa - *pb).length())
}
