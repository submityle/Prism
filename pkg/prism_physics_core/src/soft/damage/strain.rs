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

/// Returns the tensile strain `(length - rest_length) / rest_length` for one
/// edge from its current `rest_length` and current `length` (the separation of
/// its two endpoints), or `None` when the rest length is degenerate
/// (`<= EPS_REST`).
///
/// This is the scalar primitive behind both tearing and plasticity: a render
/// (or GPU) caller that already knows an edge's rest length and endpoint
/// separation routes its strain query through here instead of re-deriving the
/// ratio, so every consumer agrees on the same floor and divide guard. A
/// positive value is tension, a negative value is compression.
#[must_use]
pub fn tensile_strain(rest_length: Real, length: Real) -> Option<Real> {
    if rest_length <= EPS_REST {
        return None;
    }
    Some((length - rest_length) / rest_length)
}

/// Returns the current separation `|p_a - p_b|` of a distance `edge`, or `None`
/// when either endpoint index is out of range.
#[must_use]
pub(crate) fn edge_length(edge: &DistanceConstraint, positions: &[Vec3]) -> Option<Real> {
    let pa = positions.get(edge.a.index())?;
    let pb = positions.get(edge.b.index())?;
    Some((*pa - *pb).length())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensile_strain_is_signed_ratio() {
        // Stretched: 50% tension.
        let s = tensile_strain(1.0, 1.5).unwrap();
        assert!((s - 0.5).abs() < 1e-6);
        // Compressed: negative strain.
        let c = tensile_strain(1.0, 0.25).unwrap();
        assert!((c + 0.75).abs() < 1e-6);
    }

    #[test]
    fn tensile_strain_rejects_degenerate_rest() {
        assert_eq!(tensile_strain(0.0, 10.0), None);
        assert_eq!(tensile_strain(EPS_REST, 10.0), None);
        assert!(tensile_strain(EPS_REST * 2.0, 10.0).is_some());
    }
}
