//! Constraint tearing: permanent removal of over-stretched fabric edges.
//!
//! Tearing drops a distance edge from the solve once its tensile strain exceeds
//! a break threshold, so an over-stretched garment rips instead of stretching
//! without bound. This mirrors the constraint-breaking used in production cloth
//! (`Chaos` Cloth, Houdini `Vellum`) where a failed edge is simply removed from
//! the constraint graph, expressed here from first principles.
//!
//! The pass is `O(edges)`, deterministic (edges are visited in list order), and
//! retains surviving edges in their original relative order so any downstream
//! graph colouring stays stable. Tensile strain is `(len - rest) / rest`; a
//! degenerate rest length or an out-of-range endpoint is never torn.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Removing a
//! constraint whose strain exceeds a threshold is a standard, publicly
//! documented position-based-dynamics technique.

use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::constraint::DistanceConstraint;

use super::strain::{edge_length, edge_strain, EPS_REST};

/// Tuning for distance-constraint tearing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TearingParams {
    /// Tensile strain `(len - rest) / rest` above which an edge breaks and is
    /// removed from the graph. Non-negative; compression never tears.
    pub break_strain: Real,
}

impl Default for TearingParams {
    /// A default break strain of 50% elongation.
    fn default() -> Self {
        TearingParams { break_strain: 0.5 }
    }
}

impl TearingParams {
    /// Creates tearing parameters from an explicit break strain.
    #[must_use]
    pub fn new(break_strain: Real) -> Self {
        TearingParams { break_strain }
    }

    /// Returns a copy with `break_strain` forced finite and non-negative, so a
    /// mis-authored `NaN` or negative threshold cannot tear every edge; such a
    /// value is mapped to infinity (nothing tears).
    #[must_use]
    pub fn sanitized(self) -> Self {
        let break_strain = if self.break_strain.is_nan() || self.break_strain < 0.0 {
            Real::INFINITY
        } else {
            self.break_strain
        };
        TearingParams { break_strain }
    }
}

/// A summary of the tensile state of the distance-constraint graph.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TearReport {
    /// Number of edges inspected (edges with a valid rest length and in-range
    /// endpoints).
    pub inspected: usize,
    /// Number of inspected edges whose tensile strain exceeds the threshold.
    pub over_threshold: usize,
    /// The largest tensile strain observed across inspected edges (`0` when none
    /// were inspected).
    pub max_strain: Real,
}

/// Returns, for every constraint in order, whether [`apply_tearing`] would tear
/// it (remove it) at `params`: `true` exactly when the edge has a valid rest
/// length, in-range endpoints, and a tensile strain exceeding
/// `params.break_strain`.
///
/// This is the single source of truth for the tear decision: both
/// [`apply_tearing`] and any GPU tearing kernel consume it, and it is the golden
/// that a per-edge flag kernel is checked against.
#[must_use]
pub fn tear_flags(
    constraints: &[DistanceConstraint],
    positions: &[Vec3],
    params: TearingParams,
) -> Vec<bool> {
    let params = params.sanitized();
    constraints
        .iter()
        .map(|c| match edge_length(c, positions) {
            Some(len) => tear_flag(c.rest_length, len, params.break_strain),
            None => false,
        })
        .collect()
}

/// Decides whether a single distance edge tears, from its current `rest_length`,
/// its current `length` (the separation of its two endpoints), and a
/// `break_strain` threshold.
///
/// Returns `true` exactly when the edge has a valid rest length
/// (`> EPS_REST`) and a tensile strain `(length - rest_length) / rest_length`
/// strictly exceeding `break_strain`; a degenerate rest length never tears.
/// `break_strain` is sanitized internally (a `NaN` or negative threshold maps to
/// infinity, so nothing tears), so the function is safe to call with raw
/// authored values.
///
/// This is the scalar kernel shared by the sequential [`tear_flags`]/
/// [`apply_tearing`] passes and the `prism_physics_gpu` cloth tearing twin, so
/// both agree on the break decision up to floating-point rounding. Out-of-range
/// endpoints are handled by the caller (an inert edge never tears).
#[must_use]
pub fn tear_flag(rest_length: Real, length: Real, break_strain: Real) -> bool {
    let break_strain = if break_strain.is_nan() || break_strain < 0.0 {
        Real::INFINITY
    } else {
        break_strain
    };
    if rest_length <= EPS_REST {
        return false;
    }
    let strain = (length - rest_length) / rest_length;
    strain > break_strain
}

/// Removes every distance edge whose tensile strain exceeds
/// `params.break_strain`, in place, and returns the number of edges torn.
///
/// Degenerate edges and edges with out-of-range endpoints are always kept.
/// Retained edges preserve their original relative order.
pub fn apply_tearing(
    constraints: &mut Vec<DistanceConstraint>,
    positions: &[Vec3],
    params: TearingParams,
) -> usize {
    let flags = tear_flags(constraints, positions, params);
    let before = constraints.len();
    let mut index = 0;
    constraints.retain(|_| {
        let keep = !flags[index];
        index += 1;
        keep
    });
    before - constraints.len()
}

/// Inspects the graph without modifying it and returns a [`TearReport`] of the
/// tensile state against `params.break_strain`.
#[must_use]
pub fn tear_report(
    constraints: &[DistanceConstraint],
    positions: &[Vec3],
    params: TearingParams,
) -> TearReport {
    let params = params.sanitized();
    let mut report = TearReport::default();
    for c in constraints {
        if let Some(strain) = edge_strain(c, positions) {
            report.inspected += 1;
            if strain > report.max_strain {
                report.max_strain = strain;
            }
            if strain > params.break_strain {
                report.over_threshold += 1;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::soft::particle::ParticleHandle;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    fn edge(a: u32, b: u32, rest: Real) -> DistanceConstraint {
        DistanceConstraint::new(h(a), h(b), rest, 0.0)
    }

    #[test]
    fn default_break_strain_is_fifty_percent() {
        assert_eq!(TearingParams::default().break_strain, 0.5);
    }

    #[test]
    fn negative_and_nan_thresholds_sanitize_to_infinity() {
        assert_eq!(
            TearingParams::new(-1.0).sanitized().break_strain,
            Real::INFINITY
        );
        assert_eq!(
            TearingParams::new(Real::NAN).sanitized().break_strain,
            Real::INFINITY
        );
    }

    #[test]
    fn tearing_removes_over_strained_edges_only() {
        let positions = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.05, 0.0),
        ];
        // Edge 0-1: strain 1.0 (rest 1, len 2). Edge 1-2: strain ~0.05.
        let mut constraints = vec![edge(0, 1, 1.0), edge(1, 2, 1.0)];
        let torn = apply_tearing(&mut constraints, &positions, TearingParams::new(0.5));
        assert_eq!(torn, 1);
        assert_eq!(constraints.len(), 1);
        // The surviving edge is the low-strain 1-2 edge.
        assert_eq!(constraints[0].a.index(), 1);
        assert_eq!(constraints[0].b.index(), 2);
    }

    #[test]
    fn compression_never_tears() {
        let positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 1.0)];
        let torn = apply_tearing(&mut constraints, &positions, TearingParams::new(0.5));
        assert_eq!(torn, 0);
        assert_eq!(constraints.len(), 1);
    }

    #[test]
    fn degenerate_and_out_of_range_edges_are_kept() {
        let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let mut constraints = vec![edge(0, 1, 0.0), edge(0, 9, 1.0)];
        let torn = apply_tearing(&mut constraints, &positions, TearingParams::new(0.1));
        assert_eq!(torn, 0);
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn tear_flags_match_apply() {
        let positions = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.05, 0.0),
        ];
        let constraints = vec![edge(0, 1, 1.0), edge(1, 2, 1.0)];
        let flags = tear_flags(&constraints, &positions, TearingParams::new(0.5));
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn tear_report_summarizes_the_graph() {
        let positions = [
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.05, 0.0),
        ];
        let constraints = vec![edge(0, 1, 1.0), edge(1, 2, 1.0), edge(0, 9, 1.0)];
        let report = tear_report(&constraints, &positions, TearingParams::new(0.5));
        assert_eq!(report.inspected, 2);
        assert_eq!(report.over_threshold, 1);
        assert!((report.max_strain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tear_flag_scalar_matches_edge_decision() {
        // Stretched past the threshold tears.
        assert!(tear_flag(1.0, 2.0, 0.5));
        // Within the threshold stays.
        assert!(!tear_flag(1.0, 1.4, 0.5));
        // Compression never tears.
        assert!(!tear_flag(1.0, 0.1, 0.5));
        // Degenerate rest length never tears.
        assert!(!tear_flag(0.0, 10.0, 0.5));
        assert!(!tear_flag(1e-12, 10.0, 0.5));
        // NaN / negative threshold sanitizes to infinity (nothing tears).
        assert!(!tear_flag(1.0, 100.0, Real::NAN));
        assert!(!tear_flag(1.0, 100.0, -1.0));
        // Boundary is strict: strain exactly at the threshold does not tear.
        assert!(!tear_flag(1.0, 1.5, 0.5));
    }
}
