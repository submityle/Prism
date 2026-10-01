//! The `CPU` golden twin for the cloth tearing (break-flag) kernel.
//!
//! The authoritative per-edge break decision lives in [`prism_physics_core`] as
//! [`tear_flag`](prism_physics_core::tear_flag); rather than copy that predicate
//! and risk it drifting, [`cpu_cloth_tearing`] *delegates* every edge to it and
//! only owns the per-edge length sample and the torn tally — exactly the work
//! the [`GpuClothTearing`](super::gpu::GpuClothTearing) kernel performs.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the `(len - rest) / rest > break`
//! predicate inline, closing the loop from first principles (no fake parity).
//!
//! # Provenance
//!
//! Removing a constraint whose strain exceeds a threshold is a standard,
//! publicly documented position-based-dynamics technique. No Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{tear_flag, TearingParams};

use super::ClothTearEdge;

/// Runs one tearing (break-flag) pass on the `CPU`, returning the per-edge break
/// flag (`1` = the edge tears, `0` = it survives) and the number of torn edges.
///
/// This is the golden twin of
/// [`GpuClothTearing::solve`](super::gpu::GpuClothTearing::solve): each edge
/// independently samples its endpoint separation from the read-only `positions`,
/// then delegates the break decision to [`prism_physics_core::tear_flag`]. The
/// returned vector is parallel to `edges` — an edge within the break threshold,
/// with a degenerate rest length, or referencing an out-of-range particle keeps
/// a `0` flag.
///
/// An empty edge slice returns an empty vector and a zero count.
#[must_use]
pub fn cpu_cloth_tearing(
    positions: &[Vec3],
    edges: &[ClothTearEdge],
    params: TearingParams,
) -> (Vec<u32>, u32) {
    let break_strain = params.sanitized().break_strain;
    let mut flags = Vec::with_capacity(edges.len());
    let mut torn = 0u32;
    for edge in edges {
        if edge_tears(positions, edge, break_strain) {
            flags.push(1);
            torn += 1;
        } else {
            flags.push(0);
        }
    }
    (flags, torn)
}

/// Returns whether a single `edge` tears at `break_strain`. The break decision
/// is delegated to [`prism_physics_core::tear_flag`]; only the
/// endpoint-separation sample is owned here. An out-of-range endpoint makes the
/// edge inert (it never tears).
#[inline]
fn edge_tears(positions: &[Vec3], edge: &ClothTearEdge, break_strain: f32) -> bool {
    let (Some(pa), Some(pb)) = (
        positions.get(edge.a as usize),
        positions.get(edge.b as usize),
    ) else {
        return false;
    };
    let length = (*pa - *pb).length();
    tear_flag(edge.rest_length, length, break_strain)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Scalar type shared with [`prism_physics_core`] (`f32`).
    type Real = f32;

    /// Independent brute-force reference re-deriving the break predicate inline,
    /// so the delegating twin is anchored against first-principles math rather
    /// than against the same `prism_physics_core` function it calls.
    fn brute_force(
        positions: &[Vec3],
        edges: &[ClothTearEdge],
        params: TearingParams,
    ) -> (Vec<u32>, u32) {
        const EPS_REST: Real = 1e-9;
        let break_strain = params.sanitized().break_strain;
        let mut flags = Vec::with_capacity(edges.len());
        let mut torn = 0u32;
        for e in edges {
            let in_range = (e.a as usize) < positions.len() && (e.b as usize) < positions.len();
            let tears = if in_range && e.rest_length > EPS_REST {
                let len = (positions[e.a as usize] - positions[e.b as usize]).length();
                let strain = (len - e.rest_length) / e.rest_length;
                strain > break_strain
            } else {
                false
            };
            if tears {
                flags.push(1);
                torn += 1;
            } else {
                flags.push(0);
            }
        }
        (flags, torn)
    }

    fn edge(a: u32, b: u32, rest: f32) -> ClothTearEdge {
        ClothTearEdge::new(a, b, rest)
    }

    #[test]
    fn empty_edges_return_empty() {
        let (flags, torn) = cpu_cloth_tearing(&[], &[], TearingParams::default());
        assert!(flags.is_empty());
        assert_eq!(torn, 0);
    }

    #[test]
    fn over_strained_edge_tears() {
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (flags, torn) = cpu_cloth_tearing(&positions, &edges, TearingParams::new(0.5));
        assert_eq!(flags, vec![1]);
        assert_eq!(torn, 1);
    }

    #[test]
    fn within_threshold_survives() {
        let positions = [Vec3::ZERO, Vec3::new(1.4, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (flags, torn) = cpu_cloth_tearing(&positions, &edges, TearingParams::new(0.5));
        assert_eq!(flags, vec![0]);
        assert_eq!(torn, 0);
    }

    #[test]
    fn compression_never_tears() {
        let positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (flags, torn) = cpu_cloth_tearing(&positions, &edges, TearingParams::new(0.5));
        assert_eq!(flags, vec![0]);
        assert_eq!(torn, 0);
    }

    #[test]
    fn degenerate_and_out_of_range_never_tear() {
        let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 0.0), edge(0, 9, 1.0)];
        let (flags, torn) = cpu_cloth_tearing(&positions, &edges, TearingParams::new(0.1));
        assert_eq!(flags, vec![0, 0]);
        assert_eq!(torn, 0);
    }

    #[test]
    fn nan_and_negative_threshold_tear_nothing() {
        let positions = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (flags_nan, torn_nan) =
            cpu_cloth_tearing(&positions, &edges, TearingParams::new(Real::NAN));
        assert_eq!(flags_nan, vec![0]);
        assert_eq!(torn_nan, 0);
        let (flags_neg, torn_neg) = cpu_cloth_tearing(&positions, &edges, TearingParams::new(-1.0));
        assert_eq!(flags_neg, vec![0]);
        assert_eq!(torn_neg, 0);
    }

    #[test]
    fn matches_brute_force_over_mixed_edges() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.3, 0.0),
            Vec3::new(0.1, 1.0, 0.4),
            Vec3::new(-1.5, 0.2, 0.9),
        ];
        let edges = [
            edge(0, 1, 1.0),  // stretched hard -> tears
            edge(1, 2, 1.3),  // at rest
            edge(2, 3, 0.5),  // stretched hard -> tears
            edge(3, 4, 3.0),  // compressed
            edge(0, 4, 0.0),  // degenerate
            edge(0, 42, 1.0), // out of range
        ];
        let params = TearingParams::new(0.4);
        let (flags, torn) = cpu_cloth_tearing(&positions, &edges, params);
        let (ref_flags, ref_torn) = brute_force(&positions, &edges, params);
        assert_eq!(torn, ref_torn);
        assert_eq!(flags, ref_flags);
    }
}
