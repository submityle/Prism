//! The `CPU` golden twin for the cloth plasticity kernel.
//!
//! The authoritative per-edge creep arithmetic lives in [`prism_physics_core`]
//! as [`plastic_rest_length`](prism_physics_core::plastic_rest_length); rather
//! than copy that math and risk it drifting, [`cpu_cloth_plasticity`] *delegates*
//! every edge to it and only owns the per-edge length sample and the modified
//! tally — exactly the work the
//! [`GpuClothPlasticity`](super::gpu::GpuClothPlasticity) kernel performs.
//!
//! The twin is in turn anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the creep-and-clamp formula inline,
//! closing the loop from first principles (no fake parity).
//!
//! # Provenance
//!
//! Rest-length creep past a yield strain is a standard, publicly documented
//! plastic-set model for position-based cloth. No Unreal Engine source or
//! derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{plastic_rest_length, PlasticParams};

use super::ClothPlasticEdge;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs one plasticity pass on the `CPU`, returning the per-edge updated rest
/// lengths and the number of edges that plastically crept.
///
/// This is the golden twin of
/// [`GpuClothPlasticity::solve`](super::gpu::GpuClothPlasticity::solve): each
/// edge independently samples its endpoint separation from the read-only
/// `positions`, then delegates the creep-and-clamp decision to
/// [`prism_physics_core::plastic_rest_length`]. The returned vector is parallel
/// to `edges` — an edge that stays within the yield band, has a degenerate rest
/// length, or references an out-of-range particle keeps its input rest length.
///
/// An empty edge slice returns an empty vector and a zero count.
#[must_use]
pub fn cpu_cloth_plasticity(
    positions: &[Vec3],
    edges: &[ClothPlasticEdge],
    params: PlasticParams,
) -> (Vec<Real>, u32) {
    let mut rest_lengths = Vec::with_capacity(edges.len());
    let mut modified = 0u32;
    for edge in edges {
        match edge_creep(positions, edge, params) {
            Some(new_rest) => {
                modified += 1;
                rest_lengths.push(new_rest);
            }
            None => rest_lengths.push(edge.rest_length),
        }
    }
    (rest_lengths, modified)
}

/// Returns `Some(new_rest)` when a single `edge` plastically creeps, or `None`
/// when it does not (out-of-range endpoint, degenerate rest length, within the
/// yield band, or clamped collapse). The creep decision is delegated to
/// [`prism_physics_core::plastic_rest_length`]; only the endpoint-separation
/// sample is owned here.
#[inline]
fn edge_creep(positions: &[Vec3], edge: &ClothPlasticEdge, params: PlasticParams) -> Option<Real> {
    let pa = positions.get(edge.a as usize)?;
    let pb = positions.get(edge.b as usize)?;
    let length = (*pa - *pb).length();
    plastic_rest_length(edge.rest_length, length, params)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Independent brute-force reference re-deriving the creep-and-clamp formula
    /// inline, so the delegating twin is anchored against first-principles math
    /// rather than against the same `prism_physics_core` function it calls.
    fn brute_force(
        positions: &[Vec3],
        edges: &[ClothPlasticEdge],
        params: PlasticParams,
    ) -> (Vec<Real>, u32) {
        const EPS_REST: Real = 1e-9;
        let p = params.sanitized();
        let mut out = Vec::with_capacity(edges.len());
        let mut modified = 0u32;
        for e in edges {
            let mut rest = e.rest_length;
            let in_range = (e.a as usize) < positions.len() && (e.b as usize) < positions.len();
            if in_range && rest > EPS_REST {
                let len = (positions[e.a as usize] - positions[e.b as usize]).length();
                let strain = (len - rest) / rest;
                if strain.abs() > p.yield_strain {
                    let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
                    let excess = strain - sign * p.yield_strain;
                    let mut new_rest = rest * (1.0 + p.creep * excess);
                    if new_rest <= EPS_REST {
                        new_rest = EPS_REST;
                    }
                    let residual = (len - new_rest) / new_rest;
                    if residual.abs() > p.max_strain {
                        let rsign = if residual >= 0.0 { 1.0 } else { -1.0 };
                        new_rest = len / (1.0 + rsign * p.max_strain);
                    }
                    if new_rest > EPS_REST {
                        rest = new_rest;
                        modified += 1;
                    }
                }
            }
            out.push(rest);
        }
        (out, modified)
    }

    fn edge(a: u32, b: u32, rest: f32) -> ClothPlasticEdge {
        ClothPlasticEdge::new(a, b, rest)
    }

    #[test]
    fn empty_edges_return_empty() {
        let (rest, modified) = cpu_cloth_plasticity(&[], &[], PlasticParams::default());
        assert!(rest.is_empty());
        assert_eq!(modified, 0);
    }

    #[test]
    fn within_yield_band_is_untouched() {
        let positions = [Vec3::ZERO, Vec3::new(1.05, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (rest, modified) =
            cpu_cloth_plasticity(&positions, &edges, PlasticParams::new(0.1, 0.5, 1.0));
        assert_eq!(modified, 0);
        assert_eq!(rest, vec![1.0]);
    }

    #[test]
    fn beyond_yield_creeps_up() {
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (rest, modified) =
            cpu_cloth_plasticity(&positions, &edges, PlasticParams::new(0.1, 0.5, 10.0));
        assert_eq!(modified, 1);
        // excess = 1.0 - 0.1 = 0.9; new_rest = 1 * (1 + 0.5*0.9) = 1.45.
        assert!((rest[0] - 1.45).abs() < 1e-5, "rest {}", rest[0]);
    }

    #[test]
    fn out_of_range_and_degenerate_are_untouched() {
        let positions = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 0.0), edge(0, 9, 1.0)];
        let (rest, modified) =
            cpu_cloth_plasticity(&positions, &edges, PlasticParams::new(0.1, 0.5, 1.0));
        assert_eq!(modified, 0);
        assert_eq!(rest, vec![0.0, 1.0]);
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
            edge(0, 1, 1.0),  // stretched
            edge(1, 2, 1.3),  // at rest
            edge(2, 3, 0.5),  // stretched hard
            edge(3, 4, 3.0),  // compressed
            edge(0, 4, 0.0),  // degenerate
            edge(0, 42, 1.0), // out of range
        ];
        let params = PlasticParams::new(0.08, 0.4, 0.25);
        let (rest, modified) = cpu_cloth_plasticity(&positions, &edges, params);
        let (ref_rest, ref_modified) = brute_force(&positions, &edges, params);
        assert_eq!(modified, ref_modified);
        assert_eq!(rest.len(), ref_rest.len());
        for (i, (g, r)) in rest.iter().zip(ref_rest.iter()).enumerate() {
            assert!((g - r).abs() < 1e-6, "edge {i}: golden {g} vs brute {r}");
        }
    }

    #[test]
    fn residual_strain_is_capped() {
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let edges = [edge(0, 1, 1.0)];
        let (rest, _) = cpu_cloth_plasticity(&positions, &edges, PlasticParams::new(0.1, 1.0, 0.2));
        let residual = (2.0 - rest[0]) / rest[0];
        assert!(residual.abs() <= 0.2 + 1e-5, "residual {residual}");
    }
}
