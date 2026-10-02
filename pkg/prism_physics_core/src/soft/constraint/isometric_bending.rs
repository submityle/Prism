//! Isometric (quadratic) dihedral bending constraint for triangle-mesh cloth.
//!
//! Where [`super::bending`] holds a three-particle *point-to-midpoint* joint
//! (ideal for ropes, hair, and regular cloth grids), this module provides the
//! production-grade **isometric bending energy** used for arbitrary
//! triangle-mesh garments. Every interior edge shared by two triangles becomes
//! a four-vertex hinge stencil `[edge0, edge1, apex_a, apex_b]`. From the rest
//! geometry we precompute a per-stencil cotangent-Laplacian weight vector `w`
//! and an area scale such that the bend measure `S = Σ wᵢ · xᵢ` vanishes when
//! the stencil is flat and grows as it folds. For a developable (planar-rest)
//! panel the energy `E = ½ · scale · |S|²` is rotation invariant, so rigidly
//! rotating the whole stencil leaves `S = 0`.
//!
//! The model resists bending without ever evaluating a dihedral *angle* — only
//! dot, cross, and `sqrt` — so it is portable and deterministic across
//! platforms, which is exactly what a parallel `GPU` twin needs. This is the
//! single authoritative implementation of isometric bending for the whole
//! engine; CPU golden, `GPU` twin, and the render pipeline all project through
//! [`project_isometric_bending`] so their arithmetic can never drift.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! isometric (quadratic) bending energy is the publicly documented Bergou et
//! al. 2006 model, and the compliant projection is the standard Müller et al.
//! `XPBD` formulation.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::{ParticleConstraint, SoftConstraintKind};

/// Squared-length epsilon below which a vector is treated as degenerate.
const EPS_LEN_SQ: Real = 1e-12;

/// One four-vertex isometric bending hinge across an interior edge.
///
/// `vertices` is `[edge0, edge1, apex_a, apex_b]`: the two endpoints of the
/// shared edge followed by the opposite vertex of each adjoining triangle.
/// `weights` is the precomputed cotangent-Laplacian stencil (`w` above) and
/// `scale` folds in the hinge area; together they define the flat-rest bending
/// energy. `compliance` is the `XPBD` inverse stiffness (`0` = perfectly stiff).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct IsometricBendingConstraint {
    /// Particle indices `[edge0, edge1, apex_a, apex_b]`.
    pub vertices: [u32; 4],
    /// Per-vertex bend-Laplacian weights, aligned with `vertices`.
    pub weights: [Real; 4],
    /// Area-derived energy scale (`3 / (area_a + area_b)`).
    pub scale: Real,
    /// `XPBD` compliance (inverse bending stiffness).
    pub compliance: Real,
}

impl IsometricBendingConstraint {
    /// Returns the current bend vector `S = Σ wᵢ · xᵢ` for this stencil.
    ///
    /// `S` is [`Vec3::ZERO`] for a flat (or rigidly transformed flat) stencil
    /// and grows with the fold; its squared length drives the energy. Indices
    /// outside `positions` contribute nothing rather than panicking.
    #[must_use]
    pub fn bend_vector(&self, positions: &[Vec3]) -> Vec3 {
        let mut s = Vec3::ZERO;
        for (idx, weight) in self.vertices.iter().zip(self.weights.iter()) {
            if let Some(p) = positions.get(*idx as usize) {
                s += *p * *weight;
            }
        }
        s
    }

    /// Returns the scalar bending energy `E = ½ · scale · |S|²` of this stencil.
    #[must_use]
    pub fn energy(&self, positions: &[Vec3]) -> Real {
        0.5 * self.scale * self.bend_vector(positions).length_squared()
    }
}

impl ParticleConstraint for IsometricBendingConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Bending
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        // The isometric projection is stateless (it recomputes the full energy
        // each call), so there is no accumulated multiplier to clear.
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let _ = project_isometric_bending(
            positions,
            inverse_masses,
            self.vertices,
            self.weights,
            self.scale,
            self.compliance,
            dt,
        );
    }
}

/// Cotangent of the angle between `a` and `b`, `cot θ = (a·b) / |a×b|`.
///
/// Returns `0` for a degenerate (collinear or zero-length) pair so a sliver
/// triangle can never produce a non-finite weight.
fn cot_angle(a: Vec3, b: Vec3) -> Real {
    let cross_len_sq = a.cross(b).length_squared();
    if cross_len_sq <= EPS_LEN_SQ {
        return 0.0;
    }
    a.dot(b) / cross_len_sq.sqrt()
}

/// Builds the isometric-bending stencil for one interior edge from rest
/// positions, or `None` when the hinge is degenerate (zero total area).
fn build_hinge(
    positions: &[Vec3],
    edge0: u32,
    edge1: u32,
    apex_a: u32,
    apex_b: u32,
    compliance: Real,
) -> Option<IsometricBendingConstraint> {
    let x0 = *positions.get(edge0 as usize)?;
    let x1 = *positions.get(edge1 as usize)?;
    let x2 = *positions.get(apex_a as usize)?;
    let x3 = *positions.get(apex_b as usize)?;

    let e0 = x1 - x0;
    let e1 = x2 - x0;
    let e2 = x3 - x0;
    let e3 = x2 - x1;
    let e4 = x3 - x1;
    let neg_e0 = -e0;

    let c01 = cot_angle(e0, e1);
    let c02 = cot_angle(e0, e2);
    let c03 = cot_angle(neg_e0, e3);
    let c04 = cot_angle(neg_e0, e4);

    // Triangle areas: ½|e0×e1| and ½|e0×e2|.
    let area_a = 0.5 * e0.cross(e1).length();
    let area_b = 0.5 * e0.cross(e2).length();
    let area = area_a + area_b;
    if area <= EPS_LEN_SQ.sqrt() {
        return None;
    }

    let weights = [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04];
    let scale = 3.0 / area;
    Some(IsometricBendingConstraint {
        vertices: [edge0, edge1, apex_a, apex_b],
        weights,
        scale,
        compliance: compliance.max(0.0),
    })
}

/// Normalizes an unordered edge into a `(min, max)` key for adjacency lookup.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Builds one isometric bending hinge per interior edge of a triangle mesh.
///
/// `positions` are the rest-pose particle positions; `triangles` are index
/// triples into `positions`. Every edge shared by exactly two triangles yields
/// one [`IsometricBendingConstraint`]; boundary edges (one triangle) and
/// non-manifold edges (three or more triangles) are skipped, so an open panel
/// or a messy authored mesh degrades gracefully instead of panicking.
/// Degenerate hinges (zero area) are dropped. The output order is
/// deterministic: hinges are emitted in ascending `(min, max)` edge order.
#[must_use]
pub fn build_dihedral_bending(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    compliance: Real,
) -> Vec<IsometricBendingConstraint> {
    // Map each undirected edge to the apex vertices of its incident triangles.
    let mut edges: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
    for tri in triangles {
        let [a, b, c] = *tri;
        // Skip degenerate triangles with a repeated vertex.
        if a == b || b == c || a == c {
            continue;
        }
        edges.entry(edge_key(a, b)).or_default().push(c);
        edges.entry(edge_key(b, c)).or_default().push(a);
        edges.entry(edge_key(a, c)).or_default().push(b);
    }

    let mut constraints = Vec::new();
    for ((edge0, edge1), apexes) in &edges {
        if apexes.len() != 2 {
            continue;
        }
        if let Some(hinge) =
            build_hinge(positions, *edge0, *edge1, apexes[0], apexes[1], compliance)
        {
            constraints.push(hinge);
        }
    }
    constraints
}

/// One stateless compliant `XPBD` projection of the isometric bending energy,
/// written over raw particle indices so a parallel `GPU` twin and the render
/// pipeline can share byte-identical arithmetic from this single source.
///
/// The constraint value is the bending energy `C = ½ · scale · |S|²`, its
/// per-vertex gradient is `scale · wᵢ · S`, and the mass-weighted correction
/// `Δxᵢ = wᵢ⁻¹ · Δλ · gradᵢ` pulls the stencil back toward flat. Pinned
/// particles (`inverse_mass ≤ 0`) never move, and a flat or degenerate stencil
/// is a no-op. Returns the energy that was corrected (before the step) so a
/// caller can monitor convergence.
#[must_use]
pub fn project_isometric_bending(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    vertices: [u32; 4],
    weights: [Real; 4],
    scale: Real,
    compliance: Real,
    dt: Real,
) -> Real {
    if dt <= 0.0 {
        return 0.0;
    }
    // S = Σ wᵢ · xᵢ.
    let mut s = Vec3::ZERO;
    for (idx, weight) in vertices.iter().zip(weights.iter()) {
        if let Some(p) = positions.get(*idx as usize) {
            s += *p * *weight;
        }
    }
    let s_len_sq = s.length_squared();
    if s_len_sq <= EPS_LEN_SQ {
        return 0.0;
    }
    let energy = 0.5 * scale * s_len_sq;

    // Denominator Σ wmassᵢ · |gradᵢ|² where gradᵢ = scale·wᵢ·S.
    let mut sum_w_grad = 0.0;
    for (idx, weight) in vertices.iter().zip(weights.iter()) {
        let inv_mass = inverse_masses.get(*idx as usize).copied().unwrap_or(0.0);
        if inv_mass <= 0.0 {
            continue;
        }
        let grad_scalar = scale * *weight;
        sum_w_grad += inv_mass * grad_scalar * grad_scalar * s_len_sq;
    }
    let alpha_tilde = compliance / (dt * dt);
    let denom = sum_w_grad + alpha_tilde;
    if denom <= 0.0 {
        return energy;
    }
    let d_lambda = -energy / denom;

    for (idx, weight) in vertices.iter().zip(weights.iter()) {
        let index = *idx as usize;
        let inv_mass = inverse_masses.get(index).copied().unwrap_or(0.0);
        if inv_mass <= 0.0 {
            continue;
        }
        // gradᵢ = scale·wᵢ·S ; Δxᵢ = wmassᵢ·Δλ·gradᵢ.
        if let Some(p) = positions.get_mut(index) {
            *p += s * (inv_mass * d_lambda * scale * *weight);
        }
    }
    energy
}

/// Runs `iterations` Gauss-Seidel sweeps of [`project_isometric_bending`] over
/// a hinge set, returning the total bending energy corrected on the first
/// sweep.
///
/// `iterations` is clamped to at least one; the sweep order follows the
/// constraint slice, so the result is deterministic.
#[must_use]
pub fn apply_isometric_bending(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    constraints: &[IsometricBendingConstraint],
    iterations: u32,
    dt: Real,
) -> Real {
    let iterations = iterations.max(1);
    let mut first_energy = 0.0;
    for iteration in 0..iterations {
        for c in constraints {
            let energy = project_isometric_bending(
                positions,
                inverse_masses,
                c.vertices,
                c.weights,
                c.scale,
                c.compliance,
                dt,
            );
            if iteration == 0 {
                first_energy += energy;
            }
        }
    }
    first_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Two triangles sharing the diagonal edge `1-2` of a unit square in the
    /// `XY` plane: vertices `0=(0,0)`, `1=(1,0)`, `2=(0,1)`, `3=(1,1)`.
    fn flat_quad() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2], [1, 3, 2]];
        (positions, triangles)
    }

    #[test]
    fn build_emits_one_hinge_for_shared_edge() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        assert_eq!(hinges.len(), 1);
        assert_eq!(hinges[0].vertices, [1, 2, 0, 3]);
    }

    #[test]
    fn boundary_and_nonmanifold_edges_are_skipped() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let hinges = build_dihedral_bending(&positions, &[[0, 1, 2]], 0.0);
        assert!(hinges.is_empty());

        let fan_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let fan = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let fan_hinges = build_dihedral_bending(&fan_positions, &fan, 0.0);
        assert!(fan_hinges.iter().all(|h| h.vertices[0..2] != [0, 1]));
    }

    #[test]
    fn flat_stencil_has_zero_energy() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        assert!(hinges[0].energy(&positions) < 1.0e-9);
    }

    #[test]
    fn folding_raises_energy_and_projection_reduces_it() {
        let (mut positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        positions[3] = Vec3::new(1.0, 1.0, 0.6);
        let inv = vec![1.0; positions.len()];

        let before = hinges[0].energy(&positions);
        assert!(before > 1.0e-4);

        let returned = project_isometric_bending(
            &mut positions,
            &inv,
            hinges[0].vertices,
            hinges[0].weights,
            hinges[0].scale,
            hinges[0].compliance,
            1.0 / 60.0,
        );
        assert!((returned - before).abs() < 1.0e-5);
        let after = hinges[0].energy(&positions);
        assert!(after < before);
    }

    #[test]
    fn energy_is_invariant_under_rigid_rotation() {
        let (mut positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        positions[3] = Vec3::new(1.0, 1.0, 0.6);
        let base_energy = hinges[0].energy(&positions);

        // Rotate every particle 90 degrees about Y: (x,y,z) -> (z,y,-x).
        let rotated: Vec<Vec3> = positions
            .iter()
            .map(|p| Vec3::new(p.z, p.y, -p.x))
            .collect();
        let rotated_energy = hinges[0].energy(&rotated);
        assert!((base_energy - rotated_energy).abs() < 1.0e-5);
    }

    #[test]
    fn pinned_particles_never_move() {
        let (mut positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        positions[3] = Vec3::new(1.0, 1.0, 0.6);
        let inv = vec![0.0; positions.len()];
        let snapshot = positions.clone();
        let _ = project_isometric_bending(
            &mut positions,
            &inv,
            hinges[0].vertices,
            hinges[0].weights,
            hinges[0].scale,
            hinges[0].compliance,
            1.0 / 60.0,
        );
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn apply_is_deterministic_and_converges() {
        let (mut positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        positions[3] = Vec3::new(1.0, 1.0, 0.6);
        let inv = vec![1.0; positions.len()];

        let mut a = positions.clone();
        let mut b = positions.clone();
        let first = apply_isometric_bending(&mut a, &inv, &hinges, 8, 1.0 / 60.0);
        let first_again = apply_isometric_bending(&mut b, &inv, &hinges, 8, 1.0 / 60.0);
        assert_eq!(a, b);
        assert!(first > 0.0);
        assert!((first - first_again).abs() < 1.0e-9);
        assert!(hinges[0].energy(&a) < first);
    }

    #[test]
    fn empty_mesh_yields_no_constraints() {
        let hinges = build_dihedral_bending(&[], &[], 0.0);
        assert!(hinges.is_empty());
    }

    #[test]
    fn trait_project_matches_free_function() {
        let (mut positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, 0.0);
        positions[3] = Vec3::new(1.0, 1.0, 0.6);
        let inv = vec![1.0; positions.len()];

        let mut via_trait = positions.clone();
        let mut c = hinges[0];
        c.reset();
        c.project(&mut via_trait, &inv, 1.0 / 60.0);

        let mut via_free = positions.clone();
        let _ = project_isometric_bending(
            &mut via_free,
            &inv,
            hinges[0].vertices,
            hinges[0].weights,
            hinges[0].scale,
            hinges[0].compliance,
            1.0 / 60.0,
        );
        assert_eq!(via_trait, via_free);
    }
}
