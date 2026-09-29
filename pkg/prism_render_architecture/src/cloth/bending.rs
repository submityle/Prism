//! Dihedral / isometric bending constraints for triangle-mesh cloth.
//!
//! Stretch and shear (see [`super::constraints`]) hold a garment's in-plane
//! shape; *bending* controls how sharply it can fold along an interior edge —
//! the difference between stiff felt and limp silk. Production cloth engines
//! (`Bridson`'s 2003 dihedral model, the `Bergou` 2006 isometric-bending model
//! used by Houdini `Vellum` and UE5 `Chaos` Cloth) resist bending without ever
//! evaluating a dihedral *angle*, so no `acos`/`atan2` is needed — only dot,
//! cross and `sqrt`, which is exactly what this dependency-free crate allows.
//!
//! The model here is the isometric (quadratic) bending energy. Every interior
//! edge shared by two triangles becomes a four-vertex hinge stencil
//! `[edge0, edge1, apex_a, apex_b]`. From the *rest* geometry we precompute a
//! per-stencil weight vector `w` (a discrete Laplacian built from cotangents)
//! and an area scale, such that the bend measure `S = Σ wᵢ · xᵢ` vanishes when
//! the stencil is flat and grows as it folds. Because a developable garment
//! panel is planar at rest (which is exactly what [`super::constraints`] and
//! [`super::pipeline`] build), the rest bend is zero, so the energy
//! `E = ½ · scale · |S|²` is rotation invariant: rigidly rotating the whole
//! stencil leaves `S = 0`. This is the standard cloth-bending assumption.
//!
//! [`project_bending`] is an XPBD projection in the same stateless,
//! compliance-parameterized form as [`super::dynamics`]'s distance solve, so
//! bending drops straight into the existing per-color / per-substep loop and
//! maps cleanly onto a GPU dispatch.

use alloc::vec::Vec;

use super::{ClothParticle, Compliance, Vec3, EPS_LEN_SQ};

/// One four-vertex bending hinge across an interior edge.
///
/// `vertices` is `[edge0, edge1, apex_a, apex_b]`: the two endpoints of the
/// shared edge followed by the opposite vertex of each adjoining triangle.
/// `weights` is the precomputed cotangent Laplacian stencil (`w` above) and
/// `scale` folds in the hinge area; together they define the flat-rest bending
/// energy. `compliance` is the XPBD inverse stiffness (`0` = perfectly stiff).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BendingConstraint {
    /// Particle indices `[edge0, edge1, apex_a, apex_b]`.
    pub vertices: [u32; 4],
    /// Per-vertex bend-Laplacian weights, aligned with `vertices`.
    pub weights: [f32; 4],
    /// Area-derived energy scale (`3 / (area_a + area_b)`).
    pub scale: f32,
    /// XPBD compliance (inverse bending stiffness).
    pub compliance: Compliance,
}

impl BendingConstraint {
    /// Returns the current bend vector `S = Σ wᵢ · xᵢ` for this stencil.
    ///
    /// `S` is [`Vec3::ZERO`] for a flat (or rigidly transformed flat) stencil
    /// and grows with the fold; its squared length drives the energy. Indices
    /// outside `particles` contribute nothing rather than panicking.
    #[must_use]
    pub fn bend_vector(self, particles: &[ClothParticle]) -> Vec3 {
        let mut s = Vec3::ZERO;
        for (idx, weight) in self.vertices.iter().zip(self.weights.iter()) {
            if let Some(particle) = particles.get(*idx as usize) {
                s = s.add(particle.position.scale(*weight));
            }
        }
        s
    }

    /// Returns the scalar bending energy `E = ½ · scale · |S|²` of this stencil.
    #[must_use]
    pub fn energy(self, particles: &[ClothParticle]) -> f32 {
        0.5 * self.scale * self.bend_vector(particles).length_squared()
    }
}

/// Cotangent of the angle between `a` and `b`, `cot θ = (a·b) / |a×b|`.
///
/// Returns `0` for a degenerate (collinear or zero-length) pair so a sliver
/// triangle can never produce a non-finite weight.
fn cot_angle(a: Vec3, b: Vec3) -> f32 {
    let cross_len_sq = a.cross(b).length_squared();
    if cross_len_sq <= EPS_LEN_SQ {
        return 0.0;
    }
    a.dot(b) / cross_len_sq.sqrt()
}

/// Builds the isometric-bending stencil for one interior edge from rest
/// positions, or `None` when the hinge is degenerate (zero total area).
///
/// `edge0`/`edge1` are the shared-edge endpoints; `apex_a`/`apex_b` are the
/// opposite vertices of the two triangles. The returned constraint stores the
/// cotangent weights and area scale evaluated at this rest pose.
fn build_hinge(
    positions: &[Vec3],
    edge0: u32,
    edge1: u32,
    apex_a: u32,
    apex_b: u32,
    compliance: Compliance,
) -> Option<BendingConstraint> {
    let x0 = *positions.get(edge0 as usize)?;
    let x1 = *positions.get(edge1 as usize)?;
    let x2 = *positions.get(apex_a as usize)?;
    let x3 = *positions.get(apex_b as usize)?;

    let e0 = x1.sub(x0);
    let e1 = x2.sub(x0);
    let e2 = x3.sub(x0);
    let e3 = x2.sub(x1);
    let e4 = x3.sub(x1);
    let neg_e0 = Vec3::ZERO.sub(e0);

    let c01 = cot_angle(e0, e1);
    let c02 = cot_angle(e0, e2);
    let c03 = cot_angle(neg_e0, e3);
    let c04 = cot_angle(neg_e0, e4);

    let area_a = 0.5 * e0.cross(e1).length();
    let area_b = 0.5 * e0.cross(e2).length();
    let area = area_a + area_b;
    if area <= EPS_LEN_SQ.sqrt() {
        return None;
    }

    let weights = [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04];
    let scale = 3.0 / area;
    Some(BendingConstraint {
        vertices: [edge0, edge1, apex_a, apex_b],
        weights,
        scale,
        compliance,
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
/// one [`BendingConstraint`]; boundary edges (one triangle) and non-manifold
/// edges (three or more triangles) are skipped, so an open panel or a messy
/// authored mesh degrades gracefully instead of panicking. Degenerate hinges
/// (zero area) are dropped. The output order is deterministic: hinges are
/// emitted in ascending `(min, max)` edge order.
#[must_use]
pub fn build_dihedral_bending(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    compliance: Compliance,
) -> Vec<BendingConstraint> {
    use alloc::collections::BTreeMap;

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

/// Projects one bending constraint in place with a single stateless XPBD step.
///
/// Mirrors the distance projection in [`super::dynamics`]: the constraint value
/// is the bending energy `C = ½ · scale · |S|²`, its per-vertex gradient is
/// `scale · wᵢ · S`, and the mass-weighted XPBD correction
/// `Δxᵢ = wᵢ⁻¹ · Δλ · gradᵢ` pulls the stencil back toward flat. Pinned
/// particles (`inverse_mass ≤ 0`) never move, and a flat or degenerate stencil
/// is a no-op. Returns the energy that was corrected (before the step) so a
/// caller can monitor convergence.
pub fn project_bending(
    particles: &mut [ClothParticle],
    constraint: BendingConstraint,
    dt_sub: f32,
) -> f32 {
    if dt_sub <= 0.0 {
        return 0.0;
    }
    let s = constraint.bend_vector(particles);
    let s_len_sq = s.length_squared();
    if s_len_sq <= EPS_LEN_SQ {
        return 0.0;
    }
    let energy = 0.5 * constraint.scale * s_len_sq;

    // Accumulate the denominator Σ wmassᵢ · |gradᵢ|² where gradᵢ = scale·wᵢ·S.
    let mut sum_w_grad = 0.0;
    for (idx, weight) in constraint.vertices.iter().zip(constraint.weights.iter()) {
        let Some(particle) = particles.get(*idx as usize) else {
            continue;
        };
        let inv_mass = effective_inverse_mass(*particle);
        if inv_mass <= 0.0 {
            continue;
        }
        let grad_scalar = constraint.scale * *weight;
        sum_w_grad += inv_mass * grad_scalar * grad_scalar * s_len_sq;
    }
    let alpha_tilde = constraint.compliance.value() / (dt_sub * dt_sub);
    let denom = sum_w_grad + alpha_tilde;
    if denom <= 0.0 {
        return energy;
    }
    let d_lambda = -energy / denom;

    for (idx, weight) in constraint.vertices.iter().zip(constraint.weights.iter()) {
        let index = *idx as usize;
        let Some(particle) = particles.get(index) else {
            continue;
        };
        let inv_mass = effective_inverse_mass(*particle);
        if inv_mass <= 0.0 {
            continue;
        }
        // gradᵢ = scale·wᵢ·S ; Δxᵢ = wmassᵢ·Δλ·gradᵢ.
        let correction = s.scale(inv_mass * d_lambda * constraint.scale * *weight);
        particles[index].position = particles[index].position.add(correction);
    }
    energy
}

/// Runs `iterations` Gauss-Seidel sweeps of [`project_bending`] over a hinge
/// set, returning the total bending energy corrected on the first sweep.
///
/// This is the batch entry point the pipeline calls per substep after the
/// stretch/shear projection. `iterations` is clamped to at least one; the sweep
/// order follows the constraint slice, so the result is deterministic.
pub fn apply_bending(
    particles: &mut [ClothParticle],
    constraints: &[BendingConstraint],
    iterations: u32,
    dt_sub: f32,
) -> f32 {
    let iterations = iterations.max(1);
    let mut first_energy = 0.0;
    for iteration in 0..iterations {
        for constraint in constraints {
            let energy = project_bending(particles, *constraint, dt_sub);
            if iteration == 0 {
                first_energy += energy;
            }
        }
    }
    first_energy
}

/// Effective inverse mass of a particle: zero when pinned.
fn effective_inverse_mass(particle: ClothParticle) -> f32 {
    if particle.is_pinned() {
        0.0
    } else {
        particle.inverse_mass
    }
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

    fn particles_from(positions: &[Vec3]) -> Vec<ClothParticle> {
        positions
            .iter()
            .map(|p| ClothParticle::new(*p, 1.0))
            .collect()
    }

    #[test]
    fn build_emits_one_hinge_for_shared_edge() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);
        assert_eq!(hinges.len(), 1);
        // The shared edge is 1-2; apexes are 0 and 3 in ascending triangle order.
        assert_eq!(hinges[0].vertices, [1, 2, 0, 3]);
    }

    #[test]
    fn boundary_and_nonmanifold_edges_are_skipped() {
        // A single triangle has only boundary edges -> no hinge.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let hinges = build_dihedral_bending(&positions, &[[0, 1, 2]], Compliance::RIGID);
        assert!(hinges.is_empty());

        // Three triangles fanning one edge (0-1) is non-manifold -> skipped.
        let fan_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let fan = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let fan_hinges = build_dihedral_bending(&fan_positions, &fan, Compliance::RIGID);
        assert!(fan_hinges.iter().all(|h| h.vertices[0..2] != [0, 1]));
    }

    #[test]
    fn flat_stencil_has_zero_energy() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);
        let particles = particles_from(&positions);
        assert!(hinges[0].energy(&particles) < 1.0e-9);
    }

    #[test]
    fn folding_raises_energy_and_projection_reduces_it() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);
        let mut particles = particles_from(&positions);
        // Fold apex 3 out of plane.
        particles[3].position = Vec3::new(1.0, 1.0, 0.6);

        let before = hinges[0].energy(&particles);
        assert!(before > 1.0e-4);

        project_bending(&mut particles, hinges[0], 1.0 / 60.0);
        let after = hinges[0].energy(&particles);
        assert!(after < before);
    }

    #[test]
    fn energy_is_invariant_under_rigid_rotation() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);
        let mut folded = particles_from(&positions);
        folded[3].position = Vec3::new(1.0, 1.0, 0.6);
        let base_energy = hinges[0].energy(&folded);

        // Rotate every particle 90 degrees about the Y axis: (x,y,z) -> (z,y,-x).
        let rotated: Vec<ClothParticle> = folded
            .iter()
            .map(|p| {
                let q = Vec3::new(p.position.z, p.position.y, -p.position.x);
                ClothParticle::new(q, 1.0)
            })
            .collect();
        let rotated_energy = hinges[0].energy(&rotated);
        assert!((base_energy - rotated_energy).abs() < 1.0e-5);
    }

    #[test]
    fn pinned_particles_never_move() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);
        let mut particles = particles_from(&positions);
        particles[3].position = Vec3::new(1.0, 1.0, 0.6);
        for particle in &mut particles {
            *particle = ClothParticle::pinned(particle.position);
        }
        let snapshot: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        project_bending(&mut particles, hinges[0], 1.0 / 60.0);
        for (particle, prev) in particles.iter().zip(snapshot.iter()) {
            assert_eq!(particle.position, *prev);
        }
    }

    #[test]
    fn apply_bending_is_deterministic_and_converges() {
        let (positions, triangles) = flat_quad();
        let hinges = build_dihedral_bending(&positions, &triangles, Compliance::RIGID);

        let mut a = particles_from(&positions);
        a[3].position = Vec3::new(1.0, 1.0, 0.6);
        let mut b = a.clone();

        let first = apply_bending(&mut a, &hinges, 8, 1.0 / 60.0);
        let first_again = apply_bending(&mut b, &hinges, 8, 1.0 / 60.0);
        // Determinism: identical inputs give identical positions.
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
        assert!(first > 0.0);
        assert!((first - first_again).abs() < 1.0e-9);
        // Energy after several sweeps is below the initial fold energy.
        assert!(hinges[0].energy(&a) < first);
    }

    #[test]
    fn empty_mesh_yields_no_constraints() {
        let hinges = build_dihedral_bending(&[], &[], Compliance::RIGID);
        assert!(hinges.is_empty());
    }
}
