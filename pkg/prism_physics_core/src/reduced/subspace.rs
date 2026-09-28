//! Construction of a reduced modal subspace for a spring-lattice soft body.
//!
//! A [`ReducedModel`] captures a deformable body by its lowest-frequency linear
//! vibration modes. Given the rest shape, per-vertex mass, the spring topology,
//! and a set of *fixed* (anchored) vertices, it assembles the linear stiffness
//! matrix `K` and lumped mass matrix `M` over the free degrees of freedom and
//! solves the generalised symmetric eigenproblem
//!
//! ```text
//! K u = lambda M u.
//! ```
//!
//! Using the lumped (diagonal, positive) mass matrix the problem is reduced to a
//! standard symmetric one by the mass-scaling `A = M^{-1/2} K M^{-1/2}`: its
//! eigenvectors `y` give the mass-orthonormal mode shapes `u = M^{-1/2} y`
//! (so `U^T M U = I` and `U^T K U = diag(lambda)`). The lowest `num_modes` modes
//! are kept. Fixed vertices carry zero displacement in every mode, which is how
//! a cantilever anchored at one end is modelled.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Linear
//! modal analysis, mass-scaling of the generalised eigenproblem, and modal
//! truncation are standard, publicly documented structural-dynamics results.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::vbd::element::SpringSet;

use super::modes::SymmetricMatrix;

/// A single retained vibration mode.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReducedMode {
    /// The eigenvalue `lambda = omega^2` (angular frequency squared), clamped to
    /// be non-negative. Higher means a stiffer, faster-oscillating mode.
    pub frequency_squared: Real,
    /// The mass-orthonormal mode shape as a per-vertex displacement field.
    /// Fixed vertices are exactly [`Vec3::ZERO`].
    pub shape: Vec<Vec3>,
}

/// A reduced-order model: a rest shape plus a truncated modal basis.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReducedModel {
    /// Rest (undeformed) vertex positions.
    rest: Vec<Vec3>,
    /// Per-vertex lumped mass, retained so body forces such as gravity can be
    /// projected onto the modal basis without the caller re-supplying it.
    masses: Vec<Real>,
    /// Retained modes, ordered from lowest to highest frequency.
    modes: Vec<ReducedMode>,
}

impl ReducedModel {
    /// Builds a reduced model from a spring lattice.
    ///
    /// * `rest` — rest positions, one per vertex.
    /// * `masses` — per-vertex mass (`> 0` for a free vertex).
    /// * `fixed` — per-vertex anchor flags; a fixed vertex contributes no DOF and
    ///   stays at its rest position in every mode.
    /// * `springs` — the lattice edges; each contributes a linear stiffness block
    ///   `k n n^T` about the rest direction `n`.
    /// * `num_modes` — how many of the lowest-frequency modes to keep.
    ///
    /// Returns an empty model (no modes) when there are no free degrees of
    /// freedom.
    #[must_use]
    pub fn from_springs(
        rest: &[Vec3],
        masses: &[Real],
        fixed: &[bool],
        springs: &SpringSet,
        num_modes: usize,
    ) -> ReducedModel {
        let vertex_count = rest.len();
        // Map each free vertex to a compact DOF block; fixed vertices get none.
        let mut free_of_vertex = vec![usize::MAX; vertex_count];
        let mut free_vertices = Vec::new();
        for (v, &is_fixed) in fixed.iter().enumerate().take(vertex_count) {
            if !is_fixed {
                free_of_vertex[v] = free_vertices.len();
                free_vertices.push(v);
            }
        }
        let free_count = free_vertices.len();
        let dof = free_count * 3;
        if dof == 0 {
            return ReducedModel {
                rest: rest.to_vec(),
                masses: masses.to_vec(),
                modes: Vec::new(),
            };
        }

        let stiffness = assemble_stiffness(rest, springs, &free_of_vertex, dof);
        // Inverse square-root lumped mass per DOF.
        let inv_sqrt_mass = build_inv_sqrt_mass(masses, &free_vertices);

        // Mass-scaled matrix A = M^{-1/2} K M^{-1/2}.
        let mut scaled = SymmetricMatrix::zeros(dof);
        for row in 0..dof {
            for col in 0..dof {
                let value = inv_sqrt_mass[row] * stiffness.get(row, col) * inv_sqrt_mass[col];
                scaled.set(row, col, value);
            }
        }

        let eigen = scaled.eigen();
        let kept = num_modes.min(dof);
        let mut modes = Vec::with_capacity(kept);
        for k in 0..kept {
            let y = &eigen.vectors[k];
            // Physical (mass-normalised) shape u = M^{-1/2} y, scattered to the
            // full vertex list with zeros at fixed vertices.
            let mut shape = vec![Vec3::ZERO; vertex_count];
            for (local, &vertex) in free_vertices.iter().enumerate() {
                let x = y[local * 3] * inv_sqrt_mass[local * 3];
                let ysc = y[local * 3 + 1] * inv_sqrt_mass[local * 3 + 1];
                let z = y[local * 3 + 2] * inv_sqrt_mass[local * 3 + 2];
                shape[vertex] = Vec3::new(x, ysc, z);
            }
            modes.push(ReducedMode {
                frequency_squared: eigen.values[k].max(0.0),
                shape,
            });
        }

        ReducedModel {
            rest: rest.to_vec(),
            masses: masses.to_vec(),
            modes,
        }
    }

    /// Returns the number of retained modes.
    #[must_use]
    pub fn num_modes(&self) -> usize {
        self.modes.len()
    }

    /// Returns the number of vertices.
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.rest.len()
    }

    /// Returns `true` when the model has no modes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modes.is_empty()
    }

    /// Returns the retained modes.
    #[must_use]
    pub fn modes(&self) -> &[ReducedMode] {
        &self.modes
    }

    /// Returns the rest positions.
    #[must_use]
    pub fn rest(&self) -> &[Vec3] {
        &self.rest
    }

    /// Returns the per-vertex lumped masses used to build the model.
    #[must_use]
    pub fn masses(&self) -> &[Real] {
        &self.masses
    }

    /// Reconstructs world positions from generalised coordinates `q`.
    ///
    /// The displacement of vertex `i` is `sum_k q[k] * mode[k].shape[i]`, added
    /// to the rest position. Extra or missing coordinates are ignored / treated
    /// as zero, so callers may pass a shorter slice.
    #[must_use]
    pub fn reconstruct(&self, q: &[Real]) -> Vec<Vec3> {
        let mut out = self.rest.clone();
        let count = q.len().min(self.modes.len());
        for (k, mode) in self.modes.iter().enumerate().take(count) {
            let qk = q[k];
            if qk == 0.0 {
                continue;
            }
            for (o, s) in out.iter_mut().zip(mode.shape.iter()) {
                *o += *s * qk;
            }
        }
        out
    }

    /// Projects a per-vertex physical force field onto the modal coordinates.
    ///
    /// For mass-orthonormal modes the generalised force on mode `k` is
    /// `phi_k = shape_k . f`, summed over vertices.
    #[must_use]
    pub fn modal_force(&self, force_per_vertex: &[Vec3]) -> Vec<Real> {
        self.modes
            .iter()
            .map(|mode| {
                mode.shape
                    .iter()
                    .zip(force_per_vertex.iter())
                    .map(|(s, f)| s.dot(*f))
                    .sum()
            })
            .collect()
    }
}

/// Assembles the `dof x dof` linear stiffness matrix over the free DOFs.
fn assemble_stiffness(
    rest: &[Vec3],
    springs: &SpringSet,
    free_of_vertex: &[usize],
    dof: usize,
) -> SymmetricMatrix {
    let mut k = SymmetricMatrix::zeros(dof);
    for spring in &springs.springs {
        let a = spring.a.index();
        let b = spring.b.index();
        let d = rest[a] - rest[b];
        let len = d.length();
        if len <= Real::MIN_POSITIVE {
            continue;
        }
        let n = d / len;
        // Rest tangent stiffness block is k n n^T (the stretch Hessian at C = 0).
        let block = [
            [n.x * n.x, n.x * n.y, n.x * n.z],
            [n.y * n.x, n.y * n.y, n.y * n.z],
            [n.z * n.x, n.z * n.y, n.z * n.z],
        ];
        let stiffness = spring.stiffness;
        let fa = free_of_vertex[a];
        let fb = free_of_vertex[b];
        for (i, brow) in block.iter().enumerate() {
            for (j, &bij) in brow.iter().enumerate() {
                let value = stiffness * bij;
                // K_aa += block, K_bb += block, K_ab -= block, K_ba -= block.
                if fa != usize::MAX {
                    k.add(fa * 3 + i, fa * 3 + j, value);
                }
                if fb != usize::MAX {
                    k.add(fb * 3 + i, fb * 3 + j, value);
                }
                if fa != usize::MAX && fb != usize::MAX {
                    k.add(fa * 3 + i, fb * 3 + j, -value);
                    k.add(fb * 3 + i, fa * 3 + j, -value);
                }
            }
        }
    }
    k
}

/// Builds the per-DOF inverse square-root lumped mass vector.
fn build_inv_sqrt_mass(masses: &[Real], free_vertices: &[usize]) -> Vec<Real> {
    let mut inv = Vec::with_capacity(free_vertices.len() * 3);
    for &vertex in free_vertices {
        let mass = masses[vertex].max(Real::MIN_POSITIVE);
        let value = 1.0 / mass.sqrt();
        inv.push(value);
        inv.push(value);
        inv.push(value);
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleHandle;
    use crate::vbd::element::SpringElement;

    fn handle(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn no_free_vertices_gives_empty_model() {
        let rest = vec![Vec3::ZERO, Vec3::X];
        let masses = vec![1.0, 1.0];
        let fixed = vec![true, true];
        let springs = SpringSet::new();
        let model = ReducedModel::from_springs(&rest, &masses, &fixed, &springs, 4);
        assert!(model.is_empty());
        assert_eq!(model.num_modes(), 0);
    }

    #[test]
    fn single_free_vertex_axis_modes_have_expected_frequency() {
        // Vertex 0 fixed, vertex 1 free, joined by a spring of stiffness k along
        // X. The free vertex has a stiff mode along the spring (lambda = k / m)
        // and two zero modes transverse to it.
        let rest = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let masses = vec![1.0, 2.0];
        let fixed = vec![true, false];
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(handle(0), handle(1), 1.0, 8.0));
        let model = ReducedModel::from_springs(&rest, &masses, &fixed, &springs, 3);
        assert_eq!(model.num_modes(), 3);
        // Highest mode should be the axial one with lambda = k / m = 8 / 2 = 4.
        let top = &model.modes()[2];
        assert!(
            (top.frequency_squared - 4.0).abs() < 1e-3,
            "axial lambda {}",
            top.frequency_squared
        );
        // Its shape displaces the free vertex along X only.
        let s = top.shape[1];
        assert!(s.x.abs() > 1e-3);
        assert!(s.y.abs() < 1e-4 && s.z.abs() < 1e-4);
        // Fixed vertex never moves.
        assert_eq!(model.modes()[2].shape[0], Vec3::ZERO);
    }

    #[test]
    fn reconstruct_adds_scaled_mode_shapes() {
        let rest = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let masses = vec![1.0, 1.0];
        let fixed = vec![true, false];
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(handle(0), handle(1), 1.0, 10.0));
        let model = ReducedModel::from_springs(&rest, &masses, &fixed, &springs, 3);
        let zero = model.reconstruct(&[0.0, 0.0, 0.0]);
        assert_eq!(zero, rest);
        // Non-zero coordinates move the free vertex but never the anchor.
        let moved = model.reconstruct(&[0.1, 0.1, 0.1]);
        assert_eq!(moved[0], Vec3::ZERO);
        assert!((moved[1] - rest[1]).length() > 0.0);
    }

    #[test]
    fn modes_are_mass_orthonormal() {
        // U^T M U = I: each mode has unit modal mass and distinct modes are
        // M-orthogonal.
        let rest = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let masses = vec![1.0, 1.5, 2.0];
        let fixed = vec![true, false, false];
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(handle(0), handle(1), 1.0, 30.0));
        springs.push(SpringElement::new(handle(1), handle(2), 1.0, 30.0));
        let model = ReducedModel::from_springs(&rest, &masses, &fixed, &springs, 6);
        let modes = model.modes();
        for (i, mi) in modes.iter().enumerate() {
            for (j, mj) in modes.iter().enumerate() {
                let mut dot = 0.0;
                for v in 0..rest.len() {
                    dot += masses[v] * mi.shape[v].dot(mj.shape[v]);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-3, "U^T M U[{i}][{j}] = {dot}");
            }
        }
    }
}
