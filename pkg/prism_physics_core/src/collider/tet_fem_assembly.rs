//! Global sparse stiffness matrix of a tetrahedral finite-element mesh.
//!
//! An implicit (backward-Euler / Newmark) `FEM` integrator does not advance one
//! element at a time; it solves a single linear system in all nodal
//! displacements at once. That system is driven by the *global* stiffness
//! matrix `K`, the sum of every element's `12x12` contribution scattered into
//! the `3 N`-dimensional space of nodal degrees of freedom (`DOF`s), where `N`
//! is the number of mesh vertices.
//!
//! `K` is large but extremely sparse: `DOF`s of vertices `a` and `b` couple
//! only when `a` and `b` share at least one tetrahedron. This module assembles
//! `K` directly in compressed-sparse-row (`CSR`) form:
//!
//! 1. the block sparsity pattern is read from the mesh's vertex adjacency (two
//!    vertices share a tetrahedron exactly when they are adjacent, because a
//!    tetrahedron is a complete graph on its four vertices), and
//! 2. each element's `12x12` [`TetStiffness`] (built from the constant-strain
//!    `CST` basis) is scattered into the pattern.
//!
//! The result is a symmetric positive-semi-definite operator exposed through a
//! sparse matrix-vector product. This module holds no simulation state and
//! performs no time integration, so it is fully decoupled from any solver. It
//! is standard linear finite-element assembly; nothing here is derived from
//! Unreal Engine source.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_stiffness::{element_stiffness, IsotropicElasticity};
use super::tet_vertex_adjacency::build_tet_vertex_adjacency;

/// The assembled global stiffness matrix of a tetrahedral mesh, stored in
/// compressed-sparse-row (`CSR`) form over `3 N` scalar degrees of freedom.
///
/// Degree of freedom `3 v + c` is the `c`-th Cartesian displacement of vertex
/// `v`. The matrix is symmetric; both the upper and lower triangles are stored
/// explicitly so the matrix-vector product needs no special casing.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalStiffness {
    /// Number of scalar degrees of freedom, equal to three times the vertex
    /// count.
    pub n_dofs: usize,
    /// `CSR` row offsets, length `n_dofs + 1`. Row `i` occupies the value /
    /// column range `row_offsets[i] .. row_offsets[i + 1]`.
    pub row_offsets: Vec<usize>,
    /// Column index of each stored entry, ascending within every row.
    pub col_indices: Vec<usize>,
    /// Stored matrix entries, parallel to [`GlobalStiffness::col_indices`].
    pub values: Vec<f32>,
}

impl GlobalStiffness {
    /// Number of scalar degrees of freedom (`3 N`).
    #[must_use]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of explicitly stored (structurally non-zero) entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The entry `K[i][j]`, or `0.0` when that position is outside the sparsity
    /// pattern.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i >= self.n_dofs {
            return 0.0;
        }
        let start = self.row_offsets[i];
        let end = self.row_offsets[i + 1];
        let cols = &self.col_indices[start..end];
        match cols.binary_search(&j) {
            Ok(offset) => self.values[start + offset],
            Err(_) => 0.0,
        }
    }

    /// The sparse matrix-vector product `K x`.
    ///
    /// Returns `None` when `x` does not have exactly [`GlobalStiffness::n_dofs`]
    /// entries. The accumulation is carried out in `f64`.
    #[must_use]
    pub fn apply(&self, x: &[f32]) -> Option<Vec<f32>> {
        if x.len() != self.n_dofs {
            return None;
        }
        let mut out = vec![0.0f32; self.n_dofs];
        for (row, out_i) in out.iter_mut().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let mut acc = 0.0f64;
            for k in start..end {
                acc += f64::from(self.values[k]) * f64::from(x[self.col_indices[k]]);
            }
            *out_i = acc as f32;
        }
        Some(out)
    }

    /// The elastic strain energy `0.5 * x^T K x` at displacement `x`.
    ///
    /// Returns `None` when `x` does not have exactly [`GlobalStiffness::n_dofs`]
    /// entries. Non-negative for every `x` because `K` is positive
    /// semi-definite.
    #[must_use]
    pub fn energy(&self, x: &[f32]) -> Option<f32> {
        let kx = self.apply(x)?;
        let mut acc = 0.0f64;
        for (&xi, &kxi) in x.iter().zip(kx.iter()) {
            acc += f64::from(xi) * f64::from(kxi);
        }
        Some((0.5 * acc) as f32)
    }
}

/// Assembles the global stiffness matrix of a tetrahedral mesh.
///
/// * `basis` provides one rest-pose element per tetrahedron (its length must
///   equal `tets.len()`).
/// * `tets` lists the four vertex indices of each tetrahedron, in the same
///   order as `basis`.
/// * `material` is the single isotropic material shared by every element.
/// * `n_vertices` is the vertex count; every index in `tets` must be smaller.
///
/// Returns `None` when the mesh is empty, when `basis` and `tets` disagree in
/// length, or when any tetrahedron references a vertex at or beyond
/// `n_vertices`.
#[must_use]
pub fn assemble_global_stiffness(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    n_vertices: usize,
) -> Option<GlobalStiffness> {
    if tets.is_empty() || basis.elements.len() != tets.len() {
        return None;
    }

    // Block sparsity: vertex v couples to itself and to every adjacent vertex.
    let adjacency = build_tet_vertex_adjacency(n_vertices, tets)?;
    let mut block_cols: Vec<Vec<u32>> = Vec::with_capacity(n_vertices);
    for v in 0..n_vertices {
        let mut cols = Vec::with_capacity(adjacency.degree(v) + 1);
        cols.push(v as u32);
        cols.extend_from_slice(adjacency.neighbours_of(v));
        cols.sort_unstable();
        block_cols.push(cols);
    }

    // Expand the block pattern into a scalar CSR structure over 3 N DOFs.
    let n_dofs = 3 * n_vertices;
    let mut row_offsets = Vec::with_capacity(n_dofs + 1);
    let mut col_indices: Vec<usize> = Vec::new();
    row_offsets.push(0);
    for cols in &block_cols {
        // Three identical scalar rows per vertex block (one per component).
        let row_cols: Vec<usize> = cols
            .iter()
            .flat_map(|&w| {
                let base = 3 * w as usize;
                [base, base + 1, base + 2]
            })
            .collect();
        for _ in 0..3 {
            col_indices.extend_from_slice(&row_cols);
            row_offsets.push(col_indices.len());
        }
    }

    // Scatter each element's 12x12 stiffness into the pattern (f64 accumulator).
    let mut accum = vec![0.0f64; col_indices.len()];
    for (element, tet) in basis.elements.iter().zip(tets.iter()) {
        let ke = element_stiffness(element, material);
        for a in 0..12 {
            let row = 3 * tet[a / 3] as usize + a % 3;
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            let cols = &col_indices[start..end];
            for b in 0..12 {
                let col = 3 * tet[b / 3] as usize + b % 3;
                let offset = cols
                    .binary_search(&col)
                    .expect("scattered DOF must lie inside the assembled pattern");
                accum[start + offset] += f64::from(ke.get(a, b));
            }
        }
    }

    let values: Vec<f32> = accum.iter().map(|&x| x as f32).collect();
    Some(GlobalStiffness {
        n_dofs,
        row_offsets,
        col_indices,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use glam::Vec3;

    /// A single non-degenerate tetrahedron (4 vertices, 1 cell).
    fn single_tet() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(1.3, 0.2, -0.1),
            Vec3::new(-0.2, 1.1, 0.4),
            Vec3::new(0.3, 0.5, 1.4),
        ];
        let tets = vec![[0u32, 1, 2, 3]];
        (verts, tets)
    }

    /// Two tetrahedra sharing a triangular face (5 vertices, 2 cells).
    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    #[test]
    fn single_tet_matches_the_element_matrix() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(3.0e3, 0.3).unwrap();
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let ke = element_stiffness(&basis.elements[0], &mat);

        // For a mesh of one tet with vertices 0..3 the DOF maps are the
        // identity, so the global operator must equal the element operator.
        let u = [
            0.3f32, -0.1, 0.2, 0.5, 0.9, -0.4, -0.7, 0.1, 0.6, 0.2, -0.3, 0.8,
        ];
        let global_f = global.apply(&u).unwrap();
        let elem_f = ke.apply(&u);
        for i in 0..12 {
            assert!(
                (global_f[i] - elem_f[i]).abs() <= 1e-4 * (1.0 + elem_f[i].abs()),
                "dof {i}: {} vs {}",
                global_f[i],
                elem_f[i]
            );
        }
    }

    #[test]
    fn assembled_matrix_is_symmetric() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(2.0e3, 0.28).unwrap();
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let n = global.n_dofs();
        let mut scale = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                scale = scale.max(global.get(i, j).abs());
            }
        }
        assert!(scale > 0.0);
        for i in 0..n {
            for j in 0..n {
                let diff = (global.get(i, j) - global.get(j, i)).abs();
                assert!(diff <= 1e-4 * scale, "asymmetry at ({i},{j})");
            }
        }
    }

    #[test]
    fn rigid_translation_is_in_the_null_space() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(1.0e3, 0.33).unwrap();
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let mut u = vec![0.0f32; global.n_dofs()];
        for v in 0..verts.len() {
            u[3 * v] = 0.7;
            u[3 * v + 1] = -0.4;
            u[3 * v + 2] = 1.2;
        }
        let f = global.apply(&u).unwrap();
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() <= 1e-2, "translation force[{i}] = {fi}");
        }
        assert!(global.energy(&u).unwrap().abs() <= 1e-3);
    }

    #[test]
    fn infinitesimal_rotation_is_in_the_null_space() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(1.5e3, 0.25).unwrap();
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        // Linearized rigid rotation about the origin: a linear field, so each
        // constant-strain element sees zero strain.
        let omega = Vec3::new(0.2, -0.5, 0.9);
        let mut u = vec![0.0f32; global.n_dofs()];
        for (v, x) in verts.iter().enumerate() {
            let w = omega.cross(*x);
            u[3 * v] = w.x;
            u[3 * v + 1] = w.y;
            u[3 * v + 2] = w.z;
        }
        let f = global.apply(&u).unwrap();
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() <= 1e-3, "rotation force[{i}] = {fi}");
        }
        assert!(global.energy(&u).unwrap().abs() <= 1e-4);
    }

    #[test]
    fn energy_is_non_negative_on_a_real_mesh() {
        let verts = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let tris = vec![
            [0u32, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mesh = tetrahedralize(&verts, &tris, &TetMeshParams::new(8)).unwrap();
        let basis = basis_of(&mesh.vertices, &mesh.tets);
        let mat = IsotropicElasticity::new(5.0e3, 0.2).unwrap();
        let global =
            assemble_global_stiffness(&basis, &mesh.tets, &mat, mesh.vertices.len()).unwrap();

        let n = global.n_dofs();
        let seeds = [1u64, 7, 42, 1234];
        for &seed in &seeds {
            // Cheap deterministic pseudo-random displacement in [-1, 1].
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut u = vec![0.0f32; n];
            for value in &mut u {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                *value = (unit * 2.0 - 1.0) as f32;
            }
            let energy = global.energy(&u).unwrap();
            assert!(energy >= -1e-2, "negative energy {energy} for seed {seed}");
        }
    }

    #[test]
    fn pattern_covers_only_shared_tet_vertices() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::default();
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        // Vertices 0 and 4 never share a tetrahedron, so their coupling blocks
        // must be absent from the pattern (exactly zero, not merely small).
        for ci in 0..3 {
            for cj in 0..3 {
                assert_eq!(global.get(3 * 0 + ci, 3 * 4 + cj), 0.0);
                assert_eq!(global.get(3 * 4 + ci, 3 * 0 + cj), 0.0);
            }
        }
        // Vertices 1 and 2 share both tets, so their coupling block is present.
        let mut any = false;
        for ci in 0..3 {
            for cj in 0..3 {
                if global.get(3 * 1 + ci, 3 * 2 + cj).abs() > 0.0 {
                    any = true;
                }
            }
        }
        assert!(any, "shared-vertex block unexpectedly empty");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::default();
        // Empty mesh.
        assert!(assemble_global_stiffness(&basis, &[], &mat, verts.len()).is_none());
        // Out-of-range vertex count.
        assert!(assemble_global_stiffness(&basis, &tets, &mat, 2).is_none());
        // Mismatched vector length into apply/energy.
        let global = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        assert!(global.apply(&[0.0; 5]).is_none());
        assert!(global.energy(&[0.0; 5]).is_none());
    }

    #[test]
    fn assembly_is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mat = IsotropicElasticity::new(4.0e3, 0.3).unwrap();
        let a = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let b = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        assert_eq!(a, b);
    }
}
