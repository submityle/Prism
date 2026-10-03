//! Connected components of the tetrahedral dual graph.
//!
//! A single tetrahedral index buffer may describe several disconnected solids:
//! pre-fractured debris, a body plus loose fragments, or several volumetric
//! chunks baked together. Volumetric solvers want those pieces separated into
//! independent simulation islands before stepping, just as the surface cooker
//! splits triangle soup into shells
//! ([`crate::collider::connectivity`]). This module performs the equivalent
//! split on the *volume*: two tets belong to the same component when a chain of
//! shared faces connects them.
//!
//! The traversal runs over the face dual graph produced by
//! [`crate::collider::tet_adjacency`]. Component ids are assigned
//! deterministically: tets are scanned in ascending index order and each
//! unvisited tet seeds the next component, which is then flood-filled across
//! face neighbours. A fixed mesh therefore always yields the same labels.
//!
//! This is standard mesh connectivity; nothing here is derived from Unreal
//! Engine source.

use super::tet_adjacency::{build_tet_adjacency, TetAdjacency};

/// A labelling of every tet with the connected component it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetComponents {
    /// `labels[t]` is the component id of tet `t`, in `0..component_count`.
    pub labels: Vec<u32>,
    /// Number of distinct connected components.
    pub component_count: usize,
}

impl TetComponents {
    /// Number of tets labelled.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.labels.len()
    }

    /// The tet count of each component, indexed by component id.
    #[must_use]
    pub fn sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.component_count];
        for &c in &self.labels {
            sizes[c as usize] += 1;
        }
        sizes
    }

    /// Returns `true` when the whole mesh is a single connected solid.
    #[must_use]
    pub fn is_single_component(&self) -> bool {
        self.component_count == 1
    }
}

/// Labels the connected components of a tet dual graph with a deterministic
/// flood fill.
///
/// Component ids follow ascending seed-tet order; the component containing the
/// lowest-indexed tet is `0`.
#[must_use]
pub fn label_tet_components(adjacency: &TetAdjacency) -> TetComponents {
    let n = adjacency.tet_count();
    let mut labels = vec![u32::MAX; n];
    let mut component_count = 0usize;
    let mut stack: Vec<usize> = Vec::new();

    for seed in 0..n {
        if labels[seed] != u32::MAX {
            continue;
        }
        let id = component_count as u32;
        component_count += 1;
        labels[seed] = id;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            for nb in adjacency.neighbours[t].iter().flatten() {
                let nb = *nb as usize;
                if labels[nb] == u32::MAX {
                    labels[nb] = id;
                    stack.push(nb);
                }
            }
        }
    }

    TetComponents {
        labels,
        component_count,
    }
}

/// Builds the dual graph of the mesh and labels its connected components.
///
/// Returns `None` under the same conditions as
/// [`build_tet_adjacency`]: empty `tets`, an out-of-range vertex index, or a
/// non-manifold face shared by three or more tets.
#[must_use]
pub fn tet_components(num_vertices: usize, tets: &[[u32; 4]]) -> Option<TetComponents> {
    let adjacency = build_tet_adjacency(num_vertices, tets)?;
    Some(label_tet_components(&adjacency))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use glam::Vec3;

    fn cube_surface(h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let idx = vec![
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
        (verts, idx)
    }

    #[test]
    fn single_tet_is_one_component() {
        let comp = tet_components(4, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(comp.component_count, 1);
        assert_eq!(comp.labels, vec![0]);
        assert!(comp.is_single_component());
    }

    #[test]
    fn two_disjoint_tets_are_two_components() {
        // No shared face: distinct vertex sets.
        let tets = vec![[0u32, 1, 2, 3], [4, 5, 6, 7]];
        let comp = tet_components(8, &tets).unwrap();
        assert_eq!(comp.component_count, 2);
        assert_ne!(comp.labels[0], comp.labels[1]);
        assert_eq!(comp.sizes(), vec![1, 1]);
    }

    #[test]
    fn two_tets_sharing_a_face_are_one_component() {
        let tets = vec![[0u32, 1, 2, 3], [0, 2, 1, 4]];
        let comp = tet_components(5, &tets).unwrap();
        assert_eq!(comp.component_count, 1);
    }

    #[test]
    fn tetrahedralized_cube_is_connected() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let comp = tet_components(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(comp.component_count, 1, "a solid cube must be one island");
        assert_eq!(comp.sizes(), vec![mesh.tets.len()]);
    }

    #[test]
    fn sizes_sum_to_tet_count_and_labels_in_range() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let comp = tet_components(mesh.vertices.len(), &mesh.tets).unwrap();
        let sizes = comp.sizes();
        assert_eq!(sizes.iter().sum::<usize>(), comp.tet_count());
        assert!(comp
            .labels
            .iter()
            .all(|&c| (c as usize) < comp.component_count));
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = tet_components(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = tet_components(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_or_invalid_returns_none() {
        assert!(tet_components(0, &[]).is_none());
        assert!(tet_components(4, &[[0u32, 1, 2, 9]]).is_none());
    }
}
