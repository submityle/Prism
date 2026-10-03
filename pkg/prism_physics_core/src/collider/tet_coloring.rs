//! Greedy colouring of the tetrahedral dual graph for parallel element sweeps.
//!
//! Many volumetric solvers (FEM, shape-matching, strain-based softbody) relax
//! one tetrahedral *element* at a time, where each element reads and writes the
//! nodes it owns. Two tets that share a face also share nodes, so relaxing them
//! simultaneously is a read-after-write hazard. Partitioning the tets into
//! colours such that no two *face-adjacent* tets share a colour removes that
//! hazard: every tet in a colour writes nodes disjoint from the others across
//! the shared faces, so a whole colour can be relaxed in parallel. Applying
//! colours one after another is a valid Gauss-Seidel sweep whose result is
//! independent of the within-colour order.
//!
//! The colouring is a deterministic greedy pass over the dual graph produced by
//! [`crate::collider::tet_adjacency`]: tets are visited in ascending index
//! order and each takes the lowest colour not already used by a previously
//! coloured face-neighbour. A fixed mesh therefore always yields the same
//! colours, which lets a CPU reference and a parallel dispatch agree. The dual
//! graph has maximum degree four (a tet has four faces), so greedy uses at most
//! five colours.
//!
//! This mirrors the crate's constraint-graph and vertex colourings
//! ([`crate::solver::xpbd::graph_color`], [`crate::vbd::coloring`]) but operates
//! on the *element* dual graph rather than a constraint or vertex graph. Greedy
//! graph colouring for parallel Gauss-Seidel is a standard, publicly documented
//! technique; nothing here is derived from Unreal Engine source.

use super::tet_adjacency::{build_tet_adjacency, TetAdjacency};

/// A colour assignment over the tets of a tetrahedral mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetColoring {
    /// `colours[t]` is the colour assigned to tet `t`. No two face-adjacent
    /// tets share a colour.
    pub colours: Vec<u32>,
}

impl TetColoring {
    /// Number of tets coloured.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.colours.len()
    }

    /// Number of distinct colours used (the largest colour index plus one, or
    /// zero when there are no tets).
    #[must_use]
    pub fn colour_count(&self) -> usize {
        self.colours
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1)
    }

    /// Groups the tet indices by colour. `groups()[c]` lists every tet assigned
    /// colour `c`, in ascending tet-index order. The returned groups partition
    /// `0..tet_count()`.
    #[must_use]
    pub fn groups(&self) -> Vec<Vec<u32>> {
        let mut groups = vec![Vec::new(); self.colour_count()];
        for (t, &c) in self.colours.iter().enumerate() {
            groups[c as usize].push(t as u32);
        }
        groups
    }
}

/// Colours the tets of the given dual graph with a deterministic greedy pass.
///
/// Face-adjacent tets never receive the same colour. Tets are visited in
/// ascending index order; each takes the lowest colour not used by an
/// already-coloured face-neighbour. Uses at most five colours (dual-graph
/// degree is at most four).
#[must_use]
pub fn colour_tet_adjacency(adjacency: &TetAdjacency) -> TetColoring {
    let n = adjacency.tet_count();
    let mut colours = vec![u32::MAX; n];
    // Scratch set of colours already taken by this tet's neighbours.
    let mut used: Vec<bool> = Vec::new();

    for t in 0..n {
        used.clear();
        for nb in adjacency.neighbours[t].iter().flatten() {
            let c = colours[*nb as usize];
            if c != u32::MAX {
                let ci = c as usize;
                if ci >= used.len() {
                    used.resize(ci + 1, false);
                }
                used[ci] = true;
            }
        }

        let mut chosen = 0usize;
        while chosen < used.len() && used[chosen] {
            chosen += 1;
        }
        colours[t] = chosen as u32;
    }

    TetColoring { colours }
}

/// Builds the dual graph of the mesh and colours it in one call.
///
/// Returns `None` under the same conditions as
/// [`build_tet_adjacency`]: empty `tets`, an out-of-range vertex index, or a
/// non-manifold face shared by three or more tets.
#[must_use]
pub fn colour_tet_mesh(num_vertices: usize, tets: &[[u32; 4]]) -> Option<TetColoring> {
    let adjacency = build_tet_adjacency(num_vertices, tets)?;
    Some(colour_tet_adjacency(&adjacency))
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
    fn single_tet_uses_one_colour() {
        let adj = build_tet_adjacency(4, &[[0u32, 1, 2, 3]]).unwrap();
        let col = colour_tet_adjacency(&adj);
        assert_eq!(col.tet_count(), 1);
        assert_eq!(col.colour_count(), 1);
        assert_eq!(col.colours, vec![0]);
    }

    #[test]
    fn adjacent_tets_get_different_colours() {
        let tets = vec![[0u32, 1, 2, 3], [0, 2, 1, 4]];
        let adj = build_tet_adjacency(5, &tets).unwrap();
        let col = colour_tet_adjacency(&adj);
        assert_eq!(col.colour_count(), 2);
        assert_ne!(col.colours[0], col.colours[1]);
    }

    #[test]
    fn colouring_is_valid_on_cube() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let col = colour_tet_adjacency(&adj);
        // No face-adjacent pair shares a colour.
        for t in 0..adj.tet_count() {
            for nb in adj.neighbours[t].iter().flatten() {
                assert_ne!(
                    col.colours[t], col.colours[*nb as usize],
                    "tet {t} and neighbour {nb} share colour {}",
                    col.colours[t]
                );
            }
        }
    }

    #[test]
    fn colour_count_is_bounded_by_five() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let col = colour_tet_mesh(mesh.vertices.len(), &mesh.tets).unwrap();
        assert!(
            col.colour_count() <= 5,
            "greedy on degree-4 graph used {} colours",
            col.colour_count()
        );
    }

    #[test]
    fn groups_partition_all_tets() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let col = colour_tet_mesh(mesh.vertices.len(), &mesh.tets).unwrap();
        let groups = col.groups();
        let mut seen = vec![false; col.tet_count()];
        let mut total = 0usize;
        for (c, g) in groups.iter().enumerate() {
            for &t in g {
                assert!(!seen[t as usize], "tet {t} appears in two groups");
                seen[t as usize] = true;
                assert_eq!(col.colours[t as usize] as usize, c);
                total += 1;
            }
        }
        assert_eq!(total, col.tet_count());
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = colour_tet_mesh(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = colour_tet_mesh(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_mesh_returns_none() {
        assert!(colour_tet_mesh(0, &[]).is_none());
        assert!(colour_tet_mesh(4, &[[0u32, 1, 2, 9]]).is_none());
    }
}
