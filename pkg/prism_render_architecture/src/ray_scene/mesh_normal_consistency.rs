//! Triangle winding-consistency repair for the `CPU` golden path.
//!
//! Meshes merged from several sources, or exported by tools with mixed winding
//! conventions, frequently contain triangles whose vertex order disagrees with
//! their neighbours: adjacent faces wind in opposite directions, so their
//! geometric normals point to opposite sides of the same surface. That breaks
//! back-face culling, flips shading, and corrupts any signed-volume or
//! two-sided test. This module rewinds the index buffer so every edge-connected
//! patch of triangles agrees on a single orientation, and — for patches that
//! form a closed surface — flips the whole patch outward using its signed
//! volume.
//!
//! [`make_winding_consistent`] walks the dual graph of the mesh. It builds the
//! undirected-edge → triangle adjacency, then breadth-first floods each
//! edge-connected patch from its lowest-indexed seed: whenever a neighbour
//! shares the seed's directed edge in the *same* direction (the signature of
//! opposite winding) its triangle is reversed. Edges shared by more than two
//! triangles are non-manifold; the flood does not propagate across them (and
//! they are reported). After a patch is made internally consistent, if every
//! one of its edges is shared by exactly two triangles — i.e. the patch is
//! closed — its signed volume is computed and the entire patch is flipped when
//! that volume is negative, so closed bodies end up outward-facing.
//!
//! Only the winding (triangle index order) is rewritten; stored per-vertex
//! normals and `UV`s are preserved untouched, so recompute shading normals via
//! [`super::mesh_smooth_normals`] afterwards if they must match the repaired
//! winding. Adjacency and flooding are pure integer bookkeeping; only the
//! optional outward-orientation step touches floating point, and it uses a
//! triple product (no transcendental functions), carried in `f64` for
//! stability — honouring the golden-path ban.

use alloc::collections::VecDeque;
use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Result of a winding-consistency repair: the rewound mesh plus statistics
/// describing what was changed.
pub struct WindingFix {
    /// The mesh with its index buffer rewound to a consistent orientation.
    mesh: TriangleMesh,
    /// Number of triangles whose winding was reversed (both consistency and
    /// outward-orientation flips).
    flipped: u32,
    /// Number of edge-connected patches discovered.
    patches: u32,
    /// Number of undirected edges shared by more than two triangles.
    non_manifold_edges: u32,
}

impl WindingFix {
    /// Returns the rewound mesh.
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Consumes the result, returning the rewound mesh.
    pub fn into_mesh(self) -> TriangleMesh {
        self.mesh
    }

    /// Returns the number of triangles that were reversed.
    pub fn flipped(&self) -> u32 {
        self.flipped
    }

    /// Returns the number of edge-connected patches.
    pub fn patches(&self) -> u32 {
        self.patches
    }

    /// Returns the number of non-manifold edges (shared by more than two
    /// triangles) encountered.
    pub fn non_manifold_edges(&self) -> u32 {
        self.non_manifold_edges
    }
}

/// Rewinds `mesh` so every edge-connected patch of triangles shares one
/// orientation, additionally flipping each closed patch outward by its signed
/// volume.
///
/// Only triangle winding (index order) changes; positions, normals, and `UV`s
/// are preserved. See the module docs for the flood algorithm.
///
/// # Errors
///
/// Returns [`TriangleMeshError`] if the rewound mesh fails validation; by
/// construction the index buffer only permutes existing indices, so this is not
/// expected in practice.
pub fn make_winding_consistent(mesh: &TriangleMesh) -> Result<WindingFix, TriangleMeshError> {
    let mut oriented: Vec<[u32; 3]> = mesh.indices().to_vec();
    let triangle_count = oriented.len();

    // Undirected edge -> triangles touching it.
    let mut edge_map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (triangle, tri) in oriented.iter().enumerate() {
        for &(u, v) in &undirected_edges(tri) {
            edge_map.entry((u, v)).or_default().push(triangle);
        }
    }
    let non_manifold_edges = edge_map.values().filter(|t| t.len() > 2).count() as u32;

    let mut visited = vec![false; triangle_count];
    let mut flipped: u32 = 0;
    let mut patch_count: u32 = 0;

    for seed in 0..triangle_count {
        if visited[seed] {
            continue;
        }
        patch_count += 1;
        visited[seed] = true;
        let mut patch = vec![seed];
        let mut queue = VecDeque::new();
        queue.push_back(seed);

        while let Some(current) = queue.pop_front() {
            for &(u, v) in &undirected_edges(&oriented[current]) {
                let sharers = &edge_map[&(u, v)];
                // Only propagate across manifold edges (exactly two faces).
                if sharers.len() != 2 {
                    continue;
                }
                for &neighbour in sharers {
                    if neighbour == current || visited[neighbour] {
                        continue;
                    }
                    let cur_dir = directed_edge(&oriented[current], u, v)
                        .expect("current triangle owns this edge");
                    let nbr_dir = directed_edge(&oriented[neighbour], u, v)
                        .expect("neighbour triangle owns this edge");
                    // Same directed edge means opposite winding → reverse.
                    if cur_dir == nbr_dir {
                        reverse_winding(&mut oriented[neighbour]);
                        flipped += 1;
                    }
                    visited[neighbour] = true;
                    patch.push(neighbour);
                    queue.push_back(neighbour);
                }
            }
        }

        if patch_is_closed(&patch, &oriented, &edge_map)
            && signed_volume(&patch, &oriented, mesh.positions()) < 0.0
        {
            for &triangle in &patch {
                reverse_winding(&mut oriented[triangle]);
                flipped += 1;
            }
        }
    }

    let mesh = TriangleMesh::new(
        mesh.positions().to_vec(),
        mesh.normals().to_vec(),
        mesh.uvs().to_vec(),
        oriented,
    )?;
    Ok(WindingFix {
        mesh,
        flipped,
        patches: patch_count,
        non_manifold_edges,
    })
}

/// Returns the three undirected (sorted-pair) edges of a triangle.
fn undirected_edges(tri: &[u32; 3]) -> [(u32, u32); 3] {
    let [a, b, c] = *tri;
    [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)]
}

/// Returns the sorted `(min, max)` endpoint pair keying a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Returns the directed edge (ordered as stored) that triangle `tri` uses for
/// the undirected edge `{a, b}`, or `None` if the triangle does not own it.
fn directed_edge(tri: &[u32; 3], a: u32, b: u32) -> Option<(u32, u32)> {
    let [c0, c1, c2] = *tri;
    for &(x, y) in &[(c0, c1), (c1, c2), (c2, c0)] {
        if (x == a && y == b) || (x == b && y == a) {
            return Some((x, y));
        }
    }
    None
}

/// Reverses the winding of a triangle in place, swapping its last two corners.
fn reverse_winding(tri: &mut [u32; 3]) {
    tri.swap(1, 2);
}

/// Returns whether every edge touched by `patch` is shared by exactly two
/// triangles — the condition for the patch to form a closed surface.
fn patch_is_closed(
    patch: &[usize],
    oriented: &[[u32; 3]],
    edge_map: &HashMap<(u32, u32), Vec<usize>>,
) -> bool {
    for &triangle in patch {
        for &edge in &undirected_edges(&oriented[triangle]) {
            if edge_map[&edge].len() != 2 {
                return false;
            }
        }
    }
    true
}

/// Returns six times the signed volume enclosed by `patch`, via the sum of
/// per-triangle origin tetra triple products, accumulated in `f64`.
fn signed_volume(patch: &[usize], oriented: &[[u32; 3]], positions: &[[f32; 3]]) -> f64 {
    let mut volume = 0.0_f64;
    for &triangle in patch {
        let [a, b, c] = oriented[triangle];
        let p0 = positions[a as usize];
        let p1 = positions[b as usize];
        let p2 = positions[c as usize];
        let v0 = [p0[0] as f64, p0[1] as f64, p0[2] as f64];
        let v1 = [p1[0] as f64, p1[1] as f64, p1[2] as f64];
        let v2 = [p2[0] as f64, p2[1] as f64, p2[2] as f64];
        let cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        volume += v0[0] * cross[0] + v0[1] * cross[1] + v0[2] * cross[2];
    }
    volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

    /// Returns whether every manifold edge is traversed in opposite directions
    /// by its two triangles — the invariant a repaired mesh must satisfy.
    fn all_edges_consistent(mesh: &TriangleMesh) -> bool {
        let mut edge_dirs: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
        for tri in mesh.indices() {
            let [a, b, c] = *tri;
            for &(x, y) in &[(a, b), (b, c), (c, a)] {
                edge_dirs.entry(sorted_pair(x, y)).or_default().push((x, y));
            }
        }
        for dirs in edge_dirs.values() {
            if dirs.len() == 2 && dirs[0] == dirs[1] {
                return false;
            }
        }
        true
    }

    /// A consistently wound unit square (two triangles) sharing diagonal (1,2).
    fn consistent_quad() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [2, 1, 3]],
        )
        .unwrap()
    }

    /// A closed tetrahedron whose four faces are all wound inward (negative
    /// signed volume), so outward reorientation must flip them.
    fn inward_tetrahedron() -> TriangleMesh {
        // Outward faces would be [0,2,1],[0,1,3],[0,3,2],[1,2,3]; reverse each.
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        )
        .unwrap()
    }

    #[test]
    fn already_consistent_quad_is_untouched() {
        let fix = make_winding_consistent(&consistent_quad()).unwrap();
        assert_eq!(fix.flipped(), 0);
        assert_eq!(fix.patches(), 1);
        assert!(all_edges_consistent(fix.mesh()));
    }

    #[test]
    fn reversed_neighbour_is_rewound() {
        // Second triangle wound the same way across the shared diagonal.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 2, 3]],
        )
        .unwrap();
        assert!(!all_edges_consistent(&mesh), "fixture should be inconsistent");
        let fix = make_winding_consistent(&mesh).unwrap();
        assert!(fix.flipped() >= 1);
        assert!(all_edges_consistent(fix.mesh()));
    }

    #[test]
    fn open_patch_is_not_volume_flipped() {
        // Two open triangles, second reversed: exactly one consistency flip,
        // no closed-surface volume flip.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 2, 3]],
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.flipped(), 1, "only the one inconsistent neighbour");
    }

    #[test]
    fn closed_patch_is_oriented_outward() {
        let fix = make_winding_consistent(&inward_tetrahedron()).unwrap();
        assert_eq!(fix.patches(), 1);
        assert!(all_edges_consistent(fix.mesh()));
        // Positive signed volume ⇒ outward-facing closed surface.
        let all: Vec<usize> = (0..fix.mesh().triangle_count()).collect();
        let vol = signed_volume(all.as_slice(), fix.mesh().indices(), fix.mesh().positions());
        assert!(vol > 0.0, "closed surface should be outward (vol = {vol})");
    }

    #[test]
    fn disjoint_patches_are_counted() {
        // Two separate quads → two patches.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [11.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [2, 1, 3], [4, 5, 6], [6, 5, 7]],
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.patches(), 2);
        assert!(all_edges_consistent(fix.mesh()));
    }

    #[test]
    fn non_manifold_edge_is_reported() {
        // Three triangles sharing edge (0,1): one non-manifold edge.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.non_manifold_edges(), 1);
    }

    #[test]
    fn single_triangle_is_untouched() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.flipped(), 0);
        assert_eq!(fix.patches(), 1);
        assert_eq!(fix.mesh().indices(), &[[0, 1, 2]]);
    }

    #[test]
    fn attributes_are_preserved() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.mesh().normals(), mesh.normals());
        assert_eq!(fix.mesh().uvs(), mesh.uvs());
    }

    #[test]
    fn empty_mesh_has_no_patches() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let fix = make_winding_consistent(&mesh).unwrap();
        assert_eq!(fix.patches(), 0);
        assert_eq!(fix.flipped(), 0);
    }

    #[test]
    fn repaired_mesh_is_ray_traceable() {
        let fix = make_winding_consistent(&consistent_quad()).unwrap();
        let bvh = TriangleMeshBvh::build(fix.into_mesh());
        let ray = Ray::infinite([0.53, 0.47, 1.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit repaired quad");
        assert!(hit.position[2].abs() < 1e-6, "hit z = {}", hit.position[2]);
    }
}
