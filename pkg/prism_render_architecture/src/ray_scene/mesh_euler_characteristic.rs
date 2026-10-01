//! Integer topology summary (Euler characteristic / genus) for the `CPU`
//! golden path.
//!
//! Before a mesh enters an `AAA` pipeline it is validated for topological
//! health: is it closed, how many separate shells does it have, how many
//! boundary loops (holes) remain, and — for a clean closed surface — what is
//! its genus (handle count)? These are all read off the Euler characteristic
//! `V - E + F` together with the boundary-loop and component counts, using only
//! integer arithmetic and no floating-point or transcendental operation.
//!
//! [`mesh_topology`] counts the referenced vertices, unique undirected edges,
//! and faces; detects boundary edges (one incident face) and non-manifold
//! edges (more than two); groups boundary edges into loops and faces into
//! connected shells; and derives the genus for the well-behaved case of a
//! single closed orientable manifold component.

use std::collections::HashMap;
use std::collections::HashSet;

use super::triangle_mesh::TriangleMesh;

/// Integer topology summary of a [`TriangleMesh`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshTopology {
    /// Number of distinct vertices referenced by at least one triangle.
    vertex_count: usize,
    /// Number of unique undirected edges.
    edge_count: usize,
    /// Number of triangles (faces).
    face_count: usize,
    /// Number of boundary edges (exactly one incident face).
    boundary_edge_count: usize,
    /// Number of closed boundary loops formed by the boundary edges.
    boundary_loop_count: usize,
    /// Number of non-manifold edges (more than two incident faces).
    nonmanifold_edge_count: usize,
    /// Number of connected shells (faces linked through shared edges).
    component_count: usize,
}

impl MeshTopology {
    /// Returns the number of distinct referenced vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Returns the number of unique undirected edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Returns the number of faces (triangles).
    pub fn face_count(&self) -> usize {
        self.face_count
    }

    /// Returns the number of boundary edges (one incident face).
    pub fn boundary_edge_count(&self) -> usize {
        self.boundary_edge_count
    }

    /// Returns the number of closed boundary loops (holes / open borders).
    pub fn boundary_loop_count(&self) -> usize {
        self.boundary_loop_count
    }

    /// Returns the number of non-manifold edges (more than two faces).
    pub fn nonmanifold_edge_count(&self) -> usize {
        self.nonmanifold_edge_count
    }

    /// Returns the number of connected shells.
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// Returns the Euler characteristic `V - E + F`.
    pub fn euler_characteristic(&self) -> i64 {
        self.vertex_count as i64 - self.edge_count as i64 + self.face_count as i64
    }

    /// Returns whether the surface is a closed manifold: no boundary edges and
    /// no non-manifold edges, with at least one face.
    pub fn is_closed_manifold(&self) -> bool {
        self.face_count > 0
            && self.boundary_edge_count == 0
            && self.nonmanifold_edge_count == 0
    }

    /// Returns the genus (handle count) for a single closed orientable manifold
    /// component, or `None` when the surface is open, non-manifold, empty, or
    /// has more than one shell.
    ///
    /// For such a surface `V - E + F = 2 - 2g`, so `g = (2 - chi) / 2`.
    pub fn genus(&self) -> Option<u32> {
        if !self.is_closed_manifold() || self.component_count != 1 {
            return None;
        }
        let twice_g = 2 - self.euler_characteristic();
        if twice_g < 0 || twice_g % 2 != 0 {
            return None;
        }
        Some((twice_g / 2) as u32)
    }
}

/// Orders two vertex indices into a canonical `(min, max)` undirected key.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Minimal union-find over `u32` keys for grouping boundary vertices and faces.
struct UnionFind {
    /// Parent link per element, keyed by dense index.
    parent: Vec<usize>,
}

impl UnionFind {
    /// Creates a forest of `n` singletons.
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }

    /// Returns the representative root of `x`, with path halving.
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merges the sets containing `a` and `b`.
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Computes the integer topology summary of `mesh`.
pub fn mesh_topology(mesh: &TriangleMesh) -> MeshTopology {
    let indices = mesh.indices();
    let face_count = indices.len();

    // Count incident faces per undirected edge and tally referenced vertices.
    let mut edge_faces: HashMap<(u32, u32), usize> = HashMap::new();
    let mut referenced: HashSet<u32> = HashSet::new();
    for tri in indices {
        for &v in tri {
            referenced.insert(v);
        }
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            *edge_faces.entry(sorted_pair(a, b)).or_insert(0) += 1;
        }
    }

    let edge_count = edge_faces.len();
    let vertex_count = referenced.len();
    let mut boundary_edge_count = 0usize;
    let mut nonmanifold_edge_count = 0usize;
    let mut boundary_edges: Vec<(u32, u32)> = Vec::new();
    for (&edge, &count) in &edge_faces {
        match count {
            1 => {
                boundary_edge_count += 1;
                boundary_edges.push(edge);
            }
            2 => {}
            _ => nonmanifold_edge_count += 1,
        }
    }

    let boundary_loop_count = count_boundary_loops(&boundary_edges);
    let component_count = count_components(indices, &edge_faces);

    MeshTopology {
        vertex_count,
        edge_count,
        face_count,
        boundary_edge_count,
        boundary_loop_count,
        nonmanifold_edge_count,
        component_count,
    }
}

/// Counts closed boundary loops by grouping boundary vertices that share a
/// boundary edge; each connected group of boundary vertices is one loop.
fn count_boundary_loops(boundary_edges: &[(u32, u32)]) -> usize {
    if boundary_edges.is_empty() {
        return 0;
    }
    // Dense-index the boundary vertices.
    let mut index_of: HashMap<u32, usize> = HashMap::new();
    for &(a, b) in boundary_edges {
        let next = index_of.len();
        index_of.entry(a).or_insert(next);
        let next = index_of.len();
        index_of.entry(b).or_insert(next);
    }
    let mut uf = UnionFind::new(index_of.len());
    for &(a, b) in boundary_edges {
        uf.union(index_of[&a], index_of[&b]);
    }
    let mut roots: HashSet<usize> = HashSet::new();
    let keys: Vec<usize> = index_of.values().copied().collect();
    for k in keys {
        roots.insert(uf.find(k));
    }
    roots.len()
}

/// Counts connected shells by unioning faces that share an undirected edge.
fn count_components(indices: &[[u32; 3]], edge_faces: &HashMap<(u32, u32), usize>) -> usize {
    let face_count = indices.len();
    if face_count == 0 {
        return 0;
    }
    // Map each edge to the faces touching it, then union faces per edge.
    let mut edge_to_faces: HashMap<(u32, u32), Vec<usize>> =
        HashMap::with_capacity(edge_faces.len());
    for (face_index, tri) in indices.iter().enumerate() {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edge_to_faces.entry(sorted_pair(a, b)).or_default().push(face_index);
        }
    }
    let mut uf = UnionFind::new(face_count);
    for faces in edge_to_faces.values() {
        for window in faces.windows(2) {
            uf.union(window[0], window[1]);
        }
    }
    let mut roots: HashSet<usize> = HashSet::new();
    for f in 0..face_count {
        roots.insert(uf.find(f));
    }
    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a closed tetrahedron: 4 vertices, 4 outward triangles.
    fn tetrahedron() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        )
        .expect("valid tetrahedron")
    }

    /// Builds a closed axis-aligned unit box: 8 vertices, 12 triangles.
    fn unit_box() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [0, 4, 7], [0, 7, 3],
            ],
        )
        .expect("valid box")
    }

    #[test]
    fn single_triangle_is_open_disc() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .expect("valid triangle");
        let t = mesh_topology(&mesh);
        assert_eq!((t.vertex_count(), t.edge_count(), t.face_count()), (3, 3, 1));
        assert_eq!(t.euler_characteristic(), 1);
        assert_eq!(t.boundary_edge_count(), 3);
        assert_eq!(t.boundary_loop_count(), 1);
        assert_eq!(t.component_count(), 1);
        assert!(!t.is_closed_manifold());
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn tetrahedron_is_genus_zero_sphere() {
        let t = mesh_topology(&tetrahedron());
        assert_eq!((t.vertex_count(), t.edge_count(), t.face_count()), (4, 6, 4));
        assert_eq!(t.euler_characteristic(), 2);
        assert_eq!(t.boundary_edge_count(), 0);
        assert_eq!(t.boundary_loop_count(), 0);
        assert_eq!(t.nonmanifold_edge_count(), 0);
        assert!(t.is_closed_manifold());
        assert_eq!(t.genus(), Some(0));
    }

    #[test]
    fn box_is_genus_zero_sphere() {
        let t = mesh_topology(&unit_box());
        assert_eq!((t.vertex_count(), t.edge_count(), t.face_count()), (8, 18, 12));
        assert_eq!(t.euler_characteristic(), 2);
        assert!(t.is_closed_manifold());
        assert_eq!(t.component_count(), 1);
        assert_eq!(t.genus(), Some(0));
    }

    #[test]
    fn open_quad_has_one_boundary_loop() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .expect("valid quad");
        let t = mesh_topology(&mesh);
        assert_eq!((t.vertex_count(), t.edge_count(), t.face_count()), (4, 5, 2));
        assert_eq!(t.euler_characteristic(), 1);
        assert_eq!(t.boundary_loop_count(), 1);
        assert_eq!(t.boundary_edge_count(), 4);
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn two_disjoint_triangles_have_two_components_and_loops() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .expect("valid pair");
        let t = mesh_topology(&mesh);
        assert_eq!(t.component_count(), 2);
        assert_eq!(t.boundary_loop_count(), 2);
        assert_eq!(t.euler_characteristic(), 2);
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn nonmanifold_fan_is_flagged() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0], [0.5, -1.0, 0.0], [0.5, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
        )
        .expect("valid fan");
        let t = mesh_topology(&mesh);
        assert_eq!(t.nonmanifold_edge_count(), 1);
        assert!(!t.is_closed_manifold());
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn open_box_with_missing_face_has_one_loop() {
        // Unit box minus its two bottom triangles: a 10-triangle open shell.
        let full = unit_box();
        let kept: Vec<[u32; 3]> = full.indices()[2..].to_vec();
        let mesh = TriangleMesh::new(
            full.positions().to_vec(),
            Vec::new(),
            Vec::new(),
            kept,
        )
        .expect("valid open box");
        let t = mesh_topology(&mesh);
        assert_eq!(t.face_count(), 10);
        assert_eq!(t.boundary_loop_count(), 1);
        assert!(!t.is_closed_manifold());
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn empty_mesh_has_zero_topology() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
            .expect("valid empty mesh");
        let t = mesh_topology(&mesh);
        assert_eq!(t.euler_characteristic(), 0);
        assert_eq!(t.component_count(), 0);
        assert_eq!(t.boundary_loop_count(), 0);
        assert!(!t.is_closed_manifold());
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn two_tetrahedra_have_two_closed_shells_but_no_genus() {
        // Two disjoint tetrahedra: closed per shell, but two components, so the
        // single-component genus is undefined.
        let a = tetrahedron();
        let mut positions = a.positions().to_vec();
        let mut indices = a.indices().to_vec();
        let offset = positions.len() as u32;
        for p in a.positions() {
            positions.push([p[0] + 10.0, p[1], p[2]]);
        }
        for tri in a.indices() {
            indices.push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices)
            .expect("valid two tetrahedra");
        let t = mesh_topology(&mesh);
        assert_eq!(t.component_count(), 2);
        assert_eq!(t.boundary_edge_count(), 0);
        assert!(t.is_closed_manifold());
        assert_eq!(t.euler_characteristic(), 4);
        assert_eq!(t.genus(), None);
    }

    #[test]
    fn referenced_vertices_ignore_unused_positions() {
        // Four positions, but the triangle uses only three.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .expect("valid mesh");
        let t = mesh_topology(&mesh);
        assert_eq!(t.vertex_count(), 3);
    }
}
