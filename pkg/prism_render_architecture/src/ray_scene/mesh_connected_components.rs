//! Connected-component labelling and island extraction for triangle meshes on
//! the `CPU` golden path.
//!
//! A mesh assembled from several objects — or produced by boolean/clipping
//! operations — often contains multiple disjoint surface *islands* that share
//! no vertices. Many downstream passes want to treat each island on its own:
//! normal-orientation fixes, per-island simplification budgets, physics
//! convex-decomposition seeds, or simply culling stray debris triangles. This
//! module computes which triangles and vertices belong to the same island and
//! can split the mesh into one [`TriangleMesh`] per island.
//!
//! [`connected_components`] runs a union–find over the vertex set, uniting the
//! three corners of every triangle, so two triangles land in the same
//! component exactly when a chain of shared vertices links them.
//! [`MeshComponents`] then exposes a dense component index per triangle and per
//! referenced vertex (vertices touched by no triangle report `None`).
//! [`split_components`] rebuilds each island as a standalone mesh with its
//! vertex pool compacted and its attributes carried along, returning the
//! islands ordered by descending triangle count so the largest body is first.
//!
//! The analysis is pure integer bookkeeping — union–find with path compression
//! and union by rank, plus index remapping — touching no floating-point math
//! and therefore trivially honouring the golden-path transcendental ban.

use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Disjoint-set forest over vertex indices with path compression and union by
/// rank, used to group triangle corners into connected components.
struct UnionFind {
    /// Parent pointer for each element; a root points at itself.
    parent: Vec<u32>,
    /// Union-by-rank upper bound on each subtree's height.
    rank: Vec<u8>,
}

impl UnionFind {
    /// Creates a forest of `n` singleton sets.
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    /// Returns the representative root of `x`, compressing the path to it.
    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Second pass: point every node on the path directly at the root.
        let mut node = x;
        while self.parent[node as usize] != root {
            let next = self.parent[node as usize];
            self.parent[node as usize] = root;
            node = next;
        }
        root
    }

    /// Merges the sets containing `a` and `b`, attaching the shorter tree under
    /// the taller to keep the forest shallow.
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (ra, rb) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
    }
}

/// Connected-component labelling of a [`TriangleMesh`], parallel to its index
/// and vertex buffers.
pub struct MeshComponents {
    /// Dense component index for each triangle, in `0..count`, parallel to the
    /// mesh index buffer.
    triangle_component: Vec<u32>,
    /// Dense component index for each vertex, or `None` when the vertex is
    /// referenced by no triangle.
    vertex_component: Vec<Option<u32>>,
    /// Triangle count of each component, indexed by component id.
    triangle_counts: Vec<u32>,
}

impl MeshComponents {
    /// Returns the number of connected components that contain at least one
    /// triangle.
    pub fn count(&self) -> u32 {
        self.triangle_counts.len() as u32
    }

    /// Returns the component index of triangle `triangle`.
    ///
    /// # Panics
    ///
    /// Panics if `triangle` is out of range of the source mesh's index buffer.
    pub fn triangle_component(&self, triangle: usize) -> u32 {
        self.triangle_component[triangle]
    }

    /// Returns the component index of vertex `vertex`, or `None` if the vertex
    /// is referenced by no triangle.
    ///
    /// # Panics
    ///
    /// Panics if `vertex` is out of range of the source mesh's vertex pool.
    pub fn vertex_component(&self, vertex: usize) -> Option<u32> {
        self.vertex_component[vertex]
    }

    /// Returns the per-component triangle counts, indexed by component id.
    pub fn triangle_counts(&self) -> &[u32] {
        &self.triangle_counts
    }
}

/// Labels the connected components of `mesh` by uniting the corners of every
/// triangle, assigning each component a dense index in order of first
/// appearance while scanning triangles.
pub fn connected_components(mesh: &TriangleMesh) -> MeshComponents {
    let vertex_count = mesh.vertex_count();
    let mut uf = UnionFind::new(vertex_count);
    for tri in mesh.indices() {
        let [a, b, c] = *tri;
        uf.union(a, b);
        uf.union(b, c);
    }

    // Snapshot each vertex's root once, then assign dense component ids in the
    // order triangles first reach a new root (deterministic labelling).
    let mut root = vec![0_u32; vertex_count];
    for (vertex, slot) in root.iter_mut().enumerate() {
        *slot = uf.find(vertex as u32);
    }

    let mut root_to_component: HashMap<u32, u32> = HashMap::new();
    let mut triangle_counts: Vec<u32> = Vec::new();
    let mut triangle_component = Vec::with_capacity(mesh.indices().len());
    for tri in mesh.indices() {
        let r = root[tri[0] as usize];
        let component = *root_to_component.entry(r).or_insert_with(|| {
            let id = triangle_counts.len() as u32;
            triangle_counts.push(0);
            id
        });
        triangle_counts[component as usize] += 1;
        triangle_component.push(component);
    }

    let mut referenced = vec![false; vertex_count];
    for tri in mesh.indices() {
        for &corner in tri {
            referenced[corner as usize] = true;
        }
    }
    let vertex_component = (0..vertex_count)
        .map(|v| {
            if referenced[v] {
                Some(root_to_component[&root[v]])
            } else {
                None
            }
        })
        .collect();

    MeshComponents {
        triangle_component,
        vertex_component,
        triangle_counts,
    }
}

/// Splits `mesh` into one standalone [`TriangleMesh`] per connected component,
/// each with its vertex pool compacted to only the vertices it uses and its
/// attributes carried along.
///
/// Islands are returned ordered by descending triangle count (ties broken by
/// ascending component id), so index `0` is the largest body. A mesh with no
/// triangles yields an empty vector.
///
/// # Errors
///
/// Returns [`TriangleMeshError`] if any extracted island fails validation; by
/// construction the compacted pools are consistent, so this is not expected in
/// practice.
pub fn split_components(mesh: &TriangleMesh) -> Result<Vec<TriangleMesh>, TriangleMeshError> {
    let components = connected_components(mesh);
    let count = components.count() as usize;
    if count == 0 {
        return Ok(Vec::new());
    }

    let has_normals = mesh.has_normals();
    let has_uvs = mesh.has_uvs();

    // Per-component builders accumulate a compacted vertex pool via a remap of
    // source index -> local index.
    let mut remaps: Vec<HashMap<u32, u32>> = vec![HashMap::new(); count];
    let mut positions: Vec<Vec<[f32; 3]>> = vec![Vec::new(); count];
    let mut normals: Vec<Vec<[f32; 3]>> = vec![Vec::new(); count];
    let mut uvs: Vec<Vec<[f32; 2]>> = vec![Vec::new(); count];
    let mut indices: Vec<Vec<[u32; 3]>> = vec![Vec::new(); count];

    for (triangle, tri) in mesh.indices().iter().enumerate() {
        let component = components.triangle_component(triangle) as usize;
        let mut local = [0_u32; 3];
        for (slot, &source) in local.iter_mut().zip(tri.iter()) {
            let next = positions[component].len() as u32;
            let mapped = *remaps[component].entry(source).or_insert(next);
            if mapped == next {
                positions[component].push(mesh.positions()[source as usize]);
                if has_normals {
                    normals[component].push(mesh.normals()[source as usize]);
                }
                if has_uvs {
                    uvs[component].push(mesh.uvs()[source as usize]);
                }
            }
            *slot = mapped;
        }
        indices[component].push(local);
    }

    // Build the island meshes, then order them largest-first.
    let mut order: Vec<usize> = (0..count).collect();
    let counts = components.triangle_counts();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));

    let mut islands = Vec::with_capacity(count);
    for component in order {
        let pos = core::mem::take(&mut positions[component]);
        let nrm = core::mem::take(&mut normals[component]);
        let uv = core::mem::take(&mut uvs[component]);
        let idx = core::mem::take(&mut indices[component]);
        islands.push(TriangleMesh::new(pos, nrm, uv, idx)?);
    }
    Ok(islands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

    /// Two triangles forming a unit square, sharing the diagonal (1, 2): a
    /// single connected island.
    fn connected_quad() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 3, 2]],
        )
        .unwrap()
    }

    /// Two triangles with no shared vertices: two disjoint islands.
    fn two_islands() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap()
    }

    #[test]
    fn single_island_is_one_component() {
        let comps = connected_components(&connected_quad());
        assert_eq!(comps.count(), 1);
        assert_eq!(comps.triangle_component(0), 0);
        assert_eq!(comps.triangle_component(1), 0);
        assert_eq!(comps.triangle_counts(), &[2]);
    }

    #[test]
    fn disjoint_triangles_are_two_components() {
        let comps = connected_components(&two_islands());
        assert_eq!(comps.count(), 2);
        assert_ne!(
            comps.triangle_component(0),
            comps.triangle_component(1),
            "disjoint triangles must differ"
        );
    }

    #[test]
    fn shared_vertex_merges_components() {
        // Three triangles chained through shared corner 2 → one component.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 3, 2], [3, 4, 2]],
        )
        .unwrap();
        let comps = connected_components(&mesh);
        assert_eq!(comps.count(), 1);
    }

    #[test]
    fn unreferenced_vertex_has_no_component() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [9.0, 9.0, 9.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let comps = connected_components(&mesh);
        assert_eq!(comps.vertex_component(0), Some(0));
        assert_eq!(comps.vertex_component(3), None);
    }

    #[test]
    fn empty_mesh_has_no_components() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let comps = connected_components(&mesh);
        assert_eq!(comps.count(), 0);
        assert!(split_components(&mesh).unwrap().is_empty());
    }

    #[test]
    fn split_extracts_each_island_compacted() {
        let islands = split_components(&two_islands()).unwrap();
        assert_eq!(islands.len(), 2);
        for island in &islands {
            // Each island uses exactly its three corners — no stray vertices.
            assert_eq!(island.vertex_count(), 3);
            assert_eq!(island.triangle_count(), 1);
        }
    }

    #[test]
    fn split_orders_largest_island_first() {
        // Island A: two triangles. Island B: one triangle, placed first in the
        // buffer so ordering is not incidental.
        let mesh = TriangleMesh::new(
            vec![
                // Island B (one tri) at the front.
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                // Island A (two tris).
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [3, 4, 5], [4, 6, 5]],
        )
        .unwrap();
        let islands = split_components(&mesh).unwrap();
        assert_eq!(islands.len(), 2);
        assert_eq!(islands[0].triangle_count(), 2, "largest first");
        assert_eq!(islands[1].triangle_count(), 1);
    }

    #[test]
    fn split_carries_attributes() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let islands = split_components(&mesh).unwrap();
        assert_eq!(islands.len(), 1);
        assert!(islands[0].has_normals());
        assert!(islands[0].has_uvs());
        assert_eq!(islands[0].normals().len(), islands[0].vertex_count());
        assert_eq!(islands[0].uvs().len(), islands[0].vertex_count());
    }

    #[test]
    fn triangle_components_are_in_range() {
        let comps = connected_components(&two_islands());
        for t in 0..2 {
            assert!(comps.triangle_component(t) < comps.count());
        }
    }

    #[test]
    fn extracted_island_is_ray_traceable() {
        let islands = split_components(&connected_quad()).unwrap();
        let island = islands.into_iter().next().unwrap();
        let bvh = TriangleMeshBvh::build(island);
        let ray = Ray::infinite([0.53, 0.47, 1.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit island");
        assert!(hit.position[2].abs() < 1e-6, "hit z = {}", hit.position[2]);
    }
}
