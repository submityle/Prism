//! Coplanar-face recovery for triangulated convex hulls.
//!
//! [`convex_hull`](super::convex_hull) emits a *simplicial* surface: every face
//! is a triangle, so a cube arrives as 12 triangles rather than 6 squares. The
//! solver's `SAT` clipping, contact-manifold generation, and the GPU host packer
//! all want the hull's true **polygonal faces** -- the maximal sets of coplanar
//! triangles merged into one convex polygon with a single support plane. This is
//! the representation `PhysX` exposes through `PxConvexMesh::getPolygonData`,
//! Jolt through `ConvexHullShape::GetFace`, and Chaos through
//! `FConvexBuilder`'s planar faces. [`merge_coplanar_faces`] recovers it.
//!
//! # Algorithm
//!
//! 1. Compute each triangle's outward normal.
//! 2. Union triangles that share an edge and whose normals agree within an
//!    angular tolerance -- on a convex polytope that is exactly coplanarity.
//! 3. For each merged group, keep the directed edges whose reverse is absent
//!    from the group: those are the polygon boundary. Because the triangles are
//!    wound counter-clockwise as seen from outside, the boundary directed edges
//!    form a single counter-clockwise cycle, which is threaded into an ordered
//!    vertex loop.
//! 4. The face plane is the area-weighted mean triangle normal (re-normalized)
//!    with the offset taken at the boundary centroid.
//!
//! All grouping and ordering is derived from deterministic index loops (hash
//! maps are used only for `O(1)` edge lookups, never iterated for output), so
//! the recovered faces are bit-for-bit reproducible -- the same requirement the
//! hull builder and state hashing impose.
//!
//! # Provenance
//!
//! Coplanar-triangle merging and boundary-loop extraction are textbook
//! polygon-mesh operations. This module contains **no Unreal Engine source or
//! derived code**.

use std::collections::HashMap;

use glam::Vec3;

/// A single convex polygonal face of a hull: its outward support plane
/// `normal . x = offset` and the boundary vertex loop (indices into the hull
/// vertices) wound counter-clockwise as seen from outside.
#[derive(Clone, Debug, PartialEq)]
pub struct PolygonFace {
    /// Outward unit face normal.
    pub normal: Vec3,
    /// Plane offset: `normal . x == offset` for every boundary vertex.
    pub offset: f32,
    /// Boundary vertex indices, counter-clockwise seen from outside.
    pub vertices: Vec<u32>,
}

impl PolygonFace {
    /// Number of boundary vertices (edges) of the polygon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// True when the face carries no boundary vertices (never happens for a
    /// well-formed hull face, but keeps the API total).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// Default coplanarity threshold: triangles whose unit normals have a dot
/// product at least this large are treated as coplanar. `0.9998` corresponds to
/// roughly a `1.15` degree cone -- the order of magnitude content tools use for
/// convex face welding. Expressed as a cosine (rather than an angle in degrees)
/// so the merge stays free of transcendental calls and bit-for-bit
/// deterministic across platforms.
pub const DEFAULT_COPLANAR_DOT: f32 = 0.9998;

/// Merges the coplanar triangles of a convex hull surface into polygonal faces.
///
/// `vertices` / `triangles` are a hull surface as produced by
/// [`convex_hull`](super::convex_hull): triangles index into vertices and are
/// wound counter-clockwise as seen from outside. `min_normal_dot` is the
/// smallest dot product (of unit normals) at which two adjacent triangles are
/// merged; pass [`DEFAULT_COPLANAR_DOT`] for the standard behaviour.
///
/// Returns one [`PolygonFace`] per maximal coplanar group. A tetrahedron yields
/// four triangular faces; a cube yields six quads. Returns an empty vector when
/// `triangles` is empty.
#[must_use]
pub fn merge_coplanar_faces(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    min_normal_dot: f32,
) -> Vec<PolygonFace> {
    if triangles.is_empty() {
        return Vec::new();
    }

    let normals: Vec<Vec3> = triangles
        .iter()
        .map(|&t| triangle_normal(vertices, t))
        .collect();

    let mut parent: Vec<usize> = (0..triangles.len()).collect();

    // Union adjacent triangles (sharing an undirected edge) with near-equal
    // normals. Each undirected edge of a closed convex surface is shared by
    // exactly two triangles.
    let mut edge_owner: HashMap<(u32, u32), usize> = HashMap::new();
    for (tri_idx, &tri) in triangles.iter().enumerate() {
        for &(a, b) in &directed_edges(tri) {
            let key = undirected(a, b);
            if let Some(&other) = edge_owner.get(&key) {
                let dot = normals[tri_idx]
                    .normalize_or_zero()
                    .dot(normals[other].normalize_or_zero());
                if dot >= min_normal_dot {
                    union(&mut parent, tri_idx, other);
                }
            } else {
                edge_owner.insert(key, tri_idx);
            }
        }
    }

    // Collect triangle indices per group root, preserving ascending triangle
    // order so the output is deterministic.
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for tri_idx in 0..triangles.len() {
        let root = find(&mut parent, tri_idx);
        let slot = *group_of.entry(root).or_insert_with(|| {
            groups.push((root, Vec::new()));
            groups.len() - 1
        });
        groups[slot].1.push(tri_idx);
    }

    groups
        .into_iter()
        .filter_map(|(_, tris)| build_face(vertices, triangles, &normals, &tris))
        .collect()
}

/// Builds a single polygonal face from a coplanar triangle group.
fn build_face(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    normals: &[Vec3],
    group: &[usize],
) -> Option<PolygonFace> {
    // Area-weighted mean normal (triangle normal length is twice the area).
    let mut weighted = Vec3::ZERO;
    for &tri_idx in group {
        weighted += normals[tri_idx];
    }
    let normal = weighted.normalize_or_zero();
    if normal == Vec3::ZERO {
        return None;
    }

    // Directed boundary edges: those whose reverse is not also in the group.
    let mut present: HashMap<(u32, u32), ()> = HashMap::new();
    for &tri_idx in group {
        for &(a, b) in &directed_edges(triangles[tri_idx]) {
            present.insert((a, b), ());
        }
    }
    let mut next: HashMap<u32, u32> = HashMap::new();
    for &tri_idx in group {
        for &(a, b) in &directed_edges(triangles[tri_idx]) {
            if !present.contains_key(&(b, a)) {
                next.insert(a, b);
            }
        }
    }
    if next.is_empty() {
        return None;
    }

    // Thread the boundary loop starting from the lowest vertex index so the
    // winding start is deterministic.
    let start = *next.keys().min()?;
    let mut loop_verts = Vec::with_capacity(next.len());
    let mut current = start;
    for _ in 0..=next.len() {
        loop_verts.push(current);
        let &n = next.get(&current)?;
        if n == start {
            break;
        }
        current = n;
        if loop_verts.len() > next.len() {
            // The boundary did not close into a single simple cycle.
            return None;
        }
    }

    // Offset at the boundary centroid (all boundary vertices are coplanar).
    let mut centroid = Vec3::ZERO;
    for &v in &loop_verts {
        centroid += vertices[v as usize];
    }
    centroid /= loop_verts.len() as f32;
    let offset = normal.dot(centroid);

    Some(PolygonFace {
        normal,
        offset,
        vertices: loop_verts,
    })
}

/// Outward (unnormalized) normal of a triangle, length proportional to twice its
/// area; the hull winding makes this point outward.
fn triangle_normal(vertices: &[Vec3], tri: [u32; 3]) -> Vec3 {
    let a = vertices[tri[0] as usize];
    let b = vertices[tri[1] as usize];
    let c = vertices[tri[2] as usize];
    (b - a).cross(c - a)
}

/// The three directed edges of a triangle in winding order.
fn directed_edges(tri: [u32; 3]) -> [(u32, u32); 3] {
    [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
}

/// An undirected edge key with the lower vertex index first.
fn undirected(a: u32, b: u32) -> (u32, u32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Union-find root with path compression.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union-find merge; the lower root wins so output ordering stays deterministic.
fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
    parent[hi] = lo;
}

#[cfg(test)]
mod tests {
    use super::super::convex_hull;
    use super::*;

    fn box_corners(h: Vec3) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(8);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    pts.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        pts
    }

    /// Newell normal of a vertex loop; agrees with the stored face normal when
    /// the loop is wound counter-clockwise as seen from outside.
    fn loop_normal(vertices: &[Vec3], loop_verts: &[u32]) -> Vec3 {
        let mut n = Vec3::ZERO;
        for i in 0..loop_verts.len() {
            let a = vertices[loop_verts[i] as usize];
            let b = vertices[loop_verts[(i + 1) % loop_verts.len()] as usize];
            n += a.cross(b);
        }
        n.normalize_or_zero()
    }

    #[test]
    fn empty_surface_has_no_faces() {
        assert!(merge_coplanar_faces(&[], &[], DEFAULT_COPLANAR_DOT).is_empty());
    }

    #[test]
    fn cube_merges_to_six_quads() {
        let pts = box_corners(Vec3::ONE);
        let (verts, tris) = convex_hull(&pts).expect("cube hull");
        assert_eq!(tris.len(), 12, "cube hull is 12 triangles");

        let faces = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        assert_eq!(faces.len(), 6, "a cube has six faces");
        for face in &faces {
            assert_eq!(face.len(), 4, "each cube face is a quad");
            // Winding is CCW from outside: Newell normal matches stored normal.
            let ln = loop_normal(&verts, &face.vertices);
            assert!(ln.dot(face.normal) > 0.99, "face loop must be outward CCW");
            // Every boundary vertex lies on the face plane.
            for &v in &face.vertices {
                let d = face.normal.dot(verts[v as usize]) - face.offset;
                assert!(d.abs() < 1.0e-4, "vertex off plane by {d}");
            }
            // Normal is axis-aligned and offset is the half-extent (unit box).
            let aligned = face.normal.abs();
            let max = aligned.max_element();
            assert!(max > 0.999, "cube face normal must be axis aligned");
            assert!((face.offset - 1.0).abs() < 1.0e-4, "offset {}", face.offset);
        }

        // The six outward normals are the six axis directions, each once.
        for axis in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            let hits = faces.iter().filter(|f| f.normal.dot(axis) > 0.99).count();
            assert_eq!(hits, 1, "exactly one face faces {axis:?}");
        }
    }

    #[test]
    fn tetrahedron_keeps_triangular_faces() {
        let pts = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let (verts, tris) = convex_hull(&pts).expect("tetra hull");
        let faces = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        assert_eq!(faces.len(), 4, "tetrahedron has four faces");
        for face in &faces {
            assert_eq!(face.len(), 3, "no coplanar merge on a tetrahedron");
            let ln = loop_normal(&verts, &face.vertices);
            assert!(ln.dot(face.normal) > 0.99);
        }
    }

    #[test]
    fn octahedron_keeps_eight_triangles() {
        let pts = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let (verts, tris) = convex_hull(&pts).expect("octahedron hull");
        let faces = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        assert_eq!(faces.len(), 8);
        assert!(faces.iter().all(|f| f.len() == 3));
    }

    #[test]
    fn box_with_coplanar_filler_still_six_faces() {
        // Extra points sitting exactly on cube faces add coplanar triangles that
        // must merge away, leaving six faces (with more boundary vertices).
        let mut pts = box_corners(Vec3::new(2.0, 1.0, 1.5));
        pts.push(Vec3::new(0.0, 1.0, 0.0)); // on +Y face
        pts.push(Vec3::new(0.0, -1.0, 0.0)); // on -Y face
        let (verts, tris) = convex_hull(&pts).expect("filled box hull");
        let faces = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        assert_eq!(faces.len(), 6, "coplanar filler must not add faces");
        for face in &faces {
            let ln = loop_normal(&verts, &face.vertices);
            assert!(ln.dot(face.normal) > 0.99, "outward CCW loop");
            for &v in &face.vertices {
                let d = face.normal.dot(verts[v as usize]) - face.offset;
                assert!(d.abs() < 1.0e-4, "vertex off plane by {d}");
            }
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let pts = box_corners(Vec3::new(1.0, 2.0, 3.0));
        let (verts, tris) = convex_hull(&pts).expect("hull");
        let first = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        for _ in 0..8 {
            let again = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
            assert_eq!(first, again, "face recovery must be reproducible");
        }
    }

    #[test]
    fn face_count_matches_euler_relation() {
        // For a convex polytope V - E + F = 2. A cube: 8 - 12 + 6 = 2.
        let pts = box_corners(Vec3::ONE);
        let (verts, tris) = convex_hull(&pts).expect("hull");
        let faces = merge_coplanar_faces(&verts, &tris, DEFAULT_COPLANAR_DOT);
        let edge_total: usize = faces.iter().map(PolygonFace::len).sum();
        // Each polygon edge is shared by two faces.
        let edges = edge_total / 2;
        let v = verts.len();
        let f = faces.len();
        assert_eq!(v + f - edges, 2, "Euler characteristic must hold");
    }
}
