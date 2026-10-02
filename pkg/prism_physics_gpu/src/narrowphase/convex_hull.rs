//! Convex polyhedron geometry: the shared shape type the convex-versus-convex
//! `GJK`/`EPA` narrow phase operates on, matching the layout its `WGSL` kernel
//! uploads.
//!
//! A [`ConvexHull`] is immutable local-space geometry: a vertex cloud plus the
//! face loops and unique edges that close it into a solid. It carries no pose;
//! the narrow phase places two hulls in the world through a translation and a
//! rotation passed at query time, exactly as a shared cooked convex mesh is
//! instanced under many rigid-body transforms. This keeps the geometry upload
//! to the device once per distinct shape while the per-body pose rides in a
//! small separate array.
//!
//! # What each table is for
//!
//! * [`ConvexHull::vertices`] feed the support function
//!   ([`ConvexHull::support_local`]/[`ConvexHull::support_point`]): the farthest
//!   vertex along a direction is all `GJK` and `EPA` ever need from a convex
//!   solid, so the Minkowski-difference support is one `argmax` per hull.
//! * [`ConvexHull::faces`] carry the outward unit normal, the plane offset, and
//!   the counter-clockwise vertex loop of every face. The manifold stage picks a
//!   reference face by normal and clips the incident face against its side
//!   planes, so the winding must be outward-consistent.
//! * [`ConvexHull::edges`] are the unique undirected vertex-index pairs. The
//!   separating-axis refinement and the edge-edge contact case both enumerate
//!   them, so storing each once (not twice, once per adjoining face) halves the
//!   axis set.
//!
//! # Construction and winding
//!
//! [`ConvexHull::new`] takes the vertices and the face loops and derives each
//! face plane from the first three loop vertices: for a loop `v0, v1, v2` wound
//! counter-clockwise as seen from outside, the outward normal is
//! `normalize(cross(v1 - v0, v2 - v0))` and the offset is `dot(normal, v0)`. The
//! caller owns the winding; [`ConvexHull::from_box`] builds a correctly wound
//! box so the common case needs no hand winding and doubles as the
//! cross-check against the dedicated [`Obb`](super::obb::Obb) path.
//!
//! The edge table is derived by walking every face loop, forming the
//! consecutive vertex pair of each edge, and inserting it as an ordered
//! `(min, max)` key so the two faces that share an edge contribute it once.
//!
//! Provenance: textbook convex-polyhedron support mapping and half-edge-free
//! face/edge tables; no Unreal Engine source or derived code.

use glam::{Quat, Vec3};

/// Squared-length floor below which a face loop's derived normal is treated as
/// degenerate; a well-formed face is far above this.
const DEGENERATE_NORMAL_EPS2: f32 = 1.0e-20;

/// One face of a convex hull: its outward unit normal, its supporting-plane
/// offset, and the counter-clockwise loop of vertex indices that bound it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexFace {
    /// Outward unit normal of the face plane.
    pub normal: Vec3,
    /// Plane offset `dot(normal, v)` for any vertex `v` on the face, so the
    /// plane is `dot(normal, x) = offset`.
    pub offset: f32,
    /// Counter-clockwise loop of indices into [`ConvexHull::vertices`], as seen
    /// from outside the hull.
    pub loop_indices: Vec<u32>,
}

/// One undirected edge of a convex hull, stored as the ordered pair of its
/// endpoint vertex indices (`a < b`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvexEdge {
    /// Lower endpoint vertex index.
    pub a: u32,
    /// Higher endpoint vertex index.
    pub b: u32,
}

/// A convex polyhedron in its own local frame: the vertex cloud plus the face
/// loops and unique edges that close it.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHull {
    /// Local-space vertices; the support function reduces to an `argmax` over
    /// these.
    vertices: Vec<Vec3>,
    /// Face loops with their outward plane, used by the manifold clip.
    faces: Vec<ConvexFace>,
    /// Unique undirected edges, used by the separating-axis edge-edge case.
    edges: Vec<ConvexEdge>,
}

impl ConvexHull {
    /// Builds a hull from local vertices and counter-clockwise face loops,
    /// deriving each face's outward plane and the unique edge set.
    ///
    /// Each face loop must list at least three indices wound counter-clockwise
    /// as seen from outside the hull; the outward normal comes from the first
    /// three. The edge table walks every loop's consecutive pairs and keeps each
    /// undirected `(min, max)` pair once.
    ///
    /// # Panics
    ///
    /// Panics when a face loop has fewer than three indices, when a loop index
    /// is out of range, or when the first three loop vertices are collinear so
    /// the face normal is degenerate. These are all malformed cooked geometry,
    /// never valid runtime input.
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, loops: Vec<Vec<u32>>) -> ConvexHull {
        let faces: Vec<ConvexFace> = loops
            .into_iter()
            .map(|loop_indices| build_face(&vertices, loop_indices))
            .collect();
        let edges = derive_edges(&faces);
        ConvexHull {
            vertices,
            faces,
            edges,
        }
    }

    /// Builds an axis-aligned box hull of the given half extents, correctly
    /// wound so every face normal points outward.
    ///
    /// The eight corners are `(+-hx, +-hy, +-hz)` and the six faces are wound
    /// counter-clockwise about their outward normal, so this doubles as the
    /// convex cross-check against the dedicated [`Obb`](super::obb::Obb) path.
    #[must_use]
    pub fn from_box(half_extents: Vec3) -> ConvexHull {
        let h = half_extents;
        // Corners indexed by the sign bits (x: bit 0, y: bit 1, z: bit 2).
        let vertices = vec![
            Vec3::new(-h.x, -h.y, -h.z), // 0 ---
            Vec3::new(h.x, -h.y, -h.z),  // 1 +--
            Vec3::new(-h.x, h.y, -h.z),  // 2 -+-
            Vec3::new(h.x, h.y, -h.z),   // 3 ++-
            Vec3::new(-h.x, -h.y, h.z),  // 4 --+
            Vec3::new(h.x, -h.y, h.z),   // 5 +-+
            Vec3::new(-h.x, h.y, h.z),   // 6 -++
            Vec3::new(h.x, h.y, h.z),    // 7 +++
        ];
        // Each loop is wound counter-clockwise seen from outside its face.
        let loops = vec![
            vec![1, 3, 7, 5], // +x
            vec![2, 0, 4, 6], // -x
            vec![2, 6, 7, 3], // +y
            vec![0, 1, 5, 4], // -y
            vec![4, 5, 7, 6], // +z
            vec![0, 2, 3, 1], // -z
        ];
        ConvexHull::new(vertices, loops)
    }

    /// Builds a degenerate single-vertex hull at the local origin: the convex
    /// core of a rounded **sphere** shape.
    ///
    /// A sphere has no polygonal surface, so it carries no faces or edges; its
    /// entire geometry is a point swept by a convex radius. The support mapping
    /// of this hull returns that one point for every direction, which is exactly
    /// what the Minkowski-difference support of a sphere's centre needs. Pair it
    /// with the sphere radius as the `radius` argument of the rounded shape
    /// cast (see [`conservative_advancement_toi_rounded`](super::conservative_advancement::conservative_advancement_toi_rounded)).
    #[must_use]
    pub fn from_point() -> ConvexHull {
        ConvexHull {
            vertices: vec![Vec3::ZERO],
            faces: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Builds a degenerate two-vertex segment hull centred on the local origin:
    /// the convex core of a rounded **capsule** shape.
    ///
    /// The segment runs `+-half_height` along `axis` (which need not be
    /// normalised; only its direction matters). A capsule is this segment swept
    /// by a convex radius, so the support mapping returns whichever endpoint is
    /// farther along the query direction and the cap radius rides in the
    /// `radius` argument of the rounded shape cast. The segment has no enclosing
    /// faces, so none are derived.
    #[must_use]
    pub fn from_segment(axis: Vec3, half_height: f32) -> ConvexHull {
        let dir = axis.try_normalize().unwrap_or(Vec3::Y);
        let tip = dir * half_height;
        ConvexHull {
            vertices: vec![tip, -tip],
            faces: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Builds a degenerate flat three-vertex triangle hull from its world (or
    /// local) corners: the convex core a swept-shape mesh query tests one mesh
    /// triangle against.
    ///
    /// A single mesh triangle is a zero-thickness convex set, so like
    /// [`ConvexHull::from_point`] and [`ConvexHull::from_segment`] it carries no
    /// enclosing faces or edges: the `GJK` distance query the conservative
    /// advancement sweep runs needs only the support mapping over the three
    /// corners, never a face loop. Keeping it face-free also means a degenerate
    /// (collinear or zero-area) triangle never panics the way the general
    /// [`ConvexHull::new`] face builder would, so a malformed mesh triangle
    /// still produces a usable (if lower-dimensional) support set rather than a
    /// crash.
    ///
    /// Pair it with a zero convex radius in the rounded shape cast (see
    /// [`conservative_advancement_toi_rounded`](super::conservative_advancement::conservative_advancement_toi_rounded)):
    /// a solid triangle has no rounding of its own; any cap radius belongs to
    /// the moving shape swept against it.
    #[must_use]
    pub fn from_triangle(a: Vec3, b: Vec3, c: Vec3) -> ConvexHull {
        ConvexHull {
            vertices: vec![a, b, c],
            faces: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// The hull's local-space vertices.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The hull's faces with their outward planes and loops.
    #[must_use]
    pub fn faces(&self) -> &[ConvexFace] {
        &self.faces
    }

    /// The hull's unique undirected edges.
    #[must_use]
    pub fn edges(&self) -> &[ConvexEdge] {
        &self.edges
    }

    /// Index of the local vertex farthest along `dir` (the support vertex in
    /// local space), resolving ties to the lower index so the choice is
    /// deterministic and matches the kernel.
    ///
    /// `dir` need not be normalised; only its direction matters.
    #[must_use]
    pub fn support_local(&self, dir: Vec3) -> u32 {
        let mut best = 0_u32;
        let mut best_dot = self.vertices[0].dot(dir);
        for (i, v) in self.vertices.iter().enumerate().skip(1) {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = u32::try_from(i).unwrap_or(u32::MAX);
            }
        }
        best
    }

    /// World-space support point of the hull posed by `translation` and
    /// `rotation` along the world direction `dir`.
    ///
    /// The direction is rotated into the local frame to pick the support vertex,
    /// then that vertex is posed back out, so the result is
    /// `translation + rotation * vertex` for the local support of
    /// `inverse(rotation) * dir`.
    #[must_use]
    pub fn support_point(&self, translation: Vec3, rotation: Quat, dir: Vec3) -> Vec3 {
        let local_dir = rotation.inverse() * dir;
        let v = self.vertices[self.support_local(local_dir) as usize];
        translation + rotation * v
    }
}

/// Derives one [`ConvexFace`] from a vertex cloud and a counter-clockwise loop.
fn build_face(vertices: &[Vec3], loop_indices: Vec<u32>) -> ConvexFace {
    assert!(
        loop_indices.len() >= 3,
        "a convex face needs at least three vertices, got {}",
        loop_indices.len()
    );
    let v0 = vertices[loop_indices[0] as usize];
    let v1 = vertices[loop_indices[1] as usize];
    let v2 = vertices[loop_indices[2] as usize];
    let raw = (v1 - v0).cross(v2 - v0);
    assert!(
        raw.length_squared() > DEGENERATE_NORMAL_EPS2,
        "a convex face's first three vertices are collinear: {v0:?} {v1:?} {v2:?}"
    );
    let normal = raw.normalize();
    ConvexFace {
        normal,
        offset: normal.dot(v0),
        loop_indices,
    }
}

/// Walks every face loop and collects each undirected edge once, keyed by its
/// ordered `(min, max)` endpoint pair.
fn derive_edges(faces: &[ConvexFace]) -> Vec<ConvexEdge> {
    let mut seen: std::collections::HashSet<ConvexEdge> = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for face in faces {
        let loop_indices = &face.loop_indices;
        for i in 0..loop_indices.len() {
            let a = loop_indices[i];
            let b = loop_indices[(i + 1) % loop_indices.len()];
            let edge = if a < b {
                ConvexEdge { a, b }
            } else {
                ConvexEdge { a: b, b: a }
            };
            if seen.insert(edge) {
                edges.push(edge);
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn box_hull_has_eight_vertices_six_faces_twelve_edges() {
        let hull = ConvexHull::from_box(Vec3::splat(1.0));
        assert_eq!(hull.vertices().len(), 8, "a box has eight corners");
        assert_eq!(hull.faces().len(), 6, "a box has six faces");
        assert_eq!(hull.edges().len(), 12, "a box has twelve edges");
    }

    #[test]
    fn box_face_normals_point_outward_along_axes() {
        let hull = ConvexHull::from_box(Vec3::new(2.0, 3.0, 4.0));
        // Every face normal must be a unit axis direction and its plane offset
        // must equal the matching half extent (the face sits that far out).
        let expect = [
            (Vec3::X, 2.0),
            (Vec3::NEG_X, 2.0),
            (Vec3::Y, 3.0),
            (Vec3::NEG_Y, 3.0),
            (Vec3::Z, 4.0),
            (Vec3::NEG_Z, 4.0),
        ];
        for (normal, offset) in expect {
            let found = hull
                .faces()
                .iter()
                .find(|f| (f.normal - normal).length() < 1.0e-6);
            let face = found.unwrap_or_else(|| panic!("no face with normal {normal:?}"));
            assert!(
                (face.offset - offset).abs() < 1.0e-6,
                "face {normal:?} offset {} != {offset}",
                face.offset
            );
        }
    }

    #[test]
    fn support_picks_farthest_corner() {
        let hull = ConvexHull::from_box(Vec3::splat(1.0));
        // Along +x+y+z the +++ corner (index 7) is farthest.
        assert_eq!(hull.support_local(Vec3::new(1.0, 1.0, 1.0)), 7);
        // Along -x-y-z the --- corner (index 0) is farthest.
        assert_eq!(hull.support_local(Vec3::new(-1.0, -1.0, -1.0)), 0);
        // Along +x only, the lowest-index +x corner wins the tie (index 1).
        assert_eq!(hull.support_local(Vec3::X), 1);
    }

    #[test]
    fn support_point_poses_the_vertex() {
        let hull = ConvexHull::from_box(Vec3::splat(1.0));
        let translation = Vec3::new(10.0, 0.0, 0.0);
        let rotation = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
        // Along world +y, after a +90 deg z-rotation local +x maps to world +y,
        // so the support is a +x-corner of the box posed out to the translation.
        let p = hull.support_point(translation, rotation, Vec3::Y);
        // Local +x corners (x = +1) rotate +90 about z to world (x', y') =
        // (-y, +1), so y = 1; adding the +x translation leaves x = 10 + 1 = 11.
        assert!((p.y - 1.0).abs() < 1.0e-5, "support world y {}", p.y);
        assert!((p.x - 11.0).abs() < 1.0e-5, "support world x {}", p.x);
    }

    #[test]
    fn every_face_plane_contains_all_its_loop_vertices() {
        let hull = ConvexHull::from_box(Vec3::new(1.5, 2.5, 0.5));
        for face in hull.faces() {
            for &idx in &face.loop_indices {
                let v = hull.vertices()[idx as usize];
                let signed = face.normal.dot(v) - face.offset;
                assert!(
                    signed.abs() < 1.0e-6,
                    "loop vertex {v:?} off its face plane by {signed}"
                );
            }
        }
    }

    #[test]
    fn point_hull_is_a_single_origin_vertex_with_no_faces() {
        let hull = ConvexHull::from_point();
        assert_eq!(hull.vertices(), &[Vec3::ZERO]);
        assert!(hull.faces().is_empty(), "a sphere core carries no faces");
        assert!(hull.edges().is_empty(), "a sphere core carries no edges");
        // The support is the origin for every direction.
        assert_eq!(hull.support_local(Vec3::new(3.0, -2.0, 1.0)), 0);
    }

    #[test]
    fn segment_hull_endpoints_straddle_the_origin_along_the_axis() {
        let hull = ConvexHull::from_segment(Vec3::new(0.0, 2.0, 0.0), 1.5);
        assert_eq!(hull.vertices().len(), 2, "a capsule core is a segment");
        assert!(hull.faces().is_empty(), "a capsule core carries no faces");
        // Endpoints sit at +-half_height along the normalised axis.
        assert_eq!(hull.vertices()[0], Vec3::new(0.0, 1.5, 0.0));
        assert_eq!(hull.vertices()[1], Vec3::new(0.0, -1.5, 0.0));
        // Support resolves to the endpoint farther along the query direction.
        assert_eq!(hull.support_local(Vec3::Y), 0);
        assert_eq!(hull.support_local(-Vec3::Y), 1);
    }

    #[test]
    fn segment_hull_normalises_its_axis() {
        // An unnormalised axis must still yield unit-length endpoints scaled by
        // the half height, not by the raw axis length.
        let hull = ConvexHull::from_segment(Vec3::new(0.0, 0.0, 10.0), 2.0);
        assert_eq!(hull.vertices()[0], Vec3::new(0.0, 0.0, 2.0));
        assert_eq!(hull.vertices()[1], Vec3::new(0.0, 0.0, -2.0));
    }

    #[test]
    fn triangle_hull_is_three_corners_with_no_faces() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 3.0, 0.0);
        let hull = ConvexHull::from_triangle(a, b, c);
        assert_eq!(hull.vertices(), &[a, b, c], "corners kept in order");
        assert!(hull.faces().is_empty(), "a triangle core carries no faces");
        assert!(hull.edges().is_empty(), "a triangle core carries no edges");
        // Support resolves to the corner farthest along the query direction.
        assert_eq!(hull.support_local(Vec3::X), 1, "+x corner is b");
        assert_eq!(hull.support_local(Vec3::Y), 2, "+y corner is c");
        assert_eq!(hull.support_local(Vec3::new(-1.0, -1.0, 0.0)), 0, "origin corner is a");
    }

    #[test]
    fn degenerate_triangle_hull_does_not_panic() {
        // Collinear corners would panic the general face builder; the flat core
        // must still build and answer support queries.
        let hull = ConvexHull::from_triangle(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert_eq!(hull.vertices().len(), 3);
        assert_eq!(hull.support_local(Vec3::X), 2, "farthest +x corner wins");
    }
}
