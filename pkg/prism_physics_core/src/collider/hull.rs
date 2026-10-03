//! Convex-hull construction from an arbitrary point cloud.
//!
//! [`ConvexMeshData::from_surface`](super::ConvexMeshData::from_surface) consumes
//! an already-triangulated convex surface, but production content pipelines must
//! *cook* a convex collider directly from a raw point cloud or a render mesh's
//! vertices (`PhysX` `PxConvexMeshCookingType`, Jolt `ConvexHullShapeSettings`,
//! Chaos `FConvexBuilder`). [`convex_hull`] closes that gap: it builds the exact
//! convex hull of a point set as an outward-wound triangle surface that is ready
//! for [`ConvexMeshData::from_surface`](super::ConvexMeshData::from_surface) or
//! the [`ConvexMeshData::from_points`](super::ConvexMeshData::from_points)
//! convenience constructor.
//!
//! # Algorithm
//!
//! A randomized-incremental 3D convex hull (the Clarkson--Shor / `QuickHull`
//! family): seed a tetrahedron from four extreme, non-coplanar points, then
//! repeatedly take the point that lies farthest outside the current hull, delete
//! every face that point can "see", and stitch new faces from the point to the
//! silhouette (the *horizon*) of the deleted cap. Every face is kept wound
//! counter-clockwise as seen from outside by orienting it against a fixed,
//! strictly-interior reference point, which keeps the directed-edge horizon test
//! exact and the output winding consistent (positive signed-tetrahedron volume).
//! Each absorbed point becomes a hull vertex at most once, so the loop runs at
//! most `points.len()` times and always terminates.
//!
//! All ordering is derived from deterministic index/edge loops (hash sets are
//! used only for `O(1)` membership lookups, never iterated), so the hull is
//! bit-for-bit reproducible for a given input -- a hard requirement for the
//! engine's cross-run state hashing.
//!
//! # Provenance
//!
//! The incremental convex-hull construction and the farthest-point / horizon
//! bookkeeping are textbook computational geometry (Preparata & Shamos,
//! *Computational Geometry: An Introduction*; de Berg et al., *Computational
//! Geometry*; Barber, Dobkin & Huhdanpaa, *The Quickhull Algorithm for Convex
//! Hulls*). This module contains **no Unreal Engine source or derived code**.

use std::collections::HashSet;

use glam::Vec3;

/// A working face of the incremental hull: three vertex indices wound
/// counter-clockwise as seen from outside, its outward support plane
/// `normal . x = offset`, and the input points still strictly in front of it
/// (its "outside set").
struct Face {
    verts: [u32; 3],
    normal: Vec3,
    offset: f32,
    outside: Vec<u32>,
}

impl Face {
    /// Signed distance from the outward plane; positive is outside the hull.
    fn signed_distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.offset
    }

    /// The three directed edges of the face in winding order.
    fn edges(&self) -> [(u32, u32); 3] {
        [
            (self.verts[0], self.verts[1]),
            (self.verts[1], self.verts[2]),
            (self.verts[2], self.verts[0]),
        ]
    }
}

/// Builds the convex hull of `points`.
///
/// Returns `(vertices, triangles)` where `triangles` index into `vertices` and
/// are wound counter-clockwise as seen from outside (outward face normal) --
/// exactly the contract expected by
/// [`ConvexMeshData::from_surface`](super::ConvexMeshData::from_surface). Only
/// the points that end up on the hull are kept in `vertices`; strictly interior
/// inputs are dropped. For a simplicial convex polytope the result satisfies
/// Euler's relation `triangles.len() == 2 * vertices.len() - 4`.
///
/// Returns [`None`] for a degenerate cloud -- fewer than four points, or all
/// points collinear or coplanar within a scale-relative tolerance -- because no
/// convex *solid* exists in those cases.
#[must_use]
pub fn convex_hull(points: &[Vec3]) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    if points.len() < 4 {
        return None;
    }
    let eps = hull_epsilon(points);
    let seed = initial_tetrahedron(points, eps)?;
    let interior = (points[seed[0] as usize]
        + points[seed[1] as usize]
        + points[seed[2] as usize]
        + points[seed[3] as usize])
        * 0.25;

    let mut faces = seed_faces(points, seed, interior, eps)?;

    // Assign every non-seed point to the outside set of the first face it is in
    // front of; points behind every face are interior and dropped.
    for i in 0..points.len() as u32 {
        if seed.contains(&i) {
            continue;
        }
        assign_point(&mut faces, 0, i, points[i as usize], eps);
    }

    // Grow the hull: repeatedly absorb the point that is farthest outside.
    while let Some(face_idx) = faces.iter().position(|f| !f.outside.is_empty()) {
        let apex = farthest_point(&faces[face_idx], points);
        let apex_pos = points[apex as usize];

        // Faces whose outward side the apex is strictly in front of.
        let visible: Vec<usize> = (0..faces.len())
            .filter(|&f| faces[f].signed_distance(apex_pos) > eps)
            .collect();

        let horizon = horizon_edges(&faces, &visible);

        // Collect the outside sets of the doomed faces so they can be reassigned.
        let mut orphans: Vec<u32> = Vec::new();
        for &f in &visible {
            orphans.extend(faces[f].outside.iter().copied());
        }

        // Remove visible faces (descending indices keep the rest valid).
        let mut doomed = visible;
        doomed.sort_unstable_by(|a, b| b.cmp(a));
        for f in doomed {
            faces.swap_remove(f);
        }

        // Stitch a new triangle from each horizon edge up to the apex.
        let first_new = faces.len();
        for (a, b) in horizon {
            if let Some(face) = make_face([a, b, apex], points, interior, eps) {
                faces.push(face);
            }
        }

        // Reassign orphaned points (except the apex itself) to the new faces.
        for p in orphans {
            if p == apex {
                continue;
            }
            assign_point(&mut faces, first_new, p, points[p as usize], eps);
        }
    }

    finalize(points, &faces)
}

/// A scale-relative tolerance: a point is only treated as outside a face when it
/// is more than this far in front of the face plane.
fn hull_epsilon(points: &[Vec3]) -> f32 {
    let mut lo = points[0];
    let mut hi = points[0];
    for &p in points {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let scale = (hi - lo).max_element();
    (scale * 1.0e-6).max(1.0e-7)
}

/// Picks four extreme, non-coplanar seed vertices, or [`None`] when the cloud is
/// collinear or coplanar within `eps`.
fn initial_tetrahedron(points: &[Vec3], eps: f32) -> Option<[u32; 4]> {
    // The six axis extrema are the natural seed candidates for the first edge.
    let mut lo = [0u32; 3];
    let mut hi = [0u32; 3];
    for (i, p) in points.iter().enumerate() {
        for axis in 0..3 {
            if p[axis] < points[lo[axis] as usize][axis] {
                lo[axis] = i as u32;
            }
            if p[axis] > points[hi[axis] as usize][axis] {
                hi[axis] = i as u32;
            }
        }
    }
    let mut candidates = Vec::with_capacity(6);
    for c in lo.into_iter().chain(hi) {
        if !candidates.contains(&c) {
            candidates.push(c);
        }
    }

    // p0, p1: the farthest-apart pair among the extrema.
    let (mut p0, mut p1) = (candidates[0], candidates[0]);
    let mut best = -1.0_f32;
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let d = points[candidates[i] as usize].distance_squared(points[candidates[j] as usize]);
            if d > best {
                best = d;
                p0 = candidates[i];
                p1 = candidates[j];
            }
        }
    }
    if best <= eps * eps {
        return None; // all extrema coincide
    }

    // p2: the point farthest from the line p0->p1.
    let a = points[p0 as usize];
    let dir = points[p1 as usize] - a;
    let dir_len = dir.length();
    if dir_len <= eps {
        return None;
    }
    let dir_n = dir / dir_len;
    let mut p2 = u32::MAX;
    let mut best = eps;
    for (i, &p) in points.iter().enumerate() {
        let rel = p - a;
        let perp = (rel - dir_n * rel.dot(dir_n)).length();
        if perp > best {
            best = perp;
            p2 = i as u32;
        }
    }
    if p2 == u32::MAX {
        return None; // collinear
    }

    // p3: the point farthest from the plane (p0, p1, p2).
    let n = dir.cross(points[p2 as usize] - a);
    let n_len = n.length();
    if n_len <= eps * eps {
        return None;
    }
    let n_n = n / n_len;
    let mut p3 = u32::MAX;
    let mut best = eps;
    for (i, &p) in points.iter().enumerate() {
        let dist = (p - a).dot(n_n).abs();
        if dist > best {
            best = dist;
            p3 = i as u32;
        }
    }
    if p3 == u32::MAX {
        return None; // coplanar
    }

    Some([p0, p1, p2, p3])
}

/// Builds the four outward-wound faces of the seed tetrahedron.
fn seed_faces(points: &[Vec3], seed: [u32; 4], interior: Vec3, eps: f32) -> Option<Vec<Face>> {
    let triples = [
        [seed[0], seed[1], seed[2]],
        [seed[0], seed[1], seed[3]],
        [seed[0], seed[2], seed[3]],
        [seed[1], seed[2], seed[3]],
    ];
    let mut faces = Vec::with_capacity(4);
    for t in triples {
        faces.push(make_face(t, points, interior, eps)?);
    }
    Some(faces)
}

/// Builds a single face from three vertex indices, orienting it so its normal
/// points away from `interior`. Returns [`None`] for a degenerate (near-zero
/// area) triangle.
fn make_face(verts: [u32; 3], points: &[Vec3], interior: Vec3, eps: f32) -> Option<Face> {
    let a = points[verts[0] as usize];
    let b = points[verts[1] as usize];
    let c = points[verts[2] as usize];
    let cross = (b - a).cross(c - a);
    let len = cross.length();
    if len <= eps * eps {
        return None;
    }
    let mut normal = cross / len;
    let mut offset = normal.dot(a);
    let mut verts = verts;
    if normal.dot(interior) - offset > 0.0 {
        // Interior is in front: flip the plane and the winding.
        normal = -normal;
        offset = -offset;
        verts.swap(1, 2);
    }
    Some(Face {
        verts,
        normal,
        offset,
        outside: Vec::new(),
    })
}

/// Assigns point `idx` to the first face in `faces[start..]` whose outward side
/// it is strictly in front of; a point behind all of them is interior and left
/// unassigned.
fn assign_point(faces: &mut [Face], start: usize, idx: u32, pos: Vec3, eps: f32) {
    for face in &mut faces[start..] {
        if face.signed_distance(pos) > eps {
            face.outside.push(idx);
            return;
        }
    }
}

/// Returns the index of the point farthest in front of `face`.
fn farthest_point(face: &Face, points: &[Vec3]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut best_idx = face.outside[0];
    for &idx in &face.outside {
        let d = face.signed_distance(points[idx as usize]);
        if d > best {
            best = d;
            best_idx = idx;
        }
    }
    best_idx
}

/// Computes the horizon: the directed boundary edges of the visible region,
/// wound so that `(a, b, apex)` is a new outward face. An edge is on the horizon
/// when its reverse is not itself an edge of a visible face (its opposite face
/// is not visible). Iteration order is deterministic (over the ordered `visible`
/// list); the hash set is used only for `O(1)` membership tests.
fn horizon_edges(faces: &[Face], visible: &[usize]) -> Vec<(u32, u32)> {
    let mut present: HashSet<(u32, u32)> = HashSet::new();
    for &f in visible {
        for e in faces[f].edges() {
            present.insert(e);
        }
    }
    let mut horizon = Vec::new();
    for &f in visible {
        for (a, b) in faces[f].edges() {
            if !present.contains(&(b, a)) {
                horizon.push((a, b));
            }
        }
    }
    horizon
}

/// Remaps the surviving faces to a compact vertex list and emits the triangle
/// index set. Returns [`None`] if fewer than four faces survive (not a solid).
fn finalize(points: &[Vec3], faces: &[Face]) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    if faces.len() < 4 {
        return None;
    }
    let mut index_of = vec![u32::MAX; points.len()];
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(faces.len());
    for face in faces {
        let mut tri = [0u32; 3];
        for (k, &orig) in face.verts.iter().enumerate() {
            if index_of[orig as usize] == u32::MAX {
                index_of[orig as usize] = vertices.len() as u32;
                vertices.push(points[orig as usize]);
            }
            tri[k] = index_of[orig as usize];
        }
        triangles.push(tri);
    }
    if vertices.len() < 4 {
        return None;
    }
    Some((vertices, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::ConvexMeshData;

    /// Every input point must lie on or behind every outward face plane (i.e.
    /// inside the closed hull), the strongest correctness witness.
    fn assert_all_points_inside(points: &[Vec3], verts: &[Vec3], tris: &[[u32; 3]]) {
        let scale = {
            let mut lo = points[0];
            let mut hi = points[0];
            for &p in points {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            (hi - lo).max_element()
        };
        let tol = (scale * 1.0e-5).max(1.0e-6);
        for &p in points {
            let mut max_signed = f32::NEG_INFINITY;
            for t in tris {
                let a = verts[t[0] as usize];
                let b = verts[t[1] as usize];
                let c = verts[t[2] as usize];
                let n = (b - a).cross(c - a).normalize();
                max_signed = max_signed.max(n.dot(p) - n.dot(a));
            }
            assert!(
                max_signed <= tol,
                "point {p:?} is {max_signed} outside the hull"
            );
        }
    }

    /// Every face must be a supporting plane: all hull vertices behind it.
    fn assert_convex(verts: &[Vec3], tris: &[[u32; 3]]) {
        for t in tris {
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let n = (b - a).cross(c - a).normalize();
            let off = n.dot(a);
            for &v in verts {
                assert!(n.dot(v) - off <= 1.0e-4, "face {t:?} is not supporting");
            }
        }
    }

    #[test]
    fn rejects_degenerate_clouds() {
        assert!(convex_hull(&[]).is_none());
        assert!(convex_hull(&[Vec3::ZERO, Vec3::X, Vec3::Y]).is_none());
        // Collinear.
        let line: Vec<Vec3> = (0..6).map(|i| Vec3::X * i as f32).collect();
        assert!(convex_hull(&line).is_none());
        // Coplanar square in the z = 0 plane.
        let square = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::ZERO,
        ];
        assert!(convex_hull(&square).is_none());
    }

    #[test]
    fn tetrahedron_has_four_faces() {
        let pts = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let (verts, tris) = convex_hull(&pts).expect("tetra");
        assert_eq!(verts.len(), 4);
        assert_eq!(tris.len(), 4);
        assert_eq!(tris.len(), 2 * verts.len() - 4);
        assert_all_points_inside(&pts, &verts, &tris);
        assert_convex(&verts, &tris);
    }

    #[test]
    fn cube_corners_yield_twelve_triangles() {
        let mut pts = Vec::new();
        for x in [-1.0_f32, 1.0] {
            for y in [-1.0_f32, 1.0] {
                for z in [-1.0_f32, 1.0] {
                    pts.push(Vec3::new(x, y, z));
                }
            }
        }
        let (verts, tris) = convex_hull(&pts).expect("cube");
        assert_eq!(verts.len(), 8, "all eight corners are hull vertices");
        assert_eq!(tris.len(), 12, "a cube triangulates into 12 faces");
        assert_eq!(tris.len(), 2 * verts.len() - 4);
        assert_all_points_inside(&pts, &verts, &tris);
        assert_convex(&verts, &tris);
    }

    #[test]
    fn interior_points_are_discarded() {
        let mut pts = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        // A lattice of strictly interior points that must not become vertices.
        for i in -2..=2 {
            for j in -2..=2 {
                let f = |n: i32| n as f32 * 0.3;
                pts.push(Vec3::new(f(i), f(j), 0.0));
            }
        }
        let (verts, tris) = convex_hull(&pts).expect("cube with filler");
        assert_eq!(verts.len(), 8);
        assert_eq!(tris.len(), 12);
        assert_all_points_inside(&pts, &verts, &tris);
    }

    #[test]
    fn octahedron_is_recovered() {
        let pts = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let (verts, tris) = convex_hull(&pts).expect("octahedron");
        assert_eq!(verts.len(), 6);
        assert_eq!(tris.len(), 8);
        assert_eq!(tris.len(), 2 * verts.len() - 4);
        assert_all_points_inside(&pts, &verts, &tris);
        assert_convex(&verts, &tris);
    }

    #[test]
    fn deterministic_across_runs() {
        let pts = sample_cloud();
        let first = convex_hull(&pts).expect("hull");
        for _ in 0..8 {
            let again = convex_hull(&pts).expect("hull");
            assert_eq!(first.0, again.0, "vertices must be reproducible");
            assert_eq!(first.1, again.1, "triangles must be reproducible");
        }
    }

    #[test]
    fn scattered_cloud_is_convex_and_encloses_all_points() {
        let pts = sample_cloud();
        let (verts, tris) = convex_hull(&pts).expect("hull");
        assert_eq!(tris.len(), 2 * verts.len() - 4, "simplicial Euler relation");
        assert_all_points_inside(&pts, &verts, &tris);
        assert_convex(&verts, &tris);
    }

    #[test]
    fn from_points_matches_closed_form_box_volume() {
        let mut pts = Vec::new();
        let h = Vec3::new(1.5, 0.75, 2.0);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    pts.push(Vec3::new(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        // Interior filler must not perturb the cooked solid.
        pts.push(Vec3::ZERO);
        pts.push(Vec3::new(0.3, -0.2, 0.5));
        let mesh = ConvexMeshData::from_points(&pts).expect("cooked box");
        let expected = 8.0 * h.x * h.y * h.z;
        assert!(
            (mesh.volume() - expected).abs() < 1.0e-3,
            "volume {} != {}",
            mesh.volume(),
            expected
        );
    }

    /// A fixed, well-separated 20-point cloud (deterministic, no RNG) whose hull
    /// exercises the horizon-stitching path with several absorptions.
    fn sample_cloud() -> Vec<Vec3> {
        let mut pts = Vec::new();
        let coords = [-3.0_f32, -1.0, 1.0, 3.0];
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                // One point per (x, y) column on a tent-shaped height field so the
                // cloud is genuinely three-dimensional and non-coplanar.
                let z = 2.0 - 0.2 * (x * x + y * y) + 0.11 * (i as f32) - 0.07 * (j as f32);
                pts.push(Vec3::new(x, y, z));
            }
        }
        pts.push(Vec3::new(0.0, 0.0, -4.0));
        pts.push(Vec3::new(0.0, 0.0, 5.0));
        pts.push(Vec3::new(4.5, 0.0, 0.0));
        pts.push(Vec3::new(-4.5, 0.0, 0.0));
        pts
    }
}
