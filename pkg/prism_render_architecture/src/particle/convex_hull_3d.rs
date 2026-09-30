//! 3D convex-hull construction for the particle spatial contracts
//! (design §8.2, §12-§13).
//!
//! Several particle stages need the tight 3D convex boundary of a point set: a
//! bounds-reduction pass wants the smallest enclosing polytope of a splat
//! cluster; a collision-broadphase wants a convex separating-axis proxy; and a
//! culling reference wants the silhouette hull of a projected cloud. This
//! module owns the small, `CPU`-verifiable contract those stages share: turning
//! an unordered slice of 3D points into the triangular faces of their convex
//! hull, each face expressed as a triple of vertex indices into the input slice
//! with a consistent outward winding.
//!
//! The hull is built with the incremental method. A non-degenerate seed
//! tetrahedron is chosen from the first four points that are not collinear or
//! coplanar, and every face is oriented so its `cross`-product normal points
//! away from the tetrahedron centroid, which is a strictly interior reference
//! point. Each remaining point is then inserted: the faces it can see (the
//! faces whose outward side it lies on, judged by the signed-volume
//! `orient3d` predicate) are deleted, the horizon loop of edges bordering the
//! deleted region is found, and one new face is stitched from that point to
//! each horizon edge. Because the interior reference never leaves the growing
//! hull, every new face is re-oriented against it, so the winding stays
//! outward without any separate repair pass.
//!
//! # Strict scope
//! This module only *constructs* the 3D convex hull as an outward-oriented
//! triangle-index list. It is deliberately distinct from its siblings and
//! neither imports nor reconstructs them:
//! * [`super::convex_hull_2d`] builds a planar `CCW` vertex ring with Andrew's
//!   monotone chain and measures 2D hull metrics; it never leaves the plane.
//! * [`super::bvh`] builds a `Morton`-code bounding-volume hierarchy for
//!   broadphase traversal; it indexes boxes, not hull faces.
//! * [`super::bounds`] tracks axis-aligned bounding boxes (`AABB`s); it is a
//!   loose enclosure, not the tight convex boundary this module returns.
//!
//! # Degenerate inputs
//! Fewer than four distinct points, a fully collinear set, and a fully coplanar
//! set have no 3D hull volume; each returns an empty face list. Points that
//! coincide within [`HULL_EPS`] are collapsed before the seed is chosen, and
//! interior points are naturally skipped because they can see no face.
//!
//! # No transcendental math
//! Construction is pure `+`, `-`, `*` vector arithmetic and signed-volume
//! comparison; the only irrational operation is the `f32::sqrt` used to
//! normalize a face normal in [`hull_signed_distance`]. There is no `sin`,
//! `cos`, `atan`, `exp`, `ln`, `powf`, or any other transcendental call, and no
//! `f32` equality: near-zero magnitudes are compared against [`HULL_EPS`].

use alloc::vec::Vec;

/// Magnitude below which a signed volume, a squared edge length, or a
/// coordinate difference is treated as zero. This is the comparison rule used
/// throughout instead of `==` on `f32`: two scalars are "equal" when their
/// absolute difference does not exceed this bound, and a point sees a face only
/// when its signed volume against that face strictly exceeds it.
pub const HULL_EPS: f32 = 1.0e-6;

/// Difference `a - b` of two 3D vectors.
#[must_use]
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a x b` of two 3D vectors.
#[must_use]
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product `a . b` of two 3D vectors.
#[must_use]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared Euclidean length of a 3D vector.
#[must_use]
fn v_len_sq(a: [f32; 3]) -> f32 {
    v_dot(a, a)
}

/// Six times the signed volume of the tetrahedron `a, b, c, d`.
///
/// Equivalently the scalar triple product `((b - a) x (c - a)) . (d - a)`. Its
/// sign classifies `d` relative to the plane of `a, b, c` oriented by the
/// `cross`-product normal `(b - a) x (c - a)`: strictly positive when `d` is on
/// the normal (outward) side, strictly negative on the far side, and zero
/// (within [`HULL_EPS`]) when the four points are coplanar.
#[must_use]
pub fn orient3d(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f32 {
    let ab = v_sub(b, a);
    let ac = v_sub(c, a);
    let ad = v_sub(d, a);
    v_dot(v_cross(ab, ac), ad)
}

/// Returns `true` when two points coincide within [`HULL_EPS`] on every axis.
#[must_use]
fn points_equal(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() <= HULL_EPS
        && (a[1] - b[1]).abs() <= HULL_EPS
        && (a[2] - b[2]).abs() <= HULL_EPS
}

/// A single hull triangle as three vertex indices into the working point list.
#[derive(Clone, Copy)]
struct Face {
    v: [usize; 3],
}

/// Builds a face `a, b, c` (working indices) whose outward normal points away
/// from the interior reference `interior`.
///
/// The seed tetrahedron centroid stays strictly inside the growing hull, so a
/// face is correctly wound outward exactly when `interior` lies on the negative
/// side of its `cross`-product normal. When the reference is on the positive
/// side the winding is flipped by swapping the last two vertices.
#[must_use]
fn make_face(interior: [f32; 3], pts: &[[f32; 3]], a: usize, b: usize, c: usize) -> Face {
    let o = orient3d(pts[a], pts[b], pts[c], interior);
    if o > 0.0 {
        Face { v: [a, c, b] }
    } else {
        Face { v: [a, b, c] }
    }
}

/// Collapses points that coincide within [`HULL_EPS`], preserving input order.
///
/// Returns the deduplicated coordinates together with, for each survivor, the
/// index of its first occurrence in `points`. The returned index list is what
/// the public hull faces are ultimately expressed in.
#[must_use]
fn dedup_points(points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<usize>) {
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut orig: Vec<usize> = Vec::new();
    for (i, &p) in points.iter().enumerate() {
        if !pts.iter().any(|&q| points_equal(q, p)) {
            pts.push(p);
            orig.push(i);
        }
    }
    (pts, orig)
}

/// Picks a non-degenerate seed tetrahedron from the deduplicated points.
///
/// Returns the four distinct working indices `[i0, i1, i2, i3]` such that no
/// three are collinear and the four are not coplanar, or `None` when the whole
/// set is collinear or coplanar and therefore has no 3D hull.
#[must_use]
fn seed_tetrahedron(pts: &[[f32; 3]]) -> Option<[usize; 4]> {
    let n = pts.len();
    if n < 4 {
        return None;
    }
    let i0 = 0;
    let i1 = 1;
    let base = v_sub(pts[i1], pts[i0]);

    // First point that is not collinear with the first edge.
    let mut i2 = None;
    for k in 2..n {
        let cross = v_cross(base, v_sub(pts[k], pts[i0]));
        if v_len_sq(cross) > HULL_EPS * HULL_EPS {
            i2 = Some(k);
            break;
        }
    }
    let i2 = i2?;

    // First point that is not coplanar with the seed triangle.
    let mut i3 = None;
    for m in 2..n {
        if m == i2 {
            continue;
        }
        if orient3d(pts[i0], pts[i1], pts[i2], pts[m]).abs() > HULL_EPS {
            i3 = Some(m);
            break;
        }
    }
    let i3 = i3?;

    Some([i0, i1, i2, i3])
}

/// Constructs the convex hull of `points` as outward-wound triangles.
///
/// Each returned triple lists three vertex indices into the original `points`
/// slice; the three vertices are ordered so the triangle's `cross`-product
/// normal points out of the hull. Points that coincide within [`HULL_EPS`] are
/// collapsed to their first occurrence, so an index never refers to a
/// duplicate.
///
/// Degenerate inputs return an empty `Vec`: fewer than four distinct points, a
/// fully collinear set, and a fully coplanar set all lack hull volume. The
/// construction is deterministic — the same slice always yields the same faces
/// in the same order.
#[must_use]
pub fn convex_hull_3d(points: &[[f32; 3]]) -> Vec<[usize; 3]> {
    let (pts, orig) = dedup_points(points);
    let Some(seed) = seed_tetrahedron(&pts) else {
        return Vec::new();
    };

    let [i0, i1, i2, i3] = seed;
    let interior = [
        (pts[i0][0] + pts[i1][0] + pts[i2][0] + pts[i3][0]) * 0.25,
        (pts[i0][1] + pts[i1][1] + pts[i2][1] + pts[i3][1]) * 0.25,
        (pts[i0][2] + pts[i1][2] + pts[i2][2] + pts[i3][2]) * 0.25,
    ];

    let mut faces: Vec<Face> = Vec::from([
        make_face(interior, &pts, i0, i1, i2),
        make_face(interior, &pts, i0, i1, i3),
        make_face(interior, &pts, i0, i2, i3),
        make_face(interior, &pts, i1, i2, i3),
    ]);

    for (i, &p) in pts.iter().enumerate() {
        // Which existing faces does this point lie strictly outside of?
        let mut visible: Vec<bool> = Vec::with_capacity(faces.len());
        let mut any_visible = false;
        for face in &faces {
            let seen = orient3d(pts[face.v[0]], pts[face.v[1]], pts[face.v[2]], p) > HULL_EPS;
            visible.push(seen);
            any_visible = any_visible || seen;
        }
        if !any_visible {
            continue;
        }

        // Directed boundary edges of the visible region.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if visible[fi] {
                edges.push((face.v[0], face.v[1]));
                edges.push((face.v[1], face.v[2]));
                edges.push((face.v[2], face.v[0]));
            }
        }

        // A horizon edge is a directed edge whose reverse is absent, i.e. the
        // edge borders exactly one visible face.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &(u, v) in &edges {
            let shared = edges.iter().any(|&(a, b)| a == v && b == u);
            if !shared {
                horizon.push((u, v));
            }
        }

        // Retain only the faces the point cannot see.
        let mut kept: Vec<Face> = Vec::with_capacity(faces.len());
        for (fi, &face) in faces.iter().enumerate() {
            if !visible[fi] {
                kept.push(face);
            }
        }
        faces = kept;

        // Stitch the point to every horizon edge.
        for &(u, v) in &horizon {
            faces.push(make_face(interior, &pts, u, v, i));
        }
    }

    faces
        .iter()
        .map(|f| [orig[f.v[0]], orig[f.v[1]], orig[f.v[2]]])
        .collect()
}

/// Outward face normal (not normalized) of hull triangle `tri` over `points`.
///
/// This is the `cross`-product `(b - a) x (c - a)` of the triangle's edges,
/// where `a, b, c` are `points[tri[0]]`, `points[tri[1]]`, and `points[tri[2]]`.
/// For a face returned by [`convex_hull_3d`] it points out of the hull.
#[must_use]
pub fn face_normal(points: &[[f32; 3]], tri: [usize; 3]) -> [f32; 3] {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    v_cross(v_sub(b, a), v_sub(c, a))
}

/// Signed distance from point `p` to the plane of hull triangle `tri`.
///
/// The distance is positive when `p` lies on the triangle's outward side,
/// negative on the inward side, and zero (within rounding) on the plane. A
/// degenerate triangle whose normal has length below [`HULL_EPS`] reports
/// `0.0`. This is the only place a `sqrt` is used, to normalize the face
/// normal returned by [`face_normal`].
#[must_use]
pub fn hull_signed_distance(points: &[[f32; 3]], tri: [usize; 3], p: [f32; 3]) -> f32 {
    let n = face_normal(points, tri);
    let len_sq = v_len_sq(n);
    if len_sq <= HULL_EPS * HULL_EPS {
        return 0.0;
    }
    let inv = 1.0 / len_sq.sqrt();
    let a = points[tri[0]];
    v_dot([n[0] * inv, n[1] * inv, n[2] * inv], v_sub(p, a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TOL: f32 = 1.0e-3;

    fn tetra() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn cube() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    }

    fn octahedron() -> Vec<[f32; 3]> {
        vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    }

    fn referenced_vertices(faces: &[[usize; 3]]) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        for f in faces {
            for &idx in f {
                if !v.contains(&idx) {
                    v.push(idx);
                }
            }
        }
        v.sort_unstable();
        v
    }

    // A deterministic linear-congruential source of f32 values in [0, 1).
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_unit(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            ((self.state >> 8) as f32) / ((1u32 << 24) as f32)
        }
    }

    fn random_cloud(seed: u32, count: usize) -> Vec<[f32; 3]> {
        let mut rng = Lcg::new(seed);
        let mut pts = Vec::with_capacity(count);
        for _ in 0..count {
            let x = rng.next_unit() * 2.0 - 1.0;
            let y = rng.next_unit() * 2.0 - 1.0;
            let z = rng.next_unit() * 2.0 - 1.0;
            pts.push([x, y, z]);
        }
        pts
    }

    #[test]
    fn orient3d_positive_on_normal_side() {
        let o = orient3d(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        assert!(o > HULL_EPS);
    }

    #[test]
    fn orient3d_negative_on_far_side() {
        let o = orient3d(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        );
        assert!(o < -HULL_EPS);
    }

    #[test]
    fn orient3d_zero_when_coplanar() {
        let o = orient3d(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 3.0, 0.0],
        );
        assert!(o.abs() <= HULL_EPS);
    }

    #[test]
    fn empty_input_yields_no_faces() {
        assert!(convex_hull_3d(&[]).is_empty());
    }

    #[test]
    fn single_point_yields_no_faces() {
        assert!(convex_hull_3d(&[[1.0, 2.0, 3.0]]).is_empty());
    }

    #[test]
    fn two_points_yield_no_faces() {
        assert!(convex_hull_3d(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).is_empty());
    }

    #[test]
    fn three_points_yield_no_faces() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(convex_hull_3d(&pts).is_empty());
    }

    #[test]
    fn collinear_points_yield_no_faces() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ];
        assert!(convex_hull_3d(&pts).is_empty());
    }

    #[test]
    fn coplanar_square_yields_no_faces() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        assert!(convex_hull_3d(&pts).is_empty());
    }

    #[test]
    fn duplicate_points_collapse_but_hull_survives() {
        let pts = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = convex_hull_3d(&pts);
        assert_eq!(faces.len(), 4);
        // The collapsed duplicate at index 1 must never be referenced.
        assert!(faces.iter().all(|f| !f.contains(&1)));
    }

    #[test]
    fn near_duplicate_within_eps_collapses() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0e-7, 0.0, 0.0],
        ];
        let faces = convex_hull_3d(&pts);
        assert_eq!(faces.len(), 4);
        assert!(faces.iter().all(|f| !f.contains(&4)));
    }

    #[test]
    fn tetrahedron_has_four_faces() {
        let faces = convex_hull_3d(&tetra());
        assert_eq!(faces.len(), 4);
    }

    #[test]
    fn tetrahedron_references_all_four_vertices() {
        let faces = convex_hull_3d(&tetra());
        assert_eq!(referenced_vertices(&faces), vec![0, 1, 2, 3]);
    }

    #[test]
    fn cube_has_twelve_triangles() {
        let faces = convex_hull_3d(&cube());
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn cube_references_all_eight_corners() {
        let faces = convex_hull_3d(&cube());
        assert_eq!(referenced_vertices(&faces), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn octahedron_has_eight_faces() {
        let faces = convex_hull_3d(&octahedron());
        assert_eq!(faces.len(), 8);
    }

    #[test]
    fn octahedron_references_all_six_vertices() {
        let faces = convex_hull_3d(&octahedron());
        assert_eq!(referenced_vertices(&faces), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn interior_point_is_excluded_from_hull() {
        let mut pts = tetra();
        // Centroid of the tetrahedron lies strictly inside.
        pts.push([0.25, 0.25, 0.25]);
        let faces = convex_hull_3d(&pts);
        assert_eq!(faces.len(), 4);
        assert!(faces.iter().all(|f| !f.contains(&4)));
    }

    #[test]
    fn deep_interior_point_is_excluded_for_cube() {
        let mut pts = cube();
        pts.push([0.5, 0.5, 0.5]);
        let faces = convex_hull_3d(&pts);
        assert_eq!(faces.len(), 12);
        assert!(faces.iter().all(|f| !f.contains(&8)));
    }

    #[test]
    fn euler_formula_holds_for_cube() {
        // A triangulated convex polytope satisfies faces == 2 * vertices - 4.
        let faces = convex_hull_3d(&cube());
        let verts = referenced_vertices(&faces).len();
        assert_eq!(faces.len(), 2 * verts - 4);
    }

    #[test]
    fn euler_formula_holds_for_random_cloud() {
        let pts = random_cloud(0x1357_9BDF, 60);
        let faces = convex_hull_3d(&pts);
        let verts = referenced_vertices(&faces).len();
        assert!(verts >= 4);
        assert_eq!(faces.len(), 2 * verts - 4);
    }

    #[test]
    fn all_cloud_points_lie_inside_hull() {
        let pts = random_cloud(0x0BAD_F00D, 80);
        let faces = convex_hull_3d(&pts);
        assert!(!faces.is_empty());
        for &p in &pts {
            for &f in &faces {
                let d = hull_signed_distance(&pts, f, p);
                assert!(d <= TOL, "point outside a face by {d}");
            }
        }
    }

    #[test]
    fn every_face_normal_points_away_from_centroid() {
        let pts = random_cloud(0x00C0_FFEE, 50);
        let faces = convex_hull_3d(&pts);
        assert!(!faces.is_empty());
        // Mean of all points is a strictly interior reference here.
        let mut c = [0.0f32; 3];
        for &p in &pts {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let inv = 1.0 / (pts.len() as f32);
        let centroid = [c[0] * inv, c[1] * inv, c[2] * inv];
        for &f in &faces {
            let d = hull_signed_distance(&pts, f, centroid);
            assert!(d < -TOL, "centroid on outward side by {d}");
        }
    }

    #[test]
    fn convexity_every_point_inside_every_face() {
        let pts = random_cloud(0x2468_ACE0, 40);
        let faces = convex_hull_3d(&pts);
        assert!(!faces.is_empty());
        for (i, &p) in pts.iter().enumerate() {
            for &f in &faces {
                if f.contains(&i) {
                    continue;
                }
                let d = hull_signed_distance(&pts, f, p);
                assert!(d <= TOL, "vertex {i} outside a face by {d}");
            }
        }
    }

    #[test]
    fn face_normal_matches_cross_product() {
        let pts = tetra();
        let n = face_normal(&pts, [0, 1, 2]);
        // Triangle (0,0,0),(1,0,0),(0,1,0) has cross product +z.
        assert!(approx(n[0], 0.0));
        assert!(approx(n[1], 0.0));
        assert!(n[2].abs() > HULL_EPS);
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    #[test]
    fn hull_signed_distance_positive_outside() {
        let pts = tetra();
        let faces = convex_hull_3d(&pts);
        let far = [5.0, 5.0, 5.0];
        // At least one face must report the far point on its outward side.
        assert!(faces
            .iter()
            .any(|&f| hull_signed_distance(&pts, f, far) > TOL));
    }

    #[test]
    fn hull_signed_distance_zero_on_vertex() {
        let pts = tetra();
        let faces = convex_hull_3d(&pts);
        for &f in &faces {
            let d = hull_signed_distance(&pts, f, pts[f[0]]);
            assert!(d.abs() <= TOL, "vertex not on its own plane: {d}");
        }
    }

    #[test]
    fn translation_preserves_vertex_set_and_count() {
        let pts = cube();
        let base = convex_hull_3d(&pts);
        let shifted: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| [p[0] + 10.0, p[1] - 4.0, p[2] + 7.0])
            .collect();
        let moved = convex_hull_3d(&shifted);
        assert_eq!(base.len(), moved.len());
        assert_eq!(referenced_vertices(&base), referenced_vertices(&moved));
    }

    #[test]
    fn uniform_scaling_preserves_vertex_set_and_count() {
        let pts = octahedron();
        let base = convex_hull_3d(&pts);
        let scaled: Vec<[f32; 3]> = pts
            .iter()
            .map(|p| [p[0] * 3.5, p[1] * 3.5, p[2] * 3.5])
            .collect();
        let bigger = convex_hull_3d(&scaled);
        assert_eq!(base.len(), bigger.len());
        assert_eq!(referenced_vertices(&base), referenced_vertices(&bigger));
    }

    #[test]
    fn permutation_preserves_vertex_set() {
        let pts = cube();
        let base = convex_hull_3d(&pts);
        // Reverse the input order and remap the resulting indices back.
        let n = pts.len();
        let reversed: Vec<[f32; 3]> = pts.iter().rev().copied().collect();
        let faces = convex_hull_3d(&reversed);
        let remapped: Vec<usize> = referenced_vertices(&faces)
            .iter()
            .map(|&idx| n - 1 - idx)
            .collect();
        let mut remapped_sorted = remapped;
        remapped_sorted.sort_unstable();
        assert_eq!(remapped_sorted, referenced_vertices(&base));
    }

    #[test]
    fn construction_is_deterministic() {
        let pts = random_cloud(0x5EED_1234, 70);
        let a = convex_hull_3d(&pts);
        let b = convex_hull_3d(&pts);
        assert_eq!(a, b);
    }

    #[test]
    fn adding_exterior_point_expands_the_hull() {
        let mut pts = cube();
        let before = convex_hull_3d(&pts).len();
        // A point well beyond one corner must join the hull as a vertex.
        pts.push([2.0, 2.0, 2.0]);
        let faces = convex_hull_3d(&pts);
        assert!(faces.iter().any(|f| f.contains(&8)));
        assert!(faces.len() >= before);
    }
}
