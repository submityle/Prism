//! Gilbert-Johnson-Keerthi distance and intersection test for two posed convex
//! hulls, built on the Minkowski-difference support ([`super::minkowski`]).
//!
//! `GJK` walks a simplex (one to four points) of the Minkowski difference
//! `A (-) B` toward the origin. The pair overlaps exactly when that difference
//! contains the origin; when it does not, the simplex converges onto the feature
//! nearest the origin and its barycentric weights reconstruct the closest point
//! on each body.
//!
//! # Outcome
//!
//! * [`GjkStatus::Separated`] carries the separation distance, the closest point
//!   on each hull in world space, and the unit normal pointing from `B` toward
//!   `A` (the direction that separates them). This is the shortest-distance
//!   query a broad phase or a speculative-contact margin consumes directly.
//! * [`GjkStatus::Intersecting`] carries the terminating simplex (one to four
//!   support points whose convex hull contains the origin). This is the seed the
//!   expanding-polytope algorithm ([`super::epa`]) grows to recover the
//!   penetration normal and depth; the stored witnesses ride along so the
//!   contact points survive.
//!
//! # Method
//!
//! Each iteration finds the point `v` on the current simplex closest to the
//! origin via the Voronoi-region sub-distance of Christer Ericson, *Real-Time
//! Collision Detection* (2005): the closest feature of a segment, triangle, or
//! tetrahedron to a query point, specialised to the origin. The reduced simplex
//! keeps only the vertices whose barycentric weight is non-zero. The next
//! support is taken toward `-v`; when it fails to make progress past `v`
//! (within a relative tolerance) the pair is separated, and when the simplex
//! grows to enclose the origin the pair intersects. The iteration count is
//! capped so a pathological input terminates rather than spins.
//!
//! Provenance: Gilbert, Johnson, and Keerthi, *A Fast Procedure for Computing
//! the Distance Between Complex Objects* (1988), with the Voronoi sub-distance
//! from Ericson (2005). No Unreal Engine source or derived code.

use glam::Vec3;

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::minkowski::{support, SupportPoint};

/// Squared length below which a vector is treated as the zero vector (origin
/// reached or a degenerate direction).
const ZERO_EPS2: f32 = 1.0e-12;

/// Relative progress tolerance: the search stops when a new support advances the
/// closest distance by less than this fraction of the current squared distance.
const PROGRESS_TOL: f32 = 1.0e-8;

/// Hard cap on simplex iterations so a degenerate pair terminates.
const MAX_ITERS: u32 = 64;

/// The result of a `GJK` query between two posed convex hulls.
#[derive(Clone, Debug)]
pub enum GjkStatus {
    /// The hulls are disjoint; carries the separation geometry.
    Separated {
        /// Shortest distance between the two surfaces.
        distance: f32,
        /// Closest point on hull `A` in world space.
        point_a: Vec3,
        /// Closest point on hull `B` in world space.
        point_b: Vec3,
        /// Unit normal pointing from `B` toward `A`.
        normal: Vec3,
    },
    /// The hulls overlap; carries the simplex (one to four support points) whose
    /// convex hull contains the origin, as the seed for `EPA`.
    Intersecting(Vec<SupportPoint>),
}

/// Runs `GJK` between `(hull_a, pose_a)` and `(hull_b, pose_b)`.
///
/// Returns [`GjkStatus::Separated`] with the closest-feature geometry when the
/// hulls are disjoint, or [`GjkStatus::Intersecting`] with the enclosing simplex
/// when they overlap.
#[must_use]
pub fn gjk(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
) -> GjkStatus {
    // Seed the search along the line between the two hull origins.
    let seed_dir = {
        let d = pose_a.translation - pose_b.translation;
        if d.length_squared() < ZERO_EPS2 {
            Vec3::X
        } else {
            d
        }
    };
    let mut simplex: Vec<SupportPoint> = vec![support(hull_a, pose_a, hull_b, pose_b, seed_dir)];
    let mut closest = simplex[0].diff;

    for _ in 0..MAX_ITERS {
        if closest.length_squared() < ZERO_EPS2 {
            return GjkStatus::Intersecting(simplex);
        }
        let dir = -closest;
        let w = support(hull_a, pose_a, hull_b, pose_b, dir);

        // Progress test: if the new support is no farther toward the origin than
        // the current closest point, the pair is separated and {closest} is the
        // nearest point of the Minkowski difference to the origin.
        let advance = closest.dot(closest) - closest.dot(w.diff);
        if advance <= PROGRESS_TOL * closest.dot(closest).max(1.0) {
            return separated(&simplex);
        }
        // Guard against re-adding a vertex already in the simplex.
        if simplex
            .iter()
            .any(|s| (s.diff - w.diff).length_squared() < ZERO_EPS2)
        {
            return separated(&simplex);
        }

        simplex.push(w);
        match reduce(&simplex) {
            Reduction::Contained => return GjkStatus::Intersecting(simplex),
            Reduction::Sub { kept, closest: c } => {
                simplex = kept;
                closest = c;
            }
        }
    }
    separated(&simplex)
}

/// Builds the separated outcome from the terminal simplex: the closest point to
/// the origin, split back into per-body witnesses through its barycentric
/// weights.
fn separated(simplex: &[SupportPoint]) -> GjkStatus {
    let (weights, closest) = barycentric(simplex);
    let mut point_a = Vec3::ZERO;
    let mut point_b = Vec3::ZERO;
    for (s, w) in simplex.iter().zip(weights.iter()) {
        point_a += s.on_a * *w;
        point_b += s.on_b * *w;
    }
    let distance = closest.length();
    let normal = if distance > ZERO_EPS2.sqrt() {
        closest / distance
    } else {
        Vec3::X
    };
    GjkStatus::Separated {
        distance,
        point_a,
        point_b,
        normal,
    }
}

/// The reduction of a simplex to the sub-feature nearest the origin.
enum Reduction {
    /// The simplex (a tetrahedron) encloses the origin.
    Contained,
    /// The nearest sub-feature and the origin's closest point on it.
    Sub {
        /// The retained support points (those with non-zero weight).
        kept: Vec<SupportPoint>,
        /// The point on the sub-feature closest to the origin.
        closest: Vec3,
    },
}

/// Reduces a one-to-four-point simplex to the sub-feature closest to the origin.
fn reduce(simplex: &[SupportPoint]) -> Reduction {
    match simplex.len() {
        1 => Reduction::Sub {
            kept: simplex.to_vec(),
            closest: simplex[0].diff,
        },
        2 => reduce_segment(simplex),
        3 => reduce_triangle(simplex),
        4 => reduce_tetrahedron(simplex),
        _ => unreachable!("GJK simplex never exceeds four points"),
    }
}

/// Barycentric weights of the origin's closest point on a one-to-four-point
/// simplex, together with that closest point. Used to reconstruct witnesses.
fn barycentric(simplex: &[SupportPoint]) -> (Vec<f32>, Vec3) {
    match reduce(simplex) {
        Reduction::Contained => {
            // Only reachable for a tetrahedron that encloses the origin, where
            // the closest point is the origin itself; weight the first vertex.
            let mut w = vec![0.0; simplex.len()];
            w[0] = 1.0;
            (w, Vec3::ZERO)
        }
        Reduction::Sub { kept, closest } => {
            // Map the kept vertices back onto the full simplex by matching diff.
            let weights_kept = segment_or_face_weights(&kept, closest);
            let mut full = vec![0.0; simplex.len()];
            for (k, wk) in kept.iter().zip(weights_kept.iter()) {
                if let Some(idx) = simplex
                    .iter()
                    .position(|s| (s.diff - k.diff).length_squared() < ZERO_EPS2)
                {
                    full[idx] = *wk;
                }
            }
            (full, closest)
        }
    }
}

/// Barycentric weights of {closest} over a one-to-three-vertex kept feature.
fn segment_or_face_weights(kept: &[SupportPoint], closest: Vec3) -> Vec<f32> {
    match kept.len() {
        1 => vec![1.0],
        2 => {
            let a = kept[0].diff;
            let b = kept[1].diff;
            let ab = b - a;
            let t = (closest - a).dot(ab) / ab.dot(ab).max(ZERO_EPS2);
            vec![1.0 - t, t]
        }
        _ => {
            let a = kept[0].diff;
            let b = kept[1].diff;
            let c = kept[2].diff;
            let (u, v, w) = triangle_barycentric(a, b, c, closest);
            vec![u, v, w]
        }
    }
}

/// Closest sub-feature of a two-point simplex (segment) to the origin.
fn reduce_segment(simplex: &[SupportPoint]) -> Reduction {
    let a = simplex[0].diff;
    let b = simplex[1].diff;
    let ab = b - a;
    let t = (-a).dot(ab);
    if t <= 0.0 {
        return Reduction::Sub {
            kept: vec![simplex[0]],
            closest: a,
        };
    }
    let denom = ab.dot(ab);
    if t >= denom {
        return Reduction::Sub {
            kept: vec![simplex[1]],
            closest: b,
        };
    }
    let s = t / denom;
    Reduction::Sub {
        kept: vec![simplex[0], simplex[1]],
        closest: a + ab * s,
    }
}

/// Closest sub-feature of a three-point simplex (triangle) to the origin.
fn reduce_triangle(simplex: &[SupportPoint]) -> Reduction {
    let a = simplex[0].diff;
    let b = simplex[1].diff;
    let c = simplex[2].diff;
    let region = triangle_region(a, b, c);
    reduction_from_region(simplex, region)
}

/// Closest sub-feature of a four-point simplex (tetrahedron) to the origin, or
/// [`Reduction::Contained`] when the origin lies inside it.
fn reduce_tetrahedron(simplex: &[SupportPoint]) -> Reduction {
    let a = simplex[0].diff;
    let b = simplex[1].diff;
    let c = simplex[2].diff;
    let d = simplex[3].diff;

    let mut best: Option<(f32, Reduction)> = None;
    // Each face is wound so its reference fourth vertex is on the inside.
    let faces = [
        ([0, 1, 2], d),
        ([0, 3, 1], c),
        ([0, 2, 3], b),
        ([1, 3, 2], a),
    ];
    let mut inside_all = true;
    for (idx, inner) in faces {
        let p0 = simplex[idx[0]].diff;
        let p1 = simplex[idx[1]].diff;
        let p2 = simplex[idx[2]].diff;
        if !origin_outside_face(p0, p1, p2, inner) {
            continue;
        }
        inside_all = false;
        let face = [simplex[idx[0]], simplex[idx[1]], simplex[idx[2]]];
        let region = triangle_region(p0, p1, p2);
        let reduction = reduction_from_region(&face, region);
        let dist2 = match &reduction {
            Reduction::Sub { closest, .. } => closest.length_squared(),
            Reduction::Contained => continue,
        };
        if best.as_ref().is_none_or(|(bd, _)| dist2 < *bd) {
            best = Some((dist2, reduction));
        }
    }
    if inside_all {
        return Reduction::Contained;
    }
    best.map_or(Reduction::Contained, |(_, r)| r)
}

/// Whether the origin lies on the outward side of the face `(p0, p1, p2)`, i.e.
/// the opposite side from the tetrahedron's fourth vertex `inner`.
fn origin_outside_face(p0: Vec3, p1: Vec3, p2: Vec3, inner: Vec3) -> bool {
    let n = (p1 - p0).cross(p2 - p0);
    let origin_side = n.dot(-p0);
    let inner_side = n.dot(inner - p0);
    // Outside when the origin and the inner vertex are on opposite sides.
    origin_side * inner_side < 0.0
}

/// Which Voronoi region of a triangle the origin falls in, as the set of kept
/// vertex indices.
#[derive(Clone, Copy)]
enum TriRegion {
    /// A single vertex `i`.
    Vertex(usize),
    /// The edge between vertices `i` and `j`.
    Edge(usize, usize),
    /// The triangle face itself.
    Face,
}

/// Classifies the origin against a triangle's Voronoi regions (Ericson 2005,
/// specialised to the query point at the origin).
fn triangle_region(a: Vec3, b: Vec3, c: Vec3) -> TriRegion {
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return TriRegion::Vertex(0);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return TriRegion::Vertex(1);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return TriRegion::Edge(0, 1);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return TriRegion::Vertex(2);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return TriRegion::Edge(0, 2);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return TriRegion::Edge(1, 2);
    }
    TriRegion::Face
}

/// Barycentric weights `(u, v, w)` of the origin's closest point on triangle
/// `(a, b, c)` assuming the origin projects inside the face.
fn triangle_barycentric(a: Vec3, b: Vec3, c: Vec3, closest: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = closest - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = (d00 * d11 - d01 * d01).max(ZERO_EPS2);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

/// Turns a triangle Voronoi classification into a [`Reduction`] over the given
/// three support points, computing the closest point on the chosen feature.
fn reduction_from_region(face: &[SupportPoint], region: TriRegion) -> Reduction {
    match region {
        TriRegion::Vertex(i) => Reduction::Sub {
            kept: vec![face[i]],
            closest: face[i].diff,
        },
        TriRegion::Edge(i, j) => {
            let a = face[i].diff;
            let b = face[j].diff;
            let ab = b - a;
            let t = ((-a).dot(ab) / ab.dot(ab).max(ZERO_EPS2)).clamp(0.0, 1.0);
            Reduction::Sub {
                kept: vec![face[i], face[j]],
                closest: a + ab * t,
            }
        }
        TriRegion::Face => {
            let a = face[0].diff;
            let b = face[1].diff;
            let c = face[2].diff;
            // Project the origin onto the plane of the face. The face passes
            // through {a} with normal {n}, so the origin's projection is
            // {n * (dot(n, a) / dot(n, n))}.
            let n = (b - a).cross(c - a);
            let n2 = n.dot(n).max(ZERO_EPS2);
            let proj = n * (n.dot(a) / n2);
            Reduction::Sub {
                kept: vec![face[0], face[1], face[2]],
                closest: proj,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    /// Builds a unit cube (half-extent 1) at the given translation with no
    /// rotation.
    fn boxed_at(x: f32, y: f32, z: f32) -> (ConvexHull, ConvexPose) {
        (
            ConvexHull::from_box(Vec3::splat(1.0)),
            ConvexPose::new(Vec3::new(x, y, z), Quat::IDENTITY),
        )
    }

    fn separated_fields(status: &GjkStatus) -> (f32, Vec3, Vec3, Vec3) {
        match status {
            GjkStatus::Separated {
                distance,
                point_a,
                point_b,
                normal,
            } => (*distance, *point_a, *point_b, *normal),
            GjkStatus::Intersecting(_) => panic!("expected Separated, got Intersecting"),
        }
    }

    #[test]
    fn axis_separated_boxes_report_gap_and_witnesses() {
        // A spans x in [2, 4]; B spans x in [-1, 1]; the gap along +x is 1.
        let (a, pose_a) = boxed_at(3.0, 0.0, 0.0);
        let (b, pose_b) = boxed_at(0.0, 0.0, 0.0);
        let status = gjk(&a, &pose_a, &b, &pose_b);
        let (distance, point_a, point_b, normal) = separated_fields(&status);
        assert!((distance - 1.0).abs() < 1.0e-4, "distance {distance}");
        // Normal points from B toward A, i.e. +x.
        assert!((normal - Vec3::X).length() < 1.0e-4, "normal {normal:?}");
        // Witness on A sits on its -x face (x = 2); on B on its +x face (x = 1).
        assert!((point_a.x - 2.0).abs() < 1.0e-4, "point_a {point_a:?}");
        assert!((point_b.x - 1.0).abs() < 1.0e-4, "point_b {point_b:?}");
    }

    #[test]
    fn diagonally_separated_boxes_report_corner_distance() {
        // A spans [2,4] in x and y; the nearest corner-to-corner gap to B is
        // the diagonal from (1,1) to (2,2): length sqrt(2).
        let (a, pose_a) = boxed_at(3.0, 3.0, 0.0);
        let (b, pose_b) = boxed_at(0.0, 0.0, 0.0);
        let status = gjk(&a, &pose_a, &b, &pose_b);
        let (distance, point_a, point_b, normal) = separated_fields(&status);
        let expected = core::f32::consts::SQRT_2;
        assert!((distance - expected).abs() < 1.0e-4, "distance {distance}");
        let want_n = Vec3::new(1.0, 1.0, 0.0).normalize();
        assert!((normal - want_n).length() < 1.0e-4, "normal {normal:?}");
        assert!((point_a.x - 2.0).abs() < 1.0e-4 && (point_a.y - 2.0).abs() < 1.0e-4, "point_a {point_a:?}");
        assert!((point_b.x - 1.0).abs() < 1.0e-4 && (point_b.y - 1.0).abs() < 1.0e-4, "point_b {point_b:?}");
    }

    #[test]
    fn overlapping_boxes_intersect() {
        // A spans [-0.5, 1.5]; B spans [-1, 1]; they overlap in x.
        let (a, pose_a) = boxed_at(0.5, 0.0, 0.0);
        let (b, pose_b) = boxed_at(0.0, 0.0, 0.0);
        let status = gjk(&a, &pose_a, &b, &pose_b);
        assert!(
            matches!(status, GjkStatus::Intersecting(ref s) if (1..=4).contains(&s.len())),
            "expected Intersecting with a 1..=4 simplex, got {status:?}"
        );
    }

    #[test]
    fn concentric_boxes_intersect() {
        // Coincident centres: the Minkowski difference straddles the origin.
        let (a, pose_a) = boxed_at(0.0, 0.0, 0.0);
        let (b, pose_b) = boxed_at(0.0, 0.0, 0.0);
        let status = gjk(&a, &pose_a, &b, &pose_b);
        assert!(matches!(status, GjkStatus::Intersecting(_)), "got {status:?}");
    }

    #[test]
    fn rotated_box_still_separated_reports_positive_distance() {
        // A turned 45 degrees about z and pushed far along +x stays disjoint.
        let a = ConvexHull::from_box(Vec3::splat(1.0));
        let pose_a = ConvexPose::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        );
        let (b, pose_b) = boxed_at(0.0, 0.0, 0.0);
        let status = gjk(&a, &pose_a, &b, &pose_b);
        let (distance, _, _, normal) = separated_fields(&status);
        // A rotated 45 degrees reaches x = 5 - sqrt(2); B reaches x = 1; the gap
        // is 5 - sqrt(2) - 1 = 4 - sqrt(2).
        let expected = 4.0 - core::f32::consts::SQRT_2;
        assert!((distance - expected).abs() < 1.0e-3, "distance {distance}");
        assert!((normal - Vec3::X).length() < 1.0e-3, "normal {normal:?}");
    }
}
