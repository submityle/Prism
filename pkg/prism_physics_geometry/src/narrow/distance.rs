//! Closest-distance query between two separated convex shapes via GJK.
//!
//! [`gjk_closest_points`] returns the shortest distance between two convex
//! shapes together with the pair of witness points that realise it, one on
//! each shape. When the shapes overlap it returns `None`; penetration recovery
//! is the job of [`crate::narrow::gjk_contact`] (EPA). This is a clean-room
//! implementation of the publicly documented GJK distance sub-algorithm and
//! contains no Unreal Engine source or derived code.

use alloc::vec::Vec;
use glam::Vec3;

use crate::narrow::minkowski::{support, SupportVertex};
use crate::narrow::support::SupportMap;

/// Maximum GJK refinement iterations before returning the best estimate.
const MAX_ITERATIONS: usize = 32;

/// Squared tolerance treating the origin as reached (shapes touch/overlap).
const INTERSECT_EPSILON_SQ: f32 = 1.0e-10;

/// Progress tolerance: when a new support advances the simplex less than this
/// along the search direction, the closest feature has been found.
const PROGRESS_EPSILON: f32 = 1.0e-6;

/// The closest pair of points between two separated convex shapes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ClosestPoints {
    /// Distance between the shapes (always positive on success).
    pub distance: f32,
    /// Witness point on the surface of shape `a`.
    pub point_a: Vec3,
    /// Witness point on the surface of shape `b`.
    pub point_b: Vec3,
    /// Unit direction from `point_a` toward `point_b`.
    pub normal: Vec3,
}

/// Returns the closest points between convex shapes `a` and `b`.
///
/// Returns `None` when the shapes overlap or touch (distance zero), in which
/// case the penetration query [`crate::narrow::gjk_contact`] should be used
/// instead.
pub fn gjk_closest_points<A: SupportMap, B: SupportMap>(a: &A, b: &B) -> Option<ClosestPoints> {
    let mut simplex: Vec<SupportVertex> = Vec::with_capacity(4);
    let first = support(a, b, Vec3::X);
    simplex.push(first);

    let mut closest = first.v;
    let mut witness_a = first.a;
    let mut witness_b = first.b;

    for _ in 0..MAX_ITERATIONS {
        let dir = -closest;
        if dir.length_squared() <= INTERSECT_EPSILON_SQ {
            // The origin lies on the simplex: the shapes overlap.
            return None;
        }
        let w = support(a, b, dir);

        // No measurable progress toward the origin: closest feature found.
        if closest.length_squared() - closest.dot(w.v) <= PROGRESS_EPSILON {
            break;
        }
        if simplex
            .iter()
            .any(|q| (q.v - w.v).length_squared() < INTERSECT_EPSILON_SQ)
        {
            break;
        }

        simplex.push(w);
        match reduce_simplex(&mut simplex) {
            // Origin enclosed by a tetrahedron: shapes overlap.
            None => return None,
            Some((point, wa, wb)) => {
                closest = point;
                witness_a = wa;
                witness_b = wb;
            }
        }
    }

    let distance = closest.length();
    if distance <= INTERSECT_EPSILON_SQ.sqrt() {
        return None;
    }
    Some(ClosestPoints {
        distance,
        point_a: witness_a,
        point_b: witness_b,
        // `closest == point_a - point_b`, so the a->b direction is its negation.
        normal: -closest / distance,
    })
}

/// Reduces the simplex to the feature closest to the origin, returning the
/// closest point and the interpolated witness points on each shape.
///
/// Returns `None` when a tetrahedron encloses the origin (the shapes overlap).
fn reduce_simplex(simplex: &mut Vec<SupportVertex>) -> Option<(Vec3, Vec3, Vec3)> {
    match simplex.len() {
        1 => {
            let s = simplex[0];
            Some((s.v, s.a, s.b))
        }
        2 => Some(reduce_segment(simplex)),
        3 => Some(reduce_triangle(simplex)),
        4 => reduce_tetrahedron(simplex),
        _ => None,
    }
}

/// Interpolates a witness point pair from barycentric weights.
fn witness(verts: &[SupportVertex], weights: &[f32]) -> (Vec3, Vec3, Vec3) {
    let mut v = Vec3::ZERO;
    let mut a = Vec3::ZERO;
    let mut b = Vec3::ZERO;
    for (vert, &w) in verts.iter().zip(weights) {
        v += vert.v * w;
        a += vert.a * w;
        b += vert.b * w;
    }
    (v, a, b)
}

/// Barycentric weights of the origin's closest point on segment `[0, 1]`.
/// Returns the surviving vertex indices and matching weights.
fn segment_bary(p0: Vec3, p1: Vec3) -> (Vec<usize>, Vec<f32>) {
    let ab = p1 - p0;
    let denom = ab.dot(ab);
    if denom <= f32::EPSILON {
        return (alloc::vec![0], alloc::vec![1.0]);
    }
    let t = (-p0).dot(ab) / denom;
    if t <= 0.0 {
        (alloc::vec![0], alloc::vec![1.0])
    } else if t >= 1.0 {
        (alloc::vec![1], alloc::vec![1.0])
    } else {
        (alloc::vec![0, 1], alloc::vec![1.0 - t, t])
    }
}

fn reduce_segment(simplex: &mut Vec<SupportVertex>) -> (Vec3, Vec3, Vec3) {
    let (idx, weights) = segment_bary(simplex[0].v, simplex[1].v);
    let kept: Vec<SupportVertex> = idx.iter().map(|&i| simplex[i]).collect();
    *simplex = kept;
    witness(simplex, &weights)
}

/// Barycentric reduction of the origin's closest point on triangle `[0, 1, 2]`,
/// following the Voronoi-region analysis in Ericson's *Real-Time Collision
/// Detection*. Returns surviving local indices and weights.
fn triangle_bary(a: Vec3, b: Vec3, c: Vec3) -> (Vec<usize>, Vec<f32>) {
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (alloc::vec![0], alloc::vec![1.0]);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (alloc::vec![1], alloc::vec![1.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (alloc::vec![0, 1], alloc::vec![1.0 - v, v]);
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (alloc::vec![2], alloc::vec![1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (alloc::vec![0, 2], alloc::vec![1.0 - w, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (alloc::vec![1, 2], alloc::vec![1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (alloc::vec![0, 1, 2], alloc::vec![1.0 - v - w, v, w])
}

fn reduce_triangle(simplex: &mut Vec<SupportVertex>) -> (Vec3, Vec3, Vec3) {
    let (idx, weights) = triangle_bary(simplex[0].v, simplex[1].v, simplex[2].v);
    let kept: Vec<SupportVertex> = idx.iter().map(|&i| simplex[i]).collect();
    *simplex = kept;
    witness(simplex, &weights)
}

/// Signed six-times volume of the tetrahedron `(p, q, r, s)`.
fn signed_volume(p: Vec3, q: Vec3, r: Vec3, s: Vec3) -> f32 {
    (q - p).dot((r - p).cross(s - p))
}

/// Reduces a tetrahedron, returning `None` when it encloses the origin.
fn reduce_tetrahedron(simplex: &mut Vec<SupportVertex>) -> Option<(Vec3, Vec3, Vec3)> {
    let p = [simplex[0].v, simplex[1].v, simplex[2].v, simplex[3].v];
    let whole = signed_volume(p[0], p[1], p[2], p[3]);
    if whole.abs() <= f32::EPSILON {
        // Degenerate (flat) tetrahedron: fall back to its largest face.
        return Some(reduce_best_face(simplex));
    }
    // Barycentric coordinates of the origin relative to the tetrahedron.
    let b0 = signed_volume(Vec3::ZERO, p[1], p[2], p[3]) / whole;
    let b1 = signed_volume(p[0], Vec3::ZERO, p[2], p[3]) / whole;
    let b2 = signed_volume(p[0], p[1], Vec3::ZERO, p[3]) / whole;
    let b3 = signed_volume(p[0], p[1], p[2], Vec3::ZERO) / whole;
    if b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0 {
        // Origin is inside: the shapes overlap.
        return None;
    }
    Some(reduce_best_face(simplex))
}

/// Picks the tetrahedron face whose closest point to the origin is nearest and
/// reduces the simplex to that face's feature.
fn reduce_best_face(simplex: &mut Vec<SupportVertex>) -> (Vec3, Vec3, Vec3) {
    const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut best_point = Vec3::ZERO;
    let mut best_a = Vec3::ZERO;
    let mut best_b = Vec3::ZERO;
    let mut best_kept: Vec<SupportVertex> = Vec::new();
    let mut best_dist = f32::INFINITY;
    for face in FACES {
        let [i, j, k] = face;
        let (local, weights) = triangle_bary(simplex[i].v, simplex[j].v, simplex[k].v);
        let kept: Vec<SupportVertex> = local.iter().map(|&l| simplex[face[l]]).collect();
        let (point, wa, wb) = witness(&kept, &weights);
        let dist = point.length_squared();
        if dist < best_dist {
            best_dist = dist;
            best_point = point;
            best_a = wa;
            best_b = wb;
            best_kept = kept;
        }
    }
    *simplex = best_kept;
    (best_point, best_a, best_b)
}

#[cfg(test)]
mod tests {
    use super::gjk_closest_points;
    use crate::bounding::{Aabb, BoundingSphere, Obb};
    use glam::{Quat, Vec3};

    #[test]
    fn separated_spheres_distance_and_witnesses() {
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let b = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r = gjk_closest_points(&a, &b).expect("separated");
        // Centres 5 apart, radii 1 each: surface gap = 3.
        assert!((r.distance - 3.0).abs() < 1.0e-3, "distance = {}", r.distance);
        assert!(r.point_a.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-2));
        assert!(r.point_b.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1.0e-2));
        assert!(r.normal.dot(Vec3::X) > 0.99, "normal = {:?}", r.normal);
    }

    #[test]
    fn overlapping_spheres_report_none() {
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let b = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(gjk_closest_points(&a, &b).is_none());
    }

    #[test]
    fn separated_boxes_distance_along_axis() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::new(3.0, -1.0, -1.0), Vec3::new(5.0, 1.0, 1.0));
        let r = gjk_closest_points(&a, &b).expect("separated");
        // Faces at x = 1 and x = 3: gap = 2.
        assert!((r.distance - 2.0).abs() < 1.0e-3, "distance = {}", r.distance);
        assert!((r.point_a.x - 1.0).abs() < 1.0e-3, "point_a = {:?}", r.point_a);
        assert!((r.point_b.x - 3.0).abs() < 1.0e-3, "point_b = {:?}", r.point_b);
    }

    #[test]
    fn diagonal_boxes_distance_is_corner_to_corner() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let b = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(4.0, 4.0, 4.0));
        let r = gjk_closest_points(&a, &b).expect("separated");
        // Nearest corners (1,1,1) and (2,2,2): gap = sqrt(3).
        let expected = (3.0_f32).sqrt();
        assert!((r.distance - expected).abs() < 1.0e-3, "distance = {}", r.distance);
        assert!(r.point_a.abs_diff_eq(Vec3::splat(1.0), 1.0e-2));
        assert!(r.point_b.abs_diff_eq(Vec3::splat(2.0), 1.0e-2));
    }

    #[test]
    fn touching_obbs_report_none() {
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        let b = Obb::new(Vec3::new(0.6, 0.0, 0.0), Vec3::splat(0.5), Quat::IDENTITY);
        // Overlapping by 0.4 along x -> not separated.
        assert!(gjk_closest_points(&a, &b).is_none());
    }
}
