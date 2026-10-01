//! Boolean convex intersection via the Gilbert-Johnson-Keerthi (GJK) algorithm.
//!
//! GJK searches the Minkowski difference `A (-) B` for the origin. If the
//! origin is contained the shapes overlap. The implementation evolves a
//! simplex of up to four points using the standard Voronoi-region case
//! analysis and is a clean-room implementation of the publicly documented
//! algorithm (no Unreal Engine source or derived code).

use alloc::vec::Vec;
use glam::Vec3;

use crate::narrow::support::SupportMap;

/// Maximum simplex refinement iterations before conceding to the fallback.
const MAX_ITERATIONS: usize = 64;

/// Squared tolerance for treating a freshly generated support point as a
/// duplicate (no further progress toward the origin).
const DUPLICATE_EPSILON_SQ: f32 = 1.0e-10;

/// Returns `true` when the convex shapes `a` and `b` overlap.
///
/// Shapes that merely touch at a boundary may report either result because the
/// test is strict about progress toward the origin; callers needing a margin
/// should inflate one shape's support mapping.
pub fn gjk_intersect<A: SupportMap, B: SupportMap>(a: &A, b: &B) -> bool {
    // Support point of the Minkowski difference along `dir`.
    let minkowski = |dir: Vec3| a.support_point(dir) - b.support_point(-dir);

    let mut simplex: Vec<Vec3> = Vec::with_capacity(4);
    let first = minkowski(Vec3::X);
    simplex.push(first);
    let mut dir = -first;

    for _ in 0..MAX_ITERATIONS {
        if dir.length_squared() <= DUPLICATE_EPSILON_SQ {
            // The current simplex already straddles the origin.
            return true;
        }
        let point = minkowski(dir);
        if point.dot(dir) < 0.0 {
            // The farthest point in the search direction fell short of the
            // origin, so a separating axis exists.
            return false;
        }
        if simplex
            .iter()
            .any(|q| (*q - point).length_squared() < DUPLICATE_EPSILON_SQ)
        {
            // No progress: the closest feature has been found and it does not
            // enclose the origin.
            return false;
        }
        simplex.push(point);
        if evolve_simplex(&mut simplex, &mut dir) {
            return true;
        }
    }

    // Degenerate convergence (e.g. grazing contact): treat as overlapping.
    true
}

/// Returns `true` when `a` and `b` point in the same half-space.
fn same_direction(a: Vec3, b: Vec3) -> bool {
    a.dot(b) > 0.0
}

/// Advances the simplex toward the origin, returning `true` once the origin is
/// enclosed.
fn evolve_simplex(simplex: &mut Vec<Vec3>, dir: &mut Vec3) -> bool {
    match simplex.len() {
        2 => line_case(simplex, dir),
        3 => triangle_case(simplex, dir),
        4 => tetrahedron_case(simplex, dir),
        _ => false,
    }
}

fn line_case(simplex: &mut Vec<Vec3>, dir: &mut Vec3) -> bool {
    let a = simplex[1];
    let b = simplex[0];
    let ab = b - a;
    let ao = -a;
    if same_direction(ab, ao) {
        *dir = ab.cross(ao).cross(ab);
    } else {
        simplex.clear();
        simplex.push(a);
        *dir = ao;
    }
    false
}

fn triangle_case(simplex: &mut Vec<Vec3>, dir: &mut Vec3) -> bool {
    let a = simplex[2];
    let b = simplex[1];
    let c = simplex[0];
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);

    if same_direction(abc.cross(ac), ao) {
        if same_direction(ac, ao) {
            // Closest to edge ac.
            simplex.clear();
            simplex.push(c);
            simplex.push(a);
            *dir = ac.cross(ao).cross(ac);
        } else {
            // Reduce to edge ab.
            simplex.clear();
            simplex.push(b);
            simplex.push(a);
            return line_case(simplex, dir);
        }
    } else if same_direction(ab.cross(abc), ao) {
        // Reduce to edge ab.
        simplex.clear();
        simplex.push(b);
        simplex.push(a);
        return line_case(simplex, dir);
    } else if same_direction(abc, ao) {
        // Origin above the triangle face.
        *dir = abc;
    } else {
        // Origin below the face: flip winding so the normal points at it.
        simplex.clear();
        simplex.push(b);
        simplex.push(c);
        simplex.push(a);
        *dir = -abc;
    }
    false
}

fn tetrahedron_case(simplex: &mut Vec<Vec3>, dir: &mut Vec3) -> bool {
    let a = simplex[3];
    let b = simplex[2];
    let c = simplex[1];
    let d = simplex[0];
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if same_direction(abc, ao) {
        simplex.clear();
        simplex.push(c);
        simplex.push(b);
        simplex.push(a);
        return triangle_case(simplex, dir);
    }
    if same_direction(acd, ao) {
        simplex.clear();
        simplex.push(d);
        simplex.push(c);
        simplex.push(a);
        return triangle_case(simplex, dir);
    }
    if same_direction(adb, ao) {
        simplex.clear();
        simplex.push(b);
        simplex.push(d);
        simplex.push(a);
        return triangle_case(simplex, dir);
    }
    // Origin is on the inner side of all three faces: enclosed.
    true
}

#[cfg(test)]
mod tests {
    use super::gjk_intersect;
    use crate::bounding::{Aabb, BoundingSphere, Capsule, Obb};
    use core::f32::consts::FRAC_PI_4;
    use glam::{Quat, Vec3};

    #[test]
    fn spheres_overlap_and_separate() {
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let hit = BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        assert!(gjk_intersect(&a, &hit));
        let miss = BoundingSphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(!gjk_intersect(&a, &miss));
    }

    #[test]
    fn aabb_versus_sphere() {
        let box_ = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let touching = BoundingSphere::new(Vec3::new(1.8, 0.0, 0.0), 1.0);
        assert!(gjk_intersect(&box_, &touching));
        let apart = BoundingSphere::new(Vec3::new(2.5, 0.0, 0.0), 1.0);
        assert!(!gjk_intersect(&box_, &apart));
    }

    #[test]
    fn rotated_obbs() {
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        // Separated when axis aligned at 1.1, overlapping once spun 45°.
        let aligned = Obb::new(Vec3::new(1.1, 0.0, 0.0), Vec3::splat(0.5), Quat::IDENTITY);
        assert!(!gjk_intersect(&a, &aligned));
        let spun = Obb::new(
            Vec3::new(1.1, 0.0, 0.0),
            Vec3::splat(0.5),
            Quat::from_rotation_z(FRAC_PI_4),
        );
        assert!(gjk_intersect(&a, &spun));
    }

    #[test]
    fn capsule_versus_obb() {
        let cap = Capsule::new(Vec3::new(-2.0, 0.3, 0.0), Vec3::new(2.0, 0.3, 0.0), 0.25);
        let box_ = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        assert!(gjk_intersect(&cap, &box_));
        let high = Capsule::new(Vec3::new(-2.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.25);
        assert!(!gjk_intersect(&high, &box_));
    }
}
