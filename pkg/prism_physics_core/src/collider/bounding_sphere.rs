//! Minimum enclosing sphere (bounding ball) fitting.
//!
//! The smallest sphere containing a point set is the tightest spherical bound
//! available and the natural leaf/aggregate volume for sphere-tree broad phases
//! and distance culling. Unlike [`ConvexMeshData::bounding_radius`], which
//! measures the radius from the centroid, this module computes the *provably
//! minimal* enclosing sphere.
//!
//! [`minimal_bounding_sphere`] runs Welzl's algorithm: the minimal sphere is
//! pinned by at most four boundary points, so the fit walks the points and,
//! whenever one falls outside the current ball, rebuilds the ball with that
//! point forced onto the boundary. Interior points never define the sphere, so
//! the fit first reduces the input to its convex-hull vertices (via
//! [`convex_hull`]), bounding the combinatorial core to the handful of extreme
//! points a collider actually has.
//!
//! # Determinism
//!
//! The input is reduced and ordered deterministically (hull-vertex order, or a
//! sorted-and-deduplicated fallback for degenerate sets), and every geometric
//! primitive is a closed-form circumsphere built from dot/cross products and a
//! single 3x3 solve -- no transcendental calls, no randomization. The resulting
//! sphere is bit-for-bit reproducible, as required for cross-run state hashing.
//!
//! # Provenance
//!
//! Welzl's minimal-enclosing-ball algorithm and the circumsphere formulae are
//! textbook computational geometry (Welzl 1991). This module contains **no
//! Unreal Engine source or derived code**.

use glam::{Mat3, Vec3};

use super::convex_mesh::ConvexMeshData;
use super::hull::convex_hull;

/// Relative slack used when testing containment so floating-point round-off in
/// the circumsphere construction cannot drive Welzl into an infinite rebuild.
const CONTAIN_EPS: f32 = 1e-5;

/// Below this squared determinant/area a simplex is treated as degenerate and
/// the fit falls back to a lower-order circumsphere.
const DEGENERATE_EPS: f32 = 1e-12;

/// A sphere given by its centre and radius.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingSphere {
    /// Sphere centre.
    pub center: Vec3,
    /// Sphere radius. Non-negative.
    pub radius: f32,
}

impl BoundingSphere {
    /// A zero-radius sphere at `center`.
    #[must_use]
    pub fn point(center: Vec3) -> BoundingSphere {
        BoundingSphere {
            center,
            radius: 0.0,
        }
    }

    /// The sphere volume (`4/3 * pi * r^3`).
    #[must_use]
    pub fn volume(&self) -> f32 {
        (4.0 / 3.0) * core::f32::consts::PI * self.radius * self.radius * self.radius
    }

    /// Whether `point` lies inside the sphere, allowing a small relative slack.
    #[must_use]
    pub fn contains_point(&self, point: Vec3) -> bool {
        let slack = self.radius * CONTAIN_EPS + CONTAIN_EPS;
        let r = self.radius + slack;
        (point - self.center).length_squared() <= r * r
    }

    /// Conservative sphere-sphere overlap test.
    #[must_use]
    pub fn overlaps(&self, other: &BoundingSphere) -> bool {
        let r = self.radius + other.radius;
        (self.center - other.center).length_squared() <= r * r
    }
}

/// Computes the minimal enclosing sphere of a point cloud.
///
/// Returns `None` only when `points` is empty.
#[must_use]
pub fn minimal_bounding_sphere(points: &[Vec3]) -> Option<BoundingSphere> {
    if points.is_empty() {
        return None;
    }
    let reduced = match convex_hull(points) {
        Some((verts, _)) => verts,
        None => deduped(points),
    };
    Some(welzl(&reduced))
}

/// Computes the minimal enclosing sphere of a cooked convex mesh.
#[must_use]
pub fn mesh_bounding_sphere(mesh: &ConvexMeshData) -> BoundingSphere {
    welzl(mesh.vertices())
}

/// Sorts and removes exact duplicate points, giving a deterministic order for
/// the degenerate (non-3D-hull) fallback path.
fn deduped(points: &[Vec3]) -> Vec<Vec3> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    });
    pts.dedup();
    pts
}

/// Welzl's minimal enclosing ball over a (small) extreme-point set.
///
/// Four nested passes mirror the fact that at most four points can lie on the
/// optimal sphere; each pass forces one more point onto the boundary. Correct
/// for any point configuration and deterministic in the given order.
fn welzl(pts: &[Vec3]) -> BoundingSphere {
    if pts.is_empty() {
        return BoundingSphere::point(Vec3::ZERO);
    }
    let n = pts.len();
    let mut sphere = BoundingSphere::point(pts[0]);
    for i in 0..n {
        if sphere.contains_point(pts[i]) {
            continue;
        }
        sphere = BoundingSphere::point(pts[i]);
        for j in 0..i {
            if sphere.contains_point(pts[j]) {
                continue;
            }
            sphere = from_two(pts[i], pts[j]);
            for k in 0..j {
                if sphere.contains_point(pts[k]) {
                    continue;
                }
                sphere = from_three(pts[i], pts[j], pts[k]);
                for l in 0..k {
                    if sphere.contains_point(pts[l]) {
                        continue;
                    }
                    sphere = from_four(pts[i], pts[j], pts[k], pts[l]);
                }
            }
        }
    }
    sphere
}

/// The sphere with `a` and `b` as a diameter.
fn from_two(a: Vec3, b: Vec3) -> BoundingSphere {
    BoundingSphere {
        center: 0.5 * (a + b),
        radius: 0.5 * (b - a).length(),
    }
}

/// The circumsphere of triangle `(a, b, c)`.
fn from_three(a: Vec3, b: Vec3, c: Vec3) -> BoundingSphere {
    let u = b - a;
    let v = c - a;
    let n = u.cross(v);
    let denom = 2.0 * n.length_squared();
    if denom <= DEGENERATE_EPS {
        return widest_pair(&[a, b, c]);
    }
    let rel = (u.length_squared() * v - v.length_squared() * u).cross(n) / denom;
    BoundingSphere {
        center: a + rel,
        radius: rel.length(),
    }
}

/// The circumsphere of tetrahedron `(a, b, c, d)`.
fn from_four(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> BoundingSphere {
    let u = b - a;
    let v = c - a;
    let w = d - a;
    // Rows u, v, w so that `m * x = (u.x_, v.x_, w.x_)` for the equidistance
    // system x.u = |u|^2/2, etc.
    let m = Mat3::from_cols(
        Vec3::new(u.x, v.x, w.x),
        Vec3::new(u.y, v.y, w.y),
        Vec3::new(u.z, v.z, w.z),
    );
    if m.determinant().abs() <= DEGENERATE_EPS {
        // Coplanar boundary: the sphere is pinned by three concyclic points.
        return from_three(a, b, c);
    }
    let rhs = 0.5 * Vec3::new(u.length_squared(), v.length_squared(), w.length_squared());
    let rel = m.inverse() * rhs;
    BoundingSphere {
        center: a + rel,
        radius: rel.length(),
    }
}

/// Fallback for a degenerate (collinear) triple: the sphere on the two farthest
/// of the given points.
fn widest_pair(pts: &[Vec3]) -> BoundingSphere {
    let mut best = BoundingSphere::point(pts[0]);
    let mut best_d2 = 0.0_f32;
    for (i, &p) in pts.iter().enumerate() {
        for &q in &pts[i + 1..] {
            let d2 = (q - p).length_squared();
            if d2 >= best_d2 {
                best_d2 = d2;
                best = from_two(p, q);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn box_corners(h: Vec3) -> Vec<Vec3> {
        vec![
            Vec3::new(-h.x, -h.y, -h.z),
            Vec3::new(h.x, -h.y, -h.z),
            Vec3::new(-h.x, h.y, -h.z),
            Vec3::new(h.x, h.y, -h.z),
            Vec3::new(-h.x, -h.y, h.z),
            Vec3::new(h.x, -h.y, h.z),
            Vec3::new(-h.x, h.y, h.z),
            Vec3::new(h.x, h.y, h.z),
        ]
    }

    #[test]
    fn single_point_is_zero_radius() {
        let s = minimal_bounding_sphere(&[Vec3::new(1.0, 2.0, 3.0)]).unwrap();
        assert_eq!(s.center, Vec3::new(1.0, 2.0, 3.0));
        assert!(s.radius.abs() < 1e-6);
    }

    #[test]
    fn two_points_form_a_diameter() {
        let s = minimal_bounding_sphere(&[Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)])
            .unwrap();
        assert!(s.center.length() < 1e-6);
        assert!((s.radius - 1.0).abs() < 1e-6);
    }

    #[test]
    fn right_triangle_circumsphere_is_hypotenuse() {
        let s = minimal_bounding_sphere(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ])
        .unwrap();
        assert!((s.center - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5);
        assert!((s.radius - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn regular_tetrahedron_circumsphere() {
        // A tetrahedron whose four vertices lie on the unit sphere.
        let pts = vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let s = minimal_bounding_sphere(&pts).unwrap();
        assert!(s.center.length() < 1e-4, "center {:?}", s.center);
        let r = 3.0_f32.sqrt();
        assert!((s.radius - r).abs() < 1e-4, "radius {}", s.radius);
    }

    #[test]
    fn cube_sphere_touches_corners() {
        let s = minimal_bounding_sphere(&box_corners(Vec3::ONE)).unwrap();
        assert!(s.center.length() < 1e-4);
        assert!((s.radius - 3.0_f32.sqrt()).abs() < 1e-4);
        for &c in &box_corners(Vec3::ONE) {
            assert!(s.contains_point(c));
        }
    }

    #[test]
    fn octahedron_sphere_is_exact() {
        let r = 2.5;
        let c = Vec3::new(1.0, -2.0, 3.0);
        let pts = vec![
            c + Vec3::new(r, 0.0, 0.0),
            c - Vec3::new(r, 0.0, 0.0),
            c + Vec3::new(0.0, r, 0.0),
            c - Vec3::new(0.0, r, 0.0),
            c + Vec3::new(0.0, 0.0, r),
            c - Vec3::new(0.0, 0.0, r),
        ];
        let s = minimal_bounding_sphere(&pts).unwrap();
        assert!((s.center - c).length() < 1e-4);
        assert!((s.radius - r).abs() < 1e-4);
    }

    #[test]
    fn contains_all_points_of_rotated_box() {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.5, -0.4, 1.0);
        let t = Vec3::new(3.0, -1.0, 2.0);
        let pts: Vec<Vec3> = box_corners(Vec3::new(1.0, 2.0, 0.5))
            .into_iter()
            .map(|p| rot * p + t)
            .collect();
        let s = minimal_bounding_sphere(&pts).unwrap();
        for &p in &pts {
            assert!(s.contains_point(p), "point {p:?} escaped sphere");
        }
        // Rotation and translation preserve the circumradius of the box.
        assert!((s.center - t).length() < 1e-3);
    }

    #[test]
    fn sphere_is_tight_at_least_one_point_on_surface() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(5.0, 0.1, -0.2),
            Vec3::new(2.0, 3.0, 1.0),
            Vec3::new(-1.0, 2.0, 4.0),
            Vec3::new(1.0, -2.0, 2.0),
        ];
        let s = minimal_bounding_sphere(&pts).unwrap();
        let on_surface = pts
            .iter()
            .any(|&p| ((p - s.center).length() - s.radius).abs() < 1e-3);
        assert!(on_surface, "no point sits on the sphere surface");
        for &p in &pts {
            assert!(s.contains_point(p));
        }
    }

    #[test]
    fn fitting_is_deterministic() {
        let pts = vec![
            Vec3::new(0.3, 1.0, -2.0),
            Vec3::new(4.0, -1.0, 0.5),
            Vec3::new(-2.0, 3.0, 1.0),
            Vec3::new(1.0, 1.0, 5.0),
            Vec3::new(-1.0, -3.0, -1.0),
        ];
        let a = minimal_bounding_sphere(&pts).unwrap();
        let b = minimal_bounding_sphere(&pts).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_input_is_none() {
        assert!(minimal_bounding_sphere(&[]).is_none());
    }

    #[test]
    fn overlap_matches_distance() {
        let a = BoundingSphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let near = BoundingSphere {
            center: Vec3::new(1.5, 0.0, 0.0),
            radius: 1.0,
        };
        let far = BoundingSphere {
            center: Vec3::new(5.0, 0.0, 0.0),
            radius: 1.0,
        };
        assert!(a.overlaps(&near));
        assert!(!a.overlaps(&far));
    }
}
