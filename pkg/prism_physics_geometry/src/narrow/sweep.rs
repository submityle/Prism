//! Swept-sphere (sphere cast) narrow-phase tests.
//!
//! A moving sphere of radius `r` whose centre travels along a ray is equivalent
//! to a ray tested against the Minkowski sum of the obstacle and a sphere of
//! radius `r`. For a triangle that rounded obstacle is the union of:
//!
//! * the triangle face pushed out by `r` along each side (two parallel slabs),
//! * the three edges grown into capsules of radius `r`, and
//! * the three vertices grown into spheres of radius `r`.
//!
//! The vertex spheres are already covered by the end caps of the edge capsules,
//! so the earliest time of impact is the minimum of the offset-face hit and the
//! three edge-capsule hits (plus an immediate `t = 0` when the sphere already
//! overlaps the triangle at the start).
//!
//! This is an engine-agnostic implementation of a publicly documented algorithm
//! (Ericson, *Real-Time Collision Detection*, intersecting a moving sphere
//! against a triangle) and contains no Unreal Engine source or derived code.

use glam::Vec3;

use crate::bounding::{Capsule, Ray};

use super::closest_point::{
    closest_point_on_segment, closest_point_on_triangle, closest_point_segment_triangle,
};
use super::ray_cast::ray_capsule;

/// The result of sweeping a sphere against a triangle.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SphereSweepHit {
    /// Parametric time of impact along the sphere-centre ray (a distance, since
    /// the ray direction is unit length).
    pub t: f32,
    /// Contact point on the triangle surface where the sphere first touches.
    pub point: Vec3,
    /// Unit surface normal at the contact, pointing from the triangle toward
    /// the sphere centre.
    pub normal: Vec3,
}

/// Sweeps a sphere of `radius` whose centre follows `ray` against the triangle
/// `(a, b, c)` and returns the earliest contact within `[0, ray.tmax]`.
///
/// Returns `None` when the swept sphere never touches the triangle. When the
/// sphere already overlaps the triangle at `t = 0`, the hit reports `t = 0`
/// with the contact at the closest point on the triangle.
pub fn sweep_sphere_triangle(
    ray: &Ray,
    radius: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<SphereSweepHit> {
    let radius = radius.max(0.0);
    let origin = ray.origin;

    // Already overlapping at the start: report an immediate contact.
    let cp0 = closest_point_on_triangle(origin, a, b, c);
    let gap0 = origin - cp0;
    if gap0.length_squared() <= radius * radius {
        let normal = fallback_normal(gap0, a, b, c);
        return Some(SphereSweepHit { t: 0.0, point: cp0, normal });
    }

    let mut best: Option<SphereSweepHit> = None;

    // Offset-face phase: the sphere centre reaches the plane offset by `radius`
    // on the side it started on.
    let face_normal = (b - a).cross(c - a).normalize_or_zero();
    if face_normal != Vec3::ZERO {
        let dist0 = face_normal.dot(origin - a);
        let side = if dist0 >= 0.0 { 1.0 } else { -1.0 };
        let denom = face_normal.dot(ray.dir);
        if denom.abs() > 1e-7 {
            let t = (side * radius - dist0) / denom;
            if t >= 0.0 && t <= ray.tmax {
                let contact_center = ray.at(t);
                let p = contact_center - face_normal * (side * radius);
                if point_in_triangle(p, a, b, c) {
                    best = Some(SphereSweepHit {
                        t,
                        point: p,
                        normal: face_normal * side,
                    });
                }
            }
        }
    }

    // Edge/vertex phase: each edge becomes a capsule of `radius`; the capsule
    // end caps also cover the vertex spheres.
    for (e0, e1) in [(a, b), (b, c), (c, a)] {
        let capsule = Capsule::new(e0, e1, radius);
        if let Some(t) = ray_capsule(ray, &capsule)
            && best.is_none_or(|h| t < h.t)
        {
            let contact_center = ray.at(t);
            let cp = closest_point_on_segment(contact_center, e0, e1);
            let normal = (contact_center - cp).normalize_or_zero();
            let normal = if normal == Vec3::ZERO { face_normal } else { normal };
            best = Some(SphereSweepHit { t, point: cp, normal });
        }
    }

    best
}

/// The result of sweeping a capsule against a triangle.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CapsuleSweepHit {
    /// Parametric time of impact along the motion direction (a distance when
    /// the direction is unit length).
    pub t: f32,
    /// Contact point on the triangle surface where the capsule first touches.
    pub point: Vec3,
    /// Unit surface normal at the contact, pointing from the triangle toward
    /// the capsule's core segment.
    pub normal: Vec3,
}

/// Sweeps a capsule (core segment `[seg_a, seg_b]` inflated by `radius`) moving
/// along `dir` by up to `tmax` against the triangle `(a, b, c)` and returns the
/// earliest contact.
///
/// The solver is a conservative-advancement loop: at each step it measures the
/// exact gap between the translated core segment and the triangle, subtracts
/// `radius`, and advances time by that clearance divided by the (unit) closing
/// speed, which never overshoots the true time of impact. It converges
/// monotonically toward the first touch and reports it once the gap falls to
/// `radius`. Returns `None` when the capsule never touches the triangle within
/// `[0, tmax]`, and reports `t = 0` when the capsule already overlaps at the
/// start. `dir` is normalised internally; a zero direction performs only the
/// static overlap test.
pub fn sweep_capsule_triangle(
    seg_a: Vec3,
    seg_b: Vec3,
    radius: f32,
    dir: Vec3,
    tmax: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<CapsuleSweepHit> {
    let radius = radius.max(0.0);
    let d = dir.normalize_or_zero();

    // No motion: only a static overlap can register.
    if d == Vec3::ZERO {
        let closest = closest_point_segment_triangle(seg_a, seg_b, a, b, c);
        if closest.distance_squared <= radius * radius {
            return Some(capsule_hit(closest, 0.0, a, b, c));
        }
        return None;
    }

    let mut t = 0.0f32;
    // Conservative advancement with a hard iteration cap as a safety net; the
    // clearance-based step guarantees monotone progress toward contact.
    for _ in 0..128 {
        let sa = seg_a + d * t;
        let sb = seg_b + d * t;
        let closest = closest_point_segment_triangle(sa, sb, a, b, c);
        let dist = closest.distance_squared.sqrt();
        let sep = dist - radius;
        if sep <= 1e-5 {
            return Some(capsule_hit(closest, t, a, b, c));
        }
        // Unit closing speed bounds the time to close `sep`, so advancing by
        // `sep` never skips past the true time of impact.
        t += sep;
        if t > tmax {
            return None;
        }
    }
    None
}

/// Builds a [`CapsuleSweepHit`] from a segment/triangle closest-point result,
/// choosing the push-out normal (surface -> axis) with a face-normal fallback.
fn capsule_hit(
    closest: super::closest_point::SegmentTriangleClosest,
    t: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> CapsuleSweepHit {
    let gap = closest.on_segment - closest.on_triangle;
    let normal = fallback_normal(gap, a, b, c);
    CapsuleSweepHit {
        t,
        point: closest.on_triangle,
        normal,
    }
}

/// Picks a unit contact normal from the start-overlap separation vector,
/// falling back to the triangle face normal when the sphere centre sits exactly
/// on the surface.
fn fallback_normal(gap: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let n = gap.normalize_or_zero();
    if n != Vec3::ZERO {
        return n;
    }
    (b - a).cross(c - a).normalize_or_zero()
}

/// Tests whether the coplanar point `p` lies inside triangle `(a, b, c)` using
/// barycentric coordinates with a small negative tolerance so boundary points
/// count as inside.
fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= f32::EPSILON {
        return false;
    }
    let inv = denom.recip();
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    let eps = -1e-5;
    u >= eps && v >= eps && w >= eps
}

#[cfg(test)]
mod tests {
    use super::{sweep_capsule_triangle, sweep_sphere_triangle};
    use crate::bounding::Ray;
    use glam::Vec3;

    /// Unit triangle in the z = 0 plane spanning the XY corner.
    fn tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn head_on_face_hit_stops_at_radius() {
        let (a, b, c) = tri();
        // Sphere of radius 0.5 falling along -Z toward the face centre.
        let ray = Ray::new(Vec3::new(0.0, -0.2, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = sweep_sphere_triangle(&ray, 0.5, a, b, c).expect("hit");
        // Centre travels from z = 3 to z = 0.5 (surface + radius): t = 2.5.
        assert!((hit.t - 2.5).abs() < 1e-4, "t = {}", hit.t);
        // Contact point sits on the face (z = 0).
        assert!(hit.point.z.abs() < 1e-4, "point z = {}", hit.point.z);
        // Normal points back up toward the sphere.
        assert!(hit.normal.z > 0.99, "normal = {:?}", hit.normal);
    }

    #[test]
    fn miss_passes_beside_triangle() {
        let (a, b, c) = tri();
        // Falls through x = 5, far outside the triangle, radius too small.
        let ray = Ray::new(Vec3::new(5.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(sweep_sphere_triangle(&ray, 0.5, a, b, c).is_none());
    }

    #[test]
    fn edge_hit_when_sliding_past_side() {
        let (a, b, c) = tri();
        // Centre travels along -X at y = -1 (the a-b edge line), z = 0, from
        // x = 5. It should strike the capsule around edge a-b.
        let ray = Ray::new(Vec3::new(5.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = sweep_sphere_triangle(&ray, 0.5, a, b, c).expect("edge hit");
        // Edge a-b runs to x = 1; sphere touches at x = 1.5 → t = 3.5.
        assert!((hit.t - 3.5).abs() < 1e-3, "t = {}", hit.t);
        assert!(hit.normal.length() > 0.99);
    }

    #[test]
    fn starts_overlapping_returns_zero() {
        let (a, b, c) = tri();
        // Centre right on the face: already overlapping for any positive radius.
        let ray = Ray::new(Vec3::new(0.0, -0.2, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = sweep_sphere_triangle(&ray, 0.5, a, b, c).expect("overlap");
        assert_eq!(hit.t, 0.0);
    }

    #[test]
    fn respects_tmax_short_of_contact() {
        let (a, b, c) = tri();
        let ray = Ray::with_tmax(
            Vec3::new(0.0, -0.2, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
        );
        // Contact needs t = 2.5 but tmax is 1.0.
        assert!(sweep_sphere_triangle(&ray, 0.5, a, b, c).is_none());
    }

    #[test]
    fn vertex_hit_via_capsule_cap() {
        let (a, b, c) = tri();
        // Aim at the apex vertex c = (0,1,0) from above-ahead in +Y travel.
        let ray = Ray::new(Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = sweep_sphere_triangle(&ray, 0.5, a, b, c).expect("vertex/edge hit");
        // Must stop before reaching the apex at y = 1 (touches at <= 0.5 short).
        assert!(hit.t > 0.0 && hit.t < 4.0, "t = {}", hit.t);
    }

    #[test]
    fn capsule_sweep_head_on_face_stops_at_radius() {
        let (a, b, c) = tri();
        // Horizontal capsule (axis along X) 3 units above the z = 0 face,
        // dropping along -Z. Radius 0.5 so the surface meets the face at t = 2.5.
        let hit = sweep_capsule_triangle(
            Vec3::new(-0.4, -0.2, 3.0),
            Vec3::new(0.4, -0.2, 3.0),
            0.5,
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            a,
            b,
            c,
        )
        .expect("capsule contact");
        assert!((hit.t - 2.5).abs() < 1e-3, "t = {}", hit.t);
        assert!(hit.normal.z > 0.99, "normal points toward axis (+z): {:?}", hit.normal);
        assert!(hit.point.z.abs() < 1e-3, "contact on the face: {:?}", hit.point);
    }

    #[test]
    fn capsule_sweep_misses_when_parallel_offset() {
        let (a, b, c) = tri();
        // Capsule travels in +X, staying 3 units above the face: never touches.
        let hit = sweep_capsule_triangle(
            Vec3::new(-0.4, -0.2, 3.0),
            Vec3::new(0.4, -0.2, 3.0),
            0.5,
            Vec3::new(1.0, 0.0, 0.0),
            100.0,
            a,
            b,
            c,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn capsule_sweep_respects_tmax() {
        let (a, b, c) = tri();
        // Needs t = 2.5 to touch, but tmax is 1.0.
        let hit = sweep_capsule_triangle(
            Vec3::new(-0.4, -0.2, 3.0),
            Vec3::new(0.4, -0.2, 3.0),
            0.5,
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
            a,
            b,
            c,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn capsule_sweep_already_overlapping_returns_zero() {
        let (a, b, c) = tri();
        // Axis crosses the face at the start.
        let hit = sweep_capsule_triangle(
            Vec3::new(0.0, -0.2, -0.1),
            Vec3::new(0.0, -0.2, 0.1),
            0.5,
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            a,
            b,
            c,
        )
        .expect("overlap");
        assert_eq!(hit.t, 0.0);
    }

    #[test]
    fn capsule_sweep_zero_direction_is_static_test() {
        let (a, b, c) = tri();
        // No motion and no overlap => miss.
        assert!(sweep_capsule_triangle(
            Vec3::new(-0.4, -0.2, 3.0),
            Vec3::new(0.4, -0.2, 3.0),
            0.5,
            Vec3::ZERO,
            100.0,
            a,
            b,
            c,
        )
        .is_none());
        // No motion but overlapping => t = 0.
        let hit = sweep_capsule_triangle(
            Vec3::new(-0.4, -0.2, 0.0),
            Vec3::new(0.4, -0.2, 0.0),
            0.5,
            Vec3::ZERO,
            100.0,
            a,
            b,
            c,
        )
        .expect("static overlap");
        assert_eq!(hit.t, 0.0);
    }
}
