//! Minimal-ish bounding capsule fit for a point cloud.
//!
//! A capsule (a line segment swept by a sphere) is one of the cheapest and most
//! widely used collision primitives: character controllers, limbs, pills, and
//! thin elongated props are all approximated far better by a capsule than by a
//! sphere or an axis-aligned box. `PhysX`, Jolt and UE all expose a capsule
//! shape and need a way to cook one from an arbitrary mesh or point cloud.
//!
//! This module fits a capsule whose central segment lies along the dominant
//! elongation axis of the cloud and whose radius just encloses every point.
//! The axis is the principal direction of the point covariance (the eigenvector
//! with the largest variance), obtained from the existing symmetric eigensolver
//! [`principal_axes`](crate::collider::principal_axes). The radius is the
//! largest perpendicular distance of any point to that axis, and the segment
//! endpoints are pulled in so the spherical end caps still cover the extreme
//! points exactly rather than leaving an over-long cylinder.
//!
//! The fit is not provably the globally minimal enclosing capsule (that problem
//! is non-convex), but the covariance axis plus exact cap accounting gives a
//! tight, deterministic, and strictly enclosing capsule in a single pass, which
//! is what cooking pipelines want. It is pure point-cloud geometry with no
//! coupling to the collision pipeline, and nothing here is derived from Unreal
//! Engine source.

use glam::{Mat3, Vec3};

use crate::collider::inertia::principal_axes;

/// Minimum number of points required to attempt a capsule fit.
const MIN_POINTS: usize = 4;

/// Relative slack used by [`BoundingCapsule::contains`] when no explicit
/// epsilon is supplied through the public API.
const CONTAIN_EPS: f32 = 1e-5;

/// A capsule given by the two centres of its spherical end caps and a radius.
///
/// The capsule is the set of points within `radius` of the line segment
/// `[center_a, center_b]`. When `center_a == center_b` the capsule degenerates
/// to a sphere of the same radius.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingCapsule {
    /// Centre of the first spherical end cap.
    pub center_a: Vec3,
    /// Centre of the second spherical end cap.
    pub center_b: Vec3,
    /// Capsule radius. Non-negative.
    pub radius: f32,
}

impl BoundingCapsule {
    /// A zero-length capsule (a sphere) of `radius` centred at `center`.
    #[must_use]
    pub fn sphere(center: Vec3, radius: f32) -> BoundingCapsule {
        BoundingCapsule {
            center_a: center,
            center_b: center,
            radius,
        }
    }

    /// The unit axis direction from `center_a` to `center_b`.
    ///
    /// Returns [`Vec3::ZERO`] when the capsule has degenerated to a sphere.
    #[must_use]
    pub fn axis(&self) -> Vec3 {
        (self.center_b - self.center_a).normalize_or_zero()
    }

    /// The segment length (distance between the two cap centres).
    ///
    /// This is the cylindrical height, excluding the two hemispherical caps.
    #[must_use]
    pub fn height(&self) -> f32 {
        self.center_a.distance(self.center_b)
    }

    /// The capsule volume: a cylinder (`pi r^2 h`) plus a full sphere
    /// (`4/3 pi r^3`) for the two hemispherical caps.
    #[must_use]
    pub fn volume(&self) -> f32 {
        let r = self.radius;
        let h = self.height();
        core::f32::consts::PI * r * r * h + (4.0 / 3.0) * core::f32::consts::PI * r * r * r
    }

    /// Whether `point` lies inside the capsule, allowing a small relative slack.
    #[must_use]
    pub fn contains(&self, point: Vec3, eps: f32) -> bool {
        let slack = self.radius * CONTAIN_EPS + eps;
        let r = self.radius + slack;
        self.distance_sq_to_segment(point) <= f64::from(r) * f64::from(r)
    }

    /// Squared distance from `point` to the central segment, computed in `f64`
    /// for determinism across platforms.
    fn distance_sq_to_segment(&self, point: Vec3) -> f64 {
        let ax = f64::from(self.center_a.x);
        let ay = f64::from(self.center_a.y);
        let az = f64::from(self.center_a.z);
        let bx = f64::from(self.center_b.x);
        let by = f64::from(self.center_b.y);
        let bz = f64::from(self.center_b.z);
        let px = f64::from(point.x);
        let py = f64::from(point.y);
        let pz = f64::from(point.z);

        let dx = bx - ax;
        let dy = by - ay;
        let dz = bz - az;
        let seg_len_sq = dx * dx + dy * dy + dz * dz;

        let t = if seg_len_sq <= 0.0 {
            0.0
        } else {
            let dot = (px - ax) * dx + (py - ay) * dy + (pz - az) * dz;
            (dot / seg_len_sq).clamp(0.0, 1.0)
        };

        let cx = ax + t * dx;
        let cy = ay + t * dy;
        let cz = az + t * dz;
        let ex = px - cx;
        let ey = py - cy;
        let ez = pz - cz;
        ex * ex + ey * ey + ez * ez
    }
}

/// Fits a bounding capsule to `points`.
///
/// Returns `None` when fewer than four points are supplied or when the point
/// cloud is fully degenerate (all points coincident).
///
/// The returned capsule strictly encloses every input point (up to floating
/// point round-off) and is aligned with the dominant elongation axis of the
/// cloud. A near-spherical cloud yields a near-zero [`BoundingCapsule::height`].
#[must_use]
pub fn fit_bounding_capsule(points: &[Vec3]) -> Option<BoundingCapsule> {
    if points.len() < MIN_POINTS {
        return None;
    }

    // Centroid accumulated in f64 to avoid cancellation on large coordinates.
    let n = points.len() as f64;
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in points {
        cx += f64::from(p.x);
        cy += f64::from(p.y);
        cz += f64::from(p.z);
    }
    cx /= n;
    cy /= n;
    cz /= n;
    let centroid = Vec3::new(cx as f32, cy as f32, cz as f32);

    // Symmetric covariance (second moments about the centroid), f64 accumulate.
    let mut m00 = 0.0_f64;
    let mut m11 = 0.0_f64;
    let mut m22 = 0.0_f64;
    let mut m01 = 0.0_f64;
    let mut m02 = 0.0_f64;
    let mut m12 = 0.0_f64;
    for p in points {
        let x = f64::from(p.x) - cx;
        let y = f64::from(p.y) - cy;
        let z = f64::from(p.z) - cz;
        m00 += x * x;
        m11 += y * y;
        m22 += z * z;
        m01 += x * y;
        m02 += x * z;
        m12 += y * z;
    }
    m00 /= n;
    m11 /= n;
    m22 /= n;
    m01 /= n;
    m02 /= n;
    m12 /= n;

    let cov = Mat3::from_cols(
        Vec3::new(m00 as f32, m01 as f32, m02 as f32),
        Vec3::new(m01 as f32, m11 as f32, m12 as f32),
        Vec3::new(m02 as f32, m12 as f32, m22 as f32),
    );

    // `principal_axes` sorts moments ascending, so the largest-variance
    // eigenvector (the elongation direction) is the third column.
    let decomposition = principal_axes(cov);
    if decomposition.moments.z <= 0.0 {
        // Largest variance is zero: the cloud has no spatial extent in any
        // direction (all points coincident), so no meaningful capsule exists.
        return None;
    }
    let axis = decomposition.axes.z_axis.normalize_or_zero();
    if axis.length_squared() <= 0.0 {
        // Degenerate eigenvector guard; should not trigger once variance > 0.
        return None;
    }

    let axis_x = f64::from(axis.x);
    let axis_y = f64::from(axis.y);
    let axis_z = f64::from(axis.z);

    // Radius = largest perpendicular distance of any point to the axis line
    // through the centroid. Also cache each point's axial coordinate and
    // perpendicular distance for the cap accounting below.
    let mut radius_sq = 0.0_f64;
    for p in points {
        let rx = f64::from(p.x) - cx;
        let ry = f64::from(p.y) - cy;
        let rz = f64::from(p.z) - cz;
        let t = rx * axis_x + ry * axis_y + rz * axis_z;
        let perp_x = rx - t * axis_x;
        let perp_y = ry - t * axis_y;
        let perp_z = rz - t * axis_z;
        let d_sq = perp_x * perp_x + perp_y * perp_y + perp_z * perp_z;
        if d_sq > radius_sq {
            radius_sq = d_sq;
        }
    }
    let radius = radius_sq.sqrt();

    // Pull the segment endpoints in so the spherical caps still cover the
    // extreme points: a point at axial coord `t` with perpendicular distance
    // `d` is covered by an end cap centred at `t_cap` iff
    // `(t - t_cap)^2 + d^2 <= r^2`, i.e. the cap must reach `t -/+ sqrt(r^2 - d^2)`.
    let mut t_max_seg = f64::NEG_INFINITY;
    let mut t_min_seg = f64::INFINITY;
    for p in points {
        let rx = f64::from(p.x) - cx;
        let ry = f64::from(p.y) - cy;
        let rz = f64::from(p.z) - cz;
        let t = rx * axis_x + ry * axis_y + rz * axis_z;
        let perp_x = rx - t * axis_x;
        let perp_y = ry - t * axis_y;
        let perp_z = rz - t * axis_z;
        let d_sq = perp_x * perp_x + perp_y * perp_y + perp_z * perp_z;
        let cap = (radius_sq - d_sq).max(0.0).sqrt();
        let lo = t - cap;
        let hi = t + cap;
        if lo > t_max_seg {
            t_max_seg = lo;
        }
        if hi < t_min_seg {
            t_min_seg = hi;
        }
    }

    // When the cap requirement crosses over, every point is already covered by
    // a single sphere: collapse the segment to its midpoint.
    if t_max_seg < t_min_seg {
        let mid = 0.5 * (t_max_seg + t_min_seg);
        t_max_seg = mid;
        t_min_seg = mid;
    }

    let a = centroid + axis * (t_min_seg as f32);
    let b = centroid + axis * (t_max_seg as f32);

    Some(BoundingCapsule {
        center_a: a,
        center_b: b,
        radius: radius as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight corners of an axis-aligned box with the given half extents.
    fn box_corners(half: Vec3) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    pts.push(Vec3::new(sx * half.x, sy * half.y, sz * half.z));
                }
            }
        }
        pts
    }

    #[test]
    fn rejects_too_few_points() {
        assert!(fit_bounding_capsule(&[]).is_none());
        assert!(fit_bounding_capsule(&[Vec3::ZERO, Vec3::X, Vec3::Y]).is_none());
    }

    #[test]
    fn rejects_fully_coincident_cloud() {
        let p = Vec3::new(2.0, -1.0, 3.0);
        assert!(fit_bounding_capsule(&[p, p, p, p, p]).is_none());
    }

    #[test]
    fn axis_follows_elongation_along_x() {
        // A thin cylinder of samples stretched along x.
        let mut pts = Vec::new();
        for i in 0..40 {
            let x = -5.0 + (i as f32) * (10.0 / 39.0);
            let theta = (i as f32) * 0.9;
            let y = 0.3 * f64::from(theta).cos() as f32;
            let z = 0.3 * f64::from(theta).sin() as f32;
            pts.push(Vec3::new(x, y, z));
        }
        let cap = fit_bounding_capsule(&pts).unwrap();
        let axis = cap.axis();
        // Axis should be (anti)parallel to x.
        assert!(axis.x.abs() > 0.99, "axis = {axis:?}");
        assert!(
            axis.y.abs() < 0.05 && axis.z.abs() < 0.05,
            "axis = {axis:?}"
        );
        // The radius should match the tube radius closely.
        assert!((cap.radius - 0.3).abs() < 0.05, "radius = {}", cap.radius);
    }

    #[test]
    fn encloses_every_input_point() {
        // A slanted elongated cloud so the axis is not grid aligned.
        let mut pts = Vec::new();
        let dir = Vec3::new(1.0, 2.0, 0.5).normalize();
        for i in 0..60 {
            let t = -4.0 + (i as f32) * (8.0 / 59.0);
            let base = dir * t;
            let a = (i as f32) * 1.3;
            let b = (i as f32) * 0.7;
            let jitter = Vec3::new(
                0.25 * f64::from(a).cos() as f32,
                0.15 * f64::from(b).sin() as f32,
                0.25 * f64::from(a).sin() as f32,
            );
            pts.push(base + jitter);
        }
        let cap = fit_bounding_capsule(&pts).unwrap();
        for p in &pts {
            assert!(
                cap.contains(*p, 1e-4),
                "point {p:?} not enclosed by {cap:?}"
            );
        }
    }

    #[test]
    fn near_spherical_cloud_collapses_to_short_segment() {
        // Points roughly on a sphere: no dominant elongation.
        let mut pts = Vec::new();
        for i in 0..50 {
            let phi = (i as f32) * 2.399_963; // golden-angle-ish spread
            let z = -1.0 + (i as f32) * (2.0 / 49.0);
            let r = (1.0 - z * z).max(0.0).sqrt();
            pts.push(Vec3::new(
                r * f64::from(phi).cos() as f32,
                r * f64::from(phi).sin() as f32,
                z,
            ));
        }
        let cap = fit_bounding_capsule(&pts).unwrap();
        assert!(cap.height() < 0.3, "height = {}", cap.height());
        assert!((cap.radius - 1.0).abs() < 0.1, "radius = {}", cap.radius);
        for p in &pts {
            assert!(cap.contains(*p, 1e-4), "point {p:?} not enclosed");
        }
    }

    #[test]
    fn elongated_box_axis_follows_long_dimension() {
        let cap = fit_bounding_capsule(&box_corners(Vec3::new(5.0, 1.0, 1.0))).unwrap();
        let axis = cap.axis();
        assert!(axis.x.abs() > 0.9, "axis = {axis:?}");
        assert!(cap.height() > 0.0);
        for p in &box_corners(Vec3::new(5.0, 1.0, 1.0)) {
            assert!(cap.contains(*p, 1e-4), "corner {p:?} not enclosed");
        }
    }

    #[test]
    fn is_deterministic() {
        let pts = box_corners(Vec3::new(3.0, 1.0, 0.5));
        let a = fit_bounding_capsule(&pts).unwrap();
        let b = fit_bounding_capsule(&pts).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn volume_is_positive_for_real_fit() {
        let cap = fit_bounding_capsule(&box_corners(Vec3::new(4.0, 1.0, 1.0))).unwrap();
        assert!(cap.volume() > 0.0, "volume = {}", cap.volume());
    }
}
