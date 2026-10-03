//! Minimum-volume oriented bounding box (OBB) fitting.
//!
//! An OBB is the tightest arbitrarily-rotated box enclosing a shape. It is the
//! workhorse mid-phase bound in production engines (`PhysX` `PxBounds3` refined
//! to an OBB, Chaos `FAABB` + rotation, Jolt `OrientedBox`) because it keeps the
//! cheap slab-overlap test of an AABB while hugging rotated or elongated
//! geometry far more closely.
//!
//! [`fit_obb`] computes a minimum-*volume* OBB of a point cloud. It exploits the
//! structure of the optimum: the minimum-volume enclosing box of a convex
//! polytope always has one face flush with a face of the polytope (O'Rourke's
//! theorem). The fitter therefore:
//!
//! 1. builds the exact 3D convex hull (reusing [`convex_hull`]);
//! 2. treats every hull-face normal as a candidate box axis `w`;
//! 3. projects the hull onto the plane perpendicular to `w` and finds the
//!    minimum-*area* enclosing rectangle of that 2D projection with the
//!    rotating-calipers method (the 2D optimum has an edge flush with a hull
//!    edge, Toussaint 1983);
//! 4. multiplies the rectangle area by the `w`-extent to get the candidate box
//!    volume and keeps the smallest.
//!
//! The result is the minimum-volume box among all face-aligned orientations --
//! the standard practical optimum, and strictly tighter than the PCA boxes many
//! engines fall back to.
//!
//! # Determinism
//!
//! Hull faces, the 2D monotone-chain hull, and the calipers sweep are all walked
//! in a fixed index order; every arithmetic step is a dot/cross product or a
//! `sqrt`, with no transcendental calls and no data-dependent ordering. A box
//! fitted from the same points is therefore bit-for-bit reproducible, as
//! required for cross-run state hashing.
//!
//! # Provenance
//!
//! The face-normal enumeration (O'Rourke) and rotating-calipers minimum-area
//! rectangle (Toussaint) are textbook computational geometry. This module
//! contains **no Unreal Engine source or derived code**.

use glam::{Vec2, Vec3};

use super::hull::convex_hull;

/// Edge lengths below this (in world units) are treated as degenerate and
/// skipped when choosing a box axis.
const DEGENERATE_EPS: f32 = 1e-12;

/// An oriented bounding box: a centre, three orthonormal axes, and the box
/// half-extents measured along those axes.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Obb {
    /// Box centre in the same space as the input points.
    pub center: Vec3,
    /// Orthonormal box axes `[u, v, w]`; `axes[i]` pairs with
    /// `half_extents[i]`.
    pub axes: [Vec3; 3],
    /// Half the box size along each corresponding axis. Always non-negative.
    pub half_extents: Vec3,
}

impl Obb {
    /// The box volume (`8 * hx * hy * hz`).
    #[must_use]
    pub fn volume(&self) -> f32 {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    /// The eight corners of the box in world space.
    #[must_use]
    pub fn corners(&self) -> [Vec3; 8] {
        let [u, v, w] = self.axes;
        let hx = u * self.half_extents.x;
        let hy = v * self.half_extents.y;
        let hz = w * self.half_extents.z;
        [
            self.center - hx - hy - hz,
            self.center + hx - hy - hz,
            self.center - hx + hy - hz,
            self.center + hx + hy - hz,
            self.center - hx - hy + hz,
            self.center + hx - hy + hz,
            self.center - hx + hy + hz,
            self.center + hx + hy + hz,
        ]
    }

    /// Whether `point` lies inside the box (surface counts as inside), with a
    /// small tolerance to absorb round-off from the fitting projection.
    #[must_use]
    pub fn contains_point(&self, point: Vec3) -> bool {
        let d = point - self.center;
        let tol = 1e-4 + 1e-4 * self.half_extents.max_element();
        d.dot(self.axes[0]).abs() <= self.half_extents.x + tol
            && d.dot(self.axes[1]).abs() <= self.half_extents.y + tol
            && d.dot(self.axes[2]).abs() <= self.half_extents.z + tol
    }

    /// The world-space axis-aligned bounding box of this OBB, as `(min, max)`.
    #[must_use]
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let r = self.axes[0].abs() * self.half_extents.x
            + self.axes[1].abs() * self.half_extents.y
            + self.axes[2].abs() * self.half_extents.z;
        (self.center - r, self.center + r)
    }
}

/// Fits the minimum-volume face-aligned OBB of a point cloud.
///
/// Returns `None` when the points do not span a 3D volume (fewer than four
/// affinely independent points), i.e. when [`convex_hull`] cannot build a solid.
#[must_use]
pub fn fit_obb(points: &[Vec3]) -> Option<Obb> {
    let (verts, tris) = convex_hull(points)?;

    let mut best: Option<(f32, Obb)> = None;
    for tri in &tris {
        let a = verts[tri[0] as usize];
        let b = verts[tri[1] as usize];
        let c = verts[tri[2] as usize];
        let normal = (b - a).cross(c - a);
        let nlen = normal.length();
        if nlen <= DEGENERATE_EPS {
            continue;
        }
        let w = normal / nlen;
        let (u0, v0) = plane_basis(w);

        let mut projected = Vec::with_capacity(verts.len());
        let mut min_w = f32::INFINITY;
        let mut max_w = f32::NEG_INFINITY;
        for &p in &verts {
            projected.push(Vec2::new(p.dot(u0), p.dot(v0)));
            let dw = p.dot(w);
            min_w = min_w.min(dw);
            max_w = max_w.max(dw);
        }

        let hull2 = convex_hull_2d(&projected);
        let Some(rect) = min_area_rect(&hull2) else {
            continue;
        };

        let height = max_w - min_w;
        let volume = rect.area * height;

        let n2 = Vec2::new(-rect.axis.y, rect.axis.x);
        let axis_e = u0 * rect.axis.x + v0 * rect.axis.y;
        let axis_n = u0 * n2.x + v0 * n2.y;
        let center = axis_e * rect.center_e + axis_n * rect.center_n + w * (0.5 * (min_w + max_w));

        let obb = Obb {
            center,
            axes: [axis_e.normalize(), axis_n.normalize(), w],
            half_extents: Vec3::new(rect.half_e, rect.half_n, 0.5 * height),
        };

        if best.as_ref().is_none_or(|(bv, _)| volume < *bv) {
            best = Some((volume, obb));
        }
    }

    best.map(|(_, obb)| obb)
}

/// An orthonormal basis `(u, v)` spanning the plane perpendicular to unit `w`,
/// chosen deterministically from the least-aligned cardinal axis.
fn plane_basis(w: Vec3) -> (Vec3, Vec3) {
    let ax = w.x.abs();
    let ay = w.y.abs();
    let az = w.z.abs();
    let seed = if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let u = (seed - w * seed.dot(w)).normalize();
    let v = w.cross(u);
    (u, v)
}

/// The minimum-area rectangle found by [`min_area_rect`], expressed in the 2D
/// projection basis.
struct Rect2 {
    /// Unit direction of the rectangle's first (`e`) edge, in `(u0, v0)` coords.
    axis: Vec2,
    /// Centre coordinate along `axis`.
    center_e: f32,
    /// Half-width along `axis`.
    half_e: f32,
    /// Centre coordinate along the perpendicular of `axis`.
    center_n: f32,
    /// Half-width along the perpendicular of `axis`.
    half_n: f32,
    /// Rectangle area (`4 * half_e * half_n`).
    area: f32,
}

/// Andrew's monotone-chain convex hull of a 2D point set, returned
/// counter-clockwise with collinear points removed.
fn convex_hull_2d(points: &[Vec2]) -> Vec<Vec2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: Vec2, a: Vec2, b: Vec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    let mut lower: Vec<Vec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<Vec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The minimum-area enclosing rectangle of a convex polygon (CCW, collinear-free
/// as produced by [`convex_hull_2d`]) via rotating calipers.
fn min_area_rect(hull: &[Vec2]) -> Option<Rect2> {
    if hull.len() < 2 {
        return None;
    }
    let n = hull.len();
    let mut best: Option<Rect2> = None;

    for (i, &a) in hull.iter().enumerate() {
        let b = hull[(i + 1) % n];
        let edge = b - a;
        let len = edge.length();
        if len <= DEGENERATE_EPS {
            continue;
        }
        let e = edge / len;
        let perp = Vec2::new(-e.y, e.x);

        let mut min_e = f32::INFINITY;
        let mut max_e = f32::NEG_INFINITY;
        let mut min_n = f32::INFINITY;
        let mut max_n = f32::NEG_INFINITY;
        for &p in hull {
            let de = p.dot(e);
            let dn = p.dot(perp);
            min_e = min_e.min(de);
            max_e = max_e.max(de);
            min_n = min_n.min(dn);
            max_n = max_n.max(dn);
        }

        let width = max_e - min_e;
        let height = max_n - min_n;
        let area = width * height;
        if best.as_ref().is_none_or(|r| area < r.area) {
            best = Some(Rect2 {
                axis: e,
                center_e: 0.5 * (min_e + max_e),
                half_e: 0.5 * width,
                center_n: 0.5 * (min_n + max_n),
                half_n: 0.5 * height,
                area,
            });
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// The eight corners of an axis-aligned box with the given half-extents.
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

    fn sorted_extents(obb: &Obb) -> [f32; 3] {
        let mut e = [obb.half_extents.x, obb.half_extents.y, obb.half_extents.z];
        e.sort_by(f32::total_cmp);
        e
    }

    #[test]
    fn fits_unit_cube_tightly() {
        let obb = fit_obb(&box_corners(Vec3::splat(1.0))).expect("cube has volume");
        assert!(
            (obb.volume() - 8.0).abs() < 1e-3,
            "volume = {}",
            obb.volume()
        );
        assert!(obb.center.length() < 1e-4);
    }

    #[test]
    fn recovers_rotated_translated_box() {
        let half = Vec3::new(1.0, 2.0, 3.0);
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 1.1);
        let translation = Vec3::new(5.0, -4.0, 2.0);
        let pts: Vec<Vec3> = box_corners(half)
            .into_iter()
            .map(|p| rot * p + translation)
            .collect();

        let obb = fit_obb(&pts).expect("box has volume");

        let expected_volume = 8.0 * half.x * half.y * half.z;
        assert!(
            (obb.volume() - expected_volume).abs() / expected_volume < 1e-2,
            "volume = {} expected {}",
            obb.volume(),
            expected_volume
        );
        assert!(
            (obb.center - translation).length() < 1e-2,
            "center = {:?}",
            obb.center
        );

        let got = sorted_extents(&obb);
        let want = [1.0_f32, 2.0, 3.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-2, "extent {g} vs {w}");
        }
    }

    #[test]
    fn axes_are_orthonormal() {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.5, -0.6);
        let pts: Vec<Vec3> = box_corners(Vec3::new(1.0, 1.5, 2.5))
            .into_iter()
            .map(|p| rot * p)
            .collect();
        let obb = fit_obb(&pts).expect("box has volume");
        let [u, v, w] = obb.axes;
        assert!((u.length() - 1.0).abs() < 1e-5);
        assert!((v.length() - 1.0).abs() < 1e-5);
        assert!((w.length() - 1.0).abs() < 1e-5);
        assert!(u.dot(v).abs() < 1e-4);
        assert!(u.dot(w).abs() < 1e-4);
        assert!(v.dot(w).abs() < 1e-4);
    }

    #[test]
    fn contains_all_input_points() {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 1.2, -0.3, 0.8);
        let pts: Vec<Vec3> = box_corners(Vec3::new(0.5, 2.0, 1.0))
            .into_iter()
            .map(|p| rot * p + Vec3::new(-3.0, 1.0, 4.0))
            .collect();
        let obb = fit_obb(&pts).expect("box has volume");
        for &p in &pts {
            assert!(obb.contains_point(p), "point {p:?} escaped the box");
        }
    }

    #[test]
    fn tetrahedron_box_contains_hull() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
        ];
        let obb = fit_obb(&pts).expect("tetra has volume");
        for &p in &pts {
            assert!(obb.contains_point(p));
        }
        assert!(obb.volume() > 0.0);
    }

    #[test]
    fn obb_is_no_larger_than_axis_aligned_box() {
        // A box rotated 45 deg about Z: its AABB is larger than the true OBB,
        // which the fitter must recover.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let half = Vec3::new(2.0, 1.0, 1.0);
        let pts: Vec<Vec3> = box_corners(half).into_iter().map(|p| rot * p).collect();

        let obb = fit_obb(&pts).expect("box has volume");
        let (min, max) = {
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for &p in &pts {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            (lo, hi)
        };
        let aabb_vol = (max - min).x * (max - min).y * (max - min).z;
        assert!(obb.volume() <= aabb_vol + 1e-4);
        assert!((obb.volume() - 8.0 * half.x * half.y * half.z).abs() < 1e-2);
    }

    #[test]
    fn degenerate_planar_points_return_none() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        assert!(fit_obb(&pts).is_none());
    }

    #[test]
    fn fitting_is_deterministic() {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.2, 0.9, -0.4);
        let pts: Vec<Vec3> = box_corners(Vec3::new(1.3, 0.7, 2.1))
            .into_iter()
            .map(|p| rot * p + Vec3::new(1.0, 2.0, 3.0))
            .collect();
        let first = fit_obb(&pts).expect("box has volume");
        let second = fit_obb(&pts).expect("box has volume");
        assert_eq!(first, second);
    }

    #[test]
    fn aabb_bounds_every_corner() {
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.6, -0.6, 0.6);
        let pts: Vec<Vec3> = box_corners(Vec3::new(1.0, 2.0, 0.5))
            .into_iter()
            .map(|p| rot * p)
            .collect();
        let obb = fit_obb(&pts).expect("box has volume");
        let (lo, hi) = obb.aabb();
        for c in obb.corners() {
            assert!(c.x >= lo.x - 1e-4 && c.x <= hi.x + 1e-4);
            assert!(c.y >= lo.y - 1e-4 && c.y <= hi.y + 1e-4);
            assert!(c.z >= lo.z - 1e-4 && c.z <= hi.z + 1e-4);
        }
    }
}
