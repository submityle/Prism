//! Ritter's approximate smallest-enclosing-sphere contract for the particle
//! subsystem's broadphase, culling, and bounding-volume authoring paths
//! (design §12, §13).
//!
//! Given a cloud of 3D particle positions, many pipeline stages want a single
//! *bounding sphere* — a center and radius that provably contain every point —
//! rather than the axis-aligned box that [`crate::particle::bounds`] and
//! [`crate::particle::bvh`] build. A sphere is rotation-invariant and needs
//! only four floats, so it is the natural coarse proxy for frustum tests,
//! collision culling, and level-of-detail switching. This module owns the
//! classic two-pass construction from Jack Ritter's *Graphics Gems* note:
//!
//! 1. **Seed pass.** Track the six axis-extremal points (`min`/`max` along
//!    `x`, `y`, `z`), pick the extremal *pair* with the largest separation,
//!    and seed the sphere on the segment between them — center at the midpoint,
//!    radius at half the separation.
//! 2. **Grow pass.** Walk every point once more; whenever a point falls outside
//!    the current sphere, expand the sphere the minimal amount that swallows
//!    that point *and* keeps the previous sphere fully enclosed (the new sphere
//!    is the smallest one tangent to both the old sphere and the stray point).
//!
//! The result is an *approximation*: it is guaranteed to contain every input
//! (the returned radius carries a small relative epsilon cushion so rounding
//! never lets a boundary point escape) and is typically within a few percent of
//! optimal, but it is not the exact minimum. This module deliberately does
//! **not** implement Welzl's exact minimum-enclosing-sphere algorithm; that is
//! a separate, heavier contract. It is likewise distinct from the *axis-aligned*
//! bounds/AABB modules ([`crate::particle::bounds`], and the local box in
//! [`crate::particle::bvh`]): those produce boxes, this produces a sphere.
//!
//! Everything is a zero-dependency contract. The vector math is hand-rolled in
//! this file and every computation uses only `+ - * /`, `f32::sqrt`,
//! `f32::abs`, `f32::min`, and `f32::max`. No transcendental function is ever
//! called and no exact `==` / `!=` is ever written on a production `f32` —
//! magnitudes are always compared against an explicit epsilon — so the `CPU`
//! reference here can agree bit for bit with a future `GPU` kernel.

/// Absolute epsilon used to guard comparisons and containment tests without
/// ever writing an exact `==` / `!=` on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// Relative cushion added to the final radius so floating-point rounding in the
/// grow pass can never leave a boundary point marginally outside the sphere.
/// It is *relative* (a multiply, not an add) so the construction stays exactly
/// linear under uniform scaling and exactly invariant under translation.
const RADIUS_REL_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its three components.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum `self + rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Component-wise negation `-self`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    /// Uniform scale `self * s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Euclidean dot product `self · rhs`.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean length `self · self` (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length `sqrt(self · self)`.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

/// A bounding sphere: a center point and a non-negative radius.
///
/// Holds `f32` fields, so it derives [`PartialEq`] but not [`Eq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// The sphere center in world space.
    pub center: Vec3,
    /// The sphere radius (always `>= 0`).
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere from a center and radius.
    #[must_use]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Returns `true` when `p` lies inside or on this sphere, using a small
    /// relative-plus-absolute epsilon so a boundary point is never rejected by
    /// rounding. Never writes an exact `==`.
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        let dist = p.sub(self.center).length();
        let tol = self.radius + CMP_EPS * (self.radius.abs() + 1.0);
        dist <= tol
    }
}

/// Computes an approximate smallest-enclosing sphere for `points` using
/// Ritter's two-pass algorithm.
///
/// Returns [`None`] for an empty slice. The returned sphere is guaranteed to
/// contain every input point (the radius carries a small relative cushion so
/// rounding cannot let a boundary point escape); it is an approximation, not
/// the exact minimum sphere.
///
/// The result is invariant under translation of all inputs and scales linearly
/// under uniform positive scaling about the origin, because every branch in the
/// algorithm depends only on distance *comparisons*, which those transforms
/// preserve.
#[must_use]
pub fn ritter_bounding_sphere(points: &[Vec3]) -> Option<Sphere> {
    let &first = points.first()?;

    // --- Seed pass: track the six axis-extremal points. ---
    let mut min_x = first;
    let mut max_x = first;
    let mut min_y = first;
    let mut max_y = first;
    let mut min_z = first;
    let mut max_z = first;
    for &p in points {
        if p.x < min_x.x {
            min_x = p;
        }
        if p.x > max_x.x {
            max_x = p;
        }
        if p.y < min_y.y {
            min_y = p;
        }
        if p.y > max_y.y {
            max_y = p;
        }
        if p.z < min_z.z {
            min_z = p;
        }
        if p.z > max_z.z {
            max_z = p;
        }
    }

    // Pick the axis-extremal pair with the largest squared separation.
    let span_x = max_x.sub(min_x).length_squared();
    let span_y = max_y.sub(min_y).length_squared();
    let span_z = max_z.sub(min_z).length_squared();

    let (lo, hi) = if span_x >= span_y && span_x >= span_z {
        (min_x, max_x)
    } else if span_y >= span_z {
        (min_y, max_y)
    } else {
        (min_z, max_z)
    };

    // Seed the sphere on the chosen segment.
    let mut center = lo.add(hi).scale(0.5);
    let mut radius = hi.sub(lo).length() * 0.5;

    // --- Grow pass: expand to swallow every stray point. ---
    for &p in points {
        let offset = p.sub(center);
        let dist = offset.length();
        if dist > radius {
            // The new sphere is the smallest one tangent to both the old sphere
            // and the stray point: its diameter spans from the far side of the
            // old sphere to the stray point.
            let new_radius = (radius + dist) * 0.5;
            // dist > radius >= 0 implies dist > 0, so this division is safe.
            let t = (new_radius - radius) / dist;
            center = center.add(offset.scale(t));
            radius = new_radius;
        }
    }

    // Relative cushion: keeps scaling exactly linear and translation invariant.
    radius += radius * RADIUS_REL_EPS;

    Some(Sphere::new(center, radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loose absolute tolerance for geometric assertions in tests.
    const TEST_EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS * (a.abs() + b.abs() + 1.0)
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// Returns the maximum pairwise distance across a point slice (brute force).
    fn max_pairwise(points: &[Vec3]) -> f32 {
        let mut best = 0.0_f32;
        for (i, &a) in points.iter().enumerate() {
            for &b in &points[i + 1..] {
                let d = b.sub(a).length();
                best = best.max(d);
            }
        }
        best
    }

    #[test]
    fn empty_input_returns_none() {
        let pts: [Vec3; 0] = [];
        assert!(ritter_bounding_sphere(&pts).is_none());
    }

    #[test]
    fn single_point_centers_on_it() {
        let p = Vec3::new(3.0, -2.0, 7.0);
        let s = ritter_bounding_sphere(&[p]).expect("non-empty");
        assert!(approx_vec(s.center, p));
        assert!(s.radius >= 0.0);
        assert!(s.radius <= TEST_EPS);
        assert!(s.contains(p));
    }

    #[test]
    fn duplicate_points_collapse_to_zero_radius() {
        let p = Vec3::new(1.0, 1.0, 1.0);
        let pts = [p, p, p, p];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        assert!(approx_vec(s.center, p));
        assert!(s.radius <= TEST_EPS);
    }

    #[test]
    fn two_points_center_at_midpoint() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        let s = ritter_bounding_sphere(&[a, b]).expect("non-empty");
        assert!(approx_vec(s.center, Vec3::new(1.0, 0.0, 0.0)));
        assert!(approx(s.radius, 2.0));
    }

    #[test]
    fn two_points_contains_both() {
        let a = Vec3::new(-5.0, 2.0, 1.0);
        let b = Vec3::new(4.0, -3.0, 6.0);
        let s = ritter_bounding_sphere(&[a, b]).expect("non-empty");
        assert!(s.contains(a));
        assert!(s.contains(b));
    }

    #[test]
    fn two_points_radius_is_half_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 6.0, 8.0);
        let s = ritter_bounding_sphere(&[a, b]).expect("non-empty");
        // distance = 10, radius ~ 5.
        assert!(approx(s.radius, 5.0));
    }

    #[test]
    fn collinear_points_all_contained() {
        let pts = [
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p), "point {p:?} escaped sphere {s:?}");
        }
    }

    #[test]
    fn collinear_center_near_extremes_midpoint() {
        let pts = [
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        assert!(approx_vec(s.center, Vec3::new(1.0, 0.0, 0.0)));
        assert!(approx(s.radius, 3.0));
    }

    #[test]
    fn collinear_diagonal_all_contained() {
        let pts = [
            Vec3::new(-3.0, -3.0, -3.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn cube_eight_vertices_all_contained() {
        let pts = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p), "vertex {p:?} escaped {s:?}");
        }
    }

    #[test]
    fn cube_center_near_origin() {
        let pts = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        // Ritter is order-dependent, so the center need not land on the origin;
        // it must stay inside the cube, and the radius must be at least the
        // half-space-diagonal lower bound (sqrt(3)) yet no larger than the full
        // space diagonal (2*sqrt(3)).
        assert!(s.center.x.abs() <= 1.0 + TEST_EPS);
        assert!(s.center.y.abs() <= 1.0 + TEST_EPS);
        assert!(s.center.z.abs() <= 1.0 + TEST_EPS);
        let half_diag = 3.0_f32.sqrt();
        assert!(s.radius + TEST_EPS >= half_diag);
        assert!(s.radius <= 2.0 * half_diag + TEST_EPS);
    }

    #[test]
    fn tetrahedron_all_contained() {
        let pts = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn radius_lower_bound_is_half_diameter() {
        // Any sphere containing all points must have radius >= half the max
        // pairwise distance, since the two farthest points both fit inside.
        let pts = [
            Vec3::new(-4.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, -3.0, 2.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        let half_diam = max_pairwise(&pts) * 0.5;
        assert!(
            s.radius + TEST_EPS >= half_diam,
            "radius {} below lower bound {half_diam}",
            s.radius
        );
    }

    #[test]
    fn radius_positive_for_distinct_points() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        assert!(s.radius > 0.0);
    }

    #[test]
    fn center_inside_axis_aligned_bounds() {
        let pts = [
            Vec3::new(-2.0, -1.0, 0.0),
            Vec3::new(5.0, 3.0, 4.0),
            Vec3::new(1.0, -4.0, 2.0),
            Vec3::new(0.0, 0.0, -3.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        let mut lo = pts[0];
        let mut hi = pts[0];
        for &p in &pts {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        assert!(s.center.x >= lo.x - TEST_EPS && s.center.x <= hi.x + TEST_EPS);
        assert!(s.center.y >= lo.y - TEST_EPS && s.center.y <= hi.y + TEST_EPS);
        assert!(s.center.z >= lo.z - TEST_EPS && s.center.z <= hi.z + TEST_EPS);
    }

    #[test]
    fn translation_invariance() {
        let pts = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 0.5, 2.0),
            Vec3::new(3.0, -2.0, -1.0),
            Vec3::new(0.0, 5.0, 4.0),
        ];
        let t = Vec3::new(10.0, -7.0, 3.5);
        let base = ritter_bounding_sphere(&pts).expect("non-empty");

        let mut shifted = pts;
        for slot in shifted.iter_mut() {
            *slot = slot.add(t);
        }
        let moved = ritter_bounding_sphere(&shifted).expect("non-empty");

        assert!(approx_vec(moved.center, base.center.add(t)));
        assert!(approx(moved.radius, base.radius));
    }

    #[test]
    fn uniform_scaling_is_linear() {
        let pts = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 0.5, 2.0),
            Vec3::new(3.0, -2.0, -1.0),
            Vec3::new(0.0, 5.0, 4.0),
        ];
        let scale = 3.0_f32;
        let base = ritter_bounding_sphere(&pts).expect("non-empty");

        let mut scaled = pts;
        for slot in scaled.iter_mut() {
            *slot = slot.scale(scale);
        }
        let grown = ritter_bounding_sphere(&scaled).expect("non-empty");

        assert!(approx_vec(grown.center, base.center.scale(scale)));
        assert!(approx(grown.radius, base.radius * scale));
    }

    #[test]
    fn determinism_same_input_same_output() {
        let pts = [
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(4.0, 5.0, -6.0),
            Vec3::new(-7.0, 8.0, 9.0),
        ];
        let a = ritter_bounding_sphere(&pts).expect("non-empty");
        let b = ritter_bounding_sphere(&pts).expect("non-empty");
        assert_eq!(a, b);
    }

    #[test]
    fn random_cloud_all_contained() {
        // Deterministic LCG-generated cloud; every point must be enclosed.
        let mut state = 0x1234_5678_u32;
        let mut pts = [Vec3::ZERO; 96];
        for slot in pts.iter_mut() {
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1_u32 << 24) as f32) * 20.0 - 10.0
            };
            *slot = Vec3::new(next(), next(), next());
        }
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p), "point {p:?} escaped sphere {s:?}");
        }
    }

    #[test]
    fn random_cloud_radius_lower_bound() {
        let mut state = 0x9E37_79B9_u32;
        let mut pts = [Vec3::ZERO; 64];
        for slot in pts.iter_mut() {
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1_u32 << 24) as f32) * 8.0 - 4.0
            };
            *slot = Vec3::new(next(), next(), next());
        }
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        let half_diam = max_pairwise(&pts) * 0.5;
        assert!(s.radius + TEST_EPS >= half_diam);
    }

    #[test]
    fn points_on_known_sphere_are_contained() {
        // Six axis crossings of a radius-5 sphere plus the origin.
        let r = 5.0_f32;
        let pts = [
            Vec3::new(r, 0.0, 0.0),
            Vec3::new(-r, 0.0, 0.0),
            Vec3::new(0.0, r, 0.0),
            Vec3::new(0.0, -r, 0.0),
            Vec3::new(0.0, 0.0, r),
            Vec3::new(0.0, 0.0, -r),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
        // The tight sphere is centered at the origin with radius r.
        assert!(s.center.length() <= 0.2);
        assert!(s.radius + TEST_EPS >= r);
    }

    #[test]
    fn large_coordinates_all_contained() {
        let pts = [
            Vec3::new(1.0e5, -2.0e5, 3.0e5),
            Vec3::new(-4.0e5, 5.0e5, -6.0e5),
            Vec3::new(2.0e5, 2.0e5, 2.0e5),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn negative_octant_all_contained() {
        let pts = [
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(-4.0, -5.0, -6.0),
            Vec3::new(-2.0, -1.0, -7.0),
            Vec3::new(-9.0, -3.0, -1.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn dominant_z_axis_pair_selected() {
        // The largest spread is along z; the seed must use that pair.
        let pts = [
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
        assert!(s.radius + TEST_EPS >= 10.0);
    }

    #[test]
    fn dominant_y_axis_pair_selected() {
        let pts = [
            Vec3::new(0.0, -8.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(0.0, 8.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
        assert!(s.radius + TEST_EPS >= 8.0);
    }

    #[test]
    fn interior_point_does_not_escape() {
        let pts = [
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        assert!(s.contains(Vec3::new(0.0, 0.0, 0.0)));
        assert!(s.contains(Vec3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn adding_enclosed_point_keeps_all_contained() {
        // A point well inside should not perturb containment of the others.
        let pts = [
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.1, 0.2, -0.1),
        ];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
    }

    #[test]
    fn contains_rejects_far_point() {
        let s = Sphere::new(Vec3::ZERO, 2.0);
        assert!(!s.contains(Vec3::new(10.0, 0.0, 0.0)));
        assert!(s.contains(Vec3::new(0.0, 0.0, 1.5)));
    }

    #[test]
    fn vec3_math_helpers() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert!(approx_vec(a.add(b), Vec3::new(5.0, 1.0, 3.5)));
        assert!(approx_vec(a.sub(b), Vec3::new(-3.0, 3.0, 2.5)));
        assert!(approx_vec(a.neg(), Vec3::new(-1.0, -2.0, -3.0)));
        assert!(approx_vec(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0)));
        assert!(approx(a.dot(b), 4.0 - 2.0 + 1.5));
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
        assert!(approx(Vec3::new(0.0, 0.0, 2.0).length_squared(), 4.0));
    }

    #[test]
    fn two_points_along_diagonal_radius() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 2.0, 1.0);
        let s = ritter_bounding_sphere(&[a, b]).expect("non-empty");
        // distance = 3, radius ~ 1.5.
        assert!(approx(s.radius, 1.5));
        assert!(approx_vec(s.center, Vec3::new(1.0, 1.0, 0.5)));
    }

    #[test]
    fn many_coincident_plus_one_outlier() {
        let base = Vec3::new(2.0, 2.0, 2.0);
        let pts = [base, base, base, Vec3::new(2.0, 2.0, 8.0)];
        let s = ritter_bounding_sphere(&pts).expect("non-empty");
        for &p in &pts {
            assert!(s.contains(p));
        }
        assert!(approx(s.radius, 3.0));
    }
}
