//! Sphere-versus-axis-aligned-bounding-box (`AABB`) *proximity* query for the
//! particle subsystem (design §10, §14).
//!
//! This module answers one narrow, purely geometric question: given a sphere
//! (a particle with a radius) and an axis-aligned box, what is the closest
//! point on the box, do the two overlap, and — when they do — how deep is the
//! penetration and along which separation normal should the sphere be pushed
//! out? It is the canonical real-time-rendering *sphere-`AABB`* test (clamp the
//! centre to the box for the closest point, compare squared distance to `r²`),
//! re-derived here rather than copied from any engine.
//!
//! # Relationship to the sibling modules (strict boundary)
//!
//! Several files in this subsystem touch boxes, distances, and contacts; this
//! one is deliberately disjoint:
//!
//! * [`crate::particle::collision`] owns the **collision response** layer:
//!   push-out, restitution bounce, and Coulomb friction. It *consumes* a
//!   proximity result like the one produced here; this module never applies a
//!   response, integrates velocity, or resolves a contact.
//! * [`crate::particle::sdf`] samples a **baked scalar `SDF`/`VDB` grid** and
//!   [`crate::particle::capsule_sdf`] evaluates **analytic closed-form distance
//!   fields**. Both answer "distance to a surface" as a signed field; this
//!   module answers the discrete *sphere-vs-box* proximity/intersection query
//!   directly, returning the closest point, a boolean overlap, a penetration
//!   depth, and a separation normal rather than a raw signed-distance field.
//! * [`crate::particle::bounds`] reduces a live particle pool down to one tight
//!   box, and [`crate::particle::aabb_transform`] transforms a box under an
//!   affine matrix and provides box-vs-box set algebra. Neither tests a box
//!   against a sphere; this module does only that.
//!
//! In short: this file computes sphere-`AABB` proximity, intersection, and
//! penetration, and nothing else.
//!
//! # Determinism
//!
//! Everything is a zero-dependency contract with hand-rolled vector math. The
//! only floating-point primitive beyond ordinary `+ - * /` is `f32::sqrt` (for
//! the penetration depth and closest-point distance), plus `f32::abs`,
//! `f32::min`, `f32::max`, and `f32::clamp`. No transcendental function is ever
//! called, so the `CPU` reference here is deterministic and a future `GPU`
//! kernel evaluating the same query produces matching results. Floating-point
//! `==` / `!=` are never used; magnitudes are compared against [`CMP_EPS`].

/// Absolute tolerance used to guard divisions and to compare magnitudes without
/// ever writing an exact `==` / `!=` on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

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

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three lanes set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
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
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Component-wise product `self * rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named mul for call-site uniformity, not operator traits."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    /// Additive inverse `-self`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named neg for call-site uniformity, not operator traits."
    )]
    pub fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length (the only `sqrt` on a vector in this module).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Component-wise minimum.
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    /// Component-wise maximum.
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    /// Component-wise clamp into the inclusive box `[lo, hi]`.
    #[must_use]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new(
            self.x.clamp(lo.x, hi.x),
            self.y.clamp(lo.y, hi.y),
            self.z.clamp(lo.z, hi.z),
        )
    }

    /// Component-wise absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

/// A sphere: a centre and a radius. A particle is modelled as a sphere of its
/// collision radius for this proximity query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// World-space centre of the sphere.
    pub center: Vec3,
    /// Radius of the sphere. A negative radius is treated as zero (a point).
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere from a centre and radius.
    #[must_use]
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

/// A self-contained axis-aligned bounding box (`AABB`).
///
/// Callers are expected to keep `min <= max` on every axis; the query clamps
/// against the box as given, so a well-formed box yields a well-formed result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Per-axis minimum corner.
    pub min: Vec3,
    /// Per-axis maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box from its two corners.
    #[must_use]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Builds a box from a centre and a (non-negative) half-extent.
    #[must_use]
    pub fn from_center_half_extent(center: Vec3, half_extent: Vec3) -> Self {
        let h = half_extent.abs();
        Self::new(center.sub(h), center.add(h))
    }

    /// The geometric centre of the box.
    #[must_use]
    pub fn center(self) -> Vec3 {
        self.min.add(self.max).scale(0.5)
    }

    /// The half-extent (half the size) of the box on each axis.
    #[must_use]
    pub fn half_extent(self) -> Vec3 {
        self.max.sub(self.min).scale(0.5)
    }

    /// Whether the (inclusive) box contains the point `p`.
    #[must_use]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// For a point at or inside the box, returns the distance to the nearest
    /// face and the outward unit axis normal of that face.
    ///
    /// Ties (a point equidistant to several faces, e.g. the exact box centre of
    /// a cube) resolve deterministically to the first candidate in axis order
    /// `(-x, +x, -y, +y, -z, +z)`, so the result is reproducible.
    #[must_use]
    fn nearest_face(self, p: Vec3) -> (f32, Vec3) {
        let candidates = [
            (p.x - self.min.x, Vec3::new(-1.0, 0.0, 0.0)),
            (self.max.x - p.x, Vec3::new(1.0, 0.0, 0.0)),
            (p.y - self.min.y, Vec3::new(0.0, -1.0, 0.0)),
            (self.max.y - p.y, Vec3::new(0.0, 1.0, 0.0)),
            (p.z - self.min.z, Vec3::new(0.0, 0.0, -1.0)),
            (self.max.z - p.z, Vec3::new(0.0, 0.0, 1.0)),
        ];
        let mut best = candidates[0];
        for &cand in candidates.iter().skip(1) {
            if cand.0 < best.0 {
                best = cand;
            }
        }
        best
    }
}

/// The result of a sphere-`AABB` proximity query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Proximity {
    /// The point on the box (surface or interior) closest to the sphere centre.
    /// When the centre is inside the box this is the centre itself.
    pub closest_point: Vec3,
    /// Distance from the sphere centre to the box. Zero when the centre lies on
    /// or inside the box.
    pub outside_distance: f32,
    /// Signed surface-to-surface separation, `outside_distance - radius`.
    /// Positive when the surfaces are apart, zero at a tangent, negative when
    /// they overlap.
    pub separation: f32,
    /// Whether the sphere and box intersect (a tangent contact counts as an
    /// intersection).
    pub intersecting: bool,
    /// Whether the sphere centre lies on or inside the box.
    pub center_inside_box: bool,
    /// Depth to translate the sphere along [`Proximity::normal`] to just
    /// separate it from the box. Zero when the two do not intersect.
    pub penetration_depth: f32,
    /// Unit separation normal: the direction to push the sphere so it leaves
    /// the box. When the centre is outside it points from the box toward the
    /// centre; when the centre is inside it is the outward normal of the
    /// nearest face.
    pub normal: Vec3,
}

/// Runs the sphere-versus-`AABB` proximity query.
///
/// The method is the standard real-time-rendering test: the closest point on
/// the box to the sphere centre is the centre clamped into `[min, max]`; the
/// squared distance to that point compared against `r²` decides intersection.
///
/// * **Centre outside the box** — the separation normal is the unit vector from
///   the closest point to the centre, and the penetration depth is
///   `radius - distance` (clamped to zero when not overlapping).
/// * **Centre on or inside the box** — the closest point coincides with the
///   centre, so the normal is taken from the nearest box face and the
///   penetration depth is `face_distance + radius`.
#[must_use]
pub fn query(sphere: Sphere, aabb: Aabb) -> Proximity {
    let center = sphere.center;
    // A negative radius is meaningless; treat it as a point.
    let radius = sphere.radius.max(0.0);

    let closest = center.clamp(aabb.min, aabb.max);
    let delta = center.sub(closest);
    let dist_sq = delta.length_squared();
    let radius_sq = radius * radius;

    let outside_distance = dist_sq.sqrt();
    let intersecting = dist_sq <= radius_sq;
    let separation = outside_distance - radius;

    if outside_distance > CMP_EPS {
        // The centre is strictly outside the box: the separation direction is
        // the unit vector from the closest surface point toward the centre.
        let normal = delta.scale(1.0 / outside_distance);
        let penetration_depth = if intersecting {
            (radius - outside_distance).max(0.0)
        } else {
            0.0
        };
        Proximity {
            closest_point: closest,
            outside_distance,
            separation,
            intersecting,
            center_inside_box: false,
            penetration_depth,
            normal,
        }
    } else {
        // The centre is on the surface or inside the box: pick the nearest
        // face for the outward separation normal.
        let (face_distance, normal) = aabb.nearest_face(center);
        let penetration_depth = if intersecting {
            face_distance + radius
        } else {
            0.0
        };
        Proximity {
            closest_point: closest,
            outside_distance: 0.0,
            separation: -radius,
            intersecting,
            center_inside_box: aabb.contains_point(center),
            penetration_depth,
            normal,
        }
    }
}

/// Convenience boolean: whether the sphere and box intersect.
#[must_use]
pub fn intersects(sphere: Sphere, aabb: Aabb) -> bool {
    query(sphere, aabb).intersecting
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only tolerance, looser than [`CMP_EPS`] to absorb `sqrt` rounding.
    const TEST_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn unit_cube() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::splat(2.0))
    }

    #[test]
    fn separated_sphere_does_not_intersect() {
        let r = query(Sphere::new(Vec3::new(5.0, 1.0, 1.0), 1.0), unit_cube());
        assert!(!r.intersecting);
        assert!(approx(r.outside_distance, 3.0));
        assert!(approx(r.separation, 2.0));
        assert!(approx(r.penetration_depth, 0.0));
    }

    #[test]
    fn tangent_contact_counts_as_intersecting() {
        let r = query(Sphere::new(Vec3::new(3.0, 1.0, 1.0), 1.0), unit_cube());
        assert!(r.intersecting);
        assert!(approx(r.outside_distance, 1.0));
        assert!(approx(r.separation, 0.0));
        assert!(approx(r.penetration_depth, 0.0));
        assert!(vec_approx(r.normal, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn overlapping_outside_center_has_positive_penetration() {
        let r = query(Sphere::new(Vec3::new(2.5, 1.0, 1.0), 1.0), unit_cube());
        assert!(r.intersecting);
        assert!(!r.center_inside_box);
        assert!(approx(r.outside_distance, 0.5));
        assert!(approx(r.penetration_depth, 0.5));
        assert!(approx(r.separation, -0.5));
        assert!(vec_approx(r.normal, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn center_inside_box_penetration_and_normal() {
        // Box [0,4]^3, centre near the -x face.
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(4.0));
        let r = query(Sphere::new(Vec3::new(1.0, 2.0, 2.0), 0.5), aabb);
        assert!(r.intersecting);
        assert!(r.center_inside_box);
        assert!(approx(r.outside_distance, 0.0));
        // Nearest face is -x at distance 1; pushout = 1 + radius.
        assert!(approx(r.penetration_depth, 1.5));
        assert!(vec_approx(r.normal, Vec3::new(-1.0, 0.0, 0.0)));
        assert!(vec_approx(r.closest_point, Vec3::new(1.0, 2.0, 2.0)));
    }

    #[test]
    fn sphere_fully_contains_box() {
        let aabb = unit_cube();
        let r = query(Sphere::new(Vec3::splat(1.0), 10.0), aabb);
        assert!(r.intersecting);
        assert!(r.center_inside_box);
        // Centre at cube centre: every face is 1 away; pushout = 1 + 10.
        assert!(approx(r.penetration_depth, 11.0));
    }

    #[test]
    fn degenerate_zero_radius_point_inside() {
        let r = query(Sphere::new(Vec3::splat(1.0), 0.0), unit_cube());
        assert!(r.intersecting);
        assert!(r.center_inside_box);
        assert!(approx(r.penetration_depth, 1.0));
    }

    #[test]
    fn degenerate_zero_radius_point_outside() {
        let r = query(Sphere::new(Vec3::new(3.0, 1.0, 1.0), 0.0), unit_cube());
        assert!(!r.intersecting);
        assert!(approx(r.outside_distance, 1.0));
        assert!(approx(r.penetration_depth, 0.0));
    }

    #[test]
    fn degenerate_zero_radius_point_on_surface() {
        let r = query(Sphere::new(Vec3::new(2.0, 1.0, 1.0), 0.0), unit_cube());
        assert!(r.intersecting);
        assert!(approx(r.outside_distance, 0.0));
        assert!(approx(r.penetration_depth, 0.0));
    }

    #[test]
    fn negative_radius_treated_as_zero() {
        let a = query(Sphere::new(Vec3::new(3.0, 1.0, 1.0), -5.0), unit_cube());
        let b = query(Sphere::new(Vec3::new(3.0, 1.0, 1.0), 0.0), unit_cube());
        assert_eq!(a.intersecting, b.intersecting);
        assert!(approx(a.penetration_depth, b.penetration_depth));
    }

    #[test]
    fn closest_point_on_face() {
        let r = query(Sphere::new(Vec3::new(5.0, 1.0, 1.0), 1.0), unit_cube());
        assert!(vec_approx(r.closest_point, Vec3::new(2.0, 1.0, 1.0)));
    }

    #[test]
    fn closest_point_on_edge() {
        let r = query(Sphere::new(Vec3::new(5.0, 5.0, 1.0), 1.0), unit_cube());
        assert!(vec_approx(r.closest_point, Vec3::new(2.0, 2.0, 1.0)));
    }

    #[test]
    fn closest_point_on_corner() {
        let r = query(Sphere::new(Vec3::new(5.0, 5.0, 5.0), 1.0), unit_cube());
        assert!(vec_approx(r.closest_point, Vec3::new(2.0, 2.0, 2.0)));
    }

    #[test]
    fn closest_point_equals_center_when_inside() {
        let r = query(Sphere::new(Vec3::new(1.0, 1.5, 0.5), 0.1), unit_cube());
        assert!(vec_approx(r.closest_point, Vec3::new(1.0, 1.5, 0.5)));
    }

    #[test]
    fn normal_outside_points_from_box_to_center() {
        let r = query(Sphere::new(Vec3::new(5.0, 5.0, 5.0), 1.0), unit_cube());
        // delta = (3,3,3); normalized to equal thirds.
        let expected = Vec3::splat(3.0).scale(1.0 / (27.0_f32).sqrt());
        assert!(vec_approx(r.normal, expected));
    }

    #[test]
    fn normal_is_unit_length_when_outside() {
        let r = query(Sphere::new(Vec3::new(4.0, 3.0, 7.0), 1.0), unit_cube());
        assert!(approx(r.normal.length(), 1.0));
    }

    #[test]
    fn inside_nearest_face_axis_x() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(10.0));
        let r = query(Sphere::new(Vec3::new(9.0, 5.0, 5.0), 0.5), aabb);
        assert!(vec_approx(r.normal, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn inside_nearest_face_axis_y() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(10.0));
        let r = query(Sphere::new(Vec3::new(5.0, 1.0, 5.0), 0.5), aabb);
        assert!(vec_approx(r.normal, Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn inside_nearest_face_axis_z() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(10.0));
        let r = query(Sphere::new(Vec3::new(5.0, 5.0, 9.5), 0.5), aabb);
        assert!(vec_approx(r.normal, Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn penetration_depth_outside_value() {
        let r = query(Sphere::new(Vec3::new(2.75, 1.0, 1.0), 1.0), unit_cube());
        // distance 0.75, radius 1 => pushout 0.25.
        assert!(approx(r.penetration_depth, 0.25));
    }

    #[test]
    fn penetration_depth_inside_value() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(6.0));
        let r = query(Sphere::new(Vec3::new(2.0, 3.0, 3.0), 0.25), aabb);
        // Nearest face is -x at distance 2 => pushout 2 + 0.25.
        assert!(approx(r.penetration_depth, 2.25));
    }

    #[test]
    fn separation_positive_when_apart() {
        let r = query(Sphere::new(Vec3::new(6.0, 1.0, 1.0), 1.0), unit_cube());
        assert!(r.separation > 0.0);
        assert!(approx(r.separation, 3.0));
    }

    #[test]
    fn separation_negative_when_overlapping() {
        let r = query(Sphere::new(Vec3::new(2.5, 1.0, 1.0), 1.0), unit_cube());
        assert!(r.separation < 0.0);
    }

    #[test]
    fn translation_invariance_of_penetration_and_normal() {
        let sphere = Sphere::new(Vec3::new(2.6, 1.2, 0.7), 1.0);
        let aabb = unit_cube();
        let base = query(sphere, aabb);

        let t = Vec3::new(10.0, -5.0, 3.0);
        let moved = query(
            Sphere::new(sphere.center.add(t), sphere.radius),
            Aabb::new(aabb.min.add(t), aabb.max.add(t)),
        );

        assert_eq!(base.intersecting, moved.intersecting);
        assert!(approx(base.penetration_depth, moved.penetration_depth));
        assert!(approx(base.outside_distance, moved.outside_distance));
        assert!(vec_approx(base.normal, moved.normal));
    }

    #[test]
    fn reflection_invariance_negates_normal() {
        let sphere = Sphere::new(Vec3::new(3.0, 4.0, 5.0), 1.5);
        let aabb = unit_cube();
        let base = query(sphere, aabb);

        // Reflect through the origin: p -> -p, box [min,max] -> [-max,-min].
        let reflected = query(
            Sphere::new(sphere.center.neg(), sphere.radius),
            Aabb::new(aabb.max.neg(), aabb.min.neg()),
        );

        assert!(approx(base.penetration_depth, reflected.penetration_depth));
        assert!(vec_approx(reflected.normal, base.normal.neg()));
    }

    #[test]
    fn outside_distance_matches_manual_sqrt() {
        let r = query(Sphere::new(Vec3::new(4.0, 5.0, 1.0), 1.0), unit_cube());
        // closest = (2,2,1); delta = (2,3,0); |delta| = sqrt(13).
        let expected = (13.0_f32).sqrt();
        assert!(approx(r.outside_distance, expected));
    }

    #[test]
    fn intersects_helper_matches_query() {
        let sphere = Sphere::new(Vec3::new(2.5, 1.0, 1.0), 1.0);
        let aabb = unit_cube();
        assert_eq!(intersects(sphere, aabb), query(sphere, aabb).intersecting);

        let far = Sphere::new(Vec3::new(20.0, 20.0, 20.0), 1.0);
        assert_eq!(intersects(far, aabb), query(far, aabb).intersecting);
    }

    #[test]
    fn center_inside_flag_true_when_inside() {
        let r = query(Sphere::new(Vec3::splat(1.0), 0.3), unit_cube());
        assert!(r.center_inside_box);
    }

    #[test]
    fn center_inside_flag_false_when_outside() {
        let r = query(Sphere::new(Vec3::new(3.0, 1.0, 1.0), 2.0), unit_cube());
        assert!(!r.center_inside_box);
        assert!(r.intersecting);
    }

    #[test]
    fn ambiguous_center_picks_deterministic_face() {
        // Exact cube centre: all six faces tie; the -x face wins by axis order.
        let r = query(Sphere::new(Vec3::splat(1.0), 0.5), unit_cube());
        assert!(vec_approx(r.normal, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn from_center_half_extent_round_trips() {
        let aabb = Aabb::from_center_half_extent(Vec3::new(1.0, 2.0, 3.0), Vec3::splat(0.5));
        assert!(vec_approx(aabb.center(), Vec3::new(1.0, 2.0, 3.0)));
        assert!(vec_approx(aabb.half_extent(), Vec3::splat(0.5)));
        assert!(aabb.contains_point(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn vector_component_product_is_lanewise() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        let b = Vec3::new(5.0, 6.0, 7.0);
        assert!(vec_approx(a.mul(b), Vec3::new(10.0, 18.0, 28.0)));
    }
}
