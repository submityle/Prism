//! Analytic ray vs *oriented bounding box* (`OBB`) intersection by the slab
//! method, for the particle subsystem's picking, collision-probe, and
//! analytic-primitive raytrace contracts (design §10, §14).
//!
//! An **`OBB`** is an axis-aligned box that has been *rotated* into world space:
//! it is described by a `center`, three mutually orthogonal **unit** axes
//! `axis_u`, `axis_v`, `axis_w`, and a non-negative half-extent along each axis.
//! A point `p` is inside the box exactly when the signed projection of
//! `p - center` onto each axis lies within `[-half_i, half_i]`. Solving the ray
//! intersection therefore reduces to *projecting the ray onto the three local
//! axes* — equivalently, transforming the ray into the box's local frame — and
//! running the classic three-slab test, intersecting the per-axis parameter
//! intervals `[t_near, t_far]` into a single `[t_enter, t_exit]` span.
//!
//! [`Obb::first_hit`] returns the nearest forward surface hit (`t >= 0`) with its
//! world-space position and the *outward* unit face normal (the local axis, with
//! the sign of the face that was crossed), or `None` on a miss. When the ray
//! origin is *inside* the box the entry parameter is negative, so the first
//! forward crossing is the **exit** face at `t_exit`, and that exit face's
//! outward normal is reported. [`Obb::span`] exposes the raw ordered
//! `(t_enter, t_exit)` interval (either may be negative) for callers that need
//! the full chord, and [`Obb::intersects`] is the unsigned boolean predicate.
//!
//! # How it differs from its siblings
//! * [`crate::particle::volume_march`] intersects a ray with an *axis-aligned*
//!   box ([`super::sort_cull::Aabb`]) by the slab method — no rotation, so its
//!   slabs are the world `x`/`y`/`z` planes and its face normals are the world
//!   basis vectors. This module generalizes that to an *oriented* box: the slabs
//!   are the box's own three (possibly rotated) axes, recovered by dot products
//!   rather than by reading vector components directly, and the reported normal
//!   is one of those rotated axes. An `OBB` whose axes are the world basis is
//!   exactly an `AABB`, and [`Obb::from_aabb`] builds that degenerate case so the
//!   two solvers agree.
//! * [`crate::particle::ray_sphere`], [`crate::particle::ray_cylinder`], and
//!   [`crate::particle::ray_capsule`] are the *round* analytic primitives solved
//!   through a quadratic (and hence needing `sqrt`); the box here is a purely
//!   *planar* primitive whose faces are flat, so it is solved with only
//!   `+ - * /` and never calls `sqrt`.
//!
//! # Degenerate cases handled explicitly (never a `NaN`)
//! * **Ray parallel to a slab** — when the direction's projection onto an axis
//!   is smaller than [`EPS`] in magnitude, that axis contributes no finite
//!   `1 / d` division: the ray misses immediately if its origin projects outside
//!   the `[-half, half]` slab, and otherwise the slab imposes the sentinel
//!   interval `(-INFINITY, +INFINITY)` and is skipped.
//! * **Origin inside the box** — `t_enter < 0 <= t_exit`, so the exit face is
//!   returned as the first forward hit.
//! * **Box entirely behind the origin** — `t_exit < 0`, reported as a miss.
//! * **Zero-length direction** — rejected as a non-ray.
//!
//! The direction is *not* required to be unit length: the solver carries it
//! literally, so the ray parameter `t` reads out in units of `direction`
//! (halving `t` when the direction is doubled), and [`Ray::new_normalized`] is
//! offered for callers that want `t` to equal Euclidean distance.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file, every `f32` division guards its denominator against [`EPS`], no
//! transcendental function is ever called, and no exact `==` / `!=` is ever
//! written on a production `f32` (magnitudes are compared against an explicit
//! epsilon), so this `CPU` reference stays bit-reproducible against a future
//! `GPU` kernel that packs the same box.

/// Magnitude floor used to guard `f32` divisions (the parallel-slab `1 / d`
/// case) and to classify a projected origin as inside or outside a slab, so no
/// exact `==` / `!=` is ever written on a production `f32`.
pub const EPS: f32 = 1.0e-6;

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
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Unary negation `-self`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
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

    /// Cross product `self × rhs` (right-handed), provided for callers that
    /// build an orthonormal frame around a hit normal.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed and, unlike it, uses no `sqrt`.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length. This is the module's only `sqrt`, used purely by
    /// [`Vec3::normalize_or_zero`] for the optional [`Ray::new_normalized`]
    /// convenience; the intersection solver itself never calls it.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector in the same direction, or the zero vector when the length is
    /// below [`EPS`], so a degenerate input never produces a `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len < EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A parametric ray `origin + t * dir` with `t >= 0` denoting the forward
/// half-line.
///
/// The solver works for any non-degenerate `dir` because it projects `dir`
/// onto each box axis explicitly; construct with [`Ray::new_normalized`] when
/// the ray parameter `t` should read out as a Euclidean distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The point the ray emanates from.
    pub origin: Vec3,
    /// The ray direction (not required to be unit length).
    pub dir: Vec3,
}

impl Ray {
    /// Builds a ray from an origin and a (possibly non-unit) direction.
    #[must_use]
    pub const fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// Builds a ray whose direction is normalized to unit length, so the ray
    /// parameter `t` returned by the solver equals the Euclidean distance from
    /// the origin. A degenerate direction collapses to the zero vector, which
    /// the solver later rejects as non-intersecting.
    #[must_use]
    pub fn new_normalized(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalize_or_zero(),
        }
    }

    /// Evaluates the ray position at parameter `t`: `origin + dir * t`.
    #[must_use]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin.add(self.dir.scale(t))
    }
}

/// A resolved forward intersection: the ray parameter, the world-space hit
/// point, and the outward-facing unit surface normal at that point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayObbHit {
    /// The ray parameter at the hit (`>= 0`). Equals the hit distance when the
    /// ray direction is unit length.
    pub t: f32,
    /// The world-space intersection point `ray.at(t)`.
    pub point: Vec3,
    /// The unit surface normal pointing *out* of the box at `point`: one of the
    /// box's three axes, signed toward the face that was crossed. Outward even
    /// when the ray origin is inside the box (then it is the exit face normal).
    pub normal: Vec3,
}

/// The full result of the three-slab intersection: the entry and exit ray
/// parameters (ordered `t_enter <= t_exit`, either possibly negative) together
/// with the outward face normals at each of the two crossings.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SlabSpan {
    t_enter: f32,
    t_exit: f32,
    enter_normal: Vec3,
    exit_normal: Vec3,
}

/// An oriented bounding box: a rotated, translated axis-aligned box.
///
/// The three axes are assumed to be mutually orthogonal **unit** vectors (a
/// proper rotation frame); the half-extents give the box's reach along each
/// axis and must be non-negative. `axis_w` is redundant with
/// `axis_u × axis_v` for a right-handed frame but is stored explicitly so the
/// solver never needs a cross product on the hot path and so left-handed frames
/// are equally supported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// The box center in world space.
    pub center: Vec3,
    /// The first local unit axis.
    pub axis_u: Vec3,
    /// The second local unit axis.
    pub axis_v: Vec3,
    /// The third local unit axis.
    pub axis_w: Vec3,
    /// Non-negative half-extents along `axis_u`, `axis_v`, `axis_w`
    /// respectively.
    pub half_extents: Vec3,
}

impl Obb {
    /// Builds an `OBB` from a center, three local unit axes, and the matching
    /// per-axis half-extents.
    #[must_use]
    pub const fn new(
        center: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
        axis_w: Vec3,
        half_extents: Vec3,
    ) -> Self {
        Self {
            center,
            axis_u,
            axis_v,
            axis_w,
            half_extents,
        }
    }

    /// Builds the axis-aligned degenerate `OBB` spanning the world box
    /// `[min, max]`: its axes are the world basis and its center / half-extents
    /// are the box midpoint / half-diagonal. This is exactly the `AABB` that
    /// [`crate::particle::volume_march`] intersects, so the two solvers agree on
    /// the un-rotated case.
    #[must_use]
    pub fn from_aabb(min: Vec3, max: Vec3) -> Self {
        let center = min.add(max).scale(0.5);
        let half_extents = max.sub(min).scale(0.5);
        Self {
            center,
            axis_u: Vec3::new(1.0, 0.0, 0.0),
            axis_v: Vec3::new(0.0, 1.0, 0.0),
            axis_w: Vec3::new(0.0, 0.0, 1.0),
            half_extents,
        }
    }

    /// Runs the three-slab test, returning the ordered entry/exit span and the
    /// outward normal at each crossing, or `None` when the ray misses the box
    /// (the slab intervals do not overlap, or the ray runs parallel to and
    /// outside some slab). A zero-length direction is rejected here as a
    /// non-ray. The returned parameters may be negative; forward-hit selection
    /// is left to [`Obb::first_hit`].
    fn slab_span(&self, ray: Ray) -> Option<SlabSpan> {
        if ray.dir.length_squared() <= EPS * EPS {
            return None;
        }

        let axes = [self.axis_u, self.axis_v, self.axis_w];
        let halves = [
            self.half_extents.x,
            self.half_extents.y,
            self.half_extents.z,
        ];

        let rel = ray.origin.sub(self.center);
        let mut t_enter = -f32::INFINITY;
        let mut t_exit = f32::INFINITY;
        let mut enter_normal = Vec3::ZERO;
        let mut exit_normal = Vec3::ZERO;

        for (axis, half) in axes.into_iter().zip(halves) {
            let o = axis.dot(rel);
            let d = axis.dot(ray.dir);

            if d.abs() <= EPS {
                // Ray parallel to this slab: it misses unless the origin already
                // projects inside `[-half, half]`; otherwise the slab spans
                // `(-INFINITY, +INFINITY)` and constrains nothing.
                if o < -half - EPS || o > half + EPS {
                    return None;
                }
                continue;
            }

            let inv = 1.0 / d;
            // `t_a` reaches the `-half` face (outward normal `-axis`); `t_b`
            // reaches the `+half` face (outward normal `+axis`).
            let t_a = (-half - o) * inv;
            let t_b = (half - o) * inv;
            let (t_near, n_near, t_far, n_far) = if t_a <= t_b {
                (t_a, axis.neg(), t_b, axis)
            } else {
                (t_b, axis, t_a, axis.neg())
            };

            if t_near > t_enter {
                t_enter = t_near;
                enter_normal = n_near;
            }
            if t_far < t_exit {
                t_exit = t_far;
                exit_normal = n_far;
            }
            if t_enter > t_exit {
                return None;
            }
        }

        Some(SlabSpan {
            t_enter,
            t_exit,
            enter_normal,
            exit_normal,
        })
    }

    /// The ordered intersection interval `(t_enter, t_exit)` of the ray with the
    /// box, or `None` on a miss. Either bound may be negative (a fully-behind
    /// box gives two negatives; an origin inside the box gives `t_enter < 0`).
    #[must_use]
    pub fn span(&self, ray: Ray) -> Option<(f32, f32)> {
        self.slab_span(ray).map(|s| (s.t_enter, s.t_exit))
    }

    /// Returns the nearest forward hit (`t >= 0`) against the box surface with
    /// its point and outward unit face normal, or `None` when the box is missed
    /// or lies entirely behind the origin.
    ///
    /// When the origin is inside the box, `t_enter` is negative and the first
    /// forward crossing is the exit face at `t_exit`, whose outward normal is
    /// reported.
    #[must_use]
    pub fn first_hit(&self, ray: Ray) -> Option<RayObbHit> {
        let span = self.slab_span(ray)?;
        if span.t_exit < 0.0 {
            // The whole overlap is behind the ray origin.
            return None;
        }
        let (t, normal) = if span.t_enter >= 0.0 {
            (span.t_enter, span.enter_normal)
        } else {
            (span.t_exit, span.exit_normal)
        };
        Some(RayObbHit {
            t,
            point: ray.at(t),
            normal,
        })
    }

    /// Whether the ray's forward half-line strikes the box surface.
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        self.first_hit(ray).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparisons that are not bit-exact.
    const TOL: f32 = 1.0e-4;

    /// `1 / sqrt(2)`, the cosine/sine of 45°, used to build rotated frames
    /// without calling a transcendental function.
    const S: f32 = core::f32::consts::FRAC_1_SQRT_2;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < TOL, "expected {b}, got {a}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    /// The canonical unit cube: center at the origin, world-aligned axes, unit
    /// half-extents (so it spans `[-1, 1]` on each world axis).
    fn unit_cube() -> Obb {
        Obb::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::splat(1.0),
        )
    }

    /// Rotates a vector by +45° about the world `z` axis.
    fn rot_z_45(v: Vec3) -> Vec3 {
        Vec3::new(v.x * S - v.y * S, v.x * S + v.y * S, v.z)
    }

    /// A unit cube rotated +45° about `z`.
    fn rotated_cube() -> Obb {
        Obb::new(
            Vec3::ZERO,
            rot_z_45(Vec3::new(1.0, 0.0, 0.0)),
            rot_z_45(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::splat(1.0),
        )
    }

    #[test]
    fn vec3_add_sub_scale_are_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert_eq!(a.add(b), Vec3::new(5.0, 1.0, 3.5));
        assert_eq!(a.sub(b), Vec3::new(-3.0, 3.0, 2.5));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(Vec3::splat(3.0), Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_neg_flips_all_lanes() {
        assert_eq!(Vec3::new(1.0, -2.0, 3.0).neg(), Vec3::new(-1.0, 2.0, -3.0));
        assert_eq!(Vec3::ZERO.neg(), Vec3::ZERO);
    }

    #[test]
    fn vec3_dot_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        approx(x.dot(y), 0.0);
        approx(x.dot(x), 1.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_length_and_normalize() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        approx(a.length_squared(), 25.0);
        approx(a.length(), 5.0);
        approx_vec(a.normalize_or_zero(), Vec3::new(0.6, 0.8, 0.0));
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn ray_at_evaluates_position() {
        let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        approx_vec(r.at(3.0), Vec3::new(1.0, 6.0, 0.0));
        let n = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0));
        approx(n.dir.length(), 1.0);
        approx_vec(n.dir, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn axis_aligned_matches_aabb_front_hit() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("front hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(-1.0, 0.0, 0.0));
        approx_vec(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn from_aabb_agrees_with_explicit_cube() {
        let a = Obb::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let ha = a.first_hit(ray).expect("hit a");
        let hb = unit_cube().first_hit(ray).expect("hit b");
        approx(ha.t, hb.t);
        approx_vec(ha.point, hb.point);
        approx_vec(ha.normal, hb.normal);
    }

    #[test]
    fn axis_aligned_side_miss() {
        let cube = unit_cube();
        // Passes well above the box, parallel to the x axis.
        let ray = Ray::new(Vec3::new(-5.0, 3.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(cube.first_hit(ray).is_none());
        assert!(!cube.intersects(ray));
    }

    #[test]
    fn top_face_normal_points_plus_y() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cube.first_hit(ray).expect("top hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(0.0, 1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn negative_x_face_normal() {
        let cube = unit_cube();
        // Shoot from +x back toward -x: enters the +x face, normal +x.
        let ray = Ray::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("hit");
        approx_vec(hit.point, Vec3::new(1.0, 0.0, 0.0));
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn rotated_box_front_face_hit_along_axis() {
        let cube = rotated_cube();
        let u = rot_z_45(Vec3::new(1.0, 0.0, 0.0));
        // Fire straight down the +u axis from 5 units out, direction -u.
        let ray = Ray::new(u.scale(5.0), u.neg());
        let hit = cube.first_hit(ray).expect("rotated face hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, u);
        // Outward normal is the +u face, pointing back toward the ray.
        approx_vec(hit.normal, u);
    }

    #[test]
    fn rotated_box_normal_is_unit_length() {
        let cube = rotated_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.2, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("hit");
        approx(hit.normal.length(), 1.0);
    }

    #[test]
    fn entry_normal_faces_back_toward_ray() {
        let cube = rotated_cube();
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.1, 0.0), dir);
        let hit = cube.first_hit(ray).expect("hit");
        // A front-facing surface normal opposes the incoming direction.
        assert!(hit.normal.dot(dir) < 0.0);
    }

    #[test]
    fn grazing_edge_is_a_tangent_hit() {
        let cube = unit_cube();
        // y == 1 rides exactly along the top face while advancing in x.
        let ray = Ray::new(Vec3::new(-5.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("tangent hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(-1.0, 1.0, 0.0));
    }

    #[test]
    fn clean_miss_passes_outside_the_box() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(cube.first_hit(ray).is_none());
    }

    #[test]
    fn origin_inside_returns_exit_face() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("exit hit");
        approx(hit.t, 1.0);
        approx_vec(hit.point, Vec3::new(1.0, 0.0, 0.0));
        // Exit face outward normal still points outward (+x).
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn origin_inside_span_has_negative_entry() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let (t_enter, t_exit) = cube.span(ray).expect("span");
        approx(t_enter, -1.0);
        approx(t_exit, 1.0);
        assert!(t_enter < 0.0 && t_exit > 0.0);
    }

    #[test]
    fn parallel_axis_inside_slab_hits() {
        let cube = unit_cube();
        // dir has no y component, so the y slab is parallel; origin y is inside.
        let ray = Ray::new(Vec3::new(-5.0, 0.5, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("parallel-slab hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(-1.0, 0.5, 0.0));
    }

    #[test]
    fn parallel_axis_outside_slab_misses() {
        let cube = unit_cube();
        // Same parallel y slab, but origin y is outside `[-1, 1]`.
        let ray = Ray::new(Vec3::new(-5.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(cube.first_hit(ray).is_none());
    }

    #[test]
    fn reverse_direction_behind_origin_is_none() {
        let cube = Obb::new(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::splat(1.0),
        );
        // Box is at +x; ray points -x, so the box is entirely behind.
        let ray = Ray::new(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0));
        assert!(cube.first_hit(ray).is_none());
        let (t_enter, t_exit) = cube.span(ray).expect("span");
        assert!(t_enter < 0.0 && t_exit < 0.0);
    }

    #[test]
    fn non_unit_direction_scales_t() {
        let cube = unit_cube();
        let unit = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let doubled = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        let hu = cube.first_hit(unit).expect("hit unit");
        let hd = cube.first_hit(doubled).expect("hit doubled");
        approx(hu.t, 4.0);
        approx(hd.t, 2.0);
        // Same geometric hit point regardless of direction scaling.
        approx_vec(hu.point, hd.point);
    }

    #[test]
    fn span_is_ordered_tmin_le_tmax() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.3, -0.2), Vec3::new(1.0, 0.0, 0.0));
        let (t_enter, t_exit) = cube.span(ray).expect("span");
        assert!(t_enter <= t_exit);
        approx(t_enter, 4.0);
        approx(t_exit, 6.0);
    }

    #[test]
    fn hit_point_lies_on_the_box_face() {
        let cube = rotated_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.25, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("hit");
        // The projection onto the hit axis (the normal's axis) must equal the
        // half-extent along that axis.
        let rel = hit.point.sub(cube.center);
        let proj = rel.dot(hit.normal);
        approx(proj.abs(), 1.0);
    }

    #[test]
    fn translation_invariance() {
        let cube = unit_cube();
        let shift = Vec3::new(3.0, -2.0, 7.0);
        let moved = Obb::new(
            cube.center.add(shift),
            cube.axis_u,
            cube.axis_v,
            cube.axis_w,
            cube.half_extents,
        );
        let ray = Ray::new(Vec3::new(-5.0, 0.3, 0.1), Vec3::new(1.0, 0.0, 0.0));
        let moved_ray = Ray::new(ray.origin.add(shift), ray.dir);
        let h0 = cube.first_hit(ray).expect("base hit");
        let h1 = moved.first_hit(moved_ray).expect("moved hit");
        approx(h0.t, h1.t);
        approx_vec(h0.point.add(shift), h1.point);
        approx_vec(h0.normal, h1.normal);
    }

    #[test]
    fn rotation_invariance() {
        let base = unit_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let h0 = base.first_hit(ray).expect("base hit");

        // Rotate both the box frame and the ray by +45° about z.
        let rotated = Obb::new(
            Vec3::ZERO,
            rot_z_45(base.axis_u),
            rot_z_45(base.axis_v),
            rot_z_45(base.axis_w),
            base.half_extents,
        );
        let rot_ray = Ray::new(rot_z_45(ray.origin), rot_z_45(ray.dir));
        let h1 = rotated.first_hit(rot_ray).expect("rotated hit");

        // The parameter is invariant; point and normal are rigidly rotated.
        approx(h0.t, h1.t);
        approx_vec(rot_z_45(h0.point), h1.point);
        approx_vec(rot_z_45(h0.normal), h1.normal);
    }

    #[test]
    fn intersects_true_and_false() {
        let cube = unit_cube();
        let through = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let aside = Ray::new(Vec3::new(-5.0, 9.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(cube.intersects(through));
        assert!(!cube.intersects(aside));
    }

    #[test]
    fn zero_direction_is_none() {
        let cube = unit_cube();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO);
        assert!(cube.first_hit(ray).is_none());
        assert!(cube.span(ray).is_none());
    }

    #[test]
    fn non_cube_half_extents() {
        // A slab that is thin in y (half 0.5) and long in x (half 3).
        let obb = Obb::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(3.0, 0.5, 1.0),
        );
        let top = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let h = obb.first_hit(top).expect("top hit");
        approx(h.t, 4.5);
        approx_vec(h.point, Vec3::new(0.0, 0.5, 0.0));
        approx_vec(h.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn diagonal_ray_through_center() {
        let cube = unit_cube();
        let ray = Ray::new_normalized(Vec3::new(-3.0, -3.0, -3.0), Vec3::new(1.0, 1.0, 1.0));
        let hit = cube.first_hit(ray).expect("diagonal hit");
        // First crossing is a corner where all three slabs meet at t == entry.
        let (t_enter, _t_exit) = cube.span(ray).expect("span");
        approx(hit.t, t_enter);
        assert!(hit.t > 0.0);
    }

    #[test]
    fn far_offset_box_hit_distance() {
        let obb = Obb::new(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::splat(2.0),
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let hit = obb.first_hit(ray).expect("far hit");
        approx(hit.t, 98.0);
        approx_vec(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn rotated_corner_hit_is_reported() {
        let cube = rotated_cube();
        // Straight along -x toward the rotated corner facing -x.
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cube.first_hit(ray).expect("corner hit");
        approx(hit.t, 5.0 - S - S);
        approx_vec(hit.point, Vec3::new(-(S + S), 0.0, 0.0));
    }

    #[test]
    fn exit_span_matches_first_hit_when_inside() {
        let cube = unit_cube();
        let ray = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        let (t_enter, t_exit) = cube.span(ray).expect("span");
        let hit = cube.first_hit(ray).expect("hit");
        assert!(t_enter < 0.0);
        approx(hit.t, t_exit);
        approx_vec(hit.point, Vec3::new(0.0, 0.0, 1.0));
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, 1.0));
    }
}
