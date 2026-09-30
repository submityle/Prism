//! Analytic ray vs *axis-aligned bounding box* (`AABB`) intersection by the
//! **slab method**, for the particle subsystem's picking, culling-probe, and
//! analytic-primitive raytrace contracts (design §10, §14).
//!
//! An **`AABB`** is described by its two extreme corners `min` and `max` (with
//! `min <= max` componentwise). The three pairs of parallel faces of the box
//! lie on the world `x`/`y`/`z` planes, so each axis defines a *slab*: the set
//! of points whose coordinate on that axis lies in `[min, max]`. A ray
//! `origin + t * dir` crosses each slab over a parameter interval
//! `[t_near, t_far]`, and the ray touches the box exactly when the three
//! intervals share a common sub-interval `[t_enter, t_exit]`. The whole solve is
//! therefore three per-axis divisions and a running interval intersection —
//! there is no `sqrt`, because the faces are flat.
//!
//! # Two intersection semantics
//! This module exposes both readings of "does the ray hit the box", because
//! callers need different ones:
//! * [`Aabb::intersect_line`] treats the ray as the *infinite line* through
//!   `origin` with slope `dir`: it returns the ordered chord `[t_enter, t_exit]`
//!   whenever the line crosses the box, and **either endpoint may be negative**
//!   (the box may lie partly or wholly *behind* the origin). This is the
//!   "line intersects box" question.
//! * [`Aabb::intersect_ray`] treats the ray as the *forward half-line*
//!   `t >= 0`: it returns the same chord but only when some of it is visible
//!   (`t_exit >= 0`). When the origin is *inside* the box the chord straddles
//!   zero (`t_enter < 0 <= t_exit`) and this is still a hit. This is the
//!   "forward ray intersects box" question, and [`Aabb::first_hit_t`] collapses
//!   it to the single first *visible* crossing.
//!
//! # Degenerate cases handled explicitly (never a `NaN`)
//! * **Ray parallel to a slab** — when a direction component's magnitude is
//!   below [`EPS`] the reciprocal `1 / d` is not formed: if the origin's
//!   coordinate lies outside `[min, max]` on that axis the ray misses
//!   immediately, otherwise that axis imposes the sentinel interval
//!   `(-INFINITY, +INFINITY)` (using [`f32::INFINITY`]) and is skipped.
//! * **Origin inside the box** — `t_enter < 0 <= t_exit`; a forward ray still
//!   hits and its first visible crossing is the exit face at `t_exit`.
//! * **Box entirely behind the origin** — `t_exit < 0`; a forward ray reports a
//!   miss while the infinite line still reports the (negative) chord.
//! * **Grazing / face-tangent** — a chord that collapses to a point
//!   (`t_enter == t_exit`, compared with `<=`, never with `==`) counts as a
//!   touching hit; a ray sliding *along* a face lies inside that axis' slab and
//!   contributes the infinite sentinel interval.
//! * **Zero-length direction** — rejected as a non-ray, so a point is never
//!   reported as spanning `(-INFINITY, +INFINITY)`.
//!
//! The direction is *not* required to be unit length: the solver carries it
//! literally, so `t` reads out in units of `dir` (halving when `dir` doubles).
//!
//! # How it differs from its siblings
//! * [`crate::particle::ray_obb`] solves the same slab test for an *oriented*
//!   (rotated) box, projecting the ray onto the box's own axes; this module is
//!   the un-rotated special case whose slabs are the world planes, so the
//!   per-axis math reads box coordinates directly and needs no dot products.
//! * [`crate::particle::sweep_aabb`] is the *broad-phase* time-of-impact of
//!   **two moving `AABB`s** over `t in [0, 1]`; there is no ray there, only a
//!   relative-motion segment against a Minkowski-expanded box.
//! * [`crate::particle::ray_sphere`], [`crate::particle::ray_cylinder`],
//!   [`crate::particle::ray_capsule`], and [`crate::particle::ray_triangle`] are
//!   the *round* or *simplex* analytic primitives: the first three need a
//!   quadratic (hence `sqrt`), and the triangle needs a barycentric solve. This
//!   box primitive is purely *planar* and uses only `+ - * /` with `min`/`max`.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file, every `f32` division guards its denominator against [`EPS`], no
//! transcendental function is ever called, and no exact `==` / `!=` is ever
//! written on a production `f32` (magnitudes are compared against an explicit
//! epsilon), so this `CPU` reference stays bit-reproducible against a future
//! `GPU` kernel that packs the same box.

/// Magnitude floor used to guard `f32` divisions (the parallel-slab `1 / d`
/// case) and to classify a direction component as parallel to a slab, so no
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
}

/// Component-wise sum `a + b`. Provided as a free function (rather than an
/// inherent `add`) so the internal math type needs no operator-trait surface.
#[must_use]
pub fn v_add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

/// Component-wise difference `a - b`.
#[must_use]
pub fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// Uniform scale of `a` by the scalar `s`.
#[must_use]
pub fn v_scale(a: Vec3, s: f32) -> Vec3 {
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

/// Dot (inner) product `a . b`.
#[must_use]
pub fn v_dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// A parametric ray `origin + t * dir`, with `t >= 0` denoting the forward
/// half-line. The direction need not be unit length; construct with
/// [`Ray::new`] and read `t` in units of `dir`.
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

    /// Evaluates the ray position at parameter `t`: `origin + dir * t`.
    #[must_use]
    pub fn at(self, t: f32) -> Vec3 {
        v_add(self.origin, v_scale(self.dir, t))
    }
}

/// The resolved slab-test chord: the ordered entry and exit ray parameters
/// (`t_enter <= t_exit`). Either endpoint may be negative when the box lies
/// partly or wholly behind the origin; forward-visibility selection is left to
/// [`Aabb::intersect_ray`] and [`Aabb::first_hit_t`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayAabbHit {
    /// The parameter at which the ray *enters* the box (the far endpoint of the
    /// three per-axis near planes). Negative when the origin is already past the
    /// entry face.
    pub t_enter: f32,
    /// The parameter at which the ray *exits* the box (the near endpoint of the
    /// three per-axis far planes). Negative only when the whole box is behind
    /// the origin.
    pub t_exit: f32,
}

impl RayAabbHit {
    /// Whether the origin lies at or beyond the entry face, i.e. the chord
    /// begins behind the origin (`t_enter < 0`). For a forward hit this means
    /// the origin is *inside* the box (paired with `t_exit >= 0`).
    #[must_use]
    pub fn entered_behind(self) -> bool {
        self.t_enter < 0.0
    }

    /// The first *forward* crossing of a ray that is already known to hit: the
    /// entry face when it is in front (`t_enter >= 0`), otherwise the exit face
    /// (the origin is inside the box).
    #[must_use]
    pub fn first_visible_t(self) -> f32 {
        if self.t_enter >= 0.0 {
            self.t_enter
        } else {
            self.t_exit
        }
    }
}

/// An axis-aligned bounding box given by its two extreme corners.
///
/// `min` must be componentwise `<=` `max`; the three slabs are then
/// `[min.x, max.x]`, `[min.y, max.y]`, and `[min.z, max.z]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// The corner with the smallest coordinate on every axis.
    pub min: Vec3,
    /// The corner with the largest coordinate on every axis.
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box from its `min` and `max` corners (assumed `min <= max`).
    #[must_use]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Builds the box centered at `center` with the given non-negative
    /// `half_extents` along each axis.
    #[must_use]
    pub fn from_center_half(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: v_sub(center, half_extents),
            max: v_add(center, half_extents),
        }
    }

    /// The box center `(min + max) / 2`.
    #[must_use]
    pub fn center(self) -> Vec3 {
        v_scale(v_add(self.min, self.max), 0.5)
    }

    /// Whether `point` lies within the closed box on every axis.
    #[must_use]
    pub fn contains(self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// The core slab solve shared by both semantics: intersects the three
    /// per-axis parameter intervals of the *infinite line* through the ray, and
    /// returns the ordered chord `[t_enter, t_exit]` when they overlap. Returns
    /// `None` when the ray runs parallel to and outside some slab, when the
    /// intervals do not overlap, or when the direction is degenerate (a
    /// zero-length "ray" is rejected here so a point never spans the infinite
    /// sentinel interval).
    fn slab_chord(self, ray: Ray) -> Option<RayAabbHit> {
        if v_dot(ray.dir, ray.dir) <= EPS * EPS {
            return None;
        }

        let axes = [
            (ray.origin.x, ray.dir.x, self.min.x, self.max.x),
            (ray.origin.y, ray.dir.y, self.min.y, self.max.y),
            (ray.origin.z, ray.dir.z, self.min.z, self.max.z),
        ];

        let mut t_enter = -f32::INFINITY;
        let mut t_exit = f32::INFINITY;

        for &(o, d, lo, hi) in axes.iter() {
            if d.abs() < EPS {
                // Parallel to this slab: a hit requires the origin to lie inside
                // the slab, otherwise the box is unreachable on this axis. When
                // inside, the axis imposes (-INFINITY, +INFINITY) and is skipped.
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let t1 = (lo - o) * inv;
                let t2 = (hi - o) * inv;
                let t_near = t1.min(t2);
                let t_far = t1.max(t2);
                t_enter = t_enter.max(t_near);
                t_exit = t_exit.min(t_far);
            }
        }

        if t_enter <= t_exit {
            Some(RayAabbHit { t_enter, t_exit })
        } else {
            None
        }
    }

    /// **Infinite-line** semantics: returns the ordered chord `[t_enter, t_exit]`
    /// whenever the line through the ray crosses the box, with either endpoint
    /// possibly negative. Answers "does the line intersect the box".
    #[must_use]
    pub fn intersect_line(self, ray: Ray) -> Option<RayAabbHit> {
        self.slab_chord(ray)
    }

    /// **Forward-ray** semantics: returns the chord `[t_enter, t_exit]` only when
    /// part of it is visible on the forward half-line (`t_exit >= 0`). When the
    /// origin is inside the box the chord straddles zero
    /// (`t_enter < 0 <= t_exit`) and is still reported. Answers "does the forward
    /// ray intersect the box".
    #[must_use]
    pub fn intersect_ray(self, ray: Ray) -> Option<RayAabbHit> {
        let chord = self.slab_chord(ray)?;
        if chord.t_exit >= 0.0 {
            Some(chord)
        } else {
            None
        }
    }

    /// The single first *visible* crossing parameter of a forward ray, or `None`
    /// on a miss. Equals `t_enter` when the box is ahead, or `t_exit` when the
    /// origin is inside the box.
    #[must_use]
    pub fn first_hit_t(self, ray: Ray) -> Option<f32> {
        self.intersect_ray(ray).map(RayAabbHit::first_visible_t)
    }

    /// Unsigned boolean predicate for the *infinite line*.
    #[must_use]
    pub fn intersects_line(self, ray: Ray) -> bool {
        self.slab_chord(ray).is_some()
    }

    /// Unsigned boolean predicate for the *forward ray*.
    #[must_use]
    pub fn intersects_ray(self, ray: Ray) -> bool {
        self.intersect_ray(ray).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_T: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS_T, "expected {a} ~= {b}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    fn unit_box() -> Aabb {
        Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    fn unit_box_0_1() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::splat(1.0))
    }

    #[test]
    fn frontal_hit_known_t() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx(hit.t_enter, 4.0);
        approx(hit.t_exit, 6.0);
    }

    #[test]
    fn frontal_hit_point_is_on_face() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx_vec(ray.at(hit.t_enter), Vec3::new(-1.0, 0.0, 0.0));
        approx_vec(ray.at(hit.t_exit), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn miss_above_the_box() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(-5.0, 9.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(b.intersect_ray(ray).is_none());
        assert!(b.intersect_line(ray).is_none());
        assert!(!b.intersects_ray(ray));
        assert!(!b.intersects_line(ray));
    }

    #[test]
    fn miss_pointing_away() {
        let b = unit_box();
        // Origin left of the box but pointing further left: line hits behind,
        // forward ray misses.
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(b.intersect_ray(ray).is_none());
        let line = b.intersect_line(ray).expect("line still crosses");
        assert!(line.t_enter < 0.0);
        assert!(line.t_exit < 0.0);
    }

    #[test]
    fn box_behind_origin_line_vs_ray() {
        let b = Aabb::new(Vec3::new(-11.0, -1.0, -1.0), Vec3::new(-9.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!(b.intersect_ray(ray).is_none());
        let line = b.intersect_line(ray).expect("line crosses behind");
        approx(line.t_enter, -11.0);
        approx(line.t_exit, -9.0);
    }

    #[test]
    fn origin_inside_box_is_a_hit() {
        let b = unit_box();
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("inside hit");
        approx(hit.t_enter, -1.0);
        approx(hit.t_exit, 1.0);
        assert!(hit.entered_behind());
        approx(hit.first_visible_t(), 1.0);
    }

    #[test]
    fn first_hit_t_ahead_is_entry() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        approx(b.first_hit_t(ray).expect("t"), 4.0);
    }

    #[test]
    fn first_hit_t_inside_is_exit() {
        let b = unit_box();
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        approx(b.first_hit_t(ray).expect("t"), 1.0);
    }

    #[test]
    fn parallel_axis_outside_slab_misses() {
        let b = unit_box();
        // Direction along x, but y = 5 is outside the [-1, 1] y-slab.
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(b.intersect_line(ray).is_none());
        assert!(b.intersect_ray(ray).is_none());
    }

    #[test]
    fn parallel_axis_inside_slab_hits() {
        let b = unit_box();
        // Direction along x, y = 0.5 inside the y-slab, z = 0 inside the z-slab.
        let ray = Ray::new(Vec3::new(-5.0, 0.5, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("parallel-inside hit");
        approx(hit.t_enter, 4.0);
        approx(hit.t_exit, 6.0);
    }

    #[test]
    fn two_parallel_axes_inside() {
        let b = unit_box();
        // Only the x direction is non-zero; y and z both lie inside their slabs.
        let ray = Ray::new(Vec3::new(10.0, -0.25, 0.75), Vec3::new(-1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx(hit.t_enter, 9.0);
        approx(hit.t_exit, 11.0);
    }

    #[test]
    fn grazing_edge_tangent_counts_as_hit() {
        let b = unit_box();
        // Travels along +x exactly on the top face y = 1 (tangent to the box).
        let ray = Ray::new(Vec3::new(-5.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("tangent hit");
        approx(hit.t_enter, 4.0);
        approx(hit.t_exit, 6.0);
    }

    #[test]
    fn grazing_just_outside_misses() {
        let b = unit_box();
        // Just above the top face: y = 1 + delta is outside the y-slab.
        let ray = Ray::new(Vec3::new(-5.0, 1.0 + 1.0e-3, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(b.intersect_ray(ray).is_none());
    }

    #[test]
    fn corner_tangent_point_chord() {
        // A ray that just grazes the (x=1, y=1) edge of the unit box: it touches
        // the box at a single point, so the chord collapses (t_enter == t_exit).
        let b = unit_box_0_1();
        let ray = Ray::new(Vec3::new(-1.0, 3.0, 0.5), Vec3::new(1.0, -1.0, 0.0));
        let hit = b.intersect_ray(ray).expect("corner-grazing hit");
        approx(hit.t_enter, 2.0);
        approx(hit.t_exit, 2.0);
        approx_vec(ray.at(hit.t_enter), Vec3::new(1.0, 1.0, 0.5));
    }

    #[test]
    fn negative_direction_hit_known_t() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx(hit.t_enter, 4.0);
        approx(hit.t_exit, 6.0);
        approx_vec(ray.at(hit.t_enter), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn negative_direction_along_y() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx(hit.t_enter, 9.0);
        approx(hit.t_exit, 11.0);
    }

    #[test]
    fn non_unit_direction_scales_t() {
        let b = unit_box();
        // Direction of length 2 along x halves the parameter values.
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("hit");
        approx(hit.t_enter, 2.0);
        approx(hit.t_exit, 3.0);
        approx_vec(ray.at(hit.t_enter), Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn diagonal_ray_through_center() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(1.0, 1.0, 1.0));
        let hit = b.intersect_ray(ray).expect("diagonal hit");
        // Enters where every axis reaches -1 at the same t, exits at +1.
        approx(hit.t_enter, 1.0);
        approx(hit.t_exit, 3.0);
        approx_vec(ray.at(hit.t_enter), Vec3::new(-1.0, -1.0, -1.0));
    }

    #[test]
    fn zero_direction_is_none() {
        let b = unit_box();
        let ray = Ray::new(Vec3::ZERO, Vec3::ZERO);
        assert!(b.intersect_ray(ray).is_none());
        assert!(b.intersect_line(ray).is_none());
        assert!(b.first_hit_t(ray).is_none());
    }

    #[test]
    fn zero_direction_from_outside_is_none() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);
        assert!(b.intersect_line(ray).is_none());
    }

    #[test]
    fn offset_box_hit_distance() {
        let b = Aabb::new(Vec3::new(98.0, -2.0, -2.0), Vec3::new(102.0, 2.0, 2.0));
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let hit = b.intersect_ray(ray).expect("far hit");
        approx(hit.t_enter, 98.0);
        approx(hit.t_exit, 102.0);
    }

    #[test]
    fn intersects_line_true_and_false() {
        let b = unit_box();
        let through = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let aside = Ray::new(Vec3::new(-5.0, 9.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(b.intersects_line(through));
        assert!(!b.intersects_line(aside));
    }

    #[test]
    fn intersects_ray_true_and_false() {
        let b = unit_box();
        let ahead = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let behind = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(b.intersects_ray(ahead));
        assert!(!b.intersects_ray(behind));
    }

    #[test]
    fn line_and_ray_agree_when_ahead() {
        let b = unit_box();
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let line = b.intersect_line(ray).expect("line");
        let fwd = b.intersect_ray(ray).expect("ray");
        approx(line.t_enter, fwd.t_enter);
        approx(line.t_exit, fwd.t_exit);
    }

    #[test]
    fn contains_reports_inside_and_outside() {
        let b = unit_box();
        assert!(b.contains(Vec3::ZERO));
        assert!(b.contains(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!b.contains(Vec3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn from_center_half_matches_corners() {
        let b = Aabb::from_center_half(Vec3::new(2.0, 0.0, 0.0), Vec3::splat(1.0));
        approx_vec(b.min, Vec3::new(1.0, -1.0, -1.0));
        approx_vec(b.max, Vec3::new(3.0, 1.0, 1.0));
        approx_vec(b.center(), Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn non_cube_box_thin_slab() {
        // Thin in y (half 0.5), long in x (half 3): a downward ray hits the top.
        let b = Aabb::new(Vec3::new(-3.0, -0.5, -1.0), Vec3::new(3.0, 0.5, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = b.intersect_ray(ray).expect("top hit");
        approx(hit.t_enter, 4.5);
        approx(hit.t_exit, 5.5);
        approx_vec(ray.at(hit.t_enter), Vec3::new(0.0, 0.5, 0.0));
    }

    #[test]
    fn no_nan_when_parallel_on_two_axes() {
        // Only z varies; x and y are parallel-inside. Result must be finite.
        let b = unit_box();
        let ray = Ray::new(Vec3::new(0.3, -0.4, -9.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = b.intersect_ray(ray).expect("hit");
        assert!(hit.t_enter.abs() < f32::INFINITY);
        assert!(hit.t_exit.abs() < f32::INFINITY);
        approx(hit.t_enter, 8.0);
        approx(hit.t_exit, 10.0);
    }
}
