//! Continuous (swept) axis-aligned-bounding-box *time-of-impact* (`TOI`)
//! primitive for the particle broad phase (design §10, §13).
//!
//! This module answers one narrow, purely geometric question: given two
//! axis-aligned boxes, each moving at a constant velocity over the unit
//! timestep `t ∈ [0, 1]`, at what fraction of the step do they *first* touch,
//! and along which axis / with which contact normal? It is the canonical
//! real-time-rendering *swept-`AABB`* continuous-collision test, solved with the
//! separating-axis "slab" method: the relative motion turns the pair into a
//! moving point against a Minkowski-expanded box, and each axis contributes an
//! entry/exit time interval whose intersection is the contact window. The
//! algorithm is re-derived here rather than copied from any engine.
//!
//! # Relationship to the sibling modules (strict boundary)
//!
//! Several files in this subsystem touch boxes and contacts; this one is
//! deliberately disjoint:
//!
//! * [`crate::particle::collision`] owns the particle-versus-environment
//!   **collision response** layer (push-out, restitution, Coulomb friction). It
//!   *consumes* a broad-phase proximity/time-of-impact result like the one
//!   produced here; this module never integrates velocity, resolves a contact,
//!   or applies any response. It only reports *when* and *where* two swept boxes
//!   first meet.
//! * [`crate::particle::bvh`] builds and traverses a **hierarchical** bounding
//!   volume tree; this module is a single leaf-level pair test with no
//!   hierarchy.
//! * [`crate::particle::bounds`] reduces a live particle pool down to one tight
//!   box, and [`crate::particle::aabb_transform`] transforms a box under an
//!   affine matrix and provides static box-vs-box set algebra. Neither performs
//!   a *swept* (time-parameterised) test; this module does only that.
//!
//! In short: this file computes the continuous time-of-impact of two uniformly
//! moving `AABB`s, and nothing else.
//!
//! # Determinism
//!
//! Everything is a zero-dependency contract with hand-rolled vector math. The
//! only floating-point primitives used are ordinary `+ - * /`, `f32::abs`,
//! `f32::min`, `f32::max`, and the [`f32::INFINITY`] sentinel for an axis with
//! no relative motion. No `sqrt` and no transcendental function is ever called,
//! so the `CPU` reference here is deterministic and a future `GPU` kernel
//! evaluating the same pair produces matching results. Floating-point `==` /
//! `!=` are never used; a near-zero relative velocity is detected against
//! [`EPS`].

/// Absolute tolerance used to guard divisions and to detect a near-zero
/// relative velocity without ever writing an exact `==` / `!=` on an `f32`.
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
        reason = "The particle math API is specified with named add/sub/mul/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
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

    /// Component-wise absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

/// A self-contained axis-aligned bounding box (`AABB`).
///
/// Callers are expected to keep `min <= max` on every axis; the sweep treats
/// the box as given, so a well-formed box yields a well-formed result.
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

    /// Translates the whole box by `d`.
    #[must_use]
    pub fn translate(self, d: Vec3) -> Self {
        Self::new(self.min.add(d), self.max.add(d))
    }

    /// Whether this box overlaps `other` (inclusive touching counts as overlap).
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
}

/// An axis-aligned box moving at a constant velocity over the unit timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovingAabb {
    /// The box at the start of the timestep (`t = 0`).
    pub aabb: Aabb,
    /// Constant per-step velocity: the box translates by `vel * t`.
    pub vel: Vec3,
}

impl MovingAabb {
    /// Builds a moving box from its start pose and velocity.
    #[must_use]
    pub const fn new(aabb: Aabb, vel: Vec3) -> Self {
        Self { aabb, vel }
    }

    /// The box at time `t` along the step, translated by `vel * t`.
    #[must_use]
    pub fn at(self, t: f32) -> Aabb {
        self.aabb.translate(self.vel.scale(t))
    }
}

/// The result of a successful swept `AABB` time-of-impact query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepResult {
    /// First-contact fraction of the timestep, clamped to `[0, 1]`.
    pub toi: f32,
    /// Contact normal on `b`'s surface pointing toward `a` (the push-out
    /// direction), a unit axis vector. It is [`Vec3::ZERO`] when the boxes are
    /// already overlapping at `t = 0` (an ambiguous resting contact).
    pub normal: Vec3,
    /// Index of the last axis to enter contact: `0 = x`, `1 = y`, `2 = z`.
    pub axis: usize,
    /// Whether the boxes already overlap at the start of the step.
    pub initially_overlapping: bool,
}

/// Per-axis entry/exit time interval for the moving-point-vs-slab test.
///
/// Returns [`None`] when the axis has no relative motion *and* the boxes are
/// already separated on it, which means they can never overlap. Otherwise
/// returns `(entry, exit)`; an axis with no relative motion but current overlap
/// yields the unbounded interval `(-INF, +INF)`.
fn axis_interval(a_min: f32, a_max: f32, b_min: f32, b_max: f32, v: f32) -> Option<(f32, f32)> {
    if v.abs() < EPS {
        if a_max < b_min || a_min > b_max {
            None
        } else {
            Some((-f32::INFINITY, f32::INFINITY))
        }
    } else if v > 0.0 {
        let inv = 1.0 / v;
        Some(((b_min - a_max) * inv, (b_max - a_min) * inv))
    } else {
        let inv = 1.0 / v;
        Some(((b_max - a_min) * inv, (b_min - a_max) * inv))
    }
}

/// Builds the positive or negative unit vector along `axis`.
fn axis_normal(axis: usize, positive: bool) -> Vec3 {
    let s = if positive { 1.0 } else { -1.0 };
    match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    }
}

/// Computes the continuous time-of-impact of two uniformly moving `AABB`s over
/// the unit timestep `t ∈ [0, 1]`.
///
/// The pair is reduced to `a`'s box moving with the relative velocity
/// `a.vel - b.vel` against a stationary `b`. Each axis contributes an
/// entry/exit interval; the boxes overlap exactly when all three intervals
/// overlap simultaneously, so the first contact is the largest per-axis entry
/// time and the separation is the smallest per-axis exit time.
///
/// Returns [`Some`] with the contact fraction, normal, and axis when the boxes
/// touch within the step, and [`None`] when they never touch in `[0, 1]`
/// (moving apart, contact strictly after the step, or a genuine near-miss where
/// the axis windows do not coincide). A pair already overlapping at `t = 0`
/// returns `toi == 0.0` with `initially_overlapping == true` and a zero normal.
#[must_use]
pub fn swept_toi(a: MovingAabb, b: MovingAabb) -> Option<SweepResult> {
    let rv = a.vel.sub(b.vel);

    let (ex_e, ex_x) = axis_interval(a.aabb.min.x, a.aabb.max.x, b.aabb.min.x, b.aabb.max.x, rv.x)?;
    let (ey_e, ey_x) = axis_interval(a.aabb.min.y, a.aabb.max.y, b.aabb.min.y, b.aabb.max.y, rv.y)?;
    let (ez_e, ez_x) = axis_interval(a.aabb.min.z, a.aabb.max.z, b.aabb.min.z, b.aabb.max.z, rv.z)?;

    let t_entry = ex_e.max(ey_e).max(ez_e);
    let t_exit = ex_x.min(ey_x).min(ez_x);

    if t_entry > t_exit || t_entry > 1.0 || t_exit < 0.0 {
        return None;
    }

    // The last axis to enter contact carries the separating normal.
    let mut axis = 0usize;
    let mut best = ex_e;
    if ey_e > best {
        best = ey_e;
        axis = 1;
    }
    if ez_e > best {
        axis = 2;
    }

    let initially_overlapping = t_entry <= 0.0;
    let toi = t_entry.max(0.0);

    let rel_axis = match axis {
        0 => rv.x,
        1 => rv.y,
        _ => rv.z,
    };
    let normal = if initially_overlapping || rel_axis.abs() < EPS {
        Vec3::ZERO
    } else if rel_axis > 0.0 {
        axis_normal(axis, false)
    } else {
        axis_normal(axis, true)
    };

    Some(SweepResult {
        toi,
        normal,
        axis,
        initially_overlapping,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only tolerance for comparing floating-point results.
    const TEST_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn box3(min: Vec3, max: Vec3) -> Aabb {
        Aabb::new(min, max)
    }

    fn unit() -> Aabb {
        box3(Vec3::ZERO, Vec3::splat(1.0))
    }

    #[test]
    fn head_on_hit_at_half_step() {
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("head-on approach must hit");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 0);
        assert!(vec_approx(r.normal, Vec3::new(-1.0, 0.0, 0.0)));
        assert!(!r.initially_overlapping);
    }

    #[test]
    fn head_on_hit_at_quarter_step() {
        let a = MovingAabb::new(unit(), Vec3::new(4.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("faster approach still hits");
        assert!(approx(r.toi, 0.25));
    }

    #[test]
    fn contact_exactly_at_end_of_step_hits() {
        let a = MovingAabb::new(unit(), Vec3::new(1.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("contact at t == 1 is inside the step");
        assert!(approx(r.toi, 1.0));
    }

    #[test]
    fn contact_beyond_step_does_not_hit() {
        let a = MovingAabb::new(unit(), Vec3::new(0.5, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        // Gap of 1.0 at speed 0.5 => contact at t == 2.0, outside [0, 1].
        assert!(swept_toi(a, b).is_none());
    }

    #[test]
    fn receding_boxes_never_hit() {
        let a = MovingAabb::new(unit(), Vec3::new(-1.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        assert!(swept_toi(a, b).is_none());
    }

    #[test]
    fn separated_static_boxes_do_not_hit() {
        let a = MovingAabb::new(unit(), Vec3::ZERO);
        let b = MovingAabb::new(
            box3(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        assert!(swept_toi(a, b).is_none());
    }

    #[test]
    fn static_overlapping_boxes_report_zero_toi() {
        let a = MovingAabb::new(unit(), Vec3::ZERO);
        let b = MovingAabb::new(
            box3(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("overlapping boxes contact at t == 0");
        assert!(approx(r.toi, 0.0));
        assert!(r.initially_overlapping);
        assert!(vec_approx(r.normal, Vec3::ZERO));
    }

    #[test]
    fn initially_overlapping_while_moving_reports_zero_toi() {
        let a = MovingAabb::new(unit(), Vec3::new(3.0, 1.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("already overlapping at start");
        assert!(approx(r.toi, 0.0));
        assert!(r.initially_overlapping);
        assert!(vec_approx(r.normal, Vec3::ZERO));
    }

    #[test]
    fn approach_from_positive_x_has_positive_x_normal() {
        let a = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::new(-2.0, 0.0, 0.0),
        );
        let b = MovingAabb::new(unit(), Vec3::ZERO);
        let r = swept_toi(a, b).expect("approaching from +x hits");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 0);
        assert!(vec_approx(r.normal, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn approach_from_below_has_negative_y_normal() {
        let a = MovingAabb::new(
            box3(Vec3::new(0.0, -2.0, 0.0), Vec3::new(1.0, -1.0, 1.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let b = MovingAabb::new(unit(), Vec3::ZERO);
        let r = swept_toi(a, b).expect("approaching from -y hits");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 1);
        assert!(vec_approx(r.normal, Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn approach_from_above_has_positive_y_normal() {
        let a = MovingAabb::new(
            box3(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 3.0, 1.0)),
            Vec3::new(0.0, -2.0, 0.0),
        );
        let b = MovingAabb::new(unit(), Vec3::ZERO);
        let r = swept_toi(a, b).expect("approaching from +y hits");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 1);
        assert!(vec_approx(r.normal, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn approach_along_z_has_negative_z_normal() {
        let a = MovingAabb::new(
            box3(Vec3::new(0.0, 0.0, -2.0), Vec3::new(1.0, 1.0, -1.0)),
            Vec3::new(0.0, 0.0, 2.0),
        );
        let b = MovingAabb::new(unit(), Vec3::ZERO);
        let r = swept_toi(a, b).expect("approaching from -z hits");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 2);
        assert!(vec_approx(r.normal, Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn zero_relative_velocity_axis_does_not_block_hit() {
        // No relative motion on y/z but boxes overlap there; x drives the hit.
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.5, 0.0), Vec3::new(3.0, 1.5, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("stationary overlapping axes must not veto the hit");
        assert!(approx(r.toi, 0.5));
        assert_eq!(r.axis, 0);
    }

    #[test]
    fn zero_relative_velocity_axis_separated_blocks_hit() {
        // y has no relative motion and is separated => never overlaps.
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 5.0, 0.0), Vec3::new(3.0, 6.0, 1.0)),
            Vec3::ZERO,
        );
        assert!(swept_toi(a, b).is_none());
    }

    #[test]
    fn both_boxes_moving_toward_each_other_hit() {
        let a = MovingAabb::new(unit(), Vec3::new(1.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::new(-1.0, 0.0, 0.0),
        );
        let r = swept_toi(a, b).expect("closing at relative speed 2 must hit");
        assert!(approx(r.toi, 0.5));
        assert!(vec_approx(r.normal, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn faster_box_catches_slower_box_same_direction() {
        let a = MovingAabb::new(unit(), Vec3::new(3.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let r = swept_toi(a, b).expect("relative speed 2 closes the unit gap");
        assert!(approx(r.toi, 0.5));
    }

    #[test]
    fn diagonal_near_miss_does_not_hit() {
        // Enters x-slab at t == 1 but has already left the y-slab at t == 0.5.
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 2.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(3.0, 0.0, 0.0), Vec3::new(4.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        assert!(swept_toi(a, b).is_none());
    }

    #[test]
    fn grazing_corner_contact_is_a_hit() {
        // At t == 1 box a is [1,2] and box b touches it along a single corner.
        let a = MovingAabb::new(unit(), Vec3::new(1.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 1.0, 0.0), Vec3::new(3.0, 2.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("tangential corner touch still registers");
        assert!(approx(r.toi, 1.0));
    }

    #[test]
    fn swapping_arguments_preserves_toi() {
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let ab = swept_toi(a, b).expect("a->b hits");
        let ba = swept_toi(b, a).expect("b->a hits");
        assert!(approx(ab.toi, ba.toi));
    }

    #[test]
    fn swapping_arguments_negates_normal() {
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let ab = swept_toi(a, b).expect("a->b hits");
        let ba = swept_toi(b, a).expect("b->a hits");
        assert!(vec_approx(ab.normal, ba.normal.neg()));
    }

    #[test]
    fn translation_invariance_of_toi() {
        let off = Vec3::new(10.0, -5.0, 3.0);
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let a2 = MovingAabb::new(a.aabb.translate(off), a.vel);
        let b2 = MovingAabb::new(b.aabb.translate(off), b.vel);
        let r = swept_toi(a, b).expect("base hit");
        let r2 = swept_toi(a2, b2).expect("translated hit");
        assert!(approx(r.toi, r2.toi));
    }

    #[test]
    fn translation_invariance_of_normal() {
        let off = Vec3::new(-2.5, 7.0, -1.0);
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let a2 = MovingAabb::new(a.aabb.translate(off), a.vel);
        let b2 = MovingAabb::new(b.aabb.translate(off), b.vel);
        let r = swept_toi(a, b).expect("base hit");
        let r2 = swept_toi(a2, b2).expect("translated hit");
        assert!(vec_approx(r.normal, r2.normal));
    }

    #[test]
    fn precise_toi_value_is_one_third() {
        let a = MovingAabb::new(unit(), Vec3::new(3.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("unit gap at speed 3 hits at 1/3");
        assert!(approx(r.toi, 1.0 / 3.0));
    }

    #[test]
    fn contact_axis_is_reported_for_y() {
        let a = MovingAabb::new(
            box3(Vec3::new(0.0, -2.0, 0.0), Vec3::new(1.0, -1.0, 1.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let b = MovingAabb::new(unit(), Vec3::ZERO);
        let r = swept_toi(a, b).expect("y-axis approach hits");
        assert_eq!(r.axis, 1);
    }

    #[test]
    fn contact_geometry_matches_toi() {
        // The moving box, advanced to the reported toi, must touch the target.
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, b).expect("hit");
        let touched = a.at(r.toi);
        assert!(touched.overlaps(b.aabb));
        assert!(approx(touched.max.x, 2.0));
    }

    #[test]
    fn moving_aabb_at_translates_by_velocity() {
        let m = MovingAabb::new(unit(), Vec3::new(2.0, -4.0, 6.0));
        let at_half = m.at(0.5);
        assert!(vec_approx(at_half.min, Vec3::new(1.0, -2.0, 3.0)));
        assert!(vec_approx(at_half.max, Vec3::new(2.0, -1.0, 4.0)));
    }

    #[test]
    fn overlaps_helper_detects_overlap_and_separation() {
        let a = unit();
        let overlapping = box3(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        let separated = box3(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(a.overlaps(overlapping));
        assert!(!a.overlaps(separated));
    }

    #[test]
    fn vector_add_sub_neg_scale_are_lanewise() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert!(vec_approx(a.add(b), Vec3::new(5.0, 1.0, 3.5)));
        assert!(vec_approx(a.sub(b), Vec3::new(-3.0, 3.0, 2.5)));
        assert!(vec_approx(a.neg(), Vec3::new(-1.0, -2.0, -3.0)));
        assert!(vec_approx(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0)));
    }

    #[test]
    fn vector_mul_min_max_abs_are_lanewise() {
        let a = Vec3::new(2.0, -3.0, 4.0);
        let b = Vec3::new(5.0, 6.0, -7.0);
        assert!(vec_approx(a.mul(b), Vec3::new(10.0, -18.0, -28.0)));
        assert!(vec_approx(a.min(b), Vec3::new(2.0, -3.0, -7.0)));
        assert!(vec_approx(a.max(b), Vec3::new(5.0, 6.0, 4.0)));
        assert!(vec_approx(a.abs(), Vec3::new(2.0, 3.0, 4.0)));
    }

    #[test]
    fn from_center_half_extent_round_trips() {
        let bx = Aabb::from_center_half_extent(Vec3::new(1.0, 2.0, 3.0), Vec3::splat(0.5));
        assert!(vec_approx(bx.center(), Vec3::new(1.0, 2.0, 3.0)));
        assert!(vec_approx(bx.half_extent(), Vec3::splat(0.5)));
    }

    #[test]
    fn contact_normal_is_zero_only_on_initial_overlap() {
        let a = MovingAabb::new(unit(), Vec3::new(2.0, 0.0, 0.0));
        let b = MovingAabb::new(
            box3(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0)),
            Vec3::ZERO,
        );
        let clean = swept_toi(a, b).expect("clean hit");
        assert!(!vec_approx(clean.normal, Vec3::ZERO));

        let overlap = MovingAabb::new(
            box3(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)),
            Vec3::ZERO,
        );
        let r = swept_toi(a, overlap).expect("initial overlap");
        assert!(vec_approx(r.normal, Vec3::ZERO));
    }
}
