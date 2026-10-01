//! Barrier-based frictional contact for hair, a real-time simplification of
//! Codimensional `IPC` (`C-IPC`, Li 2021).
//!
//! Incremental Potential Contact (`IPC`) guarantees interpenetration-free
//! stepping by adding a *log barrier* to the energy: as a contact distance `d`
//! shrinks toward the activation distance `d̂`, the barrier grows without bound
//! so the optimizer can never cross `d = 0`. The textbook `C-IPC` form is
//!
//! ```text
//!     b(d) = -(d - d̂)^2 * ln(d / d̂),   for 0 < d < d̂
//! ```
//!
//! which is C¹ at `d = d̂` (both the value and its first derivative vanish
//! there) and diverges as `d -> 0`. The design (§8.5 item 4) targets the
//! *real-time simplified tier*: a soft barrier plus projection plus a hard
//! distance floor, rather than the full implicit `IPC` Newton solve.
//!
//! ## Why this module does not use `ln`
//!
//! Prism's workspace `clippy.toml` bans `f32::ln`/`log`/`exp`/`powf`/`powi`
//! (and every trig/hyperbolic function) so that all math is libm-deterministic
//! and auditable. The original `C-IPC` barrier is therefore off-limits: it is
//! built on `ln(d / d̂)`. Instead this module uses a **rational/polynomial soft
//! barrier** that is closed-form, needs only `sqrt`/`abs`/`clamp`/`min`/`max`,
//! and reproduces the two properties that matter:
//!
//! ```text
//!     b(d)  = k * (d̂ - d)^2 * (1/d - 1/d̂),   for d_floor <= d < d̂
//!     b(d)  = 0,                              for d >= d̂
//! ```
//!
//! Evaluated on the clamped distance `d_e = clamp(d, d_floor, d̂)`:
//!
//! * `b(d̂) = 0` and `b'(d̂) = 0` because the `(d̂ - d)^2` factor and its
//!   derivative both vanish at `d = d̂`, so the barrier joins the free region
//!   with C¹ continuity (no force discontinuity when a contact activates).
//! * The rational factor `1/d - 1/d̂` is positive and grows as `d` shrinks, so
//!   `b` rises steeply toward the floor; the hard lower bound `d_floor` clamps
//!   the evaluation so the value stays large-but-finite (never `NaN`/infinity),
//!   which is the "projection + distance floor" that prevents tunnelling.
//!
//! The analytic first derivative (used for the repulsive force magnitude) is
//! hand-derived below in [`barrier_force_magnitude`].
//!
//! ## Friction
//!
//! Friction is the semi-implicit `Coulomb` model: the relative velocity is
//! projected onto the contact tangent plane, a friction impulse is applied to
//! cancel the tangential slide, and that impulse is clamped to the `Coulomb` cone
//! `|j_t| <= mu * j_n`. Below the cone the contact sticks (no residual slip);
//! on the cone it slips. All distribution is by inverse mass, so a pinned
//! endpoint (`inv_mass = 0`) never moves.
//!
//! The module is self-contained (its own [`Vec3`]), zero-dependency
//! (`core`/`alloc` only), `unsafe`-free, and panic-free: empty, degenerate, and
//! non-finite inputs are sanitized rather than trusted.

use alloc::vec::Vec;

/// Vectors whose squared length is below this are treated as zero-length, so
/// normalization and direction extraction never divide by ~0.
const EPS_LEN_SQ: f32 = 1.0e-24;

/// Denominators with magnitude below this are treated as degenerate.
const EPS_DENOM: f32 = 1.0e-12;

/// A minimal hand-written 3D vector; this module intentionally does not reuse
/// any other hair module's vector type so it stays independently auditable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The contact math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses a named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
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

    /// Cross (vector) product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }

    /// Replaces any non-finite component with `0`, so a poisoned external vector
    /// cannot smuggle `NaN`/infinity into the deterministic contact solve.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            finite_or(self.x, 0.0),
            finite_or(self.y, 0.0),
            finite_or(self.z, 0.0),
        )
    }
}

/// A contact endpoint: world position, world velocity, and inverse mass.
///
/// `inv_mass = 0` marks a pinned (infinite-mass) endpoint that never moves and
/// absorbs no impulse; corrections then fall entirely on the free partner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity.
    pub velocity: Vec3,
    /// Inverse mass (`0` = pinned / infinite mass).
    pub inv_mass: f32,
}

impl ContactPoint {
    /// Constructs a contact endpoint.
    #[must_use]
    pub const fn new(position: Vec3, velocity: Vec3, inv_mass: f32) -> Self {
        Self {
            position,
            velocity,
            inv_mass,
        }
    }

    /// Returns a copy with non-finite fields sanitized: positions/velocities
    /// lose any `NaN`/infinity, and inverse mass is clamped to a finite `>= 0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            position: self.position.sanitized(),
            velocity: self.velocity.sanitized(),
            inv_mass: sanitize_nonneg(self.inv_mass),
        }
    }
}

/// Tuning for the soft barrier and `Coulomb` friction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierParams {
    /// Activation distance `d̂`: contacts at or beyond this feel no barrier.
    pub dhat: f32,
    /// Barrier stiffness `k` (`>= 0`); scales the repulsive energy and force.
    pub stiffness: f32,
    /// Hard distance floor `d_floor` in `(0, d̂)`: the barrier is evaluated on
    /// `clamp(d, d_floor, d̂)`, so repulsion stays finite instead of diverging.
    pub d_floor: f32,
    /// `Coulomb` friction coefficient `mu` (`>= 0`).
    pub friction_mu: f32,
}

impl Default for BarrierParams {
    fn default() -> Self {
        Self {
            dhat: 1.0e-2,
            stiffness: 1.0,
            d_floor: 1.0e-3,
            friction_mu: 0.3,
        }
    }
}

impl BarrierParams {
    /// Returns a copy with every field forced into its valid range so no later
    /// computation can divide by zero or chase a `NaN`:
    ///
    /// * `dhat` becomes a finite `> 0` value (falling back to the default),
    /// * `d_floor` is clamped into the open interval `(0, dhat)`,
    /// * `stiffness` and `friction_mu` are clamped to finite `>= 0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let default = Self::default();
        let dhat = sanitize_pos(self.dhat, default.dhat);
        // Keep the floor strictly inside `(0, dhat)`. The upper guard
        // `dhat * 0.5` is always below `dhat`, and `min`/`max` are used instead
        // of `clamp` so the bounds can never be out of order (which would
        // panic): the floor is pulled under the upper guard and then lifted to
        // a strictly positive, finite value.
        let hi = dhat * 0.5;
        let floor_raw = sanitize_pos(self.d_floor, default.d_floor);
        let d_floor = floor_raw.min(hi).max(f32::MIN_POSITIVE);
        Self {
            dhat,
            stiffness: sanitize_nonneg(self.stiffness),
            d_floor,
            friction_mu: sanitize_nonneg(self.friction_mu),
        }
    }
}

/// Outcome of resolving one contact, returned for deterministic inspection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactResolution {
    /// Scalar normal (repulsive) impulse applied along the contact normal.
    pub normal_impulse: f32,
    /// Scalar friction impulse applied in the tangent plane.
    pub friction_impulse: f32,
    /// `true` when the friction impulse was clamped to the `Coulomb` cone
    /// (the contact slips); `false` when it stuck.
    pub slipping: bool,
}

impl ContactResolution {
    /// A no-op resolution (no normal or friction impulse, not slipping).
    pub const NONE: Self = Self {
        normal_impulse: 0.0,
        friction_impulse: 0.0,
        slipping: false,
    };
}

// --------------------------------------------------------------------------
// Closest-distance primitives.
// --------------------------------------------------------------------------

/// Euclidean distance between two points.
#[must_use]
pub fn point_point_distance(a: Vec3, b: Vec3) -> f32 {
    a.sanitized().sub(b.sanitized()).length()
}

/// Signed distance from `p` to the plane through `plane_point` with the given
/// normal. Positive on the side the normal points toward, negative behind it.
///
/// A (numerically) zero normal has no orientation, so the result is `0`.
#[must_use]
pub fn point_plane_signed_distance(p: Vec3, plane_point: Vec3, plane_normal: Vec3) -> f32 {
    let n = plane_normal.sanitized().normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return 0.0;
    }
    p.sanitized().sub(plane_point.sanitized()).dot(n)
}

/// Closest points between the segments `[p1, q1]` and `[p2, q2]`, with the
/// distance between them. The parameters `s, t` are clamped to `[0, 1]` so the
/// result lies on the segments, and degenerate (zero-length) segments are
/// handled without dividing by zero (Ericson, *Real-Time Collision Detection*).
///
/// Returns `(closest_on_first, closest_on_second, distance)`.
#[must_use]
pub fn segment_segment_closest(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3, f32) {
    let p1 = p1.sanitized();
    let q1 = q1.sanitized();
    let p2 = p2.sanitized();
    let q2 = q2.sanitized();

    let d1 = q1.sub(p1); // direction and length of segment 1
    let d2 = q2.sub(p2); // direction and length of segment 2
    let r = p1.sub(p2);
    let a = d1.length_squared(); // squared length of segment 1
    let e = d2.length_squared(); // squared length of segment 2
    let f = d2.dot(r);

    let seg1_degenerate = a <= EPS_LEN_SQ;
    let seg2_degenerate = e <= EPS_LEN_SQ;

    let (s, t) = if seg1_degenerate && seg2_degenerate {
        // Both segments are points.
        (0.0_f32, 0.0_f32)
    } else if seg1_degenerate {
        // First segment is a point: only slide along the second.
        (0.0_f32, clamp01(f / e))
    } else {
        let c = d1.dot(r);
        if seg2_degenerate {
            // Second segment is a point: only slide along the first.
            (clamp01(-c / a), 0.0_f32)
        } else {
            // General non-degenerate case.
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s0 = if denom.abs() > EPS_DENOM {
                clamp01((b * f - c * e) / denom)
            } else {
                // Parallel segments: pick an arbitrary point on segment 1 and
                // let the clamp on t below place the closest point.
                0.0_f32
            };
            // Compute t for this s, then re-clamp s if t left [0, 1].
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                (clamp01(-c / a), 0.0_f32)
            } else if t0 > 1.0 {
                (clamp01((b - c) / a), 1.0_f32)
            } else {
                (s0, t0)
            }
        }
    };

    let c1 = p1.add(d1.scale(s));
    let c2 = p2.add(d2.scale(t));
    (c1, c2, c1.sub(c2).length())
}

// --------------------------------------------------------------------------
// Soft barrier (rational, C^1, no ln/exp/powf).
// --------------------------------------------------------------------------

/// Soft-barrier energy `b(d)` for a contact distance `d`.
///
/// Evaluated on `d_e = clamp(d, d_floor, dhat)`:
///
/// ```text
///     b = k * (dhat - d_e)^2 * (1/d_e - 1/dhat)
/// ```
///
/// Returns `0` for `d >= dhat` (free region) and a large-but-finite value as
/// `d` approaches `d_floor`. Monotonically non-increasing in `d` on the active
/// window, and C¹ at `dhat`. Never returns `NaN`/infinity.
#[must_use]
pub fn barrier_energy(d: f32, params: BarrierParams) -> f32 {
    let p = params.sanitized();
    if !d.is_finite() {
        // A non-finite distance is treated as "far": no barrier.
        return 0.0;
    }
    if d >= p.dhat {
        return 0.0;
    }
    let de = d.clamp(p.d_floor, p.dhat);
    let gap = p.dhat - de; // >= 0
    let recip = 1.0 / de - 1.0 / p.dhat; // >= 0 since de <= dhat
    let energy = p.stiffness * (gap * gap) * recip;
    finite_or(energy.max(0.0), 0.0)
}

/// Magnitude of the repulsive barrier force `-b'(d)` (always `>= 0`, pointing
/// to increase separation).
///
/// With `u = (dhat - d_e)^2`, `u' = -2 (dhat - d_e)`, `v = 1/d_e - 1/dhat`, and
/// `v' = -1/d_e^2`, the product rule on `b = k * u * v` gives
///
/// ```text
///     b'(d_e) = k * [ u' * v + u * v' ]
///             = -k * (dhat - d_e) * [ 2 (1/d_e - 1/dhat) + (dhat - d_e)/d_e^2 ]
/// ```
///
/// so the repulsive magnitude is
///
/// ```text
///     -b'(d_e) = k * (dhat - d_e) * [ 2 (1/d_e - 1/dhat) + (dhat - d_e)/d_e^2 ]
/// ```
///
/// Both bracket terms are `>= 0` for `d_floor <= d_e <= dhat`, so the force is
/// non-negative; the `(dhat - d_e)` factor makes it vanish at `d = dhat` (C¹),
/// and clamping `d_e` at `d_floor` keeps it finite instead of diverging.
#[must_use]
pub fn barrier_force_magnitude(d: f32, params: BarrierParams) -> f32 {
    let p = params.sanitized();
    if !d.is_finite() {
        return 0.0;
    }
    if d >= p.dhat {
        return 0.0;
    }
    let de = d.clamp(p.d_floor, p.dhat);
    let gap = p.dhat - de; // >= 0
    let recip = 1.0 / de - 1.0 / p.dhat; // >= 0
    let inv_de = 1.0 / de;
    let bracket = 2.0 * recip + gap * (inv_de * inv_de);
    let force = p.stiffness * gap * bracket;
    finite_or(force.max(0.0), 0.0)
}

// --------------------------------------------------------------------------
// Contact resolution: repulsion + semi-implicit `Coulomb` friction.
// --------------------------------------------------------------------------

/// Resolves one contact in place: applies a barrier-driven positional push
/// along `normal` and a semi-implicit `Coulomb` friction impulse, both split by
/// inverse mass (a pinned endpoint stays put).
///
/// `normal` points from `b` toward `a` (the repulsion pushes `a` along
/// `+normal` and `b` along `-normal`). `distance` is the current contact gap;
/// `rel_velocity` is the velocity of `a` relative to `b` used for friction.
///
/// Returns the applied impulses and whether the contact slipped. The call is a
/// deterministic no-op (returning [`ContactResolution::NONE`]) when the normal
/// is degenerate, both endpoints are pinned, or the gap is at/beyond `dhat`.
pub fn resolve_contact(
    a: &mut ContactPoint,
    b: &mut ContactPoint,
    normal: Vec3,
    distance: f32,
    rel_velocity: Vec3,
    params: BarrierParams,
) -> ContactResolution {
    let p = params.sanitized();
    let sa = a.sanitized();
    let sb = b.sanitized();
    *a = sa;
    *b = sb;

    let n = normal.sanitized().normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return ContactResolution::NONE;
    }

    let w = a.inv_mass + b.inv_mass;
    if w <= 0.0 {
        // Both endpoints pinned: nothing can move.
        return ContactResolution::NONE;
    }

    let dist = finite_or(distance, p.dhat);
    if dist >= p.dhat {
        return ContactResolution::NONE;
    }

    // --- Normal repulsion -------------------------------------------------
    let jn = barrier_force_magnitude(dist, p);
    let inv_w = 1.0 / w;
    a.position = a.position.add(n.scale(jn * a.inv_mass * inv_w));
    b.position = b.position.sub(n.scale(jn * b.inv_mass * inv_w));

    // --- Semi-implicit `Coulomb` friction ----------------------------------
    let vrel = rel_velocity.sanitized();
    let vn = vrel.dot(n);
    let vt = vrel.sub(n.scale(vn)); // tangential relative velocity
    let speed = vt.length();
    let t_hat = vt.normalize_or_zero();

    let mut friction_impulse = 0.0;
    let mut slipping = false;
    if t_hat.length_squared() > EPS_LEN_SQ && jn > 0.0 {
        // Impulse that would exactly cancel the tangential slide.
        let jt_full = speed * inv_w;
        let jt_max = p.friction_mu * jn;
        let jt = jt_full.min(jt_max);
        slipping = jt_full > jt_max + EPS_DENOM;
        friction_impulse = jt;

        // Apply along -t_hat (opposing the slide), split by inverse mass.
        a.velocity = a.velocity.sub(t_hat.scale(jt * a.inv_mass));
        b.velocity = b.velocity.add(t_hat.scale(jt * b.inv_mass));
    }

    ContactResolution {
        normal_impulse: jn,
        friction_impulse,
        slipping,
    }
}

/// Convenience batch wrapper: resolves a list of contacts in index order and
/// collects their resolutions. Deterministic for a fixed input ordering.
///
/// Each tuple is `(index_a, index_b, normal, distance, rel_velocity)`; indices
/// outside `points` or referring to the same endpoint are skipped.
#[must_use]
pub fn resolve_contacts(
    points: &mut [ContactPoint],
    contacts: &[(usize, usize, Vec3, f32, Vec3)],
    params: BarrierParams,
) -> Vec<ContactResolution> {
    let mut out = Vec::with_capacity(contacts.len());
    let len = points.len();
    for &(ia, ib, normal, distance, rel_velocity) in contacts {
        if ia >= len || ib >= len || ia == ib {
            out.push(ContactResolution::NONE);
            continue;
        }
        // Take disjoint mutable borrows via split_at_mut.
        let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
        let (left, right) = points.split_at_mut(hi);
        let (first, second) = (&mut left[lo], &mut right[0]);
        let res = if ia < ib {
            resolve_contact(first, second, normal, distance, rel_velocity, params)
        } else {
            // Normal points from b->a; flip when the stored order is reversed.
            resolve_contact(second, first, normal, distance, rel_velocity, params)
        };
        out.push(res);
    }
    out
}

// --------------------------------------------------------------------------
// Scalar sanitizers and clamps.
// --------------------------------------------------------------------------

/// Returns `x` when finite, otherwise `fallback`.
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

/// Clamp to a finite `>= 0` value (non-finite -> `0`).
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Clamp to a finite `> 0` value, falling back to `default` otherwise.
fn sanitize_pos(x: f32, default: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        default
    }
}

/// Clamp a (possibly non-finite) value into `[0, 1]`.
fn clamp01(x: f32) -> f32 {
    finite_or(x, 0.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < T_EPS
    }

    fn vclose(a: Vec3, b: Vec3) -> bool {
        a.sub(b).length() < T_EPS
    }

    #[test]
    fn vec3_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert!(vclose(a.add(b), Vec3::new(5.0, 1.0, 3.5)));
        assert!(vclose(a.sub(b), Vec3::new(-3.0, 3.0, 2.5)));
        assert!(vclose(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0)));
        assert!(close(a.dot(b), 1.0 * 4.0 + 2.0 * -1.0 + 3.0 * 0.5));
        // x cross y = z for the standard basis.
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert!(vclose(x.cross(y), Vec3::new(0.0, 0.0, 1.0)));
        assert!(close(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
        assert!(close(Vec3::new(3.0, 4.0, 0.0).length_squared(), 25.0));
        let n = Vec3::new(0.0, 0.0, 5.0).normalize_or_zero();
        assert!(vclose(n, Vec3::new(0.0, 0.0, 1.0)));
        // Zero vector normalizes to zero (no `NaN`).
        assert!(vclose(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO));
    }

    #[test]
    fn point_point_distance_is_exact() {
        let d = point_point_distance(Vec3::new(1.0, 2.0, 2.0), Vec3::new(1.0, 2.0, 2.0 + 3.0));
        assert!(close(d, 3.0));
        let d2 = point_point_distance(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0));
        assert!(close(d2, 5.0));
    }

    #[test]
    fn point_plane_signed_distance_has_correct_sign() {
        let plane_point = Vec3::new(0.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        // Above the plane (along +normal): positive.
        assert!(close(
            point_plane_signed_distance(Vec3::new(5.0, 2.0, -3.0), plane_point, normal),
            2.0
        ));
        // Below the plane: negative.
        assert!(close(
            point_plane_signed_distance(Vec3::new(0.0, -1.5, 0.0), plane_point, normal),
            -1.5
        ));
        // Unnormalized normal still gives the true signed distance.
        assert!(close(
            point_plane_signed_distance(
                Vec3::new(0.0, 2.0, 0.0),
                plane_point,
                Vec3::new(0.0, 10.0, 0.0)
            ),
            2.0
        ));
        // Degenerate (zero) normal -> 0.
        assert!(close(
            point_plane_signed_distance(Vec3::new(0.0, 2.0, 0.0), plane_point, Vec3::ZERO),
            0.0
        ));
    }

    #[test]
    fn segment_segment_closest_general_case() {
        // Segment 1 along x at y=0; segment 2 along y at x=1, offset z=2.
        let (c1, c2, dist) = segment_segment_closest(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 2.0),
            Vec3::new(1.0, 1.0, 2.0),
        );
        assert!(vclose(c1, Vec3::new(1.0, 0.0, 0.0)));
        assert!(vclose(c2, Vec3::new(1.0, 0.0, 2.0)));
        assert!(close(dist, 2.0));
    }

    #[test]
    fn segment_segment_closest_parallel() {
        // Two parallel segments along x, separated by 1 in y.
        let (_, _, dist) = segment_segment_closest(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        );
        assert!(close(dist, 1.0));
    }

    #[test]
    fn segment_segment_closest_degenerate_is_point_point() {
        // Both segments collapsed to points.
        let (c1, c2, dist) = segment_segment_closest(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(4.0, 1.0, 1.0),
            Vec3::new(4.0, 1.0, 1.0),
        );
        assert!(vclose(c1, Vec3::new(1.0, 1.0, 1.0)));
        assert!(vclose(c2, Vec3::new(4.0, 1.0, 1.0)));
        assert!(close(dist, 3.0));
        // One point vs a segment: closest is the perpendicular foot.
        let (c1b, c2b, distb) = segment_segment_closest(
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        );
        assert!(vclose(c1b, Vec3::new(1.0, 5.0, 0.0)));
        assert!(vclose(c2b, Vec3::new(1.0, 0.0, 0.0)));
        assert!(close(distb, 5.0));
    }

    #[test]
    fn barrier_is_zero_at_and_above_dhat() {
        let p = BarrierParams {
            dhat: 0.01,
            stiffness: 1000.0,
            d_floor: 0.001,
            friction_mu: 0.3,
        };
        assert!(close(barrier_energy(p.dhat, p), 0.0));
        assert!(close(barrier_energy(p.dhat + 0.5, p), 0.0));
        assert!(close(barrier_force_magnitude(p.dhat, p), 0.0));
        assert!(close(barrier_force_magnitude(p.dhat * 2.0, p), 0.0));
    }

    #[test]
    fn barrier_is_c1_continuous_near_dhat() {
        let p = BarrierParams::default();
        // Just inside the activation distance, both value and force are small
        // (they vanish exactly at dhat), confirming C^1 join with the free
        // region.
        let d = p.dhat - 1.0e-6;
        assert!(barrier_energy(d, p) < 1.0e-3);
        assert!(barrier_force_magnitude(d, p) < 1.0e-1);
        assert!(barrier_energy(d, p) >= 0.0);
        assert!(barrier_force_magnitude(d, p) >= 0.0);
    }

    #[test]
    fn barrier_increases_as_distance_shrinks() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 10.0,
            d_floor: 0.001,
            friction_mu: 0.0,
        };
        let d_far = 0.015;
        let d_mid = 0.008;
        let d_near = 0.003;
        let e_far = barrier_energy(d_far, p);
        let e_mid = barrier_energy(d_mid, p);
        let e_near = barrier_energy(d_near, p);
        assert!(e_far < e_mid, "e_far={e_far} e_mid={e_mid}");
        assert!(e_mid < e_near, "e_mid={e_mid} e_near={e_near}");
        // Force magnitude is likewise monotone increasing toward the floor.
        let f_mid = barrier_force_magnitude(d_mid, p);
        let f_near = barrier_force_magnitude(d_near, p);
        assert!(f_mid < f_near, "f_mid={f_mid} f_near={f_near}");
    }

    #[test]
    fn barrier_is_clamped_and_finite_below_floor() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 10.0,
            d_floor: 0.002,
            friction_mu: 0.0,
        };
        let at_floor = barrier_force_magnitude(p.d_floor, p);
        let below = barrier_force_magnitude(p.d_floor * 0.1, p);
        let way_below = barrier_force_magnitude(-5.0, p);
        assert!(at_floor.is_finite() && at_floor > 0.0);
        // Clamped evaluation: below the floor matches the floor value exactly.
        assert!(close(below, at_floor), "below={below} at_floor={at_floor}");
        assert!(close(way_below, at_floor));
        assert!(barrier_energy(p.d_floor * 0.1, p).is_finite());
    }

    #[test]
    fn resolve_pushes_two_free_points_apart() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 50.0,
            d_floor: 0.001,
            friction_mu: 0.0,
        };
        // a above b along +y, gap 0.005 (< dhat).
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::ZERO, 1.0);
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        let before = point_point_distance(a.position, b.position);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let res = resolve_contact(&mut a, &mut b, normal, before, Vec3::ZERO, p);
        let after = point_point_distance(a.position, b.position);
        assert!(res.normal_impulse > 0.0);
        // Equal inverse mass: separation grows by exactly the normal impulse.
        assert!(after > before);
        assert!(
            close(after - before, res.normal_impulse),
            "d={}",
            after - before
        );
    }

    #[test]
    fn coulomb_friction_clamps_to_cone_when_slipping() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 1.0,
            d_floor: 0.001,
            friction_mu: 0.1,
        };
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::new(10.0, 0.0, 0.0), 1.0);
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        // Large tangential rel velocity, small mu -> cone-limited (slips).
        let rel = Vec3::new(10.0, 0.0, 0.0);
        let res = resolve_contact(&mut a, &mut b, Vec3::new(0.0, 1.0, 0.0), 0.005, rel, p);
        assert!(res.slipping, "should slip");
        assert!(close(
            res.friction_impulse,
            p.friction_mu * res.normal_impulse
        ));
        // Residual tangential relative velocity remains (not fully cancelled).
        let vrel_t = a.velocity.sub(b.velocity).x;
        assert!(vrel_t > 1.0, "residual slide should remain: {vrel_t}");
    }

    #[test]
    fn coulomb_friction_sticks_for_small_tangential() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 1000.0,
            d_floor: 0.001,
            friction_mu: 2.0,
        };
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::new(0.01, 0.0, 0.0), 1.0);
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        // Tiny tangential velocity, big mu & stiffness -> within the cone (sticks).
        let rel = Vec3::new(0.01, 0.0, 0.0);
        let res = resolve_contact(&mut a, &mut b, Vec3::new(0.0, 1.0, 0.0), 0.005, rel, p);
        assert!(!res.slipping, "should stick");
        // Tangential relative velocity is driven to ~0.
        let vrel_t = a.velocity.sub(b.velocity).x;
        assert!(
            vrel_t.abs() < T_EPS,
            "tangential slide not cancelled: {vrel_t}"
        );
    }

    #[test]
    fn pinned_endpoint_does_not_move() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 50.0,
            d_floor: 0.001,
            friction_mu: 0.5,
        };
        let b_start_pos = Vec3::new(0.0, 0.0, 0.0);
        let b_start_vel = Vec3::ZERO;
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::new(5.0, 0.0, 0.0), 1.0);
        let mut b = ContactPoint::new(b_start_pos, b_start_vel, 0.0); // pinned
        let res = resolve_contact(
            &mut a,
            &mut b,
            Vec3::new(0.0, 1.0, 0.0),
            0.005,
            Vec3::new(5.0, 0.0, 0.0),
            p,
        );
        assert!(res.normal_impulse > 0.0);
        assert!(
            vclose(b.position, b_start_pos),
            "pinned moved: {:?}",
            b.position
        );
        assert!(vclose(b.velocity, b_start_vel), "pinned velocity changed");
        // The free partner absorbs the whole correction.
        assert!(a.position.y > 0.005);
    }

    #[test]
    fn both_pinned_is_noop() {
        let p = BarrierParams::default();
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::ZERO, 0.0);
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 0.0);
        let a0 = a;
        let b0 = b;
        let res = resolve_contact(
            &mut a,
            &mut b,
            Vec3::new(0.0, 1.0, 0.0),
            0.005,
            Vec3::ZERO,
            p,
        );
        assert_eq!(res, ContactResolution::NONE);
        assert_eq!(a, a0);
        assert_eq!(b, b0);
    }

    #[test]
    fn nonfinite_inputs_are_sanitized_without_panic() {
        let bad = BarrierParams {
            dhat: f32::NAN,
            stiffness: f32::INFINITY,
            d_floor: -1.0,
            friction_mu: f32::NAN,
        };
        // Sanitized params land in valid ranges.
        let s = bad.sanitized();
        assert!(s.dhat.is_finite() && s.dhat > 0.0);
        assert!(s.d_floor > 0.0 && s.d_floor < s.dhat);
        assert!(s.stiffness.is_finite() && s.stiffness >= 0.0);
        assert!(s.friction_mu.is_finite() && s.friction_mu >= 0.0);

        // Barrier functions never return `NaN`/inf even with poisoned distance.
        assert!(barrier_energy(f32::NAN, bad).is_finite());
        assert!(barrier_force_magnitude(f32::INFINITY, bad).is_finite());

        // Resolve with poisoned positions/velocities/normal: no panic.
        let mut a = ContactPoint::new(
            Vec3::new(f32::NAN, 0.005, 0.0),
            Vec3::new(f32::INFINITY, 0.0, 0.0),
            f32::NAN,
        );
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        let res = resolve_contact(
            &mut a,
            &mut b,
            Vec3::new(f32::NAN, 1.0, 0.0),
            f32::NAN,
            Vec3::new(f32::NAN, 0.0, 0.0),
            bad,
        );
        assert!(res.normal_impulse.is_finite());
        assert!(res.friction_impulse.is_finite());
        assert!(
            a.position.sanitized() == a.position,
            "position should be finite"
        );
    }

    #[test]
    fn degenerate_normal_is_noop() {
        let p = BarrierParams::default();
        let mut a = ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::ZERO, 1.0);
        let mut b = ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        let a0 = a;
        let b0 = b;
        let res = resolve_contact(&mut a, &mut b, Vec3::ZERO, 0.005, Vec3::ZERO, p);
        assert_eq!(res, ContactResolution::NONE);
        assert_eq!(a, a0);
        assert_eq!(b, b0);
    }

    #[test]
    fn resolve_contacts_batch_skips_bad_indices_and_pushes_apart() {
        let p = BarrierParams {
            dhat: 0.02,
            stiffness: 50.0,
            d_floor: 0.001,
            friction_mu: 0.0,
        };
        let mut points = [
            ContactPoint::new(Vec3::new(0.0, 0.005, 0.0), Vec3::ZERO, 1.0),
            ContactPoint::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 1.0),
        ];
        let before = point_point_distance(points[0].position, points[1].position);
        let contacts = [
            (0usize, 1usize, Vec3::new(0.0, 1.0, 0.0), before, Vec3::ZERO),
            (5usize, 9usize, Vec3::new(0.0, 1.0, 0.0), 0.001, Vec3::ZERO), // out of range
            (0usize, 0usize, Vec3::new(0.0, 1.0, 0.0), 0.001, Vec3::ZERO), // same index
        ];
        let res = resolve_contacts(&mut points, &contacts, p);
        assert_eq!(res.len(), 3);
        assert!(res[0].normal_impulse > 0.0);
        assert_eq!(res[1], ContactResolution::NONE);
        assert_eq!(res[2], ContactResolution::NONE);
        let after = point_point_distance(points[0].position, points[1].position);
        assert!(after > before);
    }
}
