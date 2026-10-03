//! Continuous collision detection (CCD) for soft-body particles.
//!
//! The substep XPBD solver only projects particles out of the analytic
//! [`BodyCollider`] proxies at their *end-of-step* position. For a thin garment
//! moving fast against a thin collider that is not enough: a particle can start
//! in front of a wall and finish behind it inside a single substep, tunnelling
//! through without the discrete projection ever seeing an overlap. This module
//! closes that gap by sweeping the segment `prev -> curr` of every free
//! particle against each body collider, solving for the earliest time of impact
//! (TOI) along the segment, snapping the particle onto the surface with a skin
//! offset, reflecting its normal velocity by a restitution coefficient, and
//! rubbing the tangential slide with position-level Coulomb friction.
//!
//! The TOI solvers are exact closed forms: a sphere is a single quadratic, a
//! half-space is linear, and a capsule is the union of an infinite cylinder
//! (restricted to the segment slab) with a sphere at each end cap. Only
//! [`f32::sqrt`] is used; there are no transcendental calls. Every routine is
//! `O(1)` per `(particle, collider)` pair, so [`resolve_ccd`] is
//! `O(particles * colliders)` with no hidden inner loops, and it is fully
//! deterministic (particles in index order, colliders in slice order, the
//! earliest valid hit wins). Pinned particles (`inverse_mass <= 0`) never move
//! and no path can produce a [`f32::NAN`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! closed-form swept-primitive TOI solvers are standard analytic
//! continuous-collision geometry; the tangential-friction projection reuses the
//! shared primitive published by Macklin et al. (2014), "Unified Particle
//! Physics for Real-Time Applications".

use glam::{Quat, Vec3};

use crate::math::scalar::Real;

use super::friction::{apply_coulomb_friction, sanitize_friction};
use super::{closest_point_on_segment, BodyCollider, EPS_LEN_SQ};

/// Numerical floor for treating a scalar coefficient as zero when classifying a
/// quadratic as linear or a segment as degenerate.
const EPS_COEF: Real = 1e-12;

/// Tuning for the continuous-collision sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CcdParams {
    /// How far outside the collider surface (along the outward normal) a
    /// particle is placed after a hit, so the next substep starts strictly
    /// outside and does not immediately re-penetrate. Non-negative; a value of
    /// zero snaps exactly to the surface.
    pub skin: Real,
    /// Normal restitution in `0..=1`: `0` is a fully inelastic stop (the
    /// inbound normal velocity is cancelled) and `1` is a perfect bounce (the
    /// normal velocity is mirrored). Values are clamped into range.
    pub restitution: Real,
    /// Master switch; when `false`, [`resolve_ccd`] is a no-op so callers can
    /// disable the sweep without restructuring the pipeline.
    pub enabled: bool,
}

impl Default for CcdParams {
    /// A conservative default: a small skin, no bounce, sweep enabled.
    fn default() -> Self {
        Self {
            skin: 1e-3,
            restitution: 0.0,
            enabled: true,
        }
    }
}

impl CcdParams {
    /// Returns a copy with `skin` forced non-negative, `restitution` clamped to
    /// `0..=1`, and any [`f32::NAN`] replaced by a safe value, so a
    /// mis-authored asset can never inject a `NaN` or a negative skin into the
    /// sweep.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let skin = if self.skin.is_nan() || self.skin < 0.0 {
            0.0
        } else {
            self.skin
        };
        let restitution = if self.restitution.is_nan() {
            0.0
        } else {
            self.restitution.clamp(0.0, 1.0)
        };
        Self {
            skin,
            restitution,
            enabled: self.enabled,
        }
    }
}

/// Returns the earliest time `t` in `0..=1` at which the point moving along
/// `prev -> curr` is on or inside the sphere `(center, radius)`, or [`None`]
/// when the swept segment never reaches the sphere.
///
/// The point path is `p(t) = prev + t * (curr - prev)`. Substituting into
/// `|p(t) - center|^2 == radius^2` gives a quadratic whose earlier root is the
/// entry crossing. A point that already starts on or inside the sphere reports
/// `t == 0`. A non-positive radius makes the sphere inert ([`None`]).
#[must_use]
pub fn sphere_toi(prev: Vec3, curr: Vec3, center: Vec3, radius: Real) -> Option<Real> {
    if radius <= 0.0 {
        return None;
    }
    let m = curr - prev;
    let e = prev - center;
    let a = m.dot(m);
    let b = 2.0 * e.dot(m);
    let c = e.dot(e) - radius * radius;
    first_entry_time(a, b, c)
}

/// Returns the earliest time `t` in `0..=1` at which the point moving along
/// `prev -> curr` crosses into the infeasible side of the half-space
/// `normal.dot(x) >= offset`, or [`None`] when it stays in front for the whole
/// segment.
///
/// The signed distance `s(t) = normal.dot(p(t)) - offset` is linear in `t`. The
/// crossing is where `s(t) == 0` while `s` is decreasing. A point that starts
/// behind the plane reports `t == 0`. A (near) zero normal has no defined plane
/// and returns [`None`].
#[must_use]
pub fn half_space_toi(prev: Vec3, curr: Vec3, normal: Vec3, offset: Real) -> Option<Real> {
    if normal.length_squared() <= EPS_LEN_SQ {
        return None;
    }
    let s0 = normal.dot(prev) - offset;
    if s0 <= 0.0 {
        return Some(0.0);
    }
    let ds = normal.dot(curr - prev);
    if ds >= -EPS_COEF {
        // Parallel to the plane or moving away: never crosses.
        return None;
    }
    let t = -s0 / ds;
    if t <= 1.0 {
        Some(t.max(0.0))
    } else {
        None
    }
}

/// Returns the earliest time `t` in `0..=1` at which the point moving along
/// `prev -> curr` is on or inside the capsule (segment `p0`..`p1` inflated by
/// `radius`), or [`None`] when the swept segment misses it.
///
/// A capsule is the union of an infinite cylinder about the segment axis with a
/// sphere at each end cap. This computes the cylinder entry time restricted to
/// the axis slab `0..=len` and the entry time of each end-cap sphere, then
/// returns the earliest of those. A collapsed capsule (`p0 == p1`) degenerates
/// to a single sphere. A non-positive radius makes the capsule inert.
#[must_use]
pub fn capsule_toi(prev: Vec3, curr: Vec3, p0: Vec3, p1: Vec3, radius: Real) -> Option<Real> {
    if radius <= 0.0 {
        return None;
    }
    let axis = p1 - p0;
    let len_sq = axis.length_squared();
    if len_sq <= EPS_LEN_SQ {
        // Degenerate capsule behaves like a sphere at `p0`.
        return sphere_toi(prev, curr, p0, radius);
    }
    let mut best = cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = earliest(best, sphere_toi(prev, curr, p0, radius));
    best = earliest(best, sphere_toi(prev, curr, p1, radius));
    best
}

/// Sweeps `prev -> curr` against the infinite cylinder about the axis through
/// `p0` with (unnormalized) direction `axis`, and returns the earliest entry
/// time whose contact projects onto the segment slab `0..=len`, or [`None`].
///
/// The perpendicular distance to the axis line is quadratic in `t`; its
/// sub-`radius` interval is intersected with the time interval during which the
/// axial projection lies within the slab and with `0..=1`. The lower bound of
/// the resulting interval is the earliest cylindrical-side contact; the end
/// caps are handled separately by [`capsule_toi`].
fn cylinder_slab_toi(prev: Vec3, curr: Vec3, p0: Vec3, axis: Vec3, radius: Real) -> Option<Real> {
    let len = axis.length();
    if len <= EPS_COEF {
        return None;
    }
    let u = axis * (1.0 / len);
    let e0 = prev - p0;
    let m = curr - prev;
    let mu = m.dot(u);
    let e0u = e0.dot(u);

    // Radial interval [rad_lo, rad_hi] where perpendicular distance <= radius.
    let a = m.dot(m) - mu * mu;
    let b = 2.0 * (e0.dot(m) - e0u * mu);
    let c = e0.dot(e0) - e0u * e0u - radius * radius;
    let (rad_lo, rad_hi) = if a > EPS_COEF {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let root = disc.sqrt();
        (((-b) - root) / (2.0 * a), ((-b) + root) / (2.0 * a))
    } else if c <= 0.0 {
        // Motion parallel to the axis and already within radius: radially
        // inside for the entire segment.
        (Real::NEG_INFINITY, Real::INFINITY)
    } else {
        return None;
    };

    // Axial interval [ax_lo, ax_hi] where the projection lies in [0, len].
    let (ax_lo, ax_hi) = if mu.abs() > EPS_COEF {
        let t_at_zero = -e0u / mu;
        let t_at_len = (len - e0u) / mu;
        (t_at_zero.min(t_at_len), t_at_zero.max(t_at_len))
    } else if (0.0..=len).contains(&e0u) {
        (Real::NEG_INFINITY, Real::INFINITY)
    } else {
        return None;
    };

    let lo = rad_lo.max(ax_lo).max(0.0);
    let hi = rad_hi.min(ax_hi).min(1.0);
    if lo <= hi {
        Some(lo)
    } else {
        None
    }
}

/// Returns the earliest root in `0..=1` of `a*t^2 + b*t + c <= 0` for a
/// non-negative leading coefficient `a`, i.e. the first time the value becomes
/// non-positive, or [`None`] when it stays positive over the interval.
///
/// A start value `c <= 0` means the point is already inside and reports
/// `t == 0`. When `a` is (near) zero the equation is linear; otherwise the
/// earlier quadratic root is the entry crossing.
fn first_entry_time(a: Real, b: Real, c: Real) -> Option<Real> {
    if c <= 0.0 {
        return Some(0.0);
    }
    if a <= EPS_COEF {
        // Linear: b*t + c <= 0. With c > 0 this needs b < 0.
        if b >= -EPS_COEF {
            return None;
        }
        let t = -c / b;
        return if t <= 1.0 { Some(t.max(0.0)) } else { None };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    // With c > 0 and a > 0 the earlier root is the entry into the region.
    let t = ((-b) - root) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// Returns whichever of the two optional times is earlier, preferring a present
/// value over [`None`].
fn earliest(lhs: Option<Real>, rhs: Option<Real>) -> Option<Real> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, rhs) => rhs,
    }
}

/// Returns the unit outward normal of `collider` at the surface point `surf`,
/// or [`None`] when the collider is degenerate and no direction is defined.
///
/// For a sphere this is the radial direction; for a capsule it is the direction
/// from the nearest axis point; for a half-space it is the (normalized) plane
/// normal; for an oriented box it is the outward normal of the face the surface
/// point lies on.
fn outward_normal(collider: BodyCollider, surf: Vec3) -> Option<Vec3> {
    let n = match collider {
        BodyCollider::Sphere { center, .. } => (surf - center).normalize_or_zero(),
        BodyCollider::Capsule { p0, p1, .. } => {
            let closest = closest_point_on_segment(p0, p1, surf);
            (surf - closest).normalize_or_zero()
        }
        BodyCollider::HalfSpace { normal, .. } => normal.normalize_or_zero(),
        BodyCollider::Obb {
            center,
            orientation,
            half_extents,
        } => obb_face_normal(center, orientation, half_extents, surf),
        BodyCollider::ConvexHull(proxy) => proxy.face_normal(surf),
    };
    if n.length_squared() <= EPS_LEN_SQ {
        None
    } else {
        Some(n)
    }
}

/// Returns the earliest time of impact of the swept segment `prev -> curr`
/// against `collider`, dispatching to the matching closed-form solver.
fn collider_toi(collider: BodyCollider, prev: Vec3, curr: Vec3) -> Option<Real> {
    match collider {
        BodyCollider::Sphere { center, radius } => sphere_toi(prev, curr, center, radius),
        BodyCollider::Capsule { p0, p1, radius } => capsule_toi(prev, curr, p0, p1, radius),
        BodyCollider::HalfSpace { normal, offset } => half_space_toi(prev, curr, normal, offset),
        BodyCollider::Obb {
            center,
            orientation,
            half_extents,
        } => obb_toi(prev, curr, center, orientation, half_extents),
        BodyCollider::ConvexHull(proxy) => proxy.segment_toi(prev, curr),
    }
}

/// Returns the outward unit normal of the oriented box face the surface point
/// `surf` lies on, or the zero vector when the box is degenerate.
///
/// The point is taken into the box's local frame; the face is the local axis on
/// which `surf` is most extended relative to that axis's half-extent (the axis
/// whose `|local| / half_extent` ratio is largest, which is `1` for the face
/// the point sits on). The local axis normal, signed by that coordinate, is
/// rotated back to world space. Axes with a non-positive half-extent are
/// ignored so a collapsed box never divides by zero.
fn obb_face_normal(center: Vec3, orientation: Quat, half_extents: Vec3, surf: Vec3) -> Vec3 {
    let local = orientation.conjugate() * (surf - center);
    let mut best_axis = usize::MAX;
    let mut best_ratio = Real::NEG_INFINITY;
    for axis in 0..3 {
        let he = half_extents[axis];
        if he <= 0.0 {
            continue;
        }
        let ratio = local[axis].abs() / he;
        if ratio > best_ratio {
            best_ratio = ratio;
            best_axis = axis;
        }
    }
    if best_axis == usize::MAX {
        return Vec3::ZERO;
    }
    let mut local_normal = Vec3::ZERO;
    local_normal[best_axis] = if local[best_axis] >= 0.0 { 1.0 } else { -1.0 };
    (orientation * local_normal).normalize_or_zero()
}

/// Returns the earliest time in `0..=1` at which the segment `prev -> curr`
/// enters the oriented box `(center, orientation, half_extents)`, or [`None`]
/// when the segment misses it.
///
/// The segment is taken into the box's local frame, where the box is the
/// axis-aligned slab `[-half_extents, half_extents]`, and solved with the
/// standard three-slab ray/box intersection restricted to the unit segment.
/// A segment that runs parallel to and outside any slab misses; a non-positive
/// half-extent collapses that slab so the box is inert along it. Only
/// multiplies and comparisons are used, so no path yields a [`f32::NAN`].
fn obb_toi(
    prev: Vec3,
    curr: Vec3,
    center: Vec3,
    orientation: Quat,
    half_extents: Vec3,
) -> Option<Real> {
    if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
        return None;
    }
    let inv = orientation.conjugate();
    let p = inv * (prev - center);
    let d = inv * (curr - prev);
    // Standard slab clip: `t_enter` is the latest per-axis entry, `t_exit` the
    // earliest per-axis exit. Starting unbounded (not at 0) keeps the true
    // entry time, so a segment that *starts inside* the box yields a negative
    // entry and is rejected below, matching the sphere/capsule solvers which
    // leave an already-penetrating particle to the discrete projection.
    let mut t_enter = Real::NEG_INFINITY;
    let mut t_exit = Real::INFINITY;
    for axis in 0..3 {
        let he = half_extents[axis];
        let pa = p[axis];
        let da = d[axis];
        if da.abs() <= EPS_COEF {
            // Parallel to this slab: a start outside the slab can never enter.
            if pa < -he || pa > he {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / da;
        let t1 = (-he - pa) * inv_d;
        let t2 = (he - pa) * inv_d;
        let (t_near, t_far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        t_enter = t_enter.max(t_near);
        t_exit = t_exit.min(t_far);
        if t_enter > t_exit {
            return None;
        }
    }
    // The segment must reach the box (`t_exit >= 0`) and the entry must fall on
    // the forward unit segment; an entry outside `0..=1` means the box is only
    // reached before the start or after the end, or the start is already inside.
    if t_exit < 0.0 || !(0.0..=1.0).contains(&t_enter) {
        return None;
    }
    Some(t_enter)
}

/// Sweeps every free particle from its previous position to its current
/// position against every collider and resolves the earliest tunnelling hit.
///
/// For each free particle the segment `prev_positions[i] -> positions[i]` is
/// swept against all colliders; the earliest valid TOI wins. On a hit the
/// particle is placed on the collider surface plus `params.skin` along the
/// outward normal, and its normal velocity is reflected by `params.restitution`
/// (recomputed against the corrected motion using `dt`). After the normal
/// velocity is reflected, the particle's tangential slide across the swept
/// segment is damped by Coulomb friction against the contact (Macklin et al.
/// 2014): the tangential part of `placed - prev` is cancelled inside the static
/// cone (`||dx_t|| <= mu * ||dx_n||`) and shrunk by `mu * ||dx_n||` in the
/// dynamic regime, where `||dx_n||` is the depth the TOI snap pushed the
/// particle out along the outward normal.
///
/// `friction` is the fabric's friction coefficient, clamped to `0..=1` with a
/// non-finite value treated as `0`; `0` reproduces the frictionless bounce
/// exactly. The body proxy is infinitely massive, so the whole tangential
/// correction lands on the particle.
///
/// Particles are addressed through the raw store columns: `positions` and
/// `velocities` are written in place, while `prev_positions` and
/// `inverse_masses` are read-only. Pinned particles (`inverse_mass <= 0`), a
/// disabled sweep, an empty collider slice, mismatched slice lengths, and a
/// (near) zero `dt` are all handled without panicking. Cost is
/// `O(particles * colliders)`; visiting order is deterministic.
pub fn resolve_ccd(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[Real],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: Real,
    friction: Real,
) {
    if !params.enabled || colliders.is_empty() {
        return;
    }
    let params = params.sanitized();
    let mu = sanitize_friction(friction);
    let inv_dt = if dt.abs() <= EPS_COEF { 0.0 } else { 1.0 / dt };
    // Only indices valid in every read-only input are swept; velocities are
    // written only when the column is long enough.
    let count = positions
        .len()
        .min(prev_positions.len())
        .min(inverse_masses.len());
    for i in 0..count {
        if inverse_masses[i] <= 0.0 {
            // Pinned particle: never moved.
            continue;
        }
        let prev = prev_positions[i];
        let curr = positions[i];
        if curr.distance_squared(prev) <= EPS_LEN_SQ {
            continue;
        }
        // Find the earliest hit across all colliders.
        let mut best_t: Option<Real> = None;
        let mut best_collider = colliders[0];
        for &collider in colliders {
            if let Some(t) = collider_toi(collider, prev, curr) {
                let take = match best_t {
                    Some(b) => t < b,
                    None => true,
                };
                if take {
                    best_t = Some(t);
                    best_collider = collider;
                }
            }
        }
        let Some(t) = best_t else {
            continue;
        };
        // Contact point along the swept segment, then snap out to the surface.
        let contact = prev + (curr - prev) * t;
        let surface = best_collider.project(contact);
        let placed = match outward_normal(best_collider, surface) {
            Some(n) => {
                let placed = surface + n * params.skin;
                // Reflect the inbound normal velocity by restitution.
                let v = (placed - prev) * inv_dt;
                let vn = v.dot(n);
                if vn < 0.0 && i < velocities.len() {
                    velocities[i] = v - n * ((1.0 + params.restitution) * vn);
                }
                // Damp the tangential slide against the contact. The push-out
                // depth along the outward normal is the friction normal
                // magnitude `||dx_n||`; a non-positive depth is a no-op.
                let push = (placed - curr).dot(n);
                apply_coulomb_friction(placed, prev, n, push, mu)
            }
            None => surface,
        };
        positions[i] = placed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1e-6;

    /// Resolves a single free particle swept from `prev` to `curr` against
    /// `colliders`, returning its `(position, velocity)` after the sweep.
    fn sweep_one(
        prev: Vec3,
        curr: Vec3,
        colliders: &[BodyCollider],
        params: CcdParams,
        dt: Real,
        friction: Real,
    ) -> (Vec3, Vec3) {
        let mut positions = [curr];
        let prev_positions = [prev];
        let mut velocities = [Vec3::ZERO];
        let inverse_masses = [1.0];
        resolve_ccd(
            &mut positions,
            &prev_positions,
            &mut velocities,
            &inverse_masses,
            colliders,
            params,
            dt,
            friction,
        );
        (positions[0], velocities[0])
    }

    #[test]
    fn sphere_toi_reports_entry_crossing() {
        let t = sphere_toi(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
        )
        .expect("segment crosses the sphere");
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn sphere_toi_misses_when_segment_passes_by() {
        let t = sphere_toi(
            Vec3::new(-2.0, 3.0, 0.0),
            Vec3::new(2.0, 3.0, 0.0),
            Vec3::ZERO,
            1.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn sphere_toi_zero_when_starting_inside() {
        let t = sphere_toi(Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0), Vec3::ZERO, 1.0)
            .expect("start inside reports zero");
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn sphere_toi_inert_for_non_positive_radius() {
        assert!(sphere_toi(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            0.0
        )
        .is_none());
    }

    #[test]
    fn half_space_toi_reports_crossing() {
        let t = half_space_toi(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        )
        .expect("segment crosses the plane");
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn half_space_toi_none_when_moving_away() {
        let t = half_space_toi(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn half_space_toi_zero_when_starting_behind() {
        let t = half_space_toi(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        )
        .expect("start behind reports zero");
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn half_space_toi_inert_for_degenerate_normal() {
        assert!(half_space_toi(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::ZERO,
            0.0,
        )
        .is_none());
    }

    #[test]
    fn capsule_toi_hits_cylindrical_side() {
        let t = capsule_toi(
            Vec3::new(-3.0, 0.0, 2.0),
            Vec3::new(-3.0, 0.0, -2.0),
            Vec3::new(-3.0, -1.0, 0.0),
            Vec3::new(-3.0, 1.0, 0.0),
            1.0,
        )
        .expect("segment crosses the cylindrical side");
        assert!((t - 0.25).abs() < 1e-5, "t = {t}");
    }

    #[test]
    fn capsule_toi_hits_end_cap() {
        // Sweep straight down onto the top end cap sphere at `p1`.
        let t = capsule_toi(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
        )
        .expect("segment reaches the top end cap");
        // Enters the cap sphere (center `p1`, radius 1) at y = 2 => t = 0.3.
        assert!((t - 0.3).abs() < 1e-5, "t = {t}");
    }

    #[test]
    fn capsule_toi_degenerates_to_sphere() {
        let t = capsule_toi(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            1.0,
        )
        .expect("collapsed capsule behaves like a sphere");
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn resolve_ccd_prevents_tunnelling_through_plane() {
        // Fast downward sweep crossing a thin ground plane: snapped to surface.
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.01,
            restitution: 0.0,
            enabled: true,
        };
        let (pos, vel) = sweep_one(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            &colliders,
            params,
            1.0 / 60.0,
            0.0,
        );
        assert!((pos.y - 0.01).abs() < 1e-4, "y = {}", pos.y);
        // The downward normal velocity has been cancelled (restitution 0).
        assert!(vel.y >= -1e-3, "vy = {}", vel.y);
    }

    #[test]
    fn resolve_ccd_bounces_with_restitution() {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 1.0,
            enabled: true,
        };
        let (_pos, vel) = sweep_one(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            &colliders,
            params,
            1.0 / 60.0,
            0.0,
        );
        // A perfect bounce flips the normal velocity to point away from the wall.
        assert!(vel.y > 0.0, "vy = {}", vel.y);
    }

    /// Fixture: a particle sweeping diagonally from `(0, 1, 0)` to `(2, -1, 0)`
    /// crosses the plane `y >= 0` at `t = 0.5`, is snapped to `(1, 0, 0)` with
    /// push-out depth `1` and a tangential slide of length `1` along `+X`; the
    /// friction-adjusted `x` is therefore `1 - min(mu, 1)`.
    fn diagonal_plane_hit(mu: Real) -> Vec3 {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 0.0,
            enabled: true,
        };
        let (pos, _vel) = sweep_one(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            &colliders,
            params,
            1.0 / 60.0,
            mu,
        );
        pos
    }

    #[test]
    fn resolve_ccd_zero_friction_keeps_tangential_slide() {
        let pos = diagonal_plane_hit(0.0);
        assert!((pos.x - 1.0).abs() < TOL, "x = {}", pos.x);
        assert!(pos.y.abs() < TOL, "y = {}", pos.y);
    }

    #[test]
    fn resolve_ccd_dynamic_friction_shrinks_slide_by_mu() {
        let pos = diagonal_plane_hit(0.5);
        assert!((pos.x - 0.5).abs() < TOL, "x = {}", pos.x);
    }

    #[test]
    fn resolve_ccd_full_friction_locks_tangential_slide() {
        let pos = diagonal_plane_hit(1.0);
        assert!(pos.x.abs() < TOL, "x = {}", pos.x);
    }

    #[test]
    fn resolve_ccd_more_friction_slides_less() {
        let low = diagonal_plane_hit(0.25);
        let high = diagonal_plane_hit(0.75);
        assert!(high.x < low.x);
    }

    #[test]
    fn resolve_ccd_ignores_pinned_disabled_and_empty() {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        // Pinned particle (inverse mass 0) never moves.
        let mut positions = [Vec3::new(0.0, -5.0, 0.0)];
        let prev_positions = [Vec3::new(0.0, 1.0, 0.0)];
        let mut velocities = [Vec3::ZERO];
        let inverse_masses = [0.0];
        resolve_ccd(
            &mut positions,
            &prev_positions,
            &mut velocities,
            &inverse_masses,
            &colliders,
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert!((positions[0].y - (-5.0)).abs() < TOL);

        // Empty collider slice is a no-op for a free particle.
        let (pos, _vel) = sweep_one(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            &[],
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert!((pos.y - (-5.0)).abs() < TOL);

        // Disabled sweep is a no-op even with a hit.
        let params = CcdParams {
            enabled: false,
            ..CcdParams::default()
        };
        let (pos, _vel) = sweep_one(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -5.0, 0.0),
            &colliders,
            params,
            1.0 / 60.0,
            0.0,
        );
        assert!((pos.y - (-5.0)).abs() < TOL);
    }

    #[test]
    fn resolve_ccd_is_deterministic() {
        let build = || {
            let mut positions = [Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.5, -4.0, 0.1)];
            let prev_positions = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.5, 2.0, 0.1)];
            let mut velocities = [Vec3::ZERO; 2];
            let inverse_masses = [1.0, 1.0];
            let colliders = [
                BodyCollider::HalfSpace {
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    offset: 0.0,
                },
                BodyCollider::Sphere {
                    center: Vec3::new(0.5, -3.0, 0.1),
                    radius: 0.5,
                },
            ];
            resolve_ccd(
                &mut positions,
                &prev_positions,
                &mut velocities,
                &inverse_masses,
                &colliders,
                CcdParams::default(),
                1.0 / 60.0,
                0.0,
            );
            (positions, velocities)
        };
        let a = build();
        let b = build();
        for (pa, pb) in a.0.iter().zip(b.0.iter()) {
            assert!(pa.distance_squared(*pb) < 1e-12);
        }
        for (va, vb) in a.1.iter().zip(b.1.iter()) {
            assert!(va.distance_squared(*vb) < 1e-12);
        }
    }
}
