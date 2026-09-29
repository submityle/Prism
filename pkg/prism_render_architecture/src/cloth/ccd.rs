//! Continuous collision detection (CCD) for cloth particles.
//!
//! The stateless [`dynamics`](super::dynamics) solver only projects particles
//! out of colliders at their *end-of-step* position. For a thin garment moving
//! fast against a thin collider that is not enough: a particle can start in
//! front of a wall and finish behind it in a single substep, tunnelling through
//! without the projection ever seeing an overlap. This module closes that gap
//! by sweeping the segment `prev -> curr` against each analytic body collider
//! and solving for the earliest time of impact (TOI) along the segment, then
//! snapping the particle to the surface with a skin offset and reflecting its
//! normal velocity by a restitution coefficient.
//!
//! The TOI solvers are exact closed forms: a sphere is a single quadratic, a
//! half-space is linear, and a capsule is the union of an infinite cylinder
//! (restricted to the segment slab) with a sphere at each end cap. Only
//! [`f32::sqrt`] is used; there are no transcendental calls. Every routine is
//! `O(1)` per (particle, collider) pair, so [`resolve_ccd`] is
//! `O(particles * colliders)` with no hidden inner loops, and it is fully
//! deterministic (particles in index order, colliders in slice order, the
//! earliest valid hit wins).

use super::collision::{apply_coulomb_friction, closest_point_on_segment, BodyCollider};
use super::{ClothParticle, Vec3, EPS_LEN_SQ};

/// Numerical floor for treating a scalar coefficient as zero when classifying a
/// quadratic as linear or a segment as degenerate.
const EPS_COEF: f32 = 1e-12;

/// Tuning for the continuous-collision sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcdParams {
    /// How far outside the collider surface (along the outward normal) a
    /// particle is placed after a hit, so the next substep starts strictly
    /// outside and does not immediately re-penetrate. Non-negative; a value of
    /// zero snaps exactly to the surface.
    pub skin: f32,
    /// Normal restitution in `[0, 1]`: `0` is a fully inelastic stop (the
    /// inbound normal velocity is cancelled) and `1` is a perfect bounce (the
    /// normal velocity is mirrored). Values are clamped into range.
    pub restitution: f32,
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
    /// `[0, 1]`, and any `NaN` replaced by a safe value, so a mis-authored asset
    /// can never inject a `NaN` or a negative skin into the sweep.
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

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` is on or inside the sphere `(center, radius)`, or `None` when
/// the swept segment never reaches the sphere.
///
/// The point path is `p(t) = prev + t * (curr - prev)`. Substituting into
/// `|p(t) - center|^2 = radius^2` gives a quadratic whose earlier root is the
/// entry crossing. A point that already starts on or inside the sphere reports
/// `t = 0`. A non-positive radius makes the sphere inert (`None`).
#[must_use]
pub fn sphere_toi(prev: Vec3, curr: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let m = curr.sub(prev);
    let e = prev.sub(center);
    let a = m.dot(m);
    let b = 2.0 * e.dot(m);
    let c = e.dot(e) - radius * radius;
    first_entry_time(a, b, c)
}

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` crosses into the infeasible side of the half-space
/// `normal.dot(x) >= offset`, or `None` when it stays in front for the whole
/// segment.
///
/// The signed distance `s(t) = normal.dot(p(t)) - offset` is linear in `t`. The
/// crossing is where `s(t) == 0` while `s` is decreasing. A point that starts
/// behind the plane reports `t = 0`. A (near) zero normal has no defined plane
/// and returns `None`.
#[must_use]
pub fn half_space_toi(prev: Vec3, curr: Vec3, normal: Vec3, offset: f32) -> Option<f32> {
    if normal.length_squared() <= EPS_LEN_SQ {
        return None;
    }
    let s0 = normal.dot(prev) - offset;
    if s0 <= 0.0 {
        return Some(0.0);
    }
    let ds = normal.dot(curr.sub(prev));
    if ds >= -EPS_COEF {
        // Moving away from or parallel to the plane: never crosses.
        return None;
    }
    let t = -s0 / ds;
    if t <= 1.0 {
        Some(t.max(0.0))
    } else {
        None
    }
}

/// Returns the earliest time `t` in `[0, 1]` at which the point moving along
/// `prev -> curr` is on or inside the capsule (segment `p0`..`p1` inflated by
/// `radius`), or `None` when the swept segment misses it.
///
/// A capsule is the union of an infinite cylinder about the segment axis with a
/// sphere at each end cap. This computes the cylinder entry time restricted to
/// the axis slab `[0, len]` and the entry time of each end-cap sphere, then
/// returns the earliest of those. A collapsed capsule (`p0 == p1`) degenerates
/// to a single sphere. A non-positive radius makes the capsule inert.
#[must_use]
pub fn capsule_toi(prev: Vec3, curr: Vec3, p0: Vec3, p1: Vec3, radius: f32) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let axis = p1.sub(p0);
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
/// time whose contact projects onto the segment slab `[0, len]`, or `None`.
///
/// The perpendicular distance to the axis line is quadratic in `t`; its
/// sub-`radius` interval is intersected with the time interval during which the
/// axial projection lies within the slab and with `[0, 1]`. The lower bound of
/// the resulting interval is the earliest cylindrical-side contact; the end
/// caps are handled separately by [`capsule_toi`].
fn cylinder_slab_toi(prev: Vec3, curr: Vec3, p0: Vec3, axis: Vec3, radius: f32) -> Option<f32> {
    let len = axis.length();
    if len <= EPS_COEF {
        return None;
    }
    let u = axis.scale(1.0 / len);
    let e0 = prev.sub(p0);
    let m = curr.sub(prev);
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
        (f32::NEG_INFINITY, f32::INFINITY)
    } else {
        return None;
    };

    // Axial interval [ax_lo, ax_hi] where the projection lies in [0, len].
    let (ax_lo, ax_hi) = if mu.abs() > EPS_COEF {
        let t_at_zero = -e0u / mu;
        let t_at_len = (len - e0u) / mu;
        (t_at_zero.min(t_at_len), t_at_zero.max(t_at_len))
    } else if (0.0..=len).contains(&e0u) {
        (f32::NEG_INFINITY, f32::INFINITY)
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

/// Returns the earliest root in `[0, 1]` of `a*t^2 + b*t + c <= 0` for a
/// non-negative leading coefficient `a`, i.e. the first time the value becomes
/// non-positive, or `None` when it stays positive over the interval.
///
/// A start value `c <= 0` means the point is already inside and reports
/// `t = 0`. When `a` is (near) zero the equation is linear; otherwise the
/// earlier quadratic root is the entry crossing.
fn first_entry_time(a: f32, b: f32, c: f32) -> Option<f32> {
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
/// value over `None`.
fn earliest(lhs: Option<f32>, rhs: Option<f32>) -> Option<f32> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, rhs) => rhs,
    }
}

/// Returns the unit outward normal of `collider` at the surface point `surf`,
/// or `None` when the collider is degenerate and no direction is defined.
///
/// For a sphere this is the radial direction; for a capsule it is the direction
/// from the nearest axis point; for a half-space it is the (normalized) plane
/// normal.
fn outward_normal(collider: BodyCollider, surf: Vec3) -> Option<Vec3> {
    let n = match collider {
        BodyCollider::Sphere { center, .. } => surf.sub(center).normalize_or_zero(),
        BodyCollider::Capsule { p0, p1, .. } => {
            let closest = closest_point_on_segment(p0, p1, surf);
            surf.sub(closest).normalize_or_zero()
        }
        BodyCollider::HalfSpace { normal, .. } => normal.normalize_or_zero(),
    };
    if n.length_squared() <= EPS_LEN_SQ {
        None
    } else {
        Some(n)
    }
}

/// Returns the earliest time of impact of the swept segment `prev -> curr`
/// against `collider`, dispatching to the matching closed-form solver.
fn collider_toi(collider: BodyCollider, prev: Vec3, curr: Vec3) -> Option<f32> {
    match collider {
        BodyCollider::Sphere { center, radius } => sphere_toi(prev, curr, center, radius),
        BodyCollider::Capsule { p0, p1, radius } => capsule_toi(prev, curr, p0, p1, radius),
        BodyCollider::HalfSpace { normal, offset } => half_space_toi(prev, curr, normal, offset),
    }
}

/// Sweeps every free particle from its previous position to its current
/// position against every collider and resolves the earliest tunnelling hit.
///
/// For each free particle the segment `prev_positions[i] -> particles[i]` is
/// swept against all colliders; the earliest valid TOI wins. On a hit the
/// particle is placed on the collider surface plus `params.skin` along the
/// outward normal, and its normal velocity is reflected by `params.restitution`
/// (recomputed against the corrected motion using `dt`). Pinned particles, a
/// disabled sweep, an empty collider slice, and a `prev_positions` slice
/// shorter than `particles` are all handled without panicking, and a
/// (near) zero `dt` leaves velocities untouched.
///
/// After the normal velocity is reflected the particle's tangential slide
/// across the swept segment is damped by Coulomb friction against the contact
/// (Macklin et al. 2014): the tangential part of `placed - prev` is cancelled
/// inside the static cone (`||Dx_t|| <= mu * ||Dx_n||`) and shrunk by
/// `mu * ||Dx_n||` in the dynamic regime, where `||Dx_n||` is the depth the TOI
/// snap pushed the particle out along the outward normal. `friction` is the
/// fabric's `FabricMaterial::friction` coefficient, clamped to `0..=1` with a
/// non-finite value treated as `0`; `0` reproduces the frictionless bounce
/// exactly. The body proxy is infinitely massive, so the whole tangential
/// correction lands on the particle.
///
/// Cost is `O(particles * colliders)`; visiting order is deterministic.
pub fn resolve_ccd(
    particles: &mut [ClothParticle],
    prev_positions: &[Vec3],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: f32,
    friction: f32,
) {
    if !params.enabled || colliders.is_empty() {
        return;
    }
    let params = params.sanitized();
    // Clamp the friction coefficient to `[0, 1]`; a non-finite value is treated
    // as frictionless so an unsanitised material can never inject a `NaN`.
    let mu = if friction.is_finite() {
        friction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let inv_dt = if dt.abs() <= EPS_COEF { 0.0 } else { 1.0 / dt };
    let count = particles.len().min(prev_positions.len());
    for i in 0..count {
        let particle = &mut particles[i];
        if particle.is_pinned() {
            continue;
        }
        let prev = prev_positions[i];
        let curr = particle.position;
        if curr.distance_squared(prev) <= EPS_LEN_SQ {
            continue;
        }
        // Find the earliest hit across all colliders.
        let mut best_t: Option<f32> = None;
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
        let contact = prev.add(curr.sub(prev).scale(t));
        let surface = best_collider.project(contact);
        let surface = match outward_normal(best_collider, surface) {
            Some(n) => {
                let placed = surface.add(n.scale(params.skin));
                // Reflect the inbound normal velocity by restitution.
                let v = placed.sub(prev).scale(inv_dt);
                let vn = v.dot(n);
                if vn < 0.0 {
                    let reflected = v.sub(n.scale((1.0 + params.restitution) * vn));
                    particle.velocity = reflected;
                }
                // Damp the tangential slide against the contact. The push-out
                // depth along the outward normal is the friction normal
                // magnitude `||Dx_n||`; a non-positive depth is a no-op.
                let push = placed.sub(curr).dot(n);
                apply_coulomb_friction(placed, prev, n, push, mu)
            }
            None => surface,
        };
        particle.position = surface;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a free particle at `position` with unit inverse mass.
    fn free_particle(position: Vec3) -> ClothParticle {
        ClothParticle {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
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
    fn capsule_toi_hits_cylindrical_side() {
        let t = capsule_toi(
            Vec3::new(-3.0, 0.0, 2.0),
            Vec3::new(3.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            1.0,
        )
        .expect("segment crosses the capsule side");
        assert!((t - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn capsule_toi_hits_end_cap() {
        let t = capsule_toi(
            Vec3::new(0.0, 0.0, 7.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            1.0,
        )
        .expect("segment crosses the end cap");
        assert!((t - 0.5).abs() < 1e-5);
    }

    #[test]
    fn resolve_ccd_prevents_tunnelling_through_a_plane() {
        let mut particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.01,
            restitution: 0.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, 0.0);
        // The particle ends in front of the plane at the skin offset, not below.
        assert!(particles[0].position.y >= 0.0);
        assert!((particles[0].position.y - 0.01).abs() < 1e-4);
        // The downward normal velocity has been cancelled (restitution 0).
        assert!(particles[0].velocity.y >= -1e-3);
    }

    #[test]
    fn resolve_ccd_bounces_with_restitution() {
        let mut particles = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 1.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, 0.0);
        // A perfect bounce flips the normal velocity to point away from the wall.
        assert!(particles[0].velocity.y > 0.0);
    }

    /// Fixture: a particle sweeping diagonally from `(0, 1, 0)` to `(2, -1, 0)`
    /// crosses the plane `y >= 0` at `t = 0.5`, is snapped to `(1, 0, 0)` with
    /// push-out depth `1` and a tangential slide of length `1` along `+X`; the
    /// friction-adjusted `x` is therefore `1 - min(mu, 1)`.
    fn diagonal_plane_hit(mu: f32) -> ClothParticle {
        let mut particles = [free_particle(Vec3::new(2.0, -1.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        let params = CcdParams {
            skin: 0.0,
            restitution: 0.0,
            enabled: true,
        };
        resolve_ccd(&mut particles, &prev, &colliders, params, 1.0 / 60.0, mu);
        particles[0]
    }

    #[test]
    fn resolve_ccd_zero_friction_keeps_tangential_slide() {
        // mu = 0 must reproduce the frictionless snap: the full tangential
        // slide survives, so x stays at 1.
        let hit = diagonal_plane_hit(0.0);
        assert!(
            (hit.position.x - 1.0).abs() < 1e-6,
            "x = {}",
            hit.position.x
        );
        assert!(hit.position.y.abs() < 1e-6, "y = {}", hit.position.y);
    }

    #[test]
    fn resolve_ccd_dynamic_friction_shrinks_slide_by_mu() {
        // Dynamic regime: x = 1 - mu * push / ||slide|| = 1 - 0.5.
        let hit = diagonal_plane_hit(0.5);
        assert!(
            (hit.position.x - 0.5).abs() < 1e-6,
            "x = {}",
            hit.position.x
        );
    }

    #[test]
    fn resolve_ccd_full_friction_locks_tangential_slide() {
        // mu = 1 saturates the static cone here (mu * push == ||slide||), so the
        // whole tangential slide is cancelled and x collapses to 0.
        let hit = diagonal_plane_hit(1.0);
        assert!(hit.position.x.abs() < 1e-6, "x = {}", hit.position.x);
    }

    #[test]
    fn resolve_ccd_more_friction_slides_less() {
        // Strictly monotone: heavier friction leaves less residual tangential
        // travel along +X.
        let low = diagonal_plane_hit(0.25);
        let high = diagonal_plane_hit(0.75);
        assert!(high.position.x < low.position.x);
    }

    #[test]
    fn resolve_ccd_ignores_pinned_and_empty() {
        let mut pinned = [ClothParticle::pinned(Vec3::new(0.0, -5.0, 0.0))];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }];
        resolve_ccd(
            &mut pinned,
            &prev,
            &colliders,
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert!((pinned[0].position.y - (-5.0)).abs() < 1e-6);

        let mut free = [free_particle(Vec3::new(0.0, -5.0, 0.0))];
        resolve_ccd(&mut free, &prev, &[], CcdParams::default(), 1.0 / 60.0, 0.0);
        assert!((free[0].position.y - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn resolve_ccd_is_deterministic() {
        let build = || {
            let mut p = [
                free_particle(Vec3::new(0.0, -5.0, 0.0)),
                free_particle(Vec3::new(0.5, -4.0, 0.1)),
            ];
            let prev = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.5, 2.0, 0.1)];
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
                &mut p,
                &prev,
                &colliders,
                CcdParams::default(),
                1.0 / 60.0,
                0.0,
            );
            p
        };
        let a = build();
        let b = build();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!(pa.position.distance_squared(pb.position) < 1e-12);
            assert!(pa.velocity.distance_squared(pb.velocity) < 1e-12);
        }
    }
}
