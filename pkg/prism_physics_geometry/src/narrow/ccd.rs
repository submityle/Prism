//! Continuous collision detection for translating convex shapes.
//!
//! [`conservative_advancement`] computes the time of impact (TOI) of a convex
//! shape `a` that translates by a fixed `motion` vector over the unit time
//! interval `[0, 1]` against a stationary convex shape `b`. It repeatedly
//! measures the GJK separation at the current time and advances time by the
//! largest amount that provably cannot close the gap, using the fact that the
//! separation distance of two translating convex shapes decreases no faster
//! than the closing speed along the current closest direction (the standard
//! conservative-advancement bound).
//!
//! Only linear motion is modelled; rotation is intentionally out of scope so
//! the closing-speed bound stays exact. Supply a relative `motion` (the motion
//! of `a` minus the motion of `b`) to handle two moving shapes.
//!
//! This is a clean-room implementation of the publicly documented conservative
//! advancement algorithm and contains no Unreal Engine source or derived code.

use glam::Vec3;

use crate::narrow::distance::gjk_closest_points;
use crate::narrow::epa::gjk_contact;
use crate::narrow::support::{SupportMap, Translated};

/// Maximum conservative-advancement refinement steps.
const MAX_ITERATIONS: usize = 32;

/// Below this closing speed the shapes are treated as separating/parallel.
const MIN_CLOSING_SPEED: f32 = 1.0e-6;

/// The outcome of a successful [`conservative_advancement`] query.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TimeOfImpact {
    /// Fraction of the motion interval `[0, 1]` at which contact occurs.
    pub toi: f32,
    /// Witness point on shape `a` at the moment of impact.
    pub point: Vec3,
    /// Unit contact normal pointing from `a` toward `b`.
    pub normal: Vec3,
}

/// Returns the time of impact of `a` translating by `motion` against the static
/// shape `b`, or `None` when the shapes never touch within `[0, 1]`.
///
/// `contact_tolerance` is the separation (in world units) at which the shapes
/// are declared to be in contact; it must be positive. When the shapes already
/// overlap at `t = 0`, the returned [`TimeOfImpact::toi`] is `0.0` and the
/// normal comes from penetration recovery (EPA).
pub fn conservative_advancement<A: SupportMap, B: SupportMap>(
    a: &A,
    b: &B,
    motion: Vec3,
    contact_tolerance: f32,
) -> Option<TimeOfImpact> {
    let tolerance = contact_tolerance.max(MIN_CLOSING_SPEED);

    // A stationary shape can only collide if it already overlaps at t = 0.
    if motion.length_squared() <= MIN_CLOSING_SPEED * MIN_CLOSING_SPEED {
        return gjk_contact(a, b).map(|c| TimeOfImpact {
            toi: 0.0,
            point: c.point_a,
            normal: c.normal,
        });
    }

    let mut t = 0.0f32;
    // Best known approach direction, used if a step overshoots into contact.
    let mut last_normal = motion.normalize_or_zero();
    for _ in 0..MAX_ITERATIONS {
        let moved = Translated::new(a, motion * t);
        let Some(cp) = gjk_closest_points(&moved, b) else {
            // Shapes already touch/overlap at this time: contact is now.
            let (point, normal) = match gjk_contact(&moved, b) {
                Some(c) => (c.point_a, c.normal),
                // Exact grazing has no penetration witness; fall back to the
                // last approach direction and the near support point.
                None => (moved.support_point(last_normal), last_normal),
            };
            return Some(TimeOfImpact { toi: t, point, normal });
        };

        if cp.distance <= tolerance {
            return Some(TimeOfImpact {
                toi: t,
                point: cp.point_a,
                normal: cp.normal,
            });
        }

        // Rate at which the gap closes = motion projected on the A->B normal.
        let closing = motion.dot(cp.normal);
        if closing <= MIN_CLOSING_SPEED {
            // Moving away from or tangent to `b`: no impact this interval.
            return None;
        }

        last_normal = cp.normal;
        // Conservative advance that stops within the tolerance band rather than
        // overshooting exactly onto the surface (which would lose the GJK
        // witness). The gap cannot close before the motion covers this much.
        let advance = (cp.distance - 0.5 * tolerance).max(0.0) / closing;
        t += advance;
        if t > 1.0 {
            return None;
        }
    }

    // Refinement budget exhausted while still approaching: report the best
    // bound reached as a (slightly early) contact estimate.
    let moved = Translated::new(a, motion * t);
    gjk_closest_points(&moved, b).map(|cp| TimeOfImpact {
        toi: t,
        point: cp.point_a,
        normal: cp.normal,
    })
}

#[cfg(test)]
mod tests {
    use super::conservative_advancement;
    use crate::bounding::Aabb;
    use glam::Vec3;

    #[test]
    fn head_on_translation_reports_toi() {
        // Unit box at origin; target unit box centred at x = 5.
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(0.5));
        let b = Aabb::from_center_half_extents(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5));
        // Move `a` +X by 10 over the interval; faces meet when centres are 1
        // apart, i.e. a travels 4 units => toi = 0.4.
        let toi = conservative_advancement(&a, &b, Vec3::new(10.0, 0.0, 0.0), 1e-3)
            .expect("impact");
        assert!((toi.toi - 0.4).abs() < 2e-2, "toi = {}", toi.toi);
        assert!(toi.normal.x > 0.9, "normal points toward b: {:?}", toi.normal);
    }

    #[test]
    fn motion_away_has_no_impact() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(0.5));
        let b = Aabb::from_center_half_extents(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5));
        // Move away from b.
        assert!(conservative_advancement(&a, &b, Vec3::new(-10.0, 0.0, 0.0), 1e-3).is_none());
    }

    #[test]
    fn motion_too_short_misses() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(0.5));
        let b = Aabb::from_center_half_extents(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5));
        // Only travels 1 unit over the whole step, gap is 4 => never reaches.
        assert!(conservative_advancement(&a, &b, Vec3::new(1.0, 0.0, 0.0), 1e-3).is_none());
    }

    #[test]
    fn tangential_motion_misses() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(0.5));
        let b = Aabb::from_center_half_extents(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5));
        // Perpendicular slide never closes the x-gap.
        assert!(conservative_advancement(&a, &b, Vec3::new(0.0, 10.0, 0.0), 1e-3).is_none());
    }

    #[test]
    fn initial_overlap_reports_zero_toi() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let b = Aabb::from_center_half_extents(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(1.0));
        let toi = conservative_advancement(&a, &b, Vec3::new(1.0, 0.0, 0.0), 1e-3)
            .expect("already overlapping");
        assert_eq!(toi.toi, 0.0);
    }
}
