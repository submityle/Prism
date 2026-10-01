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

use glam::{Quat, Vec3};

use crate::narrow::distance::gjk_closest_points;
use crate::narrow::epa::gjk_contact;
use crate::narrow::support::{SupportMap, Transformed, Translated};

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

/// Returns the time of impact of shape `a` undergoing a *rigid* motion against
/// the static shape `b`, or [`None`] when they never come within
/// `contact_tolerance` during the interval `[0, 1]`.
///
/// The motion is a rotation of `rotation` about `pivot` combined with a
/// `linear` translation, both applied uniformly over `[0, 1]` (the rotation is
/// spherically interpolated from identity). `radius_a` is the farthest
/// distance from `pivot` to any point of `a` in its starting pose — pass the
/// body's bounding-sphere radius about the pivot.
///
/// Unlike [`conservative_advancement`], which assumes pure translation, this
/// bounds the worst-case closing speed as the linear term projected on the
/// current separating normal plus the angular term `angle * radius_a`, where
/// `angle` is the total swept rotation. That sum is a valid upper bound on how
/// fast any point of `a` can approach `b`, so advancing time by
/// `gap / closing_bound` never steps past a real contact. When the shapes
/// already overlap at `t = 0` the returned [`TimeOfImpact::toi`] is `0.0`.
pub fn rotational_conservative_advancement<A: SupportMap, B: SupportMap>(
    a: &A,
    b: &B,
    pivot: Vec3,
    linear: Vec3,
    rotation: Quat,
    radius_a: f32,
    contact_tolerance: f32,
) -> Option<TimeOfImpact> {
    let tolerance = contact_tolerance.max(MIN_CLOSING_SPEED);
    // Total swept angle over the unit interval bounds the angular point speed.
    let (_, angle) = rotation.to_axis_angle();
    let angular_speed = angle.abs() * radius_a.max(0.0);

    // Pure rest: only a pre-existing overlap can count as contact.
    if linear.length_squared() <= MIN_CLOSING_SPEED * MIN_CLOSING_SPEED
        && angular_speed <= MIN_CLOSING_SPEED
    {
        let moved = Transformed::new(a, rotation, pivot - rotation * pivot + linear);
        return gjk_contact(&moved, b).map(|c| TimeOfImpact {
            toi: 0.0,
            point: c.point_a,
            normal: c.normal,
        });
    }

    let mut t = 0.0f32;
    let mut last_normal = linear.normalize_or_zero();
    for _ in 0..MAX_ITERATIONS {
        let r_t = Quat::IDENTITY.slerp(rotation, t);
        let translation = pivot - r_t * pivot + linear * t;
        let moved = Transformed::new(a, r_t, translation);
        let Some(cp) = gjk_closest_points(&moved, b) else {
            let (point, normal) = match gjk_contact(&moved, b) {
                Some(c) => (c.point_a, c.normal),
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

        // Upper bound on the gap's closing speed along the current normal.
        let closing = linear.dot(cp.normal) + angular_speed;
        if closing <= MIN_CLOSING_SPEED {
            // Even the worst-case rotation cannot close the gap this interval.
            return None;
        }

        last_normal = cp.normal;
        let advance = (cp.distance - 0.5 * tolerance).max(0.0) / closing;
        t += advance;
        if t > 1.0 {
            return None;
        }
    }

    // Budget exhausted while still approaching: report the best bound reached.
    let r_t = Quat::IDENTITY.slerp(rotation, t);
    let translation = pivot - r_t * pivot + linear * t;
    let moved = Transformed::new(a, r_t, translation);
    gjk_closest_points(&moved, b).map(|cp| TimeOfImpact {
        toi: t,
        point: cp.point_a,
        normal: cp.normal,
    })
}

#[cfg(test)]
mod tests {
    use super::{conservative_advancement, rotational_conservative_advancement};
    use crate::bounding::{Aabb, Obb};
    use crate::narrow::distance::gjk_closest_points;
    use crate::narrow::support::Transformed;
    use core::f32::consts::FRAC_PI_2;
    use glam::{Quat, Vec3};

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

    #[test]
    fn rotation_sweeps_bar_into_target() {
        // A long bar lying along the x axis, pivoting 90 degrees about z about
        // the origin, so its tip sweeps up toward +y and strikes a small box.
        let half = Vec3::new(2.0, 0.2, 0.2);
        let a = Obb::new(Vec3::ZERO, half, Quat::IDENTITY);
        let b = Obb::new(Vec3::new(0.0, 1.6, 0.0), Vec3::splat(0.3), Quat::IDENTITY);
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        let radius_a = half.length();
        let hit = rotational_conservative_advancement(
            &a, &b, Vec3::ZERO, Vec3::ZERO, rot, radius_a, 1e-3,
        )
        .expect("rotation should bring the bar into contact");
        assert!(hit.toi > 0.0 && hit.toi <= 1.0, "toi = {}", hit.toi);
        // Rebuild the pose at the reported TOI and confirm the gap is tiny.
        let r_t = Quat::IDENTITY.slerp(rot, hit.toi);
        let moved = Transformed::new(&a, r_t, Vec3::ZERO);
        let cp = gjk_closest_points(&moved, &b);
        if let Some(cp) = cp {
            assert!(cp.distance < 5e-2, "distance at toi = {}", cp.distance);
        }
    }

    #[test]
    fn small_rotation_far_apart_misses() {
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.3), Quat::IDENTITY);
        let b = Obb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.3), Quat::IDENTITY);
        let rot = Quat::from_rotation_z(0.1);
        let radius_a = Vec3::splat(0.3).length();
        assert!(rotational_conservative_advancement(
            &a, &b, Vec3::ZERO, Vec3::ZERO, rot, radius_a, 1e-3,
        )
        .is_none());
    }

    #[test]
    fn coincident_shapes_report_zero_toi() {
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        let b = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        let hit = rotational_conservative_advancement(
            &a, &b, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY, 0.87, 1e-3,
        )
        .expect("already overlapping");
        assert_eq!(hit.toi, 0.0);
    }

    #[test]
    fn pure_linear_motion_matches_translation_toi() {
        // With identity rotation the rigid sweep must coincide with the
        // translation-only conservative advancement result.
        let a = Obb::new(Vec3::ZERO, Vec3::splat(0.5), Quat::IDENTITY);
        let b = Obb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(0.5), Quat::IDENTITY);
        let hit = rotational_conservative_advancement(
            &a,
            &b,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.5).length(),
            1e-3,
        )
        .expect("linear sweep should impact");
        assert!((hit.toi - 0.4).abs() < 2e-2, "toi = {}", hit.toi);
        assert!(hit.normal.x > 0.9, "normal toward b: {:?}", hit.normal);
    }
}
