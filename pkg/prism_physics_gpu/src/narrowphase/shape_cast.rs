//! Jolt/PhysX-style shape cast: sweep one moving rounded convex shape through
//! a scene of rounded convex targets and report the earliest contact.
//!
//! A ray cast asks when a point first touches a scene; a shape cast asks the
//! same for a whole moving volume. It is the query behind character controllers
//! ("how far can this capsule step before it hits something?"), projectile
//! sweeps, and spawn-overlap probes. Each query reuses the per-pair
//! conservative-advancement time of impact in
//! [`conservative_advancement_toi_rounded`](super::conservative_advancement::conservative_advancement_toi_rounded),
//! which is validated slot-for-slot against its `GPU` twin, so the cast here is
//! a thin, allocation-light collector over kernel-grade math.
//!
//! Every participant is a [`RoundedConvex`]: a borrowed [`ConvexHull`] core
//! inflated by a convex radius, with a start [`ConvexPose`] and a [`BodyMotion`]
//! carrying it across the substep. A sphere is a one-vertex core plus radius, a
//! capsule a segment core plus radius, and a bevelled box a box core plus a
//! small radius, so one type covers the whole rounded-convex family. Targets
//! usually hold [`BodyMotion::still`], but a moving target is swept correctly
//! because the underlying solver advances both bodies.
//!
//! Provenance: conservative advancement (Mirtich, 2000; van den Bergen, 2004)
//! over a Gilbert-Johnson-Keerthi distance walk. No Unreal Engine source or
//! derived code.

use super::body_motion::BodyMotion;
use super::conservative_advancement::{conservative_advancement_toi_rounded, Toi};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;

/// A rounded convex instance: a convex core inflated by a convex radius, placed
/// by a start pose and swept by a motion.
///
/// The hull is borrowed so a single shared core (for example one unit sphere
/// core) can be reused across many instances without cloning its vertex table.
#[derive(Clone, Copy, Debug)]
pub struct RoundedConvex<'a> {
    /// The convex core. A single origin vertex gives a sphere, a segment gives a
    /// capsule, a box gives a (optionally bevelled) box.
    pub hull: &'a ConvexHull,
    /// World placement of the core at the start of the substep.
    pub pose: ConvexPose,
    /// Linear and angular motion carrying the core across the substep.
    pub motion: BodyMotion,
    /// Convex radius inflating the core: the sphere/capsule cap radius or a box
    /// bevel. Pass `0.0` for a sharp hull.
    pub radius: f32,
}

impl<'a> RoundedConvex<'a> {
    /// Builds a moving rounded convex from its core, start pose, motion, and
    /// convex radius.
    #[must_use]
    pub fn new(
        hull: &'a ConvexHull,
        pose: ConvexPose,
        motion: BodyMotion,
        radius: f32,
    ) -> RoundedConvex<'a> {
        RoundedConvex {
            hull,
            pose,
            motion,
            radius,
        }
    }

    /// Builds a stationary rounded convex: a core at `pose` with
    /// [`BodyMotion::still`] and the given convex radius. Convenient for the
    /// static scene geometry a cast sweeps against.
    #[must_use]
    pub fn still(hull: &'a ConvexHull, pose: ConvexPose, radius: f32) -> RoundedConvex<'a> {
        RoundedConvex {
            hull,
            pose,
            motion: BodyMotion::still(),
            radius,
        }
    }
}

/// A target struck by a shape cast, paired with the time of impact against it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Time of impact against that target: the substep fraction, world contact
    /// point, and contact normal (pointing from the target toward the cast
    /// shape).
    pub toi: Toi,
}

/// Sweeps the moving `shape` through `targets` and returns the earliest contact,
/// or `None` when nothing is reached within `dt`.
///
/// `target_sep` is a speculative separation margin: pass `0.0` for touching
/// contact, or a small positive value to report an impact just before surfaces
/// meet (useful to leave a solver skin). Ties on impact time resolve to the
/// lower target index, so the result is deterministic regardless of target
/// order.
///
/// The contact normal of the returned [`Toi`] points from the struck target
/// toward the cast shape, matching the `B` toward `A` convention of the
/// underlying narrow phase (the cast shape is `A`, the target is `B`).
#[must_use]
pub fn cast_shape(
    shape: &RoundedConvex,
    targets: &[RoundedConvex],
    dt: f32,
    target_sep: f32,
) -> Option<ShapeCastHit> {
    let mut best: Option<ShapeCastHit> = None;
    for (index, target) in targets.iter().enumerate() {
        let Some(toi) = sweep_against(shape, target, dt, target_sep) else {
            continue;
        };
        // Keep the earliest impact; on an exact time tie the lower index wins
        // because it was visited first and a strictly-less test does not replace
        // it.
        let replace = best.is_none_or(|current| toi.time < current.toi.time);
        if replace {
            best = Some(ShapeCastHit {
                target: index as u32,
                toi,
            });
        }
    }
    best
}

/// Sweeps the moving `shape` through `targets` and returns every contact within
/// `dt`, ordered by increasing time of impact (ties broken by ascending target
/// index).
///
/// Use this when a single cast must know all shapes along its path (for example
/// to apply damage to each body a projectile passes through), rather than only
/// the first blocker. See [`cast_shape`] for the `target_sep` and normal
/// conventions.
#[must_use]
pub fn cast_shape_all(
    shape: &RoundedConvex,
    targets: &[RoundedConvex],
    dt: f32,
    target_sep: f32,
) -> Vec<ShapeCastHit> {
    let mut hits: Vec<ShapeCastHit> = targets
        .iter()
        .enumerate()
        .filter_map(|(index, target)| {
            sweep_against(shape, target, dt, target_sep).map(|toi| ShapeCastHit {
                target: index as u32,
                toi,
            })
        })
        .collect();
    // Earliest first; a stable sort keeps equal-time hits in ascending index
    // order, matching the single-hit tie rule.
    hits.sort_by(|lhs, rhs| {
        lhs.toi
            .time
            .partial_cmp(&rhs.toi.time)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    hits
}

/// Sweeps the cast `shape` (as body `A`) against one `target` (as body `B`) and
/// returns the per-pair time of impact.
#[must_use]
fn sweep_against(
    shape: &RoundedConvex,
    target: &RoundedConvex,
    dt: f32,
    target_sep: f32,
) -> Option<Toi> {
    conservative_advancement_toi_rounded(
        shape.hull,
        &shape.pose,
        &shape.motion,
        shape.radius,
        target.hull,
        &target.pose,
        &target.motion,
        target.radius,
        dt,
        target_sep,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    #[test]
    fn cast_returns_the_nearest_blocker() {
        // A sphere core (point core + radius 0.5) travels +x from the origin.
        let sphere = ConvexHull::from_point();
        let shape = RoundedConvex::new(
            &sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
            0.5,
        );
        // Two static unit boxes on the path: the near one at x = 4, the far one
        // at x = 8. The sphere surface (radius 0.5) meets the near box face
        // (at x = 3.5) first.
        let box_hull = unit_box();
        let near = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let far = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hit = cast_shape(&shape, &[near, far], 1.0, 0.0).expect("the cast must hit the near box");
        assert_eq!(hit.target, 0, "the nearer box index 0 must win");
        // Surface gap is 3.0 (origin+0.5 to 3.5) closing at 10 per unit => 0.3.
        assert!((hit.toi.time - 0.3).abs() < 0.02, "time {}", hit.toi.time);
        // Normal points from the box (B) back toward the sphere (A): roughly -x.
        assert!(hit.toi.normal.x < -0.5, "normal {:?}", hit.toi.normal);
    }

    #[test]
    fn cast_misses_when_nothing_is_on_the_path() {
        let sphere = ConvexHull::from_point();
        let shape = RoundedConvex::new(
            &sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
            0.5,
        );
        // A box well off the +x line of travel.
        let box_hull = unit_box();
        let aside = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 20.0, 0.0), Quat::IDENTITY), 0.0);
        assert!(cast_shape(&shape, &[aside], 1.0, 0.0).is_none(), "an off-path target must not be hit");
    }

    #[test]
    fn cast_against_empty_scene_is_none() {
        let sphere = ConvexHull::from_point();
        let shape = RoundedConvex::new(
            &sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
            0.5,
        );
        assert!(cast_shape(&shape, &[], 1.0, 0.0).is_none(), "an empty scene cannot be hit");
    }

    #[test]
    fn cast_all_orders_every_hit_by_time() {
        // A sphere passing through two boxes it fully overlaps along +x reports
        // both, nearest first.
        let sphere = ConvexHull::from_point();
        let shape = RoundedConvex::new(
            &sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
            0.5,
        );
        let box_hull = unit_box();
        // Deliberately pass the far box first in the slice to prove ordering is
        // by time, not by input order.
        let far = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let near = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hits = cast_shape_all(&shape, &[far, near], 1.0, 0.0);
        assert_eq!(hits.len(), 2, "both boxes lie on the path");
        // Index 1 (the near box at x = 4) is struck before index 0 (x = 8).
        assert_eq!(hits[0].target, 1, "nearest hit first");
        assert_eq!(hits[1].target, 0, "farther hit second");
        assert!(hits[0].toi.time < hits[1].toi.time, "times must increase");
    }

    #[test]
    fn equal_time_ties_resolve_to_the_lower_index() {
        // Two identical boxes stacked at the same x but different y are both
        // reached at the same time by a wide sweep; the lower index must win.
        let sphere = ConvexHull::from_point();
        let shape = RoundedConvex::new(
            &sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
            0.5,
        );
        let box_hull = unit_box();
        // Both boxes centred on the travel line so each is reached at t = 0.3.
        let first = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let second = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hit = cast_shape(&shape, &[first, second], 1.0, 0.0).expect("the stacked boxes must be hit");
        assert_eq!(hit.target, 0, "an exact time tie resolves to the lower index");
    }

    #[test]
    fn rounded_radius_brings_the_impact_earlier() {
        // The same geometry hits sooner when the cast shape is fatter, because
        // its inflated surface reaches the box with less core travel.
        let sphere = ConvexHull::from_point();
        let box_hull = unit_box();
        let target = RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let make = |radius: f32| {
            RoundedConvex::new(
                &sphere,
                ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
                BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
                radius,
            )
        };
        let thin = cast_shape(&make(0.25), &[target], 1.0, 0.0).expect("thin sphere hits");
        let fat = cast_shape(&make(0.75), &[target], 1.0, 0.0).expect("fat sphere hits");
        assert!(fat.toi.time < thin.toi.time, "fat {} should precede thin {}", fat.toi.time, thin.toi.time);
    }
}
