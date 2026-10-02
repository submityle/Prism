//! Broad-phase-accelerated shape cast: the earliest-contact query of
//! [`cast_shape`](super::shape_cast::cast_shape), but a linear `BVH` over the
//! targets' motion-swept world bounds replaces the brute-force scan so a cast
//! against a large static scene costs work proportional to the handful of
//! targets the moving query could actually reach, not the full target count.
//!
//! This is the broad-phase gather `AAA` engines run before the per-pair narrow
//! phase (Jolt's `NarrowPhaseQuery` cast, `PhysX` scene sweeps, Unreal's `Chaos`
//! sweep): build an acceleration structure over the scene, descend it with the
//! moving query's swept bounds to collect candidates, then run the exact
//! per-pair time of impact only on those candidates.
//!
//! # Why the result is identical to the brute-force cast
//!
//! Each body's swept bound is a *conservative* superset of the world volume its
//! rounded core occupies across the whole substep. [`BodyMotion::pose_at`]
//! translates the core's origin linearly and turns the core about that moving
//! origin, so every core vertex stays within a ball of the core's bounding
//! radius about the origin at every sampled time; sweeping that ball along the
//! origin's straight path and inflating by the convex radius encloses every
//! point the shape can touch. A real contact at some substep time is a point
//! inside both swept volumes, hence inside both swept boxes, so the two boxes
//! overlap. The `BVH` therefore never prunes a target the brute-force cast would
//! hit: the candidate set is a superset of the truly-struck targets, and running
//! the identical per-pair sweep and the identical earliest-hit tie rule over it
//! yields the identical hit. The included unit tests assert that equivalence
//! directly, scene for scene, against [`cast_shape`](super::shape_cast::cast_shape).
//!
//! # Provenance
//!
//! Conservative advancement after Brian Mirtich, *Timewarp Rigid Body
//! Simulation* (2000) and the ray-casting formulation of Gino van den Bergen,
//! *Ray Casting against General Convex Objects* (2004), over a
//! Gilbert-Johnson-Keerthi distance walk (1988), gathered through the linear
//! `BVH` of Karras, *Maximizing Parallelism in the Construction of BVHs,
//! Octrees, and k-d Trees* (2012). No Unreal Engine source or derived code.

use glam::Vec3;

use crate::bvh::{cpu_build_lbvh, cpu_bvh_aabb_overlap, Aabb};

use super::shape_cast::{sweep_against, RoundedConvex, ShapeCastHit};

/// The farthest a core vertex sits from the core's local origin: the radius of
/// the smallest origin-centred ball containing the core.
///
/// Because [`BodyMotion::pose_at`](super::body_motion::BodyMotion::pose_at)
/// rotates the core about its origin, this ball radius is rotation-invariant, so
/// one scalar bounds the core's reach at every sampled orientation. A point core
/// returns `0`, a segment (capsule) core its half-length, a box core its
/// half-diagonal.
fn core_bounding_radius(shape: &RoundedConvex) -> f32 {
    shape
        .hull
        .vertices()
        .iter()
        .fold(0.0_f32, |acc, v| acc.max(v.length()))
}

/// Conservative world-space bound of a rounded convex swept across `[0, dt]`,
/// inflated by `extra`.
///
/// The core's origin travels the segment from the start translation to
/// `translation + linear * dt`; the core stays inside a ball of radius
/// `core_bounding_radius + convex_radius` about that origin at every time, so the
/// swept region is that segment's box grown by the ball radius. `extra` adds the
/// speculative separation margin so a target reached only within `target_sep` is
/// still gathered.
fn swept_aabb(shape: &RoundedConvex, dt: f32, extra: f32) -> Aabb {
    let c0 = shape.pose.translation;
    let c1 = shape.pose.translation + shape.motion.linear * dt;
    let r = core_bounding_radius(shape) + shape.radius + extra;
    let lo = c0.min(c1) - Vec3::splat(r);
    let hi = c0.max(c1) + Vec3::splat(r);
    Aabb::new(lo, hi)
}

/// Gathers the candidate target indices for a cast: builds a linear `BVH` over
/// the targets' swept bounds and descends it with the shape's swept query box.
///
/// The returned indices are a superset of every truly-struck target (the swept
/// bounds are conservative), in `BVH`-traversal order. The per-query capacity is
/// the full target count, which a single query can never exceed, so the gather
/// cannot overflow; the `Err` arm nonetheless falls back to the whole scene so
/// correctness never depends on the gather succeeding.
fn candidates(
    shape: &RoundedConvex,
    targets: &[RoundedConvex],
    dt: f32,
    target_sep: f32,
) -> Vec<u32> {
    if targets.is_empty() {
        return Vec::new();
    }
    let target_boxes: Vec<Aabb> = targets.iter().map(|t| swept_aabb(t, dt, 0.0)).collect();
    let lbvh = cpu_build_lbvh(&target_boxes);
    // The query box carries the full speculative margin so a target reached only
    // within target_sep is still gathered; over-inclusion only adds candidates
    // the exact per-pair sweep then rejects.
    let query = swept_aabb(shape, dt, target_sep.max(0.0));
    let capacity = u32::try_from(targets.len()).unwrap_or(u32::MAX);
    match cpu_bvh_aabb_overlap(&lbvh, &[query], capacity) {
        Ok(mut per_query) => per_query.pop().unwrap_or_default(),
        Err(_) => (0..capacity).collect(),
    }
}

/// Replaces `best` with a hit on `target` at `toi` when it is strictly earlier,
/// or at an equal time strictly lower in target index.
///
/// Candidates arrive in `BVH`-traversal order rather than ascending index, so
/// the lower-index tie rule is applied explicitly rather than relying on visit
/// order, reproducing [`cast_shape`](super::shape_cast::cast_shape) exactly. The
/// comparison goes through `partial_cmp` so it never performs a direct float
/// equality test.
fn consider(best: &mut Option<ShapeCastHit>, candidate: ShapeCastHit) {
    let replace = best.is_none_or(|current| match candidate
        .toi
        .time
        .partial_cmp(&current.toi.time)
    {
        Some(core::cmp::Ordering::Less) => true,
        Some(core::cmp::Ordering::Equal) => candidate.target < current.target,
        _ => false,
    });
    if replace {
        *best = Some(candidate);
    }
}

/// Broad-phase-accelerated form of
/// [`cast_shape`](super::shape_cast::cast_shape): the earliest contact of the
/// moving `shape` against `targets`, or `None` when nothing is reached within
/// `dt`.
///
/// The struck target index, impact time, contact point, and normal are
/// identical to the brute-force cast, including the lower-index rule on an exact
/// time tie; only the work to find them scales with genuine overlap instead of
/// the full target count. See [`cast_shape`](super::shape_cast::cast_shape) for
/// the `target_sep` and normal conventions.
#[must_use]
pub fn cast_shape_bvh(
    shape: &RoundedConvex,
    targets: &[RoundedConvex],
    dt: f32,
    target_sep: f32,
) -> Option<ShapeCastHit> {
    let mut best: Option<ShapeCastHit> = None;
    for index in candidates(shape, targets, dt, target_sep) {
        let Some(toi) = sweep_against(shape, &targets[index as usize], dt, target_sep) else {
            continue;
        };
        consider(&mut best, ShapeCastHit { target: index, toi });
    }
    best
}

/// Broad-phase-accelerated form of
/// [`cast_shape_all`](super::shape_cast::cast_shape_all): every contact within
/// `dt`, ordered by increasing time of impact with ties broken by ascending
/// target index.
///
/// The ordering key is `(time, target index)` rather than time alone, because
/// candidates arrive in `BVH`-traversal order, not input order; this reproduces
/// the stable time-then-index order of the brute-force list exactly.
#[must_use]
pub fn cast_shape_all_bvh(
    shape: &RoundedConvex,
    targets: &[RoundedConvex],
    dt: f32,
    target_sep: f32,
) -> Vec<ShapeCastHit> {
    let mut hits: Vec<ShapeCastHit> = candidates(shape, targets, dt, target_sep)
        .into_iter()
        .filter_map(|index| {
            sweep_against(shape, &targets[index as usize], dt, target_sep)
                .map(|toi| ShapeCastHit { target: index, toi })
        })
        .collect();
    hits.sort_by(|lhs, rhs| {
        lhs.toi
            .time
            .partial_cmp(&rhs.toi.time)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(lhs.target.cmp(&rhs.target))
    });
    hits
}

#[cfg(test)]
mod tests {
    use super::super::body_motion::BodyMotion;
    use super::super::convex_hull::ConvexHull;
    use super::super::convex_pose::ConvexPose;
    use super::super::shape_cast::{cast_shape, cast_shape_all, RoundedConvex};
    use super::{cast_shape_all_bvh, cast_shape_bvh};
    use glam::{Quat, Vec3};

    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    /// A moving sphere core (point core + radius) sweeping along `+x` from the
    /// origin at the given speed.
    fn mover<'a>(sphere: &'a ConvexHull, speed: f32, radius: f32) -> RoundedConvex<'a> {
        RoundedConvex::new(
            sphere,
            ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
            BodyMotion::new(Vec3::new(speed, 0.0, 0.0), Vec3::ZERO),
            radius,
        )
    }

    #[test]
    fn bvh_cast_returns_the_nearest_blocker() {
        let sphere = ConvexHull::from_point();
        let shape = mover(&sphere, 10.0, 0.5);
        let box_hull = unit_box();
        let near =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let far =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hit = cast_shape_bvh(&shape, &[near, far], 1.0, 0.0).expect("the cast must hit the near box");
        assert_eq!(hit.target, 0, "the nearer box index 0 must win");
        assert!((hit.toi.time - 0.3).abs() < 0.02, "time {}", hit.toi.time);
        assert!(hit.toi.normal.x < -0.5, "normal {:?}", hit.toi.normal);
    }

    #[test]
    fn bvh_cast_misses_an_off_path_target() {
        let sphere = ConvexHull::from_point();
        let shape = mover(&sphere, 10.0, 0.5);
        let box_hull = unit_box();
        let aside =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 20.0, 0.0), Quat::IDENTITY), 0.0);
        assert!(
            cast_shape_bvh(&shape, &[aside], 1.0, 0.0).is_none(),
            "an off-path target must not be hit"
        );
    }

    #[test]
    fn bvh_cast_against_empty_scene_is_none() {
        let sphere = ConvexHull::from_point();
        let shape = mover(&sphere, 1.0, 0.5);
        assert!(
            cast_shape_bvh(&shape, &[], 1.0, 0.0).is_none(),
            "an empty scene cannot be hit"
        );
        assert!(
            cast_shape_all_bvh(&shape, &[], 1.0, 0.0).is_empty(),
            "an empty scene yields no hits"
        );
    }

    #[test]
    fn bvh_equal_time_ties_resolve_to_the_lower_index() {
        let sphere = ConvexHull::from_point();
        let shape = mover(&sphere, 10.0, 0.5);
        let box_hull = unit_box();
        let first =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let second =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hit = cast_shape_bvh(&shape, &[first, second], 1.0, 0.0).expect("the stacked boxes must be hit");
        assert_eq!(hit.target, 0, "an exact time tie resolves to the lower index");
    }

    #[test]
    fn bvh_cast_all_orders_every_hit_by_time() {
        let sphere = ConvexHull::from_point();
        let shape = mover(&sphere, 10.0, 0.5);
        let box_hull = unit_box();
        let far =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let near =
            RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0);
        let hits = cast_shape_all_bvh(&shape, &[far, near], 1.0, 0.0);
        assert_eq!(hits.len(), 2, "both boxes lie on the path");
        assert_eq!(hits[0].target, 1, "nearest hit first");
        assert_eq!(hits[1].target, 0, "farther hit second");
        assert!(hits[0].toi.time < hits[1].toi.time, "times must increase");
    }

    /// The headline guarantee: scene for scene the broad-phase cast returns
    /// exactly what the brute-force cast returns, for both the earliest hit and
    /// the full ordered list.
    #[test]
    fn bvh_cast_matches_brute_force_across_scenes() {
        let point = ConvexHull::from_point();
        let segment = ConvexHull::from_segment(Vec3::Y, 0.5);
        let box_hull = unit_box();

        // A spread of scenes: on-path blockers at mixed distances and radii, a
        // capsule mover, off-path decoys, exact distance ties, and a clean miss.
        let scenes: Vec<(RoundedConvex, Vec<RoundedConvex>)> = vec![
            (
                mover(&point, 10.0, 0.5),
                vec![
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, 0.0, 0.0), Quat::IDENTITY), 0.1),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(6.0, 30.0, 0.0), Quat::IDENTITY), 0.0),
                ],
            ),
            (
                RoundedConvex::new(
                    &segment,
                    ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
                    BodyMotion::new(Vec3::new(8.0, 0.0, 0.0), Vec3::ZERO),
                    0.3,
                ),
                vec![
                    RoundedConvex::still(&point, ConvexPose::new(Vec3::new(6.0, 0.0, 0.0), Quat::IDENTITY), 0.4),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY), 0.1),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(5.0, 40.0, 0.0), Quat::IDENTITY), 0.0),
                ],
            ),
            (
                mover(&point, 10.0, 0.5),
                vec![
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY), 0.0),
                ],
            ),
            (
                mover(&point, 10.0, 0.5),
                vec![
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(4.0, 25.0, 0.0), Quat::IDENTITY), 0.0),
                    RoundedConvex::still(&box_hull, ConvexPose::new(Vec3::new(8.0, -25.0, 0.0), Quat::IDENTITY), 0.0),
                ],
            ),
            (mover(&point, 5.0, 0.5), vec![]),
        ];

        for (scene, (shape, targets)) in scenes.iter().enumerate() {
            for &sep in &[0.0_f32, 0.1] {
                let brute = cast_shape(shape, targets, 1.0, sep);
                let bvh = cast_shape_bvh(shape, targets, 1.0, sep);
                assert_eq!(
                    brute, bvh,
                    "scene {scene} sep {sep}: earliest hit differs: brute {brute:?} vs bvh {bvh:?}"
                );

                let brute_all = cast_shape_all(shape, targets, 1.0, sep);
                let bvh_all = cast_shape_all_bvh(shape, targets, 1.0, sep);
                assert_eq!(
                    brute_all, bvh_all,
                    "scene {scene} sep {sep}: ordered hit list differs"
                );
            }
        }
    }
}
