//! Conservative-advancement continuous collision detection for two moving
//! convex hulls: the earliest time of impact (`TOI`) within a substep.
//!
//! A discrete narrow phase samples the gap only at the substep endpoints, so a
//! thin or fast body can pass clean through another between samples (tunnelling).
//! Conservative advancement closes that gap. It repeatedly measures the current
//! separation with [`gjk`](super::gjk::gjk), bounds the fastest rate at which
//! that separation can shrink under the two bodies' [`BodyMotion`]s, and
//! advances time by the largest step that provably cannot close the gap. The
//! sequence of times increases monotonically and converges on the first contact,
//! so no impact earlier than the reported one is ever skipped.
//!
//! # Closing-speed bound
//!
//! At a sampled time the separation distance is `d` along the contact normal
//! `n` (pointing from `B` toward `A`). The relative linear velocity closes the
//! gap at `(v_a - v_b) . (-n)`. Rotation of either hull cannot move a surface
//! point toward the other faster than `|omega| * r`, where `r` is the hull's
//! circumradius about its local origin (the pivot the motion rotates about).
//! Summing the linear closing speed and both rotational bounds gives `mu`, an
//! upper bound on how fast `d` can decrease. Advancing time by `(d - target) /
//! mu` therefore cannot overshoot the first instant the surfaces reach the
//! `target` separation. When `mu` is non-positive the gap cannot close and the
//! pair never touches within the substep.
//!
//! Provenance: conservative advancement after Brian Mirtich, *Timewarp Rigid
//! Body Simulation* (2000), and the ray-casting formulation of Gino van den
//! Bergen, *Ray Casting against General Convex Objects with Application to
//! Continuous Collision Detection* (2004). No Unreal Engine source or derived
//! code.

use glam::Vec3;

use super::body_motion::BodyMotion;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::gjk::{gjk, GjkStatus};

/// Separation below which [`gjk`] is treated as touching, so the advance stops
/// rather than chasing an ever-shrinking gap.
const DISTANCE_TOL: f32 = 1.0e-4;

/// Closing-speed floor: when the bounded closing speed `mu` is at or below this
/// the gap is treated as unable to close and the pair is reported as missing.
const CLOSING_EPS: f32 = 1.0e-8;

/// Hard cap on advancement iterations so a grazing or slowly closing pair
/// terminates rather than spins.
const MAX_ITERS: u32 = 64;

/// A time of impact between two swept convex hulls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Toi {
    /// Fraction of the substep at which the surfaces reach the target
    /// separation, in `[0, dt]` (same units as the `dt` passed to the solver).
    pub time: f32,
    /// World-space contact point at the time of impact (midpoint of the two
    /// closest witnesses).
    pub point: Vec3,
    /// Unit contact normal at the time of impact, pointing from `B` toward `A`.
    pub normal: Vec3,
}

/// A directed pair of body indices to sweep against each other.
///
/// `a` and `b` index into the shared hull, pose, and motion tables the batch
/// entry points consume. The normal of any resulting [`Toi`] points from body
/// `b` toward body `a`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvexConvexSweepPair {
    /// Index of hull `A` (the body the contact normal points toward).
    pub a: u32,
    /// Index of hull `B`.
    pub b: u32,
}

impl ConvexConvexSweepPair {
    /// Builds a sweep pair from the two body indices.
    #[must_use]
    pub fn new(a: u32, b: u32) -> ConvexConvexSweepPair {
        ConvexConvexSweepPair { a, b }
    }
}

/// The circumradius of a hull about its local origin: the farthest local vertex
/// distance, i.e. the largest radius a surface point sweeps when the hull turns
/// about the origin its motion pivots on.
#[must_use]
fn circumradius(hull: &ConvexHull) -> f32 {
    hull.vertices()
        .iter()
        .map(|v| v.length())
        .fold(0.0_f32, f32::max)
}

/// Conservative-advancement time of impact for a single swept convex pair.
///
/// Advances both bodies under their motions from the start of the substep until
/// their surfaces reach `target` separation (a speculative margin; pass `0.0`
/// for touching contact) or the substep `dt` elapses. Returns the first
/// [`Toi`], or `None` when the pair stays farther apart than `target` for the
/// whole substep.
///
/// When the hulls already overlap at the start of the substep the impact is
/// reported at the current time with the overlap seed direction, so the caller
/// can resolve the existing penetration rather than miss it.
#[must_use]
pub fn conservative_advancement_toi(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    motion_a: &BodyMotion,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    motion_b: &BodyMotion,
    dt: f32,
    target: f32,
) -> Option<Toi> {
    let r_a = circumradius(hull_a);
    let r_b = circumradius(hull_b);
    // Rotational contribution to the closing-speed bound is constant across the
    // sweep: both angular rates and both circumradii are fixed.
    let ang_bound = motion_a.angular.length() * r_a + motion_b.angular.length() * r_b;
    let rel_linear = motion_a.linear - motion_b.linear;

    // The normal at the touching iteration can degenerate to a fallback when the
    // gap is numerically zero, so keep the last well-separated normal to report
    // at impact.
    let mut last_normal = rel_linear.try_normalize().map_or(Vec3::X, |n| -n);
    let mut t = 0.0_f32;
    for _ in 0..MAX_ITERS {
        let pa = motion_a.pose_at(pose_a, t);
        let pb = motion_b.pose_at(pose_b, t);
        match gjk(hull_a, &pa, hull_b, &pb) {
            GjkStatus::Separated {
                distance,
                point_a,
                point_b,
                normal,
            } => {
                // A separation above the degeneracy floor yields a trustworthy
                // normal; remember it for the impact report.
                if distance > CLOSING_EPS {
                    last_normal = normal;
                }
                if distance <= target + DISTANCE_TOL {
                    return Some(Toi {
                        time: t,
                        point: (point_a + point_b) * 0.5,
                        normal: last_normal,
                    });
                }
                // Fastest rate the gap along the normal can shrink.
                let lin_closing = rel_linear.dot(-last_normal);
                let mu = lin_closing + ang_bound;
                if mu <= CLOSING_EPS {
                    return None;
                }
                t += (distance - target) / mu;
                if t > dt {
                    return None;
                }
            }
            GjkStatus::Intersecting(_) => {
                // The surfaces have met (or already overlapped at the start of
                // the substep): report the impact now with the last trustworthy
                // approach normal, which is the direction the advance closed
                // along.
                return Some(Toi {
                    time: t,
                    point: pa.translation,
                    normal: last_normal,
                });
            }
        }
    }
    None
}

/// Conservative-advancement time of impact for a batch of swept convex pairs.
///
/// `hulls`, `poses`, and `motions` are body-indexed and share a common length;
/// each [`ConvexConvexSweepPair`] names two bodies by index. The returned vector
/// is aligned with `pairs`: entry `i` is the [`Toi`] of `pairs[i]`, or `None`
/// when that pair never reaches `target` separation within `dt`.
#[must_use]
pub fn cpu_convex_convex_toi(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    pairs: &[ConvexConvexSweepPair],
    dt: f32,
    target: f32,
) -> Vec<Option<Toi>> {
    pairs
        .iter()
        .map(|pair| {
            let ia = pair.a as usize;
            let ib = pair.b as usize;
            conservative_advancement_toi(
                &hulls[ia],
                &poses[ia],
                &motions[ia],
                &hulls[ib],
                &poses[ib],
                &motions[ib],
                dt,
                target,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    #[test]
    fn head_on_approach_reports_a_mid_substep_impact() {
        let hull = unit_box();
        // A at x = -3 moving +x, B at x = +3 moving -x. They are 5 apart
        // (gap between faces at x = -2.5 and x = +2.5) and close at 4 per unit.
        let pose_a = ConvexPose::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 2.0, 0.0,
        )
        .expect("head-on boxes must collide");
        // Gap 5, closing speed 4 => contact near t = 1.25.
        assert!(toi.time > 0.0 && toi.time < 2.0, "time {}", toi.time);
        assert!((toi.time - 1.25).abs() < 0.05, "time {}", toi.time);
        // Normal points from B (+x) toward A (-x): roughly -x.
        assert!(toi.normal.x < -0.5, "normal {:?}", toi.normal);
    }

    #[test]
    fn parallel_miss_reports_no_impact() {
        let hull = unit_box();
        // Both slide along +x, three apart on y: never approach.
        let pose_a = ConvexPose::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(-3.0, 3.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 1.0, 0.0,
        );
        assert!(toi.is_none(), "parallel slide must miss: {toi:?}");
    }

    #[test]
    fn separating_pair_reports_no_impact() {
        let hull = unit_box();
        // Already apart and moving farther apart.
        let pose_a = ConvexPose::new(Vec3::new(-2.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 1.0, 0.0,
        );
        assert!(toi.is_none(), "separating pair must miss: {toi:?}");
    }

    #[test]
    fn grazing_approach_misses_within_the_substep() {
        let hull = unit_box();
        // Approach that would collide eventually but dt is too short.
        let pose_a = ConvexPose::new(Vec3::new(-5.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO);
        // Gap 9, closing 2 => impact ~t=4.5, but dt=1.0.
        let toi = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 1.0, 0.0,
        );
        assert!(toi.is_none(), "impact beyond dt must be skipped: {toi:?}");
    }

    #[test]
    fn rotating_sweep_still_finds_the_impact() {
        let hull = ConvexHull::from_box(Vec3::new(1.0, 0.2, 0.2));
        // A long thin box at the origin spinning about z; B approaches on +x.
        let pose_a = ConvexPose::new(Vec3::ZERO, Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0));
        let motion_b = BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 2.0, 0.0,
        )
        .expect("approaching bodies must collide");
        assert!(toi.time > 0.0 && toi.time <= 2.0, "time {}", toi.time);
    }

    #[test]
    fn batch_aligns_results_with_pairs() {
        let hull = unit_box();
        let hulls = vec![hull.clone(), hull.clone(), hull];
        let poses = vec![
            ConvexPose::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY),
            ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY),
            ConvexPose::new(Vec3::new(0.0, 20.0, 0.0), Quat::IDENTITY),
        ];
        let motions = vec![
            BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO),
            BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO),
            BodyMotion::still(),
        ];
        let pairs = vec![
            ConvexConvexSweepPair::new(0, 1),
            ConvexConvexSweepPair::new(0, 2),
        ];
        let out = cpu_convex_convex_toi(&hulls, &poses, &motions, &pairs, 2.0, 0.0);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_some(), "0-1 head-on must hit");
        assert!(out[1].is_none(), "0-2 far apart must miss");
    }
}
