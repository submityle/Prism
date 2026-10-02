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

/// Conservative-advancement time of impact for a single swept pair of
/// **sharp** (zero convex-radius) hulls.
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
///
/// This is the sharp-hull special case of
/// [`conservative_advancement_toi_rounded`] with both convex radii zero.
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
    conservative_advancement_toi_rounded(
        hull_a, pose_a, motion_a, 0.0, hull_b, pose_b, motion_b, 0.0, dt, target,
    )
}

/// Conservative-advancement time of impact for a single swept pair of
/// **rounded** convex shapes: each hull inflated by its own convex radius.
///
/// A rounded shape is a convex core (the `hull`) swept by a sphere of radius
/// `radius`. This is the shape model AAA engines use for their primitives: a
/// sphere is [`ConvexHull::from_point`] plus its radius, a capsule is
/// [`ConvexHull::from_segment`] plus its cap radius, and a rounded box is a box
/// hull plus a small bevel radius. Casting the inflated shapes needs no new
/// support geometry: the Minkowski difference of two rounded convex bodies is
/// the Minkowski difference of their cores grown by the sum of the radii, so the
/// cores' separation simply has to reach `target + radius_a + radius_b` instead
/// of `target` for the surfaces to touch.
///
/// The closing-speed bound is unchanged by the radii (a constant surface offset
/// moves no faster than the core it rides on), so the advance keeps the same
/// provable no-overshoot guarantee. The reported contact point is the midpoint
/// of the two inflated surfaces, found by pushing each core witness out along
/// the contact normal by that body's radius.
///
/// Returns the first [`Toi`], or `None` when the inflated surfaces stay farther
/// apart than `target` for the whole substep. When the cores already overlap at
/// the start of the substep (so the inflated shapes certainly do) the impact is
/// reported at the current time with the overlap seed direction.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a swept rounded pair is two full (hull, pose, motion, radius) bundles \
              plus the step dt and target gap; grouping them into a struct would \
              only hide the symmetry the call site reads directly"
)]
pub fn conservative_advancement_toi_rounded(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    motion_a: &BodyMotion,
    radius_a: f32,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    motion_b: &BodyMotion,
    radius_b: f32,
    dt: f32,
    target: f32,
) -> Option<Toi> {
    let r_a = circumradius(hull_a);
    let r_b = circumradius(hull_b);
    // Rotational contribution to the closing-speed bound is constant across the
    // sweep: both angular rates and both circumradii are fixed.
    let ang_bound = motion_a.angular.length() * r_a + motion_b.angular.length() * r_b;
    let rel_linear = motion_a.linear - motion_b.linear;

    // The cores only have to close to this separation for the inflated surfaces
    // to reach the requested target gap.
    let core_target = target + radius_a + radius_b;

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
                if distance <= core_target + DISTANCE_TOL {
                    // Push each core witness out to its inflated surface along
                    // the contact normal (B toward A), then report the midpoint.
                    let surf_a = point_a - last_normal * radius_a;
                    let surf_b = point_b + last_normal * radius_b;
                    return Some(Toi {
                        time: t,
                        point: (surf_a + surf_b) * 0.5,
                        normal: last_normal,
                    });
                }
                // Fastest rate the gap along the normal can shrink.
                let lin_closing = rel_linear.dot(-last_normal);
                let mu = lin_closing + ang_bound;
                if mu <= CLOSING_EPS {
                    return None;
                }
                t += (distance - core_target) / mu;
                if t > dt {
                    return None;
                }
            }
            GjkStatus::Intersecting(_) => {
                // The cores have met (or already overlapped at the start of the
                // substep), so the inflated shapes certainly have: report the
                // impact now with the last trustworthy approach normal, which is
                // the direction the advance closed along.
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

/// Conservative-advancement time of impact for a batch of swept **sharp**
/// (zero convex-radius) convex pairs.
///
/// `hulls`, `poses`, and `motions` are body-indexed and share a common length;
/// each [`ConvexConvexSweepPair`] names two bodies by index. The returned vector
/// is aligned with `pairs`: entry `i` is the [`Toi`] of `pairs[i]`, or `None`
/// when that pair never reaches `target` separation within `dt`.
///
/// This is the sharp-hull special case of [`cpu_convex_convex_toi_rounded`]
/// with every body's convex radius zero.
#[must_use]
pub fn cpu_convex_convex_toi(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    pairs: &[ConvexConvexSweepPair],
    dt: f32,
    target: f32,
) -> Vec<Option<Toi>> {
    let radii = vec![0.0_f32; hulls.len()];
    cpu_convex_convex_toi_rounded(hulls, poses, motions, &radii, pairs, dt, target)
}

/// Conservative-advancement time of impact for a batch of swept **rounded**
/// convex pairs: every body inflated by its own convex radius.
///
/// `hulls`, `poses`, `motions`, and `radii` are body-indexed and share a common
/// length; `radii[i]` is the convex radius of body `i` (the sphere/capsule cap
/// radius, or a box bevel). Each [`ConvexConvexSweepPair`] names two bodies by
/// index. The returned vector is aligned with `pairs`: entry `i` is the [`Toi`]
/// of `pairs[i]`, or `None` when that inflated pair never reaches `target`
/// separation within `dt`. See [`conservative_advancement_toi_rounded`] for the
/// per-pair rounded-shape model.
#[must_use]
pub fn cpu_convex_convex_toi_rounded(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    radii: &[f32],
    pairs: &[ConvexConvexSweepPair],
    dt: f32,
    target: f32,
) -> Vec<Option<Toi>> {
    pairs
        .iter()
        .map(|pair| {
            let ia = pair.a as usize;
            let ib = pair.b as usize;
            conservative_advancement_toi_rounded(
                &hulls[ia],
                &poses[ia],
                &motions[ia],
                radii[ia],
                &hulls[ib],
                &poses[ib],
                &motions[ib],
                radii[ib],
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

    #[test]
    fn rounded_zero_radius_matches_the_sharp_cast() {
        // With both radii zero the rounded cast must agree with the sharp one.
        let hull = unit_box();
        let pose_a = ConvexPose::new(Vec3::new(-3.0, 0.07, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let sharp = conservative_advancement_toi(
            &hull, &pose_a, &motion_a, &hull, &pose_b, &motion_b, 2.0, 0.0,
        );
        let rounded = conservative_advancement_toi_rounded(
            &hull, &pose_a, &motion_a, 0.0, &hull, &pose_b, &motion_b, 0.0, 2.0, 0.0,
        );
        assert_eq!(sharp, rounded, "zero-radius rounded must equal sharp");
    }

    #[test]
    fn two_spheres_touch_at_the_analytic_time() {
        // Sphere cores are single points; the inflated surfaces touch when the
        // centres are radius_a + radius_b apart. Centres start 6 apart on x and
        // close at 2 + 2 = 4 per unit, so surfaces (1.0 apart core target) meet
        // at t = (6 - 1) / 4 = 1.25.
        let sphere = ConvexHull::from_point();
        let pose_a = ConvexPose::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi_rounded(
            &sphere, &pose_a, &motion_a, 0.5, &sphere, &pose_b, &motion_b, 0.5, 2.0, 0.0,
        )
        .expect("closing spheres must collide");
        assert!((toi.time - 1.25).abs() < 1.0e-3, "time {}", toi.time);
        // Contact point sits on the x axis midway between the two surfaces,
        // which at impact both land on the origin.
        assert!(toi.point.x.abs() < 1.0e-2, "point x {}", toi.point.x);
        // Normal points from B (+x) toward A (-x).
        assert!(toi.normal.x < -0.9, "normal {:?}", toi.normal);
    }

    #[test]
    fn sphere_versus_static_box_touches_at_the_analytic_time() {
        // A radius-0.5 sphere centre starts at x = -3 and moves +x at 4. The
        // static unit box spans [-0.5, 0.5]; the sphere surface reaches the box
        // face (core gap = radius 0.5) when the centre is at x = -1.0, i.e. after
        // travelling 2.0 at speed 4.0 => t = 0.5.
        let sphere = ConvexHull::from_point();
        let box_hull = unit_box();
        let pose_a = ConvexPose::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::ZERO, Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::still();
        let toi = conservative_advancement_toi_rounded(
            &sphere, &pose_a, &motion_a, 0.5, &box_hull, &pose_b, &motion_b, 0.0, 1.0, 0.0,
        )
        .expect("sphere must reach the box");
        assert!((toi.time - 0.5).abs() < 1.0e-3, "time {}", toi.time);
        assert!(toi.normal.x < -0.9, "normal {:?}", toi.normal);
    }

    #[test]
    fn larger_radii_report_an_earlier_impact() {
        // Inflating the shapes closes the surface gap sooner, so a larger convex
        // radius must never report a later impact than a smaller one.
        let sphere = ConvexHull::from_point();
        let pose_a = ConvexPose::new(Vec3::new(-5.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let small = conservative_advancement_toi_rounded(
            &sphere, &pose_a, &motion_a, 0.5, &sphere, &pose_b, &motion_b, 0.5, 5.0, 0.0,
        )
        .expect("small spheres still collide within dt");
        let large = conservative_advancement_toi_rounded(
            &sphere, &pose_a, &motion_a, 1.5, &sphere, &pose_b, &motion_b, 1.5, 5.0, 0.0,
        )
        .expect("large spheres collide within dt");
        assert!(large.time < small.time, "large {} small {}", large.time, small.time);
    }

    #[test]
    fn rounded_separating_pair_reports_no_impact() {
        // Spheres already apart and moving farther apart never touch, however
        // large the radii, as long as they stay below the opening gap.
        let sphere = ConvexHull::from_point();
        let pose_a = ConvexPose::new(Vec3::new(-2.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY);
        let motion_a = BodyMotion::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO);
        let motion_b = BodyMotion::new(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let toi = conservative_advancement_toi_rounded(
            &sphere, &pose_a, &motion_a, 0.5, &sphere, &pose_b, &motion_b, 0.5, 1.0, 0.0,
        );
        assert!(toi.is_none(), "separating spheres must miss: {toi:?}");
    }

    #[test]
    fn rounded_batch_aligns_results_with_pairs() {
        let sphere = ConvexHull::from_point();
        let hulls = vec![sphere.clone(), sphere.clone(), sphere];
        let radii = vec![0.5_f32, 0.5, 0.5];
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
        let out = cpu_convex_convex_toi_rounded(&hulls, &poses, &motions, &radii, &pairs, 2.0, 0.0);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_some(), "0-1 closing spheres must hit");
        assert!(out[1].is_none(), "0-2 far apart must miss");
    }
}
