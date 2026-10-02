//! Minkowski-difference support point: the single query `GJK` and `EPA` make of
//! a pair of posed convex hulls.
//!
//! For two convex solids `A` and `B`, the Minkowski difference `A (-) B` is itself
//! convex, and the pair overlaps exactly when that difference contains the
//! origin. Its support in a direction `d` is
//! `support_A(d) - support_B(-d)`: push `A` as far as possible along `d` and
//! `B` as far as possible along `-d`. [`support`] evaluates exactly that and
//! keeps the two world-space witness vertices alongside the difference, so once
//! the simplex that encloses or nearest-approaches the origin is known, the
//! contact points on each body fall straight out of the stored witnesses.
//!
//! Provenance: textbook Minkowski-difference support mapping for `GJK`/`EPA`;
//! no Unreal Engine source or derived code.

use glam::Vec3;

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;

/// One point of the Minkowski difference `A (-) B`, carrying the world-space
/// witness vertex on each hull so contact points survive the simplex reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportPoint {
    /// The difference `on_a - on_b`; the simplex lives in this space.
    pub diff: Vec3,
    /// World-space support vertex on hull `A`.
    pub on_a: Vec3,
    /// World-space support vertex on hull `B`.
    pub on_b: Vec3,
}

/// The Minkowski-difference support of `(hull_a, pose_a)` minus
/// `(hull_b, pose_b)` along world direction `dir`.
///
/// Picks the farthest vertex of `A` along `dir` and of `B` along `-dir`,
/// both in world space, and returns their difference with the witnesses kept.
/// `dir` need not be normalised.
#[must_use]
pub fn support(
    hull_a: &ConvexHull,
    pose_a: &ConvexPose,
    hull_b: &ConvexHull,
    pose_b: &ConvexPose,
    dir: Vec3,
) -> SupportPoint {
    let on_a = pose_a.transform_point(hull_a.vertices()[hull_a.support_local(pose_a.inverse_rotate(dir)) as usize]);
    let on_b = pose_b.transform_point(hull_b.vertices()[hull_b.support_local(pose_b.inverse_rotate(-dir)) as usize]);
    SupportPoint {
        diff: on_a - on_b,
        on_a,
        on_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn support_of_two_unit_boxes_matches_hand_computation() {
        let a = ConvexHull::from_box(Vec3::splat(1.0));
        let b = ConvexHull::from_box(Vec3::splat(1.0));
        let pose_a = ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
        let pose_b = ConvexPose::identity();
        // Along +x: A's farthest vertex is x = 3 + 1 = 4, B's along -x is x = -1,
        // so the difference's x is 4 - (-1) = 5.
        let s = support(&a, &pose_a, &b, &pose_b, Vec3::X);
        assert!((s.on_a.x - 4.0).abs() < 1.0e-5, "on_a.x {}", s.on_a.x);
        assert!((s.on_b.x + 1.0).abs() < 1.0e-5, "on_b.x {}", s.on_b.x);
        assert!((s.diff.x - 5.0).abs() < 1.0e-5, "diff.x {}", s.diff.x);
    }

    #[test]
    fn support_direction_rotates_into_each_local_frame() {
        let a = ConvexHull::from_box(Vec3::new(2.0, 1.0, 1.0));
        let b = ConvexHull::from_box(Vec3::splat(1.0));
        let pose_a = ConvexPose::new(
            Vec3::ZERO,
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        );
        let pose_b = ConvexPose::identity();
        // A is a 2x1x1 box turned +90 about z, so its long axis now lies on world
        // y: the +y support reaches y = 2.
        let s = support(&a, &pose_a, &b, &pose_b, Vec3::Y);
        assert!((s.on_a.y - 2.0).abs() < 1.0e-5, "on_a.y {}", s.on_a.y);
    }
}
