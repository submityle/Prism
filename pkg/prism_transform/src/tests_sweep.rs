//! §24.8 tests: conservative swept-transform bounds for CCD.
//!
//! The headline property is **conservativeness**: the swept box must contain
//! the proxy at every instant of the interpolated motion, not just at the
//! endpoints. We verify this by densely sampling the motion and asserting each
//! exact sampled world box is contained in the swept bound — for pure
//! translation, pure rotation, and a combined translate/rotate/scale sweep.
//! We also cover the hand-computable translation union, teleport collapse, the
//! `still` degenerate case, and the subdivision contract.

use crate::sweep::{SweepSegment, SweptMotion};
use crate::{GlobalTransform, Transform};
use prism_math::{Aabb3, Quat, Vec3};

fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb3 {
    Aabb3::new(
        Vec3::new(min[0], min[1], min[2]),
        Vec3::new(max[0], max[1], max[2]),
    )
}

fn pose(t: Transform) -> GlobalTransform {
    GlobalTransform::from_transform(&t)
}

/// Assert the swept bound contains every densely-sampled instantaneous box.
fn assert_conservative(motion: &SweptMotion, samples: usize) {
    let bounds = motion.bounds();
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let inst = motion.box_at(t);
        assert!(
            bounds.contains_aabb(inst),
            "swept bounds {bounds:?} fails to contain box at t={t}: {inst:?}",
        );
    }
}

// ---- pure translation: union with no angular expansion ----------------------

#[test]
fn pure_translation_bounds_is_endpoint_union() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let prev = pose(Transform::from_xyz(0.0, 0.0, 0.0));
    let curr = pose(Transform::from_xyz(10.0, 0.0, 0.0));
    let motion = SweptMotion::new(prev, curr, local);

    assert!((motion.angular_motion()).abs() < 1e-6);
    let bounds = motion.bounds();
    // Union of [-1,1] and [9,11] on X; Y/Z unchanged. No rotation => no expand.
    let eps = 1e-5;
    assert!((bounds.min.x - (-1.0)).abs() < eps);
    assert!((bounds.max.x - 11.0).abs() < eps);
    assert!((bounds.min.y - (-1.0)).abs() < eps);
    assert!((bounds.max.y - 1.0).abs() < eps);
    assert_conservative(&motion, 128);
}

#[test]
fn translation_delta_matches_motion() {
    let local = aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
    let prev = pose(Transform::from_xyz(1.0, 2.0, 3.0));
    let curr = pose(Transform::from_xyz(4.0, 6.0, 8.0));
    let motion = SweptMotion::new(prev, curr, local);
    let d = motion.translation_delta();
    assert!((d - Vec3::new(3.0, 4.0, 5.0)).length() < 1e-5);
}

// ---- pure rotation: arc bulge must be captured ------------------------------

#[test]
fn rotation_sweep_is_conservative_and_expands_past_union() {
    // A box sitting out on +X, swept through 120 deg about Z about the origin:
    // its midpoint pose bulges well beyond the union of the endpoint boxes.
    let local = aabb([1.0, -0.5, -0.5], [2.0, 0.5, 0.5]);
    let prev = pose(Transform::IDENTITY);
    let curr = pose(Transform::from_rotation(Quat::from_rotation_z(
        2.0 * core::f32::consts::PI / 3.0,
    )));
    let motion = SweptMotion::new(prev, curr, local);

    // Angular motion is ~120 deg.
    assert!((motion.angular_motion() - 2.094_395).abs() < 1e-3);

    // The swept bound must strictly grow past the raw endpoint union on every
    // axis (the arc-length expansion is non-zero).
    let union = motion.box_at(0.0).merge(motion.box_at(1.0));
    let bounds = motion.bounds();
    assert!(bounds.min.x < union.min.x);
    assert!(bounds.min.y < union.min.y);
    assert!(bounds.max.x > union.max.x);
    assert!(bounds.max.y > union.max.y);

    // The conservative bound contains the mid-sweep pose with room to spare.
    let mid = motion.box_at(0.5);
    assert!(bounds.contains_aabb(mid));

    assert_conservative(&motion, 256);
}

#[test]
fn combined_translate_rotate_scale_is_conservative() {
    let local = aabb([-0.5, -1.0, -0.25], [1.5, 1.0, 0.75]);
    let prev = pose(Transform {
        translation: Vec3::new(-3.0, 1.0, 0.0),
        rotation: Quat::from_rotation_y(0.3),
        scale: Vec3::new(1.0, 1.0, 1.0),
    });
    let curr = pose(Transform {
        translation: Vec3::new(6.0, -2.0, 4.0),
        rotation: Quat::from_rotation_y(0.3) * Quat::from_rotation_z(1.1),
        scale: Vec3::new(2.0, 0.5, 1.5),
    });
    let motion = SweptMotion::new(prev, curr, local);
    assert_conservative(&motion, 512);
}

// ---- teleport ---------------------------------------------------------------

#[test]
fn teleport_collapses_bounds_and_zeroes_velocity() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let curr = pose(Transform::from_xyz(100.0, 0.0, 0.0));
    let motion = SweptMotion::teleported(curr, local);

    assert!(motion.is_teleport());
    assert!(motion.translation_delta().length() < 1e-6);
    assert!(motion.angular_motion().abs() < 1e-6);

    let bounds = motion.bounds();
    let expected = motion.box_at(1.0);
    let eps = 1e-5;
    assert!((bounds.min - expected.min).length() < eps);
    assert!((bounds.max - expected.max).length() < eps);
    // No smear: the box is exactly the destination box, not a union with origin.
    assert!(bounds.min.x > 98.0);
}

#[test]
fn still_motion_is_static_box() {
    let local = aabb([-1.0, -1.0, -1.0], [2.0, 1.0, 1.0]);
    let p = pose(Transform::from_xyz(5.0, 0.0, 0.0));
    let motion = SweptMotion::still(p, local);
    assert!(motion.translation_delta().length() < 1e-6);
    let bounds = motion.bounds();
    let eps = 1e-5;
    assert!((bounds.min - Vec3::new(4.0, -1.0, -1.0)).length() < eps);
    assert!((bounds.max - Vec3::new(7.0, 1.0, 1.0)).length() < eps);
}

// ---- subdivision ------------------------------------------------------------

#[test]
fn subdivide_covers_interval_and_each_slice_is_conservative() {
    let local = aabb([1.0, -0.5, -0.5], [2.0, 0.5, 0.5]);
    let prev = pose(Transform::IDENTITY);
    let curr = pose(Transform::from_rotation(Quat::from_rotation_z(
        2.0 * core::f32::consts::PI / 3.0,
    )));
    let motion = SweptMotion::new(prev, curr, local);

    let segments = motion.subdivide(8);
    assert_eq!(segments.len(), 8);
    // Contiguous, ordered partition of [0, 1].
    assert!((segments[0].t0 - 0.0).abs() < 1e-6);
    assert!((segments[7].t1 - 1.0).abs() < 1e-6);
    for w in segments.windows(2) {
        assert!((w[0].t1 - w[1].t0).abs() < 1e-6);
    }

    let whole = motion.bounds();
    for seg in &segments {
        // Each slice box is itself conservative over its sub-interval ...
        let steps = 16;
        for i in 0..=steps {
            let t = seg.t0 + (seg.t1 - seg.t0) * (i as f32 / steps as f32);
            assert!(seg.bounds.contains_aabb(motion.box_at(t)));
        }
        // ... and never pokes outside the whole-sweep bound.
        assert!(whole.contains_aabb(seg.bounds));
    }
}

#[test]
fn finer_subdivision_tightens_total_volume() {
    // Summing slice volumes is not meaningful (overlap), but the max slice box
    // of a finer partition should be no larger than the single whole box.
    let local = aabb([1.0, -0.5, -0.5], [2.0, 0.5, 0.5]);
    let prev = pose(Transform::IDENTITY);
    let curr = pose(Transform::from_rotation(Quat::from_rotation_z(
        2.0 * core::f32::consts::PI / 3.0,
    )));
    let motion = SweptMotion::new(prev, curr, local);

    let whole = motion.bounds().volume();
    let coarse = max_slice_volume(&motion.subdivide(2));
    let fine = max_slice_volume(&motion.subdivide(16));
    assert!(coarse <= whole + 1e-3);
    assert!(fine <= coarse + 1e-3);
}

fn max_slice_volume(segments: &[SweepSegment]) -> f32 {
    let mut m = 0.0_f32;
    for s in segments {
        m = m.max(s.bounds.volume());
    }
    m
}

#[test]
fn subdivide_zero_is_treated_as_one() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let motion = SweptMotion::new(
        pose(Transform::from_xyz(0.0, 0.0, 0.0)),
        pose(Transform::from_xyz(4.0, 0.0, 0.0)),
        local,
    );
    let segs = motion.subdivide(0);
    assert_eq!(segs.len(), 1);
    assert!((segs[0].t0 - 0.0).abs() < 1e-6);
    assert!((segs[0].t1 - 1.0).abs() < 1e-6);
}

#[test]
fn pose_at_endpoints_match_prev_and_curr() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let prev = pose(Transform::from_xyz(1.0, 0.0, 0.0));
    let curr = pose(Transform::from_xyz(9.0, 0.0, 0.0));
    let motion = SweptMotion::new(prev, curr, local);
    assert!((motion.pose_at(0.0).translation() - prev.translation()).length() < 1e-6);
    assert!((motion.pose_at(1.0).translation() - curr.translation()).length() < 1e-6);
    // Clamp: t outside [0,1] saturates.
    assert!((motion.pose_at(-1.0).translation() - prev.translation()).length() < 1e-6);
    assert!((motion.pose_at(2.0).translation() - curr.translation()).length() < 1e-6);
}
