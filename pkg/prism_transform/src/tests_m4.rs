//! M4 tests: fixed-step render interpolation and teleport handling.

use prism_math::{Affine3, Quat, Vec3};

use crate::interpolation::{InterpolationBuffer, clamp_alpha};
use crate::{GlobalTransform, Transform};

fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

fn approx_quat(a: Quat, b: Quat, eps: f32) -> bool {
    // q and -q are the same rotation; compare by absolute dot near 1.
    let d = (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w).abs();
    (1.0 - d).abs() <= eps
}

#[test]
fn clamp_alpha_bounds_and_nan() {
    assert_eq!(clamp_alpha(-0.5), 0.0);
    assert_eq!(clamp_alpha(0.0), 0.0);
    assert_eq!(clamp_alpha(0.3), 0.3);
    assert_eq!(clamp_alpha(1.0), 1.0);
    assert_eq!(clamp_alpha(2.5), 1.0);
    assert_eq!(clamp_alpha(f32::NAN), 0.0);
}

#[test]
fn transform_interpolate_endpoints() {
    let a = Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::new(1.0, 1.0, 1.0));
    let b = Transform::from_xyz(10.0, -4.0, 2.0).with_scale(Vec3::new(3.0, 3.0, 3.0));

    let at0 = a.interpolate(&b, 0.0);
    assert!(approx_vec(at0.translation, a.translation, 1e-6));
    assert!(approx_vec(at0.scale, a.scale, 1e-6));

    let at1 = a.interpolate(&b, 1.0);
    assert!(approx_vec(at1.translation, b.translation, 1e-6));
    assert!(approx_vec(at1.scale, b.scale, 1e-6));
}

#[test]
fn transform_interpolate_midpoint_translation_scale() {
    let a = Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::new(1.0, 2.0, 4.0));
    let b = Transform::from_xyz(8.0, 8.0, 8.0).with_scale(Vec3::new(3.0, 4.0, 8.0));
    let m = a.interpolate(&b, 0.5);
    assert!(approx_vec(m.translation, Vec3::new(4.0, 4.0, 4.0), 1e-6));
    assert!(approx_vec(m.scale, Vec3::new(2.0, 3.0, 6.0), 1e-6));
}

#[test]
fn transform_interpolate_rotation_shortest_path() {
    // 0° -> 90° about Y, blended halfway should be ~45° about Y.
    let a = Transform::from_rotation(Quat::IDENTITY);
    let q90 = Quat::from_axis_angle(Vec3::Y, core::f32::consts::FRAC_PI_2);
    let b = Transform::from_rotation(q90);
    let m = a.interpolate(&b, 0.5);
    let q45 = Quat::from_axis_angle(Vec3::Y, core::f32::consts::FRAC_PI_4);
    assert!(approx_quat(m.rotation, q45, 1e-5));
}

#[test]
fn transform_interpolate_rotation_handles_negated_quat() {
    // b is the negation of a 90° rotation: same orientation, opposite sign.
    let q90 = Quat::from_axis_angle(Vec3::Y, core::f32::consts::FRAC_PI_2);
    let neg = Quat { x: -q90.x, y: -q90.y, z: -q90.z, w: -q90.w };
    let a = Transform::from_rotation(Quat::IDENTITY);
    let b = Transform::from_rotation(neg);
    let m = a.interpolate(&b, 0.5);
    // shortest path still lands near 45°, never the long way round.
    let q45 = Quat::from_axis_angle(Vec3::Y, core::f32::consts::FRAC_PI_4);
    assert!(approx_quat(m.rotation, q45, 1e-5));
}

#[test]
fn transform_alpha_is_clamped() {
    let a = Transform::from_xyz(0.0, 0.0, 0.0);
    let b = Transform::from_xyz(10.0, 0.0, 0.0);
    // Overstepping must not extrapolate past b.
    let over = a.interpolate(&b, 1.5);
    assert!(approx_vec(over.translation, b.translation, 1e-6));
    // Negative must not extrapolate before a.
    let under = a.interpolate(&b, -0.5);
    assert!(approx_vec(under.translation, a.translation, 1e-6));
}

#[test]
fn global_interpolate_pure_translation() {
    let a = GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, 0.0),
    ));
    let b = GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(4.0, 6.0, 8.0),
    ));
    let m = a.interpolate(&b, 0.25);
    assert!(approx_vec(m.translation(), Vec3::new(1.0, 1.5, 2.0), 1e-5));
}

#[test]
fn global_interpolate_srt_roundtrip_endpoints() {
    let rot = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let a = GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::new(2.0, 2.0, 2.0),
        rot,
        Vec3::new(1.0, -2.0, 3.0),
    ));
    let b = GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::new(3.0, 3.0, 3.0),
        Quat::from_axis_angle(Vec3::Y, 1.2),
        Vec3::new(5.0, 5.0, 5.0),
    ));
    let at0 = a.interpolate(&b, 0.0);
    assert!(approx_vec(at0.translation(), a.translation(), 1e-5));
    let at1 = a.interpolate(&b, 1.0);
    assert!(approx_vec(at1.translation(), b.translation(), 1e-5));
}

#[test]
fn buffer_resize_and_defaults() {
    let mut buf = InterpolationBuffer::new();
    assert!(buf.is_empty());
    buf.resize(3);
    assert_eq!(buf.len(), 3);
    assert!(!buf.is_empty());
    for i in 0..3 {
        assert!(!buf.is_teleport(i));
        assert_eq!(buf.current()[i].translation(), Vec3::ZERO);
    }
}

#[test]
fn buffer_double_buffer_and_sample() {
    let mut buf = InterpolationBuffer::with_len(2);
    // Step 1: current = origin.
    let step1 = [
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
        )),
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
        )),
    ];
    buf.set_current(&step1);

    // Step 2: begin_step rolls current->previous, write new current.
    buf.begin_step();
    let step2 = [
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
        )),
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(0.0, 20.0, 0.0),
        )),
    ];
    buf.set_current(&step2);

    let mut out = [GlobalTransform::IDENTITY; 2];
    buf.sample(0.5, &mut out);
    assert!(approx_vec(out[0].translation(), Vec3::new(5.0, 0.0, 0.0), 1e-5));
    assert!(approx_vec(out[1].translation(), Vec3::new(0.0, 10.0, 0.0), 1e-5));
}

#[test]
fn buffer_teleport_snaps_not_lerps() {
    let mut buf = InterpolationBuffer::with_len(1);
    buf.set_current(&[GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, 0.0),
    ))]);
    buf.begin_step();
    buf.set_current(&[GlobalTransform(Affine3::from_scale_rotation_translation(
        Vec3::ONE,
        Quat::IDENTITY,
        Vec3::new(1000.0, 0.0, 0.0),
    ))]);
    buf.mark_teleport(0);

    let mut out = [GlobalTransform::IDENTITY; 1];
    buf.sample(0.5, &mut out);
    // Snaps to current (1000), not the midpoint (500).
    assert!(approx_vec(out[0].translation(), Vec3::new(1000.0, 0.0, 0.0), 1e-4));

    // After clearing, the same alpha interpolates again.
    buf.clear_teleports();
    assert!(!buf.is_teleport(0));
    buf.sample(0.5, &mut out);
    assert!(approx_vec(out[0].translation(), Vec3::new(500.0, 0.0, 0.0), 1e-4));
}

#[test]
fn buffer_teleport_bitset_boundary() {
    // Exercise >64 nodes so the bitset uses multiple words.
    let mut buf = InterpolationBuffer::with_len(130);
    buf.mark_teleport(0);
    buf.mark_teleport(63);
    buf.mark_teleport(64);
    buf.mark_teleport(129);
    assert!(buf.is_teleport(0));
    assert!(buf.is_teleport(63));
    assert!(buf.is_teleport(64));
    assert!(buf.is_teleport(129));
    assert!(!buf.is_teleport(1));
    assert!(!buf.is_teleport(65));
    buf.clear_teleports();
    for i in [0usize, 63, 64, 129] {
        assert!(!buf.is_teleport(i));
    }
}

#[test]
fn buffer_resize_clears_stale_high_bits() {
    let mut buf = InterpolationBuffer::with_len(70);
    buf.mark_teleport(69);
    // Shrink below the flagged bit, then grow back: the flag must not resurrect.
    buf.resize(10);
    buf.resize(70);
    assert!(!buf.is_teleport(69));
}

#[test]
fn buffer_sample_to_vec_matches_sample() {
    let mut buf = InterpolationBuffer::with_len(2);
    buf.set_current(&[GlobalTransform::IDENTITY, GlobalTransform::IDENTITY]);
    buf.begin_step();
    buf.set_current(&[
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0),
        )),
        GlobalTransform(Affine3::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::IDENTITY,
            Vec3::new(0.0, 4.0, 0.0),
        )),
    ]);
    let v = buf.sample_to_vec(0.5);
    let mut out = [GlobalTransform::IDENTITY; 2];
    buf.sample(0.5, &mut out);
    assert!(approx_vec(v[0].translation(), out[0].translation(), 1e-6));
    assert!(approx_vec(v[1].translation(), out[1].translation(), 1e-6));
}

#[test]
#[should_panic(expected = "set_current length mismatch")]
fn set_current_length_mismatch_panics() {
    let mut buf = InterpolationBuffer::with_len(2);
    buf.set_current(&[GlobalTransform::IDENTITY]);
}

#[test]
#[should_panic(expected = "sample output length mismatch")]
fn sample_length_mismatch_panics() {
    let buf = InterpolationBuffer::with_len(2);
    let mut out = [GlobalTransform::IDENTITY; 1];
    buf.sample(0.5, &mut out);
}
