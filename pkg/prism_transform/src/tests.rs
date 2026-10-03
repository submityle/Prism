//! M0 tests: TRS/affine round-trips and composition associativity.

use crate::{GlobalTransform, Transform};
use prism_math::{Quat, Vec3, vec3};

fn v_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

#[test]
fn identity_is_noop() {
    let p = vec3(1.0, 2.0, 3.0);
    assert_eq!(Transform::IDENTITY.transform_point(p), p);
    assert_eq!(GlobalTransform::IDENTITY.transform_point(p), p);
}

#[test]
fn transform_point_matches_affine() {
    let t = Transform {
        translation: vec3(1.0, -2.0, 3.0),
        rotation: Quat::from_axis_angle(vec3(0.2, 1.0, 0.3).normalize(), 0.8),
        scale: vec3(2.0, 0.5, 1.5),
    };
    let p = vec3(1.0, 2.0, 3.0);
    assert!(v_approx(t.transform_point(p), t.to_affine().transform_point3(p), 1e-5));
}

#[test]
fn global_propagation_matches_affine_chain() {
    // parent world ∘ child local, compared against direct affine composition.
    let parent = Transform {
        translation: vec3(5.0, 0.0, 0.0),
        rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        scale: Vec3::ONE,
    };
    let child = Transform::from_xyz(1.0, 0.0, 0.0);

    let parent_global = GlobalTransform::from_transform(&parent);
    let child_global = parent_global.mul_transform(&child);

    // Expected: rotate (1,0,0) by 90deg about Z -> (0,1,0), then add parent pos.
    let expected = vec3(5.0, 1.0, 0.0);
    assert!(v_approx(child_global.translation(), expected, 1e-5));
}

#[test]
fn composition_associates() {
    let a = Transform {
        translation: vec3(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_x(0.3),
        scale: Vec3::splat(2.0), // uniform scale keeps TRS composition exact
    };
    let b = Transform {
        translation: vec3(-1.0, 0.5, 2.0),
        rotation: Quat::from_rotation_y(0.7),
        scale: Vec3::splat(0.5),
    };
    let c = Transform::from_xyz(4.0, -2.0, 1.0).with_rotation(Quat::from_rotation_z(1.1));

    let lhs = (a.mul_transform(&b)).mul_transform(&c);
    let rhs = a.mul_transform(&b.mul_transform(&c));
    let p = vec3(1.0, 1.0, 1.0);
    assert!(v_approx(lhs.transform_point(p), rhs.transform_point(p), 1e-4));
}

#[test]
fn global_inverse_round_trip() {
    let t = Transform {
        translation: vec3(3.0, -4.0, 5.0),
        rotation: Quat::from_axis_angle(vec3(1.0, 1.0, 0.0).normalize(), 0.9),
        scale: vec3(2.0, 1.0, 0.5),
    };
    let g = GlobalTransform::from_transform(&t);
    let p = vec3(2.0, 3.0, -1.0);
    let round = g.inverse().transform_point(g.transform_point(p));
    assert!(v_approx(round, p, 1e-4));
}

#[test]
fn compute_transform_recovers_srt() {
    let t = Transform {
        translation: vec3(1.0, 2.0, 3.0),
        rotation: Quat::from_axis_angle(vec3(0.3, 0.4, 0.5).normalize(), 0.6),
        scale: vec3(2.0, 1.5, 0.5),
    };
    let recovered = GlobalTransform::from_transform(&t).compute_transform();
    assert!(v_approx(recovered.translation, t.translation, 1e-5));
    assert!(v_approx(recovered.scale, t.scale, 1e-4));
    let v = vec3(1.0, 0.0, 0.0);
    assert!(v_approx(recovered.rotation * v, t.rotation * v, 1e-4));
}

#[test]
fn direction_helpers() {
    let t = Transform::from_rotation(Quat::from_rotation_y(core::f32::consts::FRAC_PI_2));
    // +Z rotated by +90deg about Y -> +X.
    assert!(v_approx(t.local_z(), vec3(1.0, 0.0, 0.0), 1e-6));
    assert!(v_approx(t.forward(), vec3(-1.0, 0.0, 0.0), 1e-6));
}
