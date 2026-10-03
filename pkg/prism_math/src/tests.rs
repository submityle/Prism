//! M0 correctness tests: operator algebra, matrix inverse round-trips, and
//! quaternion <-> matrix round-trips.

use crate::prelude::*;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

fn vec3_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
}

fn mat4_approx(a: Mat4, b: Mat4, eps: f32) -> bool {
    let a = [a.x_axis, a.y_axis, a.z_axis, a.w_axis];
    let b = [b.x_axis, b.y_axis, b.z_axis, b.w_axis];
    a.iter().zip(b.iter()).all(|(ca, cb)| {
        approx(ca.x, cb.x, eps)
            && approx(ca.y, cb.y, eps)
            && approx(ca.z, cb.z, eps)
            && approx(ca.w, cb.w, eps)
    })
}

#[test]
fn vec3_ops() {
    let a = vec3(1.0, 2.0, 3.0);
    let b = vec3(4.0, 5.0, 6.0);
    assert_eq!(a + b, vec3(5.0, 7.0, 9.0));
    assert_eq!(b - a, vec3(3.0, 3.0, 3.0));
    assert_eq!(a * 2.0, vec3(2.0, 4.0, 6.0));
    assert_eq!(2.0 * a, vec3(2.0, 4.0, 6.0));
    assert_eq!(a.dot(b), 32.0);
    assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
    assert!(approx(a.length_squared(), 14.0, 1e-6));
}

#[test]
fn vec3_add_commutes_and_associates() {
    let a = vec3(1.0, -2.0, 0.5);
    let b = vec3(3.0, 4.0, -1.0);
    let c = vec3(-0.25, 7.0, 2.0);
    assert_eq!(a + b, b + a);
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn vec3_normalize_is_unit() {
    let v = vec3(3.0, 4.0, 12.0);
    assert!(approx(v.normalize().length(), 1.0, 1e-6));
    assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
}

#[test]
fn mat3_mul_associates() {
    let a = Mat3::from_quat(Quat::from_rotation_z(0.3));
    let b = Mat3::from_scale(vec3(2.0, 3.0, 0.5));
    let c = Mat3::from_quat(Quat::from_rotation_x(1.1));
    let lhs = (a * b) * c;
    let rhs = a * (b * c);
    let v = vec3(1.0, -2.0, 3.0);
    assert!(vec3_approx(lhs * v, rhs * v, 1e-5));
}

#[test]
fn mat3_inverse_round_trip() {
    let m = Mat3::from_quat(Quat::from_axis_angle(vec3(1.0, 2.0, 3.0).normalize(), 0.9))
        * Mat3::from_scale(vec3(2.0, 0.5, 1.5));
    let id = m * m.inverse();
    assert!(vec3_approx(id * Vec3::X, Vec3::X, 1e-4));
    assert!(vec3_approx(id * Vec3::Y, Vec3::Y, 1e-4));
    assert!(vec3_approx(id * Vec3::Z, Vec3::Z, 1e-4));
}

#[test]
fn mat4_inverse_round_trip() {
    let m = Mat4::from_scale_rotation_translation(
        vec3(2.0, 0.5, 1.5),
        Quat::from_axis_angle(vec3(0.3, 1.0, -0.7).normalize(), 1.2),
        vec3(10.0, -5.0, 3.0),
    );
    let id = m * m.inverse();
    assert!(mat4_approx(id, Mat4::IDENTITY, 1e-4));
}

#[test]
fn mat4_transform_point_matches_affine() {
    let scale = vec3(1.5, 2.0, 0.5);
    let rot = Quat::from_axis_angle(vec3(0.2, 0.8, 0.1).normalize(), 0.7);
    let trans = vec3(3.0, -1.0, 4.0);
    let m = Mat4::from_scale_rotation_translation(scale, rot, trans);
    let a = Affine3::from_scale_rotation_translation(scale, rot, trans);
    let p = vec3(1.0, 2.0, 3.0);
    assert!(vec3_approx(m.transform_point3(p), a.transform_point3(p), 1e-5));
}

#[test]
fn quat_mat3_round_trip() {
    let q = Quat::from_axis_angle(vec3(1.0, -2.0, 0.5).normalize(), 1.3);
    let q2 = Quat::from_mat3(Mat3::from_quat(q));
    assert!(q.abs_diff_eq(q2, 1e-5));
}

#[test]
fn quat_rotates_like_matrix() {
    let q = Quat::from_axis_angle(vec3(0.3, 1.0, -0.2).normalize(), 0.95);
    let m = Mat3::from_quat(q);
    let v = vec3(1.0, 2.0, -3.0);
    assert!(vec3_approx(q * v, m * v, 1e-5));
}

#[test]
fn quat_compose_matches_rotation_order() {
    // a * b applies b first, then a.
    let a = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
    let b = Quat::from_rotation_x(core::f32::consts::FRAC_PI_2);
    let v = vec3(0.0, 0.0, 1.0);
    let composed = (a * b) * v;
    let stepwise = a * (b * v);
    assert!(vec3_approx(composed, stepwise, 1e-5));
}

#[test]
fn quat_slerp_endpoints() {
    let a = Quat::from_rotation_y(0.2);
    let b = Quat::from_rotation_y(1.1);
    assert!(a.slerp(b, 0.0).abs_diff_eq(a, 1e-5));
    assert!(a.slerp(b, 1.0).abs_diff_eq(b, 1e-5));
    let mid = a.slerp(b, 0.5);
    assert!(approx(mid.length(), 1.0, 1e-5));
}

#[test]
fn affine_inverse_round_trip() {
    let a = Affine3::from_scale_rotation_translation(
        vec3(2.0, 1.0, 0.5),
        Quat::from_axis_angle(vec3(0.1, 0.2, 1.0).normalize(), 0.6),
        vec3(5.0, 6.0, 7.0),
    );
    let p = vec3(1.0, 2.0, 3.0);
    let round = a.inverse().transform_point3(a.transform_point3(p));
    assert!(vec3_approx(round, p, 1e-4));
}

#[test]
fn affine_srt_decompose_round_trip() {
    let scale = vec3(2.0, 1.5, 0.5);
    let rot = Quat::from_axis_angle(vec3(0.3, 0.4, 0.5).normalize(), 0.8);
    let trans = vec3(1.0, 2.0, 3.0);
    let a = Affine3::from_scale_rotation_translation(scale, rot, trans);
    let (s2, r2, t2) = a.to_scale_rotation_translation();
    assert!(vec3_approx(s2, scale, 1e-4));
    assert!(vec3_approx(t2, trans, 1e-5));
    // Rotation should rotate vectors identically.
    let v = vec3(1.0, 0.0, 0.0);
    assert!(vec3_approx(r2 * v, rot * v, 1e-4));
}

#[test]
fn mat4_affine_mul4_round_trip() {
    let a = Affine3::from_scale_rotation_translation(
        vec3(1.0, 2.0, 3.0),
        Quat::from_rotation_z(0.5),
        vec3(1.0, 0.0, -1.0),
    );
    assert!(mat4_approx(Affine3::from_mat4(a.to_mat4()).to_mat4(), a.to_mat4(), 1e-5));
}
