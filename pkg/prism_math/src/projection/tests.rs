//! Oracle and property tests for the camera view/projection family.
//!
//! Projection correctness is checked by transforming known view-space points
//! and asserting their normalized-device-coordinate (NDC) image, which is the
//! API contract renderers depend on. `Mat4::transform_point3` already performs
//! the perspective divide, so it yields NDC directly.

use crate::mat::Mat4;
use crate::vec::{vec3, Vec3, Vec4};
use core::f32::consts::FRAC_PI_2;

const EPS: f32 = 1e-5;

fn close(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() <= eps, "expected {a} ~= {b} (|d|={})", (a - b).abs());
}

fn close_v3(a: Vec3, b: Vec3, eps: f32) {
    close(a.x, b.x, eps);
    close(a.y, b.y, eps);
    close(a.z, b.z, eps);
}

/// NDC image of a view-space point (implicit `w = 1`, perspective divide).
fn ndc(m: Mat4, p: Vec3) -> Vec3 {
    m.transform_point3(p)
}

// ----- perspective RH, clip depth [0, 1] -----------------------------------

#[test]
fn perspective_rh_depth_endpoints() {
    let near = 0.5;
    let far = 100.0;
    let m = super::perspective_rh(FRAC_PI_2, 16.0 / 9.0, near, far);
    // Center of the near plane -> NDC origin at depth 0.
    close_v3(ndc(m, vec3(0.0, 0.0, -near)), vec3(0.0, 0.0, 0.0), EPS);
    // Center of the far plane -> depth 1.
    close_v3(ndc(m, vec3(0.0, 0.0, -far)), vec3(0.0, 0.0, 1.0), EPS);
}

#[test]
fn perspective_rh_corner_and_fov() {
    // 90-degree vertical FOV, square aspect: the top edge of the frustum at a
    // given depth has |y_view| == |z_view|, which maps to NDC y = +1.
    let m = super::perspective_rh(FRAC_PI_2, 1.0, 1.0, 10.0);
    let d = 5.0;
    let top = ndc(m, vec3(0.0, d, -d));
    close(top.y, 1.0, EPS);
    let right = ndc(m, vec3(d, 0.0, -d));
    close(right.x, 1.0, EPS);
    // Depth is monotonically increasing with distance.
    let a = ndc(m, vec3(0.0, 0.0, -2.0)).z;
    let b = ndc(m, vec3(0.0, 0.0, -8.0)).z;
    assert!(a < b, "depth should increase with distance: {a} !< {b}");
}

#[test]
fn perspective_rh_matrix_oracle() {
    // Hand-computed entries for fovy=90deg, aspect=1, near=1, far=100.
    let m = super::perspective_rh(FRAC_PI_2, 1.0, 1.0, 100.0);
    // f = cot(45deg) = 1.
    close(m.x_axis.x, 1.0, EPS);
    close(m.y_axis.y, 1.0, EPS);
    // r = far/(near-far) = 100/(1-100) = -1.0101...
    close(m.z_axis.z, 100.0 / (1.0 - 100.0), EPS);
    close(m.z_axis.w, -1.0, EPS);
    close(m.w_axis.z, (100.0 / (1.0 - 100.0)) * 1.0, EPS);
}

// ----- perspective RH, clip depth [-1, 1] (OpenGL) -------------------------

#[test]
fn perspective_rh_gl_depth_endpoints() {
    let (near, far) = (2.0, 50.0);
    let m = super::perspective_rh_gl(FRAC_PI_2, 1.5, near, far);
    close(ndc(m, vec3(0.0, 0.0, -near)).z, -1.0, EPS);
    close(ndc(m, vec3(0.0, 0.0, -far)).z, 1.0, EPS);
}

// ----- reverse-Z -----------------------------------------------------------

#[test]
fn perspective_reverse_z_endpoints_and_monotonicity() {
    let (near, far) = (0.1, 1000.0);
    let m = super::perspective_reverse_z_rh(FRAC_PI_2, 1.0, near, far);
    close(ndc(m, vec3(0.0, 0.0, -near)).z, 1.0, EPS);
    close(ndc(m, vec3(0.0, 0.0, -far)).z, 0.0, EPS);
    // Reverse-Z depth decreases monotonically with distance.
    let a = ndc(m, vec3(0.0, 0.0, -1.0)).z;
    let b = ndc(m, vec3(0.0, 0.0, -500.0)).z;
    assert!(a > b, "reverse-Z depth should decrease with distance: {a} !> {b}");
}

// ----- infinite far --------------------------------------------------------

#[test]
fn perspective_infinite_near_and_asymptote() {
    let near = 0.25;
    let m = super::perspective_infinite_rh(FRAC_PI_2, 1.0, near);
    close(ndc(m, vec3(0.0, 0.0, -near)).z, 0.0, EPS);
    // Depth approaches but never reaches 1 as distance grows.
    let far_depth = ndc(m, vec3(0.0, 0.0, -1.0e6)).z;
    assert!(far_depth < 1.0 && far_depth > 0.999, "asymptotic depth {far_depth}");
}

#[test]
fn perspective_infinite_reverse_z_near_and_asymptote() {
    let near = 0.25;
    let m = super::perspective_infinite_reverse_z_rh(FRAC_PI_2, 1.0, near);
    close(ndc(m, vec3(0.0, 0.0, -near)).z, 1.0, EPS);
    let far_depth = ndc(m, vec3(0.0, 0.0, -1.0e6)).z;
    assert!(far_depth > 0.0 && far_depth < 0.001, "asymptotic reverse depth {far_depth}");
}

// ----- left-handed perspective --------------------------------------------

#[test]
fn perspective_lh_depth_endpoints() {
    let (near, far) = (1.0, 20.0);
    let m = super::perspective_lh(FRAC_PI_2, 1.0, near, far);
    // LH: camera looks down +Z.
    close(ndc(m, vec3(0.0, 0.0, near)).z, 0.0, EPS);
    close(ndc(m, vec3(0.0, 0.0, far)).z, 1.0, EPS);
}

// ----- orthographic --------------------------------------------------------

#[test]
fn orthographic_rh_box_to_cube() {
    let m = super::orthographic_rh(-2.0, 2.0, -1.0, 1.0, 1.0, 11.0);
    // x in [-2,2] -> [-1,1]; y in [-1,1] -> [-1,1]; z in [-near,-far] -> [0,1].
    close_v3(ndc(m, vec3(-2.0, -1.0, -1.0)), vec3(-1.0, -1.0, 0.0), EPS);
    close_v3(ndc(m, vec3(2.0, 1.0, -11.0)), vec3(1.0, 1.0, 1.0), EPS);
    close_v3(ndc(m, vec3(0.0, 0.0, -6.0)), vec3(0.0, 0.0, 0.5), EPS);
}

#[test]
fn orthographic_rh_gl_box_to_cube() {
    let m = super::orthographic_rh_gl(0.0, 4.0, 0.0, 2.0, 1.0, 5.0);
    close(ndc(m, vec3(0.0, 0.0, -1.0)).z, -1.0, EPS);
    close(ndc(m, vec3(0.0, 0.0, -5.0)).z, 1.0, EPS);
    close(ndc(m, vec3(2.0, 1.0, -3.0)).x, 0.0, EPS);
    close(ndc(m, vec3(2.0, 1.0, -3.0)).y, 0.0, EPS);
}

#[test]
fn orthographic_lh_depth() {
    let m = super::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0);
    close(ndc(m, vec3(0.0, 0.0, 1.0)).z, 0.0, EPS);
    close(ndc(m, vec3(0.0, 0.0, 10.0)).z, 1.0, EPS);
}

// ----- look-at view matrices ----------------------------------------------

#[test]
fn look_at_rh_places_eye_at_origin_and_target_in_front() {
    let eye = vec3(3.0, 4.0, 5.0);
    let target = vec3(0.0, 0.0, 0.0);
    let v = super::look_at_rh(eye, target, vec3(0.0, 1.0, 0.0));
    // Eye maps to the view-space origin.
    close_v3(v.transform_point3(eye), vec3(0.0, 0.0, 0.0), 1e-4);
    // Target lies down -Z in view space (in front of an RH camera).
    let t_view = v.transform_point3(target);
    close(t_view.x, 0.0, 1e-4);
    close(t_view.y, 0.0, 1e-4);
    assert!(t_view.z < 0.0, "target should be in front (-Z): {}", t_view.z);
    // Distance is preserved by the rigid view transform.
    close(-t_view.z, (eye - target).length(), 1e-4);
}

#[test]
fn look_at_lh_places_target_in_front_positive_z() {
    let eye = vec3(0.0, 0.0, -5.0);
    let target = vec3(0.0, 0.0, 0.0);
    let v = super::look_at_lh(eye, target, vec3(0.0, 1.0, 0.0));
    let t_view = v.transform_point3(target);
    assert!(t_view.z > 0.0, "LH target should be in front (+Z): {}", t_view.z);
    close(t_view.z, (eye - target).length(), 1e-4);
}

#[test]
fn view_projection_round_trip() {
    // A full RH view-projection: a point at the target maps to the screen
    // center at some positive clip depth in [0, 1].
    let eye = vec3(0.0, 2.0, 6.0);
    let target = vec3(0.0, 1.0, 0.0);
    let view = super::look_at_rh(eye, target, vec3(0.0, 1.0, 0.0));
    let proj = super::perspective_rh(FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0);
    let vp = proj * view;
    let clip = vp.mul_vec4(Vec4::new(target.x, target.y, target.z, 1.0));
    assert!(clip.w > 0.0, "clip w should be positive in front of camera");
    let ndc = vec3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
    close(ndc.x, 0.0, 1e-4);
    close(ndc.y, 0.0, 1e-4);
    assert!((0.0..=1.0).contains(&ndc.z), "clip depth in range: {}", ndc.z);
}
