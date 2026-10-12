//! CPU unit tests with analytically derivable expectations.

use crate::planetest::Containment;
use crate::{Camera, Frustum, Orthographic, Perspective, Projection, View};
use core::f32::consts::FRAC_PI_2;
use prism_math::{perspective_rh, perspective_rh_gl, Affine3, Mat4, Quat, Vec2, Vec3, Vec4};

/// Tight tolerance: every expectation below is an exact decimal value, and the
/// only error introduced is `tan` for the 90° field of view.
const EPS: f32 = 1e-4;

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= EPS,
        "expected {expected}, got {actual}"
    );
}

fn assert_vec3_close(actual: Vec3, expected: Vec3) {
    assert_close(actual.x, expected.x);
    assert_close(actual.y, expected.y);
    assert_close(actual.z, expected.z);
}

fn assert_mat4_close(actual: Mat4, expected: Mat4) {
    let actual = [actual.x_axis, actual.y_axis, actual.z_axis, actual.w_axis];
    let expected = [
        expected.x_axis,
        expected.y_axis,
        expected.z_axis,
        expected.w_axis,
    ];
    for (ca, ce) in actual.iter().zip(expected.iter()) {
        assert_close(ca.x, ce.x);
        assert_close(ca.y, ce.y);
        assert_close(ca.z, ce.z);
        assert_close(ca.w, ce.w);
    }
}

/// 90° vertical FOV at aspect 1, near 1, far 10, camera at the origin looking
/// down `-Z`.
fn origin_camera() -> Camera {
    Camera::new(
        View::from_transform(Affine3::IDENTITY),
        Projection::Perspective(Perspective::new(FRAC_PI_2, 1.0, 1.0, 10.0)),
    )
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

#[test]
fn perspective_matrix_matches_analytic_entries() {
    // f = 1 / tan(45°) = 1, so x_axis.x = f / aspect = 0.5 and y_axis.y = 1.
    // z_axis.z = far / (near - far) = 3 / (1 - 3) = -1.5, z_axis.w = -1.
    // w_axis.z = z_axis.z * near = -1.5, w_axis.w = 0.
    let perspective = Perspective::new(FRAC_PI_2, 2.0, 1.0, 3.0);
    let expected = Mat4::from_cols(
        Vec4::new(0.5, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.5, -1.0),
        Vec4::new(0.0, 0.0, -1.5, 0.0),
    );
    assert_mat4_close(perspective.matrix(), expected);
    // Cross-check against the prism_math constructor it is built on.
    assert_mat4_close(
        perspective.matrix(),
        perspective_rh(FRAC_PI_2, 2.0, 1.0, 3.0),
    );
    assert!(perspective.is_valid());
}

#[test]
fn perspective_maps_near_to_zero_and_far_to_one() {
    let view_projection = origin_camera().view_projection();
    // The near plane (z = -1) must land at clip depth 0 …
    let near = view_projection.mul_vec4(Vec4::new(0.0, 0.0, -1.0, 1.0));
    assert_close(near.z / near.w, 0.0);
    // … and the far plane (z = -10) at clip depth 1.
    let far = view_projection.mul_vec4(Vec4::new(0.0, 0.0, -10.0, 1.0));
    assert_close(far.z / far.w, 1.0);
}

#[test]
fn orthographic_from_size_matches_analytic_entries() {
    // A 2x2 window: rcp_w = 1/2 so 2 * rcp_w = 1; depth scale is
    // 1 / (near - far) = 1 / (1 - 3) = -0.5 and the depth bias is -0.5.
    let orthographic = Orthographic::from_size(Vec2::new(2.0, 2.0), 1.0, 3.0);
    let expected = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -0.5, 0.0),
        Vec4::new(0.0, 0.0, -0.5, 1.0),
    );
    assert_mat4_close(orthographic.matrix(), expected);
    assert!(orthographic.is_valid());
}

#[test]
fn orthographic_explicit_bounds_are_respected() {
    let orthographic = Orthographic::new(-4.0, 4.0, -2.0, 2.0, 0.5, 10.0);
    let matrix = orthographic.matrix();
    // Width 8 -> x scale 2 / 8 = 0.25; height 4 -> y scale 2 / 4 = 0.5.
    assert_close(matrix.x_axis.x, 0.25);
    assert_close(matrix.y_axis.y, 0.5);
}

#[test]
fn with_aspect_refits_each_projection() {
    let perspective =
        Projection::Perspective(Perspective::new(FRAC_PI_2, 1.0, 1.0, 10.0)).with_aspect(2.0);
    match perspective {
        Projection::Perspective(inner) => assert_close(inner.aspect, 2.0),
        Projection::Orthographic(_) => panic!("expected a perspective projection"),
    }

    // Widening a 2x2 orthographic window to aspect 4 doubles the width to 8:
    // x scale = 2 / 8 = 0.25, while the height scale stays 1.
    let orthographic =
        Projection::Orthographic(Orthographic::from_size(Vec2::new(2.0, 2.0), 1.0, 5.0))
            .with_aspect(4.0);
    match orthographic {
        Projection::Orthographic(inner) => {
            let matrix = inner.matrix();
            assert_close(matrix.x_axis.x, 0.25);
            assert_close(matrix.y_axis.y, 1.0);
        }
        Projection::Perspective(_) => panic!("expected an orthographic projection"),
    }
}

#[test]
fn degenerate_projections_are_reported_invalid() {
    assert!(!Perspective::new(FRAC_PI_2, 0.0, 1.0, 10.0).is_valid());
    assert!(!Perspective::new(FRAC_PI_2, 1.0, 0.0, 10.0).is_valid());
    assert!(!Perspective::new(FRAC_PI_2, 1.0, 10.0, 1.0).is_valid());
    assert!(!Orthographic::new(1.0, 1.0, -1.0, 1.0, 0.0, 1.0).is_valid());
    assert!(!Orthographic::new(-1.0, 1.0, 1.0, -1.0, 0.0, 1.0).is_valid());
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

#[test]
fn view_matrix_inverts_a_known_transform() {
    // A camera 5 units up +Z looking back down -Z: the view matrix must map the
    // world origin to (0, 0, -5) and the camera position to the origin.
    let view = View::from_translation_rotation(Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY);
    let matrix = view.matrix();
    assert_vec3_close(
        matrix.transform_point3(Vec3::ZERO),
        Vec3::new(0.0, 0.0, -5.0),
    );
    assert_vec3_close(
        matrix.transform_point3(Vec3::new(0.0, 0.0, 5.0)),
        Vec3::ZERO,
    );
    assert_vec3_close(view.position(), Vec3::new(0.0, 0.0, 5.0));
    // Pure translation: the fourth column is exactly -position.
    assert_eq!(matrix.w_axis, Vec4::new(0.0, 0.0, -5.0, 1.0));
}

#[test]
fn looking_at_places_target_on_negative_z() {
    let view = View::looking_at(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    let matrix = view.matrix();
    assert_vec3_close(
        matrix.transform_point3(Vec3::ZERO),
        Vec3::new(0.0, 0.0, -10.0),
    );
    assert_vec3_close(view.position(), Vec3::new(0.0, 0.0, 10.0));
}

#[test]
fn view_matrix_round_trips_the_world_transform() {
    let view = View::from_translation_rotation(
        Vec3::new(3.0, -2.0, 7.0),
        Quat::from_rotation_y(FRAC_PI_2),
    );
    let camera_to_world = view.transform_matrix();
    let world_to_camera = view.matrix();
    let point = Vec3::new(1.5, 0.25, -4.0);
    let round_trip = camera_to_world.transform_point3(world_to_camera.transform_point3(point));
    assert_vec3_close(round_trip, point);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

#[test]
fn camera_view_projection_is_projection_times_view() {
    let camera = Camera::new(
        View::from_translation_rotation(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY),
        Projection::Perspective(Perspective::new(FRAC_PI_2, 1.5, 0.1, 100.0)),
    );
    assert_mat4_close(
        camera.view_projection(),
        camera.projection_matrix() * camera.view_matrix(),
    );
}

#[test]
fn camera_frustum_matches_manual_extraction() {
    let camera = origin_camera();
    assert_eq!(
        camera.frustum(),
        Frustum::from_view_proj(camera.view_projection())
    );
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

#[test]
fn frustum_classifies_spheres() {
    let frustum = origin_camera().frustum();
    // Well inside, on the view axis at the middle of the depth range.
    assert_eq!(
        frustum.contains_sphere(Vec3::new(0.0, 0.0, -5.0), 0.5),
        Containment::Inside
    );
    // Behind the camera.
    assert_eq!(
        frustum.contains_sphere(Vec3::new(0.0, 0.0, 5.0), 0.5),
        Containment::Outside
    );
    // Far beyond the right plane (half-width at z = -5 is 5).
    assert_eq!(
        frustum.contains_sphere(Vec3::new(100.0, 0.0, -5.0), 0.5),
        Containment::Outside
    );
    // Straddling the near plane (z = -1).
    assert_eq!(
        frustum.contains_sphere(Vec3::new(0.0, 0.0, -1.0), 2.0),
        Containment::Intersecting
    );
    // Straddling the far plane (z = -10).
    assert_eq!(
        frustum.contains_sphere(Vec3::new(0.0, 0.0, -10.0), 2.0),
        Containment::Intersecting
    );
    // Straddling the left plane (x = -5 at z = -5).
    assert_eq!(
        frustum.contains_sphere(Vec3::new(-5.0, 0.0, -5.0), 1.0),
        Containment::Intersecting
    );
}

#[test]
fn frustum_point_containment_matches_sphere_classification() {
    let frustum = origin_camera().frustum();
    assert!(frustum.contains_point(Vec3::new(0.0, 0.0, -5.0)));
    assert!(frustum.contains_point(Vec3::new(0.0, 0.0, -1.0)));
    assert!(!frustum.contains_point(Vec3::new(100.0, 0.0, -5.0)));
    assert!(!frustum.contains_point(Vec3::new(0.0, 0.0, 5.0)));
}

#[test]
fn frustum_classifies_aabbs() {
    let frustum = origin_camera().frustum();
    // Fully inside near the middle of the depth range.
    assert_eq!(
        frustum.contains_aabb(Vec3::new(-0.1, -0.1, -2.1), Vec3::new(0.1, 0.1, -1.9)),
        Containment::Inside
    );
    // Entirely behind the camera.
    assert_eq!(
        frustum.contains_aabb(Vec3::new(-0.5, -0.5, 1.0), Vec3::new(0.5, 0.5, 2.0)),
        Containment::Outside
    );
    // Straddling the near plane.
    assert_eq!(
        frustum.contains_aabb(Vec3::new(-0.5, -0.5, -1.5), Vec3::new(0.5, 0.5, -0.5)),
        Containment::Intersecting
    );
    // Straddling the left plane (x = -5 at z = -5).
    assert_eq!(
        frustum.contains_aabb(Vec3::new(-6.0, -0.1, -5.1), Vec3::new(-4.0, 0.1, -4.9)),
        Containment::Intersecting
    );
}

#[test]
fn frustum_visibility_helpers_agree_with_classification() {
    let frustum = origin_camera().frustum();
    assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, -5.0), 0.5));
    assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, -10.0), 2.0));
    assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 5.0), 0.5));
    assert!(frustum.intersects_aabb(Vec3::new(-0.1, -0.1, -2.1), Vec3::new(0.1, 0.1, -1.9)));
    assert!(!frustum.intersects_aabb(Vec3::new(-0.5, -0.5, 1.0), Vec3::new(0.5, 0.5, 2.0)));
}

#[test]
fn gl_extractor_handles_the_negative_one_depth_range() {
    // An OpenGL-style projection puts the near plane at clip depth -1; the GL
    // extractor must place the near plane there, not at the [0, 1] location.
    let projection = perspective_rh_gl(FRAC_PI_2, 1.0, 1.0, 10.0);
    let clip_near = projection.mul_vec4(Vec4::new(0.0, 0.0, -1.0, 1.0));
    assert_close(clip_near.z / clip_near.w, -1.0);

    let frustum = Frustum::from_view_proj_gl(projection);
    // A sphere centered on the OpenGL near plane straddles it …
    assert_eq!(
        frustum.contains_sphere(Vec3::new(0.0, 0.0, -1.0), 1.0),
        Containment::Intersecting
    );
    // … whereas the [0, 1] extractor would misread that sphere as fully inside,
    // so the two conventions must not coincide for a GL projection.
    assert_ne!(
        Frustum::from_view_proj(projection),
        Frustum::from_view_proj_gl(projection)
    );
}
