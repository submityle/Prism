//! M2 correctness tests: intersection golden checks against analytic answers,
//! spline endpoint/tangent continuity, and easing boundary values.

use crate::curve::{
    bezier_cubic, bezier_cubic_tangent, catmull_rom, catmull_rom_tangent, cubic_in, cubic_in_out,
    cubic_out, expo_in, expo_in_out, expo_out, hermite, hermite_tangent, lerp as clerp, quad_in,
    quad_in_out, quad_out, sine_in, sine_in_out, sine_out, smootherstep, smoothstep,
};
use crate::intersect::{
    Containment, aabb_aabb, frustum_aabb, frustum_sphere, ray_aabb, ray_plane, ray_sphere,
    ray_triangle, ray_triangle_bary, sphere_aabb, sphere_sphere,
};
use crate::prelude::*;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}
fn v_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
}

// --- geometry primitives -------------------------------------------------

#[test]
fn aabb_basics() {
    let bb = Aabb3::new(vec3(1.0, 2.0, 3.0), vec3(-1.0, 5.0, 0.0));
    assert_eq!(bb.min, vec3(-1.0, 2.0, 0.0));
    assert_eq!(bb.max, vec3(1.0, 5.0, 3.0));
    assert!(v_approx(bb.center(), vec3(0.0, 3.5, 1.5), 1e-6));
    assert!(bb.contains_point(bb.center()));
    assert!(!bb.contains_point(vec3(2.0, 3.0, 1.0)));
    let merged = bb.expand_to_include(vec3(10.0, -5.0, 1.0));
    assert_eq!(merged.min, vec3(-1.0, -5.0, 0.0));
    assert_eq!(merged.max, vec3(10.0, 5.0, 3.0));
}

#[test]
fn aabb_closest_and_distance() {
    let bb = Aabb3::new(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
    assert!(v_approx(bb.closest_point(vec3(5.0, 0.0, 0.0)), vec3(1.0, 0.0, 0.0), 1e-6));
    assert!(approx(bb.distance_squared(vec3(4.0, 0.0, 0.0)), 9.0, 1e-5));
    assert!(approx(bb.distance_squared(vec3(0.0, 0.0, 0.0)), 0.0, 1e-6));
}

#[test]
fn sphere_merge_contains() {
    let a = BoundingSphere::new(vec3(0.0, 0.0, 0.0), 1.0);
    let b = BoundingSphere::new(vec3(4.0, 0.0, 0.0), 1.0);
    let m = a.merge(b);
    // Diameter spans from -1 to 5 on x => radius 3, center at x = 2.
    assert!(approx(m.radius, 3.0, 1e-5));
    assert!(v_approx(m.center, vec3(2.0, 0.0, 0.0), 1e-5));
    assert!(m.contains_sphere(a) && m.contains_sphere(b));
}

#[test]
fn plane_signed_distance() {
    let p = Plane::from_point_normal(vec3(0.0, 2.0, 0.0), vec3(0.0, 3.0, 0.0));
    assert!(approx(p.signed_distance(vec3(0.0, 5.0, 0.0)), 3.0, 1e-6));
    assert!(approx(p.signed_distance(vec3(0.0, 2.0, 0.0)), 0.0, 1e-6));
    assert!(approx(p.signed_distance(vec3(0.0, 0.0, 0.0)), -2.0, 1e-6));
}

// --- ray intersections (golden analytic checks) --------------------------

#[test]
fn ray_sphere_golden() {
    let ray = Ray3::new(vec3(0.0, 0.0, -5.0), Vec3::Z);
    let sphere = BoundingSphere::new(Vec3::ZERO, 1.0);
    let hit = ray_sphere(ray, sphere).expect("hit");
    assert!(approx(hit.t, 4.0, 1e-5));
    assert!(v_approx(hit.point, vec3(0.0, 0.0, -1.0), 1e-5));
    assert!(v_approx(hit.normal, vec3(0.0, 0.0, -1.0), 1e-5));
    // Missing ray.
    let miss = Ray3::new(vec3(0.0, 5.0, -5.0), Vec3::Z);
    assert!(ray_sphere(miss, sphere).is_none());
}

#[test]
fn ray_sphere_from_inside() {
    let ray = Ray3::new(Vec3::ZERO, Vec3::X);
    let sphere = BoundingSphere::new(Vec3::ZERO, 2.0);
    let hit = ray_sphere(ray, sphere).expect("hit");
    assert!(approx(hit.t, 2.0, 1e-5));
    // Normal flipped to face the ray from inside.
    assert!(v_approx(hit.normal, vec3(-1.0, 0.0, 0.0), 1e-5));
}

#[test]
fn ray_aabb_slab_golden() {
    let bb = Aabb3::new(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
    let ray = Ray3::new(vec3(0.0, 0.0, -5.0), Vec3::Z);
    let hit = ray_aabb(ray, bb).expect("hit");
    assert!(approx(hit.t, 4.0, 1e-5));
    assert!(v_approx(hit.point, vec3(0.0, 0.0, -1.0), 1e-5));
    assert!(v_approx(hit.normal, vec3(0.0, 0.0, -1.0), 1e-5));
    // Parallel miss.
    let miss = Ray3::new(vec3(5.0, 0.0, -5.0), Vec3::Z);
    assert!(ray_aabb(miss, bb).is_none());
}

#[test]
fn ray_aabb_from_inside_reports_exit() {
    let bb = Aabb3::new(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
    let ray = Ray3::new(Vec3::ZERO, Vec3::X);
    let hit = ray_aabb(ray, bb).expect("hit");
    assert!(approx(hit.t, 1.0, 1e-5));
    assert!(v_approx(hit.normal, vec3(1.0, 0.0, 0.0), 1e-5));
}

#[test]
fn ray_plane_golden() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let ray = Ray3::new(vec3(0.0, 5.0, 0.0), Vec3::NEG_Y);
    let hit = ray_plane(ray, plane).expect("hit");
    assert!(approx(hit.t, 5.0, 1e-5));
    assert!(v_approx(hit.point, Vec3::ZERO, 1e-5));
    assert!(v_approx(hit.normal, Vec3::Y, 1e-5));
    // Parallel ray: no hit.
    let parallel = Ray3::new(vec3(0.0, 1.0, 0.0), Vec3::X);
    assert!(ray_plane(parallel, plane).is_none());
}

#[test]
fn ray_triangle_barycentric_golden() {
    let a = vec3(0.0, 0.0, 0.0);
    let b = vec3(1.0, 0.0, 0.0);
    let c = vec3(0.0, 1.0, 0.0);
    let ray = Ray3::new(vec3(0.25, 0.25, -1.0), Vec3::Z);
    let (t, u, v) = ray_triangle_bary(ray, a, b, c).expect("hit");
    assert!(approx(t, 1.0, 1e-5));
    assert!(approx(u, 0.25, 1e-5));
    assert!(approx(v, 0.25, 1e-5));
    // Reconstruct the point from barycentric weights.
    let w = 1.0 - u - v;
    let p = a * w + b * u + c * v;
    assert!(v_approx(p, vec3(0.25, 0.25, 0.0), 1e-5));
    let hit = ray_triangle(ray, a, b, c).expect("hit");
    assert!(v_approx(hit.point, vec3(0.25, 0.25, 0.0), 1e-5));
    assert!(approx(hit.normal.dot(ray.direction).abs(), 1.0, 1e-5));
    // Ray that misses the triangle (outside the edge).
    let miss = Ray3::new(vec3(0.9, 0.9, -1.0), Vec3::Z);
    assert!(ray_triangle_bary(miss, a, b, c).is_none());
}

// --- overlap tests -------------------------------------------------------

#[test]
fn aabb_aabb_overlap() {
    let a = Aabb3::new(Vec3::ZERO, vec3(2.0, 2.0, 2.0));
    let b = Aabb3::new(vec3(1.0, 1.0, 1.0), vec3(3.0, 3.0, 3.0));
    let c = Aabb3::new(vec3(5.0, 5.0, 5.0), vec3(6.0, 6.0, 6.0));
    assert!(aabb_aabb(a, b));
    assert!(!aabb_aabb(a, c));
}

#[test]
fn sphere_overlaps() {
    let a = BoundingSphere::new(Vec3::ZERO, 1.0);
    let b = BoundingSphere::new(vec3(1.5, 0.0, 0.0), 1.0);
    let c = BoundingSphere::new(vec3(5.0, 0.0, 0.0), 1.0);
    assert!(sphere_sphere(a, b));
    assert!(!sphere_sphere(a, c));

    let bb = Aabb3::new(vec3(2.0, -1.0, -1.0), vec3(4.0, 1.0, 1.0));
    assert!(sphere_aabb(BoundingSphere::new(vec3(1.5, 0.0, 0.0), 1.0), bb));
    assert!(!sphere_aabb(BoundingSphere::new(vec3(0.0, 0.0, 0.0), 1.0), bb));
}

// --- frustum culling -----------------------------------------------------

/// OpenGL-style right-handed perspective matrix (clip depth in `[-1, 1]`).
fn perspective_rh(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / crate::float::f32::tan(fovy * 0.5);
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, (far + near) / (near - far), -1.0),
        Vec4::new(0.0, 0.0, (2.0 * far * near) / (near - far), 0.0),
    )
}

#[test]
fn frustum_cull_in_and_out() {
    let proj = perspective_rh(core::f32::consts::FRAC_PI_2, 1.0, 1.0, 100.0);
    let frustum = Frustum::from_view_proj(proj);

    // Point in front of the camera is inside.
    assert!(frustum.contains_point(vec3(0.0, 0.0, -5.0)));
    // Point behind the camera is outside.
    assert!(!frustum.contains_point(vec3(0.0, 0.0, 5.0)));
    // Point way off to the side is outside.
    assert!(!frustum.contains_point(vec3(100.0, 0.0, -5.0)));

    // Sphere classification.
    let inside = BoundingSphere::new(vec3(0.0, 0.0, -10.0), 1.0);
    let outside = BoundingSphere::new(vec3(0.0, 0.0, 10.0), 1.0);
    assert_eq!(frustum_sphere(&frustum, inside), Containment::Inside);
    assert_eq!(frustum_sphere(&frustum, outside), Containment::Outside);

    // AABB classification.
    let bb_in = Aabb3::from_center_half_extents(vec3(0.0, 0.0, -10.0), Vec3::splat(0.5));
    let bb_out = Aabb3::from_center_half_extents(vec3(0.0, 0.0, 50.0), Vec3::splat(0.5));
    assert_eq!(frustum_aabb(&frustum, bb_in), Containment::Inside);
    assert_eq!(frustum_aabb(&frustum, bb_out), Containment::Outside);

    // A box straddling the near plane should be classified as intersecting.
    let bb_straddle = Aabb3::from_center_half_extents(vec3(0.0, 0.0, -1.0), Vec3::splat(2.0));
    assert_eq!(frustum_aabb(&frustum, bb_straddle), Containment::Intersecting);
}

// --- interpolation / splines --------------------------------------------

#[test]
fn generic_lerp_vec() {
    let a = vec3(0.0, 0.0, 0.0);
    let b = vec3(2.0, 4.0, 6.0);
    assert!(v_approx(clerp(a, b, 0.0), a, 1e-6));
    assert!(v_approx(clerp(a, b, 1.0), b, 1e-6));
    assert!(v_approx(clerp(a, b, 0.5), vec3(1.0, 2.0, 3.0), 1e-6));
}

#[test]
fn hermite_endpoints_and_tangents() {
    let p0 = vec3(0.0, 0.0, 0.0);
    let p1 = vec3(1.0, 2.0, 3.0);
    let m0 = vec3(1.0, 0.0, 0.0);
    let m1 = vec3(0.0, 1.0, 0.0);
    assert!(v_approx(hermite(p0, m0, p1, m1, 0.0), p0, 1e-6));
    assert!(v_approx(hermite(p0, m0, p1, m1, 1.0), p1, 1e-6));
    assert!(v_approx(hermite_tangent(p0, m0, p1, m1, 0.0), m0, 1e-5));
    assert!(v_approx(hermite_tangent(p0, m0, p1, m1, 1.0), m1, 1e-5));
}

#[test]
fn catmull_rom_passes_through_and_is_c1() {
    let p0 = vec3(-1.0, 0.0, 0.0);
    let p1 = vec3(0.0, 0.0, 0.0);
    let p2 = vec3(1.0, 1.0, 0.0);
    let p3 = vec3(2.0, 0.0, 0.0);
    let p4 = vec3(3.0, -1.0, 0.0);
    // Interpolates the inner control points.
    assert!(v_approx(catmull_rom(p0, p1, p2, p3, 0.0), p1, 1e-6));
    assert!(v_approx(catmull_rom(p0, p1, p2, p3, 1.0), p2, 1e-6));
    // C1 continuity at the shared knot p2: end tangent of one segment equals
    // the start tangent of the next.
    let end = catmull_rom_tangent(p0, p1, p2, p3, 1.0);
    let start = catmull_rom_tangent(p1, p2, p3, p4, 0.0);
    assert!(v_approx(end, start, 1e-5));
    // And both equal the uniform central-difference tangent (p3 - p1) / 2.
    assert!(v_approx(end, (p3 - p1) * 0.5, 1e-5));
}

#[test]
fn bezier_endpoints_and_tangents() {
    let p0 = vec3(0.0, 0.0, 0.0);
    let p1 = vec3(0.0, 1.0, 0.0);
    let p2 = vec3(1.0, 1.0, 0.0);
    let p3 = vec3(1.0, 0.0, 0.0);
    assert!(v_approx(bezier_cubic(p0, p1, p2, p3, 0.0), p0, 1e-6));
    assert!(v_approx(bezier_cubic(p0, p1, p2, p3, 1.0), p3, 1e-6));
    assert!(v_approx(bezier_cubic_tangent(p0, p1, p2, p3, 0.0), (p1 - p0) * 3.0, 1e-5));
    assert!(v_approx(bezier_cubic_tangent(p0, p1, p2, p3, 1.0), (p3 - p2) * 3.0, 1e-5));
}

// --- easing boundary values ----------------------------------------------

#[test]
fn easing_boundaries() {
    let fns: [fn(f32) -> f32; 13] = [
        smoothstep,
        smootherstep,
        quad_in,
        quad_out,
        quad_in_out,
        cubic_in,
        cubic_out,
        cubic_in_out,
        sine_in,
        sine_out,
        sine_in_out,
        expo_in,
        expo_out,
    ];
    for f in fns {
        assert!(approx(f(0.0), 0.0, 1e-6), "f(0) must be 0");
        assert!(approx(f(1.0), 1.0, 1e-6), "f(1) must be 1");
    }
    // expo_in_out separately (shares the same contract).
    assert!(approx(expo_in_out(0.0), 0.0, 1e-6));
    assert!(approx(expo_in_out(1.0), 1.0, 1e-6));
    assert!(approx(expo_in_out(0.5), 0.5, 1e-6));

    // Midpoint symmetry of the in-out variants.
    assert!(approx(quad_in_out(0.5), 0.5, 1e-6));
    assert!(approx(cubic_in_out(0.5), 0.5, 1e-6));
    assert!(approx(sine_in_out(0.5), 0.5, 1e-6));

    // smoothstep clamps outside [0, 1].
    assert!(approx(smoothstep(-1.0), 0.0, 1e-6));
    assert!(approx(smoothstep(2.0), 1.0, 1e-6));
    assert!(approx(smootherstep(0.5), 0.5, 1e-6));
}
