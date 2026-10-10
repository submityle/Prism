//! Real-device parity for the §24.1 GPU ray/primitive intersection shader
//! mirror.
//!
//! Each test builds a batch of rays and primitives, intersects them on the CPU
//! with [`prism_math::intersect::ray_sphere`] / [`ray_aabb`] / [`ray_plane`],
//! runs the same
//! query on a real `GPU` from the single-sourced
//! [`WGSL_RAYCAST`](prism_math::shader_mirror::WGSL_RAYCAST) fragment, and
//! asserts the per-element hit flag matches exactly and the ray parameter `t`,
//! hit point, and normal agree within a small tolerance.
//!
//! The sphere quadratic and AABB slab divides are float arithmetic, and Metal
//! compiles WGSL under fast-math, so `t`/point/normal agree only within a small
//! absolute+relative epsilon. The suite uses geometry with a comfortable margin
//! from any tangent/grazing boundary so the discrete hit flag and axis-aligned
//! normal are unambiguous. It skips gracefully when no adapter is available so
//! it still passes on a device-less CI image while running the full dispatch on
//! a real `GPU`.

use prism_math::geom::aabb::Aabb3;
use prism_math::geom::plane::Plane;
use prism_math::geom::ray::Ray3;
use prism_math::geom::sphere::BoundingSphere;
use prism_math::intersect::{ray_aabb, ray_plane, ray_sphere, RayHit};
use prism_math::Vec3;
use prism_math_gpu::{GpuAabb, GpuContext, GpuRay, GpuRayCast, GpuRayHit};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness, matching the fast-math tolerance contract.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-4 + 1.0e-4 * a.abs().max(b.abs())
}

/// Compares a decoded GPU hit against the CPU reference `Option<RayHit>`.
fn assert_hit(gpu: GpuRayHit, cpu: Option<RayHit>, tag: &str) {
    match cpu {
        None => assert!(!gpu.hit, "{tag}: GPU reported a hit where CPU missed"),
        Some(hit) => {
            assert!(gpu.hit, "{tag}: GPU missed where CPU hit");
            assert!(
                close(gpu.t, hit.t),
                "{tag}: t mismatch gpu={} cpu={}",
                gpu.t,
                hit.t
            );
            for (axis, (g, c)) in gpu
                .point
                .iter()
                .zip([hit.point.x, hit.point.y, hit.point.z])
                .enumerate()
            {
                assert!(
                    close(*g, c),
                    "{tag}: point[{axis}] mismatch gpu={g} cpu={c}"
                );
            }
            for (axis, (g, c)) in gpu
                .normal
                .iter()
                .zip([hit.normal.x, hit.normal.y, hit.normal.z])
                .enumerate()
            {
                assert!(
                    close(*g, c),
                    "{tag}: normal[{axis}] mismatch gpu={g} cpu={c}"
                );
            }
        }
    }
}

/// A spread of rays with integer-component directions (never unit, never
/// parallel to a tested slab) so the test needs no transcendental functions.
fn ray_batch() -> [(Vec3, Vec3); 6] {
    [
        (Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        (Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, -1.0)),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
        (Vec3::new(3.0, 2.0, 6.0), Vec3::new(-1.0, -1.0, -3.0)),
        (Vec3::new(-4.0, 1.0, 4.0), Vec3::new(2.0, -1.0, -2.0)),
        (Vec3::new(0.0, -6.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    ]
}

#[test]
fn ray_sphere_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayCast::new(&ctx);

    let rays = ray_batch();
    // One sphere per ray; a mix of hits, a clear miss, and an inside-origin hit.
    let spheres = [
        (Vec3::new(0.0, 0.0, 0.0), 1.0_f32),
        (Vec3::new(0.0, 0.0, 0.0), 1.0),
        (Vec3::new(0.0, 0.0, 0.0), 2.0),
        (Vec3::new(0.0, 0.0, 0.0), 1.5),
        (Vec3::new(0.0, 0.0, 0.0), 2.0),
        (Vec3::new(0.0, 0.0, 0.0), 1.0),
    ];

    let gpu_rays: Vec<GpuRay> = rays
        .iter()
        .map(|(o, d)| GpuRay::new([o.x, o.y, o.z], [d.x, d.y, d.z]))
        .collect();
    let gpu_spheres: Vec<[f32; 4]> = spheres.iter().map(|(c, r)| [c.x, c.y, c.z, *r]).collect();

    let got = caster.cast_spheres(&ctx, &gpu_rays, &gpu_spheres);
    assert_eq!(got.len(), rays.len());

    for (i, ((o, d), (c, r))) in rays.iter().zip(spheres.iter()).enumerate() {
        let cpu = ray_sphere(Ray3::new(*o, *d), BoundingSphere::new(*c, *r));
        assert_hit(got[i], cpu, &alloc_tag("sphere", i));
    }
}

#[test]
fn ray_aabb_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayCast::new(&ctx);

    let rays = ray_batch();
    // Boxes giving a frontal hit, a miss, an inside-origin exit, and oblique hits.
    let boxes = [
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        (Vec3::new(-1.5, -1.5, -1.5), Vec3::new(1.5, 1.5, 1.5)),
        (Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0)),
        (Vec3::new(-1.0, -2.0, -1.0), Vec3::new(1.0, 2.0, 1.0)),
    ];

    let gpu_rays: Vec<GpuRay> = rays
        .iter()
        .map(|(o, d)| GpuRay::new([o.x, o.y, o.z], [d.x, d.y, d.z]))
        .collect();
    let gpu_boxes: Vec<GpuAabb> = boxes
        .iter()
        .map(|(lo, hi)| GpuAabb::new([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]))
        .collect();

    let got = caster.cast_aabbs(&ctx, &gpu_rays, &gpu_boxes);
    assert_eq!(got.len(), rays.len());

    for (i, ((o, d), (lo, hi))) in rays.iter().zip(boxes.iter()).enumerate() {
        let cpu = ray_aabb(Ray3::new(*o, *d), Aabb3::new(*lo, *hi));
        assert_hit(got[i], cpu, &alloc_tag("aabb", i));
    }
}

#[test]
fn ray_plane_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayCast::new(&ctx);

    let rays = ray_batch();
    // One plane per ray: frontal hits, a back-facing normal flip, a parallel
    // miss, and oblique hits. Normals are unit length and `d` is the raw
    // offset so `signed_distance(p) = dot(normal, p) + d`.
    let planes = [
        // z = 0 plane facing +z; ray from z=5 heading -z hits at t=5.
        Plane::new(Vec3::new(0.0, 0.0, 1.0), 0.0),
        // z = 2 plane facing +z; ray from z=5 heading -z hits at t=3.
        Plane::new(Vec3::new(0.0, 0.0, 1.0), -2.0),
        // z = -3 plane facing -z; ray from origin heading -z hits, normal flips.
        Plane::new(Vec3::new(0.0, 0.0, -1.0), -3.0),
        // Oblique plane through origin; ray (3,2,6) dir (-1,-1,-3) crosses it.
        Plane::new(Vec3::new(0.0, 0.0, 1.0), -1.0),
        // Plane parallel to the ray direction (2,-1,-2): normal perpendicular
        // to it -> a clean miss (denominator zero).
        Plane::new(Vec3::new(1.0, 2.0, 0.0).normalize(), 10.0),
        // y = 0 plane facing +y; ray from y=-6 heading +y hits, normal flips.
        Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0),
    ];

    let gpu_rays: Vec<GpuRay> = rays
        .iter()
        .map(|(o, d)| GpuRay::new([o.x, o.y, o.z], [d.x, d.y, d.z]))
        .collect();
    let gpu_planes: Vec<[f32; 4]> = planes
        .iter()
        .map(|pl| [pl.normal.x, pl.normal.y, pl.normal.z, pl.d])
        .collect();

    let got = caster.cast_planes(&ctx, &gpu_rays, &gpu_planes);
    assert_eq!(got.len(), rays.len());

    for (i, ((o, d), pl)) in rays.iter().zip(planes.iter()).enumerate() {
        let cpu = ray_plane(Ray3::new(*o, *d), *pl);
        assert_hit(got[i], cpu, &alloc_tag("plane", i));
    }
}

#[test]
fn empty_batches_return_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayCast::new(&ctx);
    assert!(caster.cast_spheres(&ctx, &[], &[]).is_empty());
    assert!(caster.cast_aabbs(&ctx, &[], &[]).is_empty());
    assert!(caster.cast_planes(&ctx, &[], &[]).is_empty());
}

/// Small heap tag so panic messages identify the failing element.
fn alloc_tag(kind: &str, i: usize) -> String {
    format!("{kind}[{i}]")
}
