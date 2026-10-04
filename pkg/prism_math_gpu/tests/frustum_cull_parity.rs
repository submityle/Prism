//! Real-device parity for the §24.1 GPU-driven view-frustum culling shader
//! mirror.
//!
//! Each test extracts the six clip planes from a view-projection matrix on the
//! CPU, classifies a batch of bounding spheres / AABBs with
//! [`prism_math::intersect::frustum_sphere`] / [`frustum_aabb`], runs the same
//! classification on a real `GPU` from the single-sourced
//! [`WGSL_FRUSTUM_CULL`](prism_math::shader_mirror::WGSL_FRUSTUM_CULL) fragment,
//! and asserts the per-element [`Containment`](prism_math::intersect::Containment)
//! discriminant matches exactly.
//!
//! The plane test is a float dot product, and Metal compiles WGSL under
//! fast-math, so a volume whose boundary lies within fast-math rounding of a
//! frustum plane could be classified one level differently from the CPU — the
//! standard conservative-culling caveat. The suite therefore uses geometry with
//! a comfortable margin from every plane, where the discrete classification is
//! unambiguous. The suite skips gracefully when no adapter is available so it
//! still passes on a device-less CI image while running the full dispatch on a
//! real `GPU`.

use prism_math::geom::aabb::Aabb3;
use prism_math::geom::frustum::Frustum;
use prism_math::geom::sphere::BoundingSphere;
use prism_math::intersect::{Containment, frustum_aabb, frustum_sphere};
use prism_math::projection::{look_at_rh, perspective_rh};
use prism_math::Vec3;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuFrustumCull;

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

/// A representative perspective view looking down -Z from the origin region.
fn test_frustum() -> Frustum {
    let proj = perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, 0.5, 100.0);
    let view = look_at_rh(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    Frustum::from_view_proj(proj * view)
}

/// Packs the six CPU planes into the `[nx, ny, nz, d]` rows the kernel expects.
fn plane_rows(frustum: &Frustum) -> [[f32; 4]; 6] {
    let mut rows = [[0.0_f32; 4]; 6];
    for (row, plane) in rows.iter_mut().zip(frustum.planes.iter()) {
        *row = [plane.normal.x, plane.normal.y, plane.normal.z, plane.d];
    }
    rows
}

/// The CPU reference discriminant as a `u32`, matching the device encoding.
fn cpu_sphere(frustum: &Frustum, c: Vec3, r: f32) -> u32 {
    frustum_sphere(frustum, BoundingSphere::new(c, r)) as u32
}

fn cpu_aabb(frustum: &Frustum, center: Vec3, half: Vec3) -> u32 {
    frustum_aabb(frustum, Aabb3::from_center_half_extents(center, half)) as u32
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let cull = GpuFrustumCull::new(&ctx);
    let frustum = test_frustum();
    let planes = plane_rows(&frustum);
    assert!(cull.cull_spheres(&ctx, &planes, &[]).is_empty());
    assert!(cull.cull_aabbs(&ctx, &planes, &[]).is_empty());
}

#[test]
fn sphere_classification_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let cull = GpuFrustumCull::new(&ctx);
    let frustum = test_frustum();
    let planes = plane_rows(&frustum);

    // Centers chosen with a comfortable margin from every plane, plus one large
    // straddling sphere that is unambiguously Intersecting.
    let spheres: &[(Vec3, f32)] = &[
        (Vec3::new(0.0, 0.0, 0.0), 0.5),      // deep inside
        (Vec3::new(0.0, 0.0, 500.0), 1.0),    // behind the near plane (eye at z=10)
        (Vec3::new(0.0, 0.0, -500.0), 1.0),   // beyond the far plane
        (Vec3::new(500.0, 0.0, 0.0), 1.0),    // far to the right
        (Vec3::new(-500.0, 0.0, 0.0), 1.0),   // far to the left
        (Vec3::new(0.0, 500.0, 0.0), 1.0),    // far above
        (Vec3::new(0.0, -500.0, 0.0), 1.0),   // far below
        (Vec3::new(0.0, 0.0, 0.0), 1000.0),   // engulfs the frustum -> Intersecting
    ];
    let input: Vec<[f32; 4]> = spheres
        .iter()
        .map(|&(c, r)| [c.x, c.y, c.z, r])
        .collect();

    let gpu = cull.cull_spheres(&ctx, &planes, &input);
    assert_eq!(gpu.len(), spheres.len());
    for (i, &(c, r)) in spheres.iter().enumerate() {
        assert_eq!(gpu[i], cpu_sphere(&frustum, c, r), "sphere {i}");
    }
    // Sanity: the first is Inside, the engulfing one is Intersecting, and at
    // least one is culled Outside.
    assert_eq!(gpu[0], Containment::Inside as u32);
    assert_eq!(gpu[7], Containment::Intersecting as u32);
    assert!(gpu.contains(&(Containment::Outside as u32)));
}

#[test]
fn aabb_classification_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let cull = GpuFrustumCull::new(&ctx);
    let frustum = test_frustum();
    let planes = plane_rows(&frustum);

    let boxes: &[(Vec3, Vec3)] = &[
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)), // deep inside
        (Vec3::new(0.0, 0.0, 500.0), Vec3::new(1.0, 1.0, 1.0)),       // behind near
        (Vec3::new(0.0, 0.0, -500.0), Vec3::new(1.0, 1.0, 1.0)),      // beyond far
        (Vec3::new(500.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),       // right
        (Vec3::new(-500.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),      // left
        (Vec3::new(0.0, 500.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),       // above
        (Vec3::new(0.0, -500.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),      // below
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1000.0, 1000.0, 1000.0)),      // engulfing -> Intersecting
    ];
    let input: Vec<[[f32; 4]; 2]> = boxes
        .iter()
        .map(|&(c, e)| [[c.x, c.y, c.z, 0.0], [e.x, e.y, e.z, 0.0]])
        .collect();

    let gpu = cull.cull_aabbs(&ctx, &planes, &input);
    assert_eq!(gpu.len(), boxes.len());
    for (i, &(c, e)) in boxes.iter().enumerate() {
        assert_eq!(gpu[i], cpu_aabb(&frustum, c, e), "aabb {i}");
    }
    assert_eq!(gpu[0], Containment::Inside as u32);
    assert_eq!(gpu[7], Containment::Intersecting as u32);
    assert!(gpu.contains(&(Containment::Outside as u32)));
}
