//! Real-device parity for the §24.1 GPU ray-triangle (Möller-Trumbore) shader
//! mirror.
//!
//! Each test builds a batch of ray/triangle pairs, intersects them on the CPU
//! with [`prism_math::intersect::ray_triangle_bary`] /
//! [`ray_triangle`](prism_math::intersect::ray_triangle), runs the same query
//! on a real `GPU` from the single-sourced
//! [`WGSL_RAYTRI`](prism_math::shader_mirror::WGSL_RAYTRI) fragment, and asserts
//! the per-element hit flag matches exactly and `t`/`u`/`v`, hit point, and
//! normal agree within a small tolerance.
//!
//! The barycentric divides and normal `normalize` are float arithmetic, and
//! Metal compiles WGSL under fast-math, so numeric fields agree only within a
//! small absolute+relative epsilon. The suite uses geometry with a comfortable
//! margin from any triangle edge / grazing ray so the discrete hit flag is
//! unambiguous. It skips gracefully when no adapter is available so it still
//! passes on a device-less CI image while running the full dispatch on a real
//! `GPU`.

use prism_math::Vec3;
use prism_math::geom::ray::Ray3;
use prism_math::intersect::{ray_triangle, ray_triangle_bary};
use prism_math_gpu::{GpuContext, GpuRay, GpuRayTri, GpuTri, GpuTriHit};

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

/// A spread of ray/triangle pairs: a center hit, two off-center (interior)
/// hits, a back-face hit, and two clean misses (outside an edge / pointing
/// away).
fn sample() -> (Vec<GpuRay>, Vec<GpuTri>, Vec<Ray3>, Vec<(Vec3, Vec3, Vec3)>) {
    // A unit triangle in the z = 0 plane.
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(4.0, 0.0, 0.0);
    let c = Vec3::new(0.0, 4.0, 0.0);

    // (origin, dir) pairs aimed from +z downward (and one from -z, one away).
    let rays = [
        (Vec3::new(1.0, 1.0, 3.0), Vec3::new(0.0, 0.0, -1.0)), // interior hit
        (Vec3::new(0.5, 0.5, 2.0), Vec3::new(0.0, 0.0, -1.0)), // near vertex a
        (Vec3::new(2.0, 1.0, 5.0), Vec3::new(0.0, 0.0, -1.0)), // interior hit
        (Vec3::new(1.0, 1.0, -3.0), Vec3::new(0.0, 0.0, 1.0)), // back-face hit
        (Vec3::new(3.0, 3.0, 3.0), Vec3::new(0.0, 0.0, -1.0)), // outside edge → miss
        (Vec3::new(1.0, 1.0, 3.0), Vec3::new(0.0, 0.0, 1.0)),  // pointing away → miss
    ];

    let mut gpu_rays = Vec::new();
    let mut gpu_tris = Vec::new();
    let mut cpu_rays = Vec::new();
    let mut cpu_tris = Vec::new();
    for (o, d) in rays {
        gpu_rays.push(GpuRay::new([o.x, o.y, o.z], [d.x, d.y, d.z]));
        gpu_tris.push(GpuTri::new([a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]));
        cpu_rays.push(Ray3::new(o, d));
        cpu_tris.push((a, b, c));
    }
    (gpu_rays, gpu_tris, cpu_rays, cpu_tris)
}

/// Compares a decoded GPU hit against the CPU bary + hit references.
fn assert_hit(gpu: GpuTriHit, ray: Ray3, tri: (Vec3, Vec3, Vec3), tag: &str) {
    let bary = ray_triangle_bary(ray, tri.0, tri.1, tri.2);
    match bary {
        None => assert!(!gpu.hit, "{tag}: GPU reported a hit where CPU missed"),
        Some((t, u, v)) => {
            assert!(gpu.hit, "{tag}: GPU missed where CPU hit");
            assert!(close(gpu.t, t), "{tag}: t mismatch gpu={} cpu={t}", gpu.t);
            assert!(close(gpu.u, u), "{tag}: u mismatch gpu={} cpu={u}", gpu.u);
            assert!(close(gpu.v, v), "{tag}: v mismatch gpu={} cpu={v}", gpu.v);
            let cpu_hit = ray_triangle(ray, tri.0, tri.1, tri.2).expect("bary hit implies hit");
            for (axis, (g, c)) in gpu
                .normal
                .iter()
                .zip([cpu_hit.normal.x, cpu_hit.normal.y, cpu_hit.normal.z])
                .enumerate()
            {
                assert!(close(*g, c), "{tag}: normal[{axis}] mismatch gpu={g} cpu={c}");
            }
        }
    }
}

#[test]
fn ray_triangle_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayTri::new(&ctx);
    let (gr, gt, cr, ct) = sample();
    let hits = caster.cast(&ctx, &gr, &gt);
    assert_eq!(hits.len(), gr.len());
    for i in 0..gr.len() {
        assert_hit(hits[i], cr[i], ct[i], &format!("tri[{i}]"));
    }
}

#[test]
fn large_batch_crosses_workgroups() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayTri::new(&ctx);
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(8.0, 0.0, 0.0);
    let c = Vec3::new(0.0, 8.0, 0.0);
    let mut gr = Vec::new();
    let mut gt = Vec::new();
    let mut cr = Vec::new();
    let mut ct = Vec::new();
    for i in 0..257u32 {
        // March the origin across the triangle interior along a diagonal.
        let x = (i % 6) as f32 + 0.5;
        let y = (i % 5) as f32 + 0.5;
        let o = Vec3::new(x, y, 10.0);
        let d = Vec3::new(0.0, 0.0, -1.0);
        gr.push(GpuRay::new([o.x, o.y, o.z], [d.x, d.y, d.z]));
        gt.push(GpuTri::new([a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]));
        cr.push(Ray3::new(o, d));
        ct.push((a, b, c));
    }
    let hits = caster.cast(&ctx, &gr, &gt);
    assert_eq!(hits.len(), gr.len());
    for i in 0..gr.len() {
        assert_hit(hits[i], cr[i], ct[i], &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let caster = GpuRayTri::new(&ctx);
    assert!(caster.cast(&ctx, &[], &[]).is_empty());
}
