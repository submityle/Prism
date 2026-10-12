//! Real-device parity for the §24.1 GPU bounding-sphere shader mirror.
//!
//! Each test evaluates the sphere helpers on the CPU with
//! [`prism_math::BoundingSphere`] and on a real `GPU` from the single-sourced
//! [`WGSL_SPHERE`](prism_math::shader_mirror::WGSL_SPHERE) fragment.
//!
//! `contains_point` is a squared-distance comparison (exact away from the
//! boundary); `merge` is compared within a small tolerance. It skips gracefully
//! when no adapter is available.

use prism_math::geom::sphere::BoundingSphere;
use prism_math::Vec3;
use prism_math_gpu::{GpuContext, GpuSphere, SpherePair, SpherePoint};

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

/// Absolute+relative closeness.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-5 + 1.0e-5 * a.abs().max(b.abs())
}

/// Spheres and query points (inside, on, and outside).
fn spheres() -> [(Vec3, f32); 4] {
    [
        (Vec3::new(0.0, 0.0, 0.0), 1.0),
        (Vec3::new(2.0, -1.0, 3.0), 0.5),
        (Vec3::new(-4.0, 0.0, 0.0), 2.0),
        (Vec3::new(1.0, 1.0, 1.0), 0.25),
    ]
}

fn points() -> [Vec3; 4] {
    [
        Vec3::new(0.5, 0.0, 0.0),  // inside
        Vec3::new(3.0, -1.0, 3.0), // outside (dist 1 > 0.5)
        Vec3::new(-2.0, 0.0, 0.0), // inside
        Vec3::new(2.0, 2.0, 2.0),  // far outside
    ]
}

#[test]
fn sphere_contains_point_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSphere::new(&ctx);
    let sp = spheres();
    let ps = points();
    let items: Vec<SpherePoint> = sp
        .iter()
        .zip(ps.iter())
        .map(|((c, r), p)| [[c.x, c.y, c.z, *r], [p.x, p.y, p.z, 0.0]])
        .collect();

    let got = kernel.contains_point(&ctx, &items);
    assert_eq!(got.len(), sp.len());
    for (i, ((c, r), p)) in sp.iter().zip(ps.iter()).enumerate() {
        let expected = BoundingSphere::new(*c, *r).contains_point(*p);
        assert_eq!(got[i], expected, "contains[{i}]");
    }
}

#[test]
fn sphere_merge_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSphere::new(&ctx);
    let sp = spheres();
    // Merge each sphere with a shifted/scaled copy to exercise all branches.
    let pairs: Vec<SpherePair> = sp
        .iter()
        .enumerate()
        .map(|(i, (c, r))| {
            let shift = (i as f32) + 0.5;
            let other = [c.x + shift, c.y - shift, c.z + 0.25, r * 0.5 + 0.1];
            [[c.x, c.y, c.z, *r], other]
        })
        .collect();

    let got = kernel.merge(&ctx, &pairs);
    assert_eq!(got.len(), pairs.len());
    for (i, pair) in pairs.iter().enumerate() {
        let a = BoundingSphere::new(Vec3::new(pair[0][0], pair[0][1], pair[0][2]), pair[0][3]);
        let b = BoundingSphere::new(Vec3::new(pair[1][0], pair[1][1], pair[1][2]), pair[1][3]);
        let m = a.merge(b);
        assert!(close(got[i][0], m.center.x), "merge[{i}].cx");
        assert!(close(got[i][1], m.center.y), "merge[{i}].cy");
        assert!(close(got[i][2], m.center.z), "merge[{i}].cz");
        assert!(close(got[i][3], m.radius), "merge[{i}].r");
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSphere::new(&ctx);
    assert!(kernel.contains_point(&ctx, &[]).is_empty());
    assert!(kernel.merge(&ctx, &[]).is_empty());
}
