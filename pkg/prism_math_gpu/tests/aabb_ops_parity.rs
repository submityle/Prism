//! Real-device parity for the §24.1 GPU AABB query shader mirror.
//!
//! Each test builds a batch of boxes / box-point queries, evaluates them on the
//! CPU with [`prism_math::Aabb3`], runs the same query on a real `GPU` from the
//! single-sourced
//! [`WGSL_AABB_OPS`](prism_math::shader_mirror::WGSL_AABB_OPS) fragment, and
//! asserts agreement.
//!
//! The ops are component-wise min/max/add/mul and a dot product; agreement is a
//! small absolute+relative epsilon. It skips gracefully when no adapter is
//! available.

use prism_math::geom::aabb::Aabb3;
use prism_math::Vec3;
use prism_math_gpu::{aabb4, aabb_pair, aabb_point, GpuAabbOps, GpuContext};

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

/// Boxes with varied extents.
fn boxes() -> [(Vec3, Vec3); 4] {
    [
        (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 4.0)),
        (Vec3::new(-5.0, -0.5, 1.0), Vec3::new(-2.0, 0.5, 3.0)),
        (Vec3::new(10.0, 10.0, 10.0), Vec3::new(11.0, 12.0, 13.0)),
    ]
}

/// Query points, some inside and some outside.
fn points() -> [Vec3; 4] {
    [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.0, 1.0, 2.0),
        Vec3::new(-10.0, 0.0, 2.0),
        Vec3::new(10.5, 11.5, 12.5),
    ]
}

#[test]
fn aabb_center_and_merge_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAabbOps::new(&ctx);
    let bx = boxes();
    let gb: Vec<_> = bx
        .iter()
        .map(|(lo, hi)| aabb4([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]))
        .collect();

    let centers = kernel.center(&ctx, &gb);
    for (i, (lo, hi)) in bx.iter().enumerate() {
        let c = Aabb3::new(*lo, *hi).center();
        for (axis, (g, e)) in centers[i].iter().zip([c.x, c.y, c.z]).enumerate() {
            assert!(close(*g, e), "center[{i}][{axis}] gpu={g} cpu={e}");
        }
    }

    // Merge consecutive boxes.
    let pairs: Vec<_> = (0..bx.len())
        .map(|i| {
            let a = gb[i];
            let b = gb[(i + 1) % bx.len()];
            aabb_pair(a, b)
        })
        .collect();
    let merged = kernel.merge(&ctx, &pairs);
    for (i, pair) in pairs.iter().enumerate() {
        let a = Aabb3::new(
            Vec3::new(pair[0][0], pair[0][1], pair[0][2]),
            Vec3::new(pair[1][0], pair[1][1], pair[1][2]),
        );
        let b = Aabb3::new(
            Vec3::new(pair[2][0], pair[2][1], pair[2][2]),
            Vec3::new(pair[3][0], pair[3][1], pair[3][2]),
        );
        let m = a.merge(b);
        for (axis, (g, e)) in merged[i][0]
            .iter()
            .zip([m.min.x, m.min.y, m.min.z])
            .enumerate()
        {
            assert!(close(*g, e), "merge_lo[{i}][{axis}] gpu={g} cpu={e}");
        }
        for (axis, (g, e)) in merged[i][1]
            .iter()
            .zip([m.max.x, m.max.y, m.max.z])
            .enumerate()
        {
            assert!(close(*g, e), "merge_hi[{i}][{axis}] gpu={g} cpu={e}");
        }
    }
}

#[test]
fn aabb_closest_and_distance_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAabbOps::new(&ctx);
    let bx = boxes();
    let ps = points();
    let queries: Vec<_> = bx
        .iter()
        .zip(ps.iter())
        .map(|((lo, hi), p)| aabb_point([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z], [p.x, p.y, p.z]))
        .collect();

    let closest = kernel.closest_point(&ctx, &queries);
    let dist2 = kernel.distance_squared(&ctx, &queries);
    for (i, ((lo, hi), p)) in bx.iter().zip(ps.iter()).enumerate() {
        let a = Aabb3::new(*lo, *hi);
        let c = a.closest_point(*p);
        for (axis, (g, e)) in closest[i].iter().zip([c.x, c.y, c.z]).enumerate() {
            assert!(close(*g, e), "closest[{i}][{axis}] gpu={g} cpu={e}");
        }
        assert!(
            close(dist2[i], a.distance_squared(*p)),
            "dist2[{i}] gpu={} cpu={}",
            dist2[i],
            a.distance_squared(*p)
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuAabbOps::new(&ctx);
    assert!(kernel.center(&ctx, &[]).is_empty());
    assert!(kernel.merge(&ctx, &[]).is_empty());
}
