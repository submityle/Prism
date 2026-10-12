//! Real-device parity for the §24.1 GPU plane-query shader mirror.
//!
//! Each test builds a batch of plane/point queries, evaluates them on the CPU
//! with [`prism_math::Plane`], runs the same query on a real `GPU` from the
//! single-sourced
//! [`WGSL_PLANE`](prism_math::shader_mirror::WGSL_PLANE) fragment, and asserts
//! agreement.
//!
//! Signed distance, distance, and projection are single dot products; flipped is
//! an exact sign flip and is compared strictly; `normalized` adds a `length`
//! divide and is compared within a small tolerance. It skips gracefully when no
//! adapter is available.

use prism_math::geom::plane::Plane;
use prism_math::Vec3;
use prism_math_gpu::{GpuContext, GpuPlane, GpuPlaneQuery};

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

/// Absolute+relative closeness for the float ops.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-5 + 1.0e-5 * a.abs().max(b.abs())
}

/// A few planes (unit and non-unit normals) and query points.
fn fixtures() -> (Vec<Plane>, Vec<Vec3>) {
    let planes = vec![
        Plane::new(Vec3::new(0.0, 0.0, 1.0), 0.0),
        Plane::from_point_normal(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0)),
        // Non-unit normal to exercise `normalized`.
        Plane::new(Vec3::new(0.0, 2.0, 0.0), -4.0),
        Plane::from_point_normal(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)),
    ];
    let points = vec![
        Vec3::new(0.5, -1.0, 2.0),
        Vec3::new(-3.0, 4.0, 0.25),
        Vec3::new(1.0, 5.0, -2.0),
        Vec3::new(2.0, -2.0, 7.0),
    ];
    (planes, points)
}

/// Packs a `Plane` into `[normal.xyz, d]`.
fn pack_plane(p: Plane) -> GpuPlane {
    [p.normal.x, p.normal.y, p.normal.z, p.d]
}

/// Packs a point into `[x, y, z]`.
fn pack_point(p: Vec3) -> [f32; 3] {
    [p.x, p.y, p.z]
}

#[test]
fn plane_signed_and_distance_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPlaneQuery::new(&ctx);
    let (planes, points) = fixtures();
    let gp: Vec<GpuPlane> = planes.iter().copied().map(pack_plane).collect();
    let pp: Vec<[f32; 3]> = points.iter().copied().map(pack_point).collect();

    let signed = kernel.signed_distance(&ctx, &gp, &pp);
    let dist = kernel.distance(&ctx, &gp, &pp);
    for (i, (plane, point)) in planes.iter().zip(points.iter()).enumerate() {
        let s = plane.signed_distance(*point);
        let d = plane.distance(*point);
        assert!(close(signed[i], s), "signed[{i}] gpu={} cpu={s}", signed[i]);
        assert!(close(dist[i], d), "distance[{i}] gpu={} cpu={d}", dist[i]);
    }
}

#[test]
fn plane_project_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPlaneQuery::new(&ctx);
    let (planes, points) = fixtures();
    let gp: Vec<GpuPlane> = planes.iter().copied().map(pack_plane).collect();
    let pp: Vec<[f32; 3]> = points.iter().copied().map(pack_point).collect();

    let got = kernel.project_points(&ctx, &gp, &pp);
    for (i, (plane, point)) in planes.iter().zip(points.iter()).enumerate() {
        let c = plane.project_point(*point);
        for (axis, (g, e)) in got[i].iter().zip([c.x, c.y, c.z]).enumerate() {
            assert!(close(*g, e), "project[{i}][{axis}] gpu={g} cpu={e}");
        }
    }
}

#[test]
fn plane_normalized_and_flipped_match_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPlaneQuery::new(&ctx);
    let (planes, _) = fixtures();
    let gp: Vec<GpuPlane> = planes.iter().copied().map(pack_plane).collect();

    let norm = kernel.normalized(&ctx, &gp);
    let flip = kernel.flipped(&ctx, &gp);
    for (i, plane) in planes.iter().enumerate() {
        let n = plane.normalized();
        assert!(
            close(norm[i][0], n.normal.x)
                && close(norm[i][1], n.normal.y)
                && close(norm[i][2], n.normal.z)
                && close(norm[i][3], n.d),
            "normalized[{i}] gpu={:?} cpu=({}, {}, {}, {})",
            norm[i],
            n.normal.x,
            n.normal.y,
            n.normal.z,
            n.d
        );
        // Flip is an exact sign flip.
        let f = plane.flipped();
        assert_eq!(
            flip[i],
            [f.normal.x, f.normal.y, f.normal.z, f.d],
            "flipped[{i}]"
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPlaneQuery::new(&ctx);
    assert!(kernel.signed_distance(&ctx, &[], &[]).is_empty());
    assert!(kernel.normalized(&ctx, &[]).is_empty());
}
