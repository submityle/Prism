//! Real-device parity for the ribbon/trail geometry twin:
//! [`GpuRibbonGeometry`](prism_volumetric_gpu::ribbon_geometry::GpuRibbonGeometry)
//! must reproduce the `CPU` golden
//! [`build_strip`](prism_render_architecture::particle::ribbon_geometry::build_strip),
//! [`build_strip_flat`](prism_render_architecture::particle::ribbon_geometry::build_strip_flat)
//! and
//! [`strip_indices`](prism_render_architecture::particle::ribbon_geometry::strip_indices)
//! across a single point, short straight runs, curved 3-D polylines, several
//! width policies (exact, short, long, empty, negative), both the camera-facing
//! and fixed-normal axis variants, and a batch of random non-degenerate
//! polylines compared vertex-for-vertex. The triangle-list indices are compared
//! bit-for-bit over a sweep of vertex counts.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each output vertex is a fixed, non-reorderable sequence of subtractions,
//! cross products, one reciprocal-`sqrt` normalization and a prefix-sum divide,
//! so `CPU` and `GPU` evaluate the same closed form in the same order. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! the continuous corner positions and `UV`.v — loose enough to admit a legal
//! fused multiply-add contraction yet tight enough to fail a genuinely wrong
//! port (a swapped tangent difference, a wrong fallback axis, a dropped
//! half-`width`). The integer index list is compared exactly.
//!
//! The continuous fixtures deliberately avoid the degeneracy zones (zero-length
//! segments and tangent-parallel-to-view collinearity) so the tolerance stays
//! meaningful; the one single-point fixture exercises the deterministic fallback
//! whose formula is identical on both sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_geometry`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ribbon_geometry::{
    build_strip, build_strip_flat, strip_indices, RibbonStripVertex,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::ribbon_geometry::{GpuRibbonGeometry, RibbonStripQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts the whole corner triple and `UV`.v of two vertices agree.
fn close_vertex(label: &str, i: usize, got: RibbonStripVertex, want: RibbonStripVertex) {
    for (axis, (g, w)) in [
        ("left.x", (got.left.x, want.left.x)),
        ("left.y", (got.left.y, want.left.y)),
        ("left.z", (got.left.z, want.left.z)),
        ("right.x", (got.right.x, want.right.x)),
        ("right.y", (got.right.y, want.right.y)),
        ("right.z", (got.right.z, want.right.z)),
        ("uv_v", (got.uv_v, want.uv_v)),
    ] {
        assert!(
            close(g, w),
            "{label} vertex {i} {axis}: gpu {g} vs cpu {w}"
        );
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a non-degenerate 3-D polyline of `count` points by accumulating random
/// steps whose magnitude is bounded away from zero, so no segment degenerates to
/// zero length.
fn random_polyline(count: usize, state: &mut u64) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(count);
    let mut cur = Vec3::new(
        lcg(state) * 2.0 - 1.0,
        lcg(state) * 2.0 - 1.0,
        lcg(state) * 2.0 - 1.0,
    );
    points.push(cur);
    for _ in 1..count {
        // Each component steps by at least 0.5 in a random direction, keeping the
        // segment length comfortably above the 1e-6 degeneracy guard.
        let step = Vec3::new(
            0.5 + lcg(state),
            (lcg(state) - 0.5) * 1.5,
            0.5 + lcg(state) * 0.75,
        );
        cur = cur.add(step);
        points.push(cur);
    }
    points
}

/// Runs the `GPU` strip expansion and asserts vertex-for-vertex parity against
/// the chosen `CPU` golden (camera-facing when `facing`, fixed-normal otherwise).
fn check_strip(
    ctx: &GpuContext,
    gpu: &GpuRibbonGeometry,
    label: &str,
    centerline: &[Vec3],
    widths: &[f32],
    axis: Vec3,
    facing: bool,
) {
    let want = if facing {
        build_strip(centerline, widths, axis)
    } else {
        build_strip_flat(centerline, widths, axis)
    };
    let query = RibbonStripQuery {
        centerline: centerline.to_vec(),
        widths: widths.to_vec(),
        axis,
        facing,
    };
    let got = gpu.build_strip(ctx, &query);
    assert_eq!(got.len(), want.len(), "{label}: vertex count mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        close_vertex(label, i, *g, *w);
    }
}

#[test]
fn empty_centerline_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    let query = RibbonStripQuery {
        centerline: Vec::new(),
        widths: Vec::new(),
        axis: Vec3::new(0.0, 0.0, 1.0),
        facing: true,
    };
    assert!(gpu.build_strip(&ctx, &query).is_empty());
}

#[test]
fn single_point_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    // A lone point has a zero tangent and exercises the deterministic fallback
    // axis; the formula is identical on both sides, so parity still holds.
    check_strip(
        &ctx,
        &gpu,
        "single-point",
        &[Vec3::new(4.0, 5.0, 6.0)],
        &[2.0],
        Vec3::new(0.0, 0.0, 0.0),
        true,
    );
}

#[test]
fn straight_run_camera_facing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    // A short run along +X viewed from +Z: the side axis is well-conditioned
    // (tangent perpendicular to the view) so this is far from any degeneracy.
    let centerline = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    ];
    check_strip(
        &ctx,
        &gpu,
        "straight-camera",
        &centerline,
        &[1.0, 1.0, 1.0],
        Vec3::new(1.5, 0.0, 5.0),
        true,
    );
}

#[test]
fn fixed_normal_variant_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    let centerline = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.5, 0.0),
        Vec3::new(2.0, 0.0, 0.3),
        Vec3::new(3.5, -0.4, 0.1),
    ];
    check_strip(
        &ctx,
        &gpu,
        "fixed-normal",
        &centerline,
        &[2.0, 2.0, 2.0, 2.0],
        Vec3::new(0.0, 0.0, 1.0),
        false,
    );
}

#[test]
fn width_policies_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    let centerline = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.2, 0.1),
        Vec3::new(2.1, 0.1, 0.4),
        Vec3::new(3.0, -0.3, 0.2),
    ];
    let axis = Vec3::new(1.5, 0.0, 6.0);
    // Exact-length, short (reuses last), long (ignores tail), empty (unit) and
    // negative (clamped to zero) width policies all resolve the same on device.
    check_strip(&ctx, &gpu, "w-exact", &centerline, &[0.5, 1.0, 1.5, 2.0], axis, true);
    check_strip(&ctx, &gpu, "w-short", &centerline, &[1.0, 3.0], axis, true);
    check_strip(
        &ctx,
        &gpu,
        "w-long",
        &centerline,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        axis,
        true,
    );
    check_strip(&ctx, &gpu, "w-empty", &centerline, &[], axis, true);
    check_strip(
        &ctx,
        &gpu,
        "w-negative",
        &centerline,
        &[-1.0, 2.0, -0.5, 1.0],
        axis,
        true,
    );
}

#[test]
fn random_polylines_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    let mut state = 0x5eed_a110_c0de_1234_u64;
    // Sweep a spread of lengths; the random polyline keeps every segment well
    // above the degeneracy guard and the camera sits off the line so the side
    // axis never collapses.
    for &count in &[2usize, 3, 5, 8, 13, 21, 64, 100] {
        let centerline = random_polyline(count, &mut state);
        let mut widths = Vec::with_capacity(count);
        for _ in 0..count {
            widths.push(0.25 + lcg(&mut state) * 3.0);
        }
        let camera = Vec3::new(
            lcg(&mut state) * 4.0 - 2.0,
            lcg(&mut state) * 4.0 - 2.0,
            8.0 + lcg(&mut state) * 4.0,
        );
        check_strip(&ctx, &gpu, "random-camera", &centerline, &widths, camera, true);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        check_strip(&ctx, &gpu, "random-flat", &centerline, &widths, normal, false);
    }
}

#[test]
fn strip_indices_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRibbonGeometry::new(&ctx);
    // Below 2 is empty (no dispatch); 2..=65 crosses the 64-thread workgroup
    // boundary so the dispatch grid is exercised past one group.
    for vertex_count in [0u32, 1, 2, 3, 4, 7, 16, 63, 64, 65, 128] {
        let got = gpu.strip_indices(&ctx, vertex_count);
        let want = strip_indices(vertex_count);
        assert_eq!(got, want, "strip_indices({vertex_count}) mismatch");
    }
}
