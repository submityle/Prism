//! Real-device parity for the integer midpoint-circle twin:
//! [`GpuMidpointCircle`](prism_volumetric_gpu::midpoint_circle::GpuMidpointCircle)
//! must reproduce the `CPU` golden
//! [`midpoint_circle`](prism_render_architecture::particle::midpoint_circle)
//! boundary rasterizer — the exact set of lattice points
//! [`rasterize`](prism_render_architecture::particle::midpoint_circle::rasterize)
//! produces for a center and radius.
//!
//! The fixtures cover the shapes the golden unit tests call out: the empty
//! negative-radius case, the single-point `r == 0` case, the four axis neighbors
//! at `r == 1`, small radii whose axis/diagonal reflections coincide (`r == 2`,
//! `3`, `5`), medium radii (`16`, `50`, `100`, `200`), the same radius under
//! positive and negative center offsets (the walk is translation invariant), and
//! a mixed batch solved in one dispatch. All inputs are integers, so the
//! fixtures are pure integer / no transcendental math and need no external math
//! library.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The boundary is a set of integer lattice points, so `CPU` and `GPU` must
//! agree *exactly*: after the kernel emits the raw octant-walk stream and the
//! host applies the reference `sort_unstable` + `dedup`, the comparison is an
//! exact `==` on the full point vector (ascending by `x`, then `y`, with no
//! duplicates). There is no tolerance: integer rasterization is bit-exact.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::midpoint_circle`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::midpoint_circle::rasterize;
use prism_volumetric_gpu::midpoint_circle::{GpuMidpointCircle, GpuMidpointCircleQuery};
use prism_volumetric_gpu::GpuContext;

/// Builds one circle query from a center and radius.
fn query(cx: i32, cy: i32, r: i32) -> GpuMidpointCircleQuery {
    GpuMidpointCircleQuery { cx, cy, r }
}

/// Asserts the `GPU` boundary for one query equals the `CPU` golden point set
/// exactly.
fn assert_parity(gpu: &GpuMidpointCircle, ctx: &GpuContext, q: &GpuMidpointCircleQuery) {
    let got = gpu.rasterize(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let expected = rasterize(q.cx, q.cy, q.r);
    assert_eq!(
        got[0].points, expected,
        "boundary mismatch for center ({}, {}) radius {}",
        q.cx, q.cy, q.r
    );
}

#[test]
fn negative_radius_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    assert_parity(&gpu, &ctx, &query(0, 0, -1));
    assert_parity(&gpu, &ctx, &query(3, -4, -100));
}

#[test]
fn radius_zero_is_single_center_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    assert_parity(&gpu, &ctx, &query(7, -3, 0));
    assert_parity(&gpu, &ctx, &query(0, 0, 0));
}

#[test]
fn radius_one_axis_neighbors() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    assert_parity(&gpu, &ctx, &query(0, 0, 1));
    assert_parity(&gpu, &ctx, &query(5, 6, 1));
}

#[test]
fn small_radii_with_coincident_reflections() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    for r in [2, 3, 4, 5, 7] {
        assert_parity(&gpu, &ctx, &query(0, 0, r));
    }
}

#[test]
fn medium_radii_at_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    for r in [16, 50, 100, 200] {
        assert_parity(&gpu, &ctx, &query(0, 0, r));
    }
}

#[test]
fn translation_invariance_positive_and_negative_offsets() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    // Same radius, several positive and negative center offsets.
    assert_parity(&gpu, &ctx, &query(40, 25, 12));
    assert_parity(&gpu, &ctx, &query(-40, 25, 12));
    assert_parity(&gpu, &ctx, &query(40, -25, 12));
    assert_parity(&gpu, &ctx, &query(-40, -25, 12));
    assert_parity(&gpu, &ctx, &query(-123, 456, 33));
}

#[test]
fn max_radius_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    // The largest radius the twin accepts; exercises the capacity headroom.
    assert_parity(
        &gpu,
        &ctx,
        &query(0, 0, prism_volumetric_gpu::midpoint_circle::MAX_RADIUS),
    );
}

#[test]
fn mixed_batch_in_one_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMidpointCircle::new(&ctx);
    let queries = [
        query(0, 0, -1),
        query(7, -3, 0),
        query(0, 0, 1),
        query(2, 2, 2),
        query(-10, 10, 16),
        query(100, -100, 50),
        query(-5, -5, 100),
    ];
    let got = gpu.rasterize(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (g, q) in got.iter().zip(queries.iter()) {
        let expected = rasterize(q.cx, q.cy, q.r);
        assert_eq!(
            g.points, expected,
            "batch boundary mismatch for center ({}, {}) radius {}",
            q.cx, q.cy, q.r
        );
    }
}
