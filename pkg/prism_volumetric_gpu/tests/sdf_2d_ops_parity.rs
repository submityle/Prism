//! Real-device parity for the analytic two-dimensional primitive/domain twin:
//! [`GpuSdf2dOps`](prism_volumetric_gpu::sdf_2d_ops::GpuSdf2dOps) must reproduce
//! the `CPU` closed forms of `prism_render_architecture::ray_scene` — the Inigo
//! Quilez `circle_2d` and `rounded_box_2d` primitives from `sdf_primitives` and
//! the `elongate_2d` / `elongate_2d_correction` domain pair from `sdf_domain` —
//! across interior, exterior and surface points, all four rounded-box quadrant
//! radius selections, and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed forms: the radial circle
//! distance, the rounded box's quadrant radius pick and corner/edge split, the
//! per-axis elongation displacement, and the non-positive interior correction.
//! Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! values the reference does.
//!
//! # Parity criterion
//!
//! `circle_2d` and `rounded_box_2d` thread through a `sqrt`, so a `GPU` result
//! may land a few units in the last place from the scalar oracle; each output
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a relative
//! floor of `1e-6` so a near-zero expected value does not inflate the relative
//! error. `elongate_2d` and `elongate_2d_correction` are pure `clamp`/`min`/
//! `max` arithmetic and should match far tighter, but share the one bound.
//!
//! # Conditioning
//!
//! The `rounded_box_2d` quadrant radius selection switches on `point.x == 0` or
//! `point.y == 0`, and the `elongate` core boundary sits at
//! `|point| == half_extent`; both are measure-zero creases where the compared
//! branches agree continuously, so no branch disagreement can produce a cliff.
//! The randomized sweep still rejects samples near the coordinate axes and near
//! any surface zero to keep the comparison far from such loci, and draws only
//! well-formed shape parameters (positive radii and half-extents).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene`（`sdf_primitives` 与 `sdf_domain`）；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_2d_ops::{GpuSdf2dOps, Sdf2dOpsQuery, Sdf2dOpsResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any output. A `GPU` `sqrt`/divide may land a few units in
/// the last place from the scalar oracle; `1e-4` admits that legal slack while
/// still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any output, applied for larger magnitudes where a few
/// units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Default `circle_2d` radius for the point-focused fixtures.
const CIRCLE_RADIUS: f32 = 0.5;

/// Default `rounded_box_2d` half-extent for the point-focused fixtures.
const BOX_HALF_EXTENT: [f32; 2] = [0.6, 0.4];

/// Default `rounded_box_2d` corner radii `[top_right, bottom_right, top_left,
/// bottom_left]`, all distinct so a wrong quadrant pick is caught.
const BOX_RADII: [f32; 4] = [0.2, 0.15, 0.1, 0.05];

/// Default `elongate_2d` / `elongate_2d_correction` half-extent.
const ELONGATE_HALF_EXTENT: [f32; 2] = [0.3, 0.2];

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Euclidean length of a 2-vector, matching the reference `length2`.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Independent reimplementation of `ray_scene::sdf_primitives::circle_2d`: the
/// radial distance to a centred circle, `|point| - radius`.
fn circle_2d_host(point: [f32; 2], radius: f32) -> f32 {
    length2(point) - radius
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::rounded_box_2d`: pick the active corner radius by
/// quadrant with ordered comparisons, then evaluate the rounded rectangle.
fn rounded_box_2d_host(point: [f32; 2], half_extent: [f32; 2], radii: [f32; 4]) -> f32 {
    let rx = if point[0] > 0.0 { radii[0] } else { radii[2] };
    let ry = if point[0] > 0.0 { radii[1] } else { radii[3] };
    let r = if point[1] > 0.0 { rx } else { ry };
    let qx = point[0].abs() - half_extent[0] + r;
    let qy = point[1].abs() - half_extent[1] + r;
    qx.max(qy).min(0.0) + length2([qx.max(0.0), qy.max(0.0)]) - r
}

/// Independent reimplementation of `ray_scene::sdf_domain::elongate_2d`: carve
/// the `[-h, h]` core out of each axis.
fn elongate_2d_host(point: [f32; 2], half_extent: [f32; 2]) -> [f32; 2] {
    [
        point[0] - point[0].clamp(-half_extent[0], half_extent[0]),
        point[1] - point[1].clamp(-half_extent[1], half_extent[1]),
    ]
}

/// Independent reimplementation of
/// `ray_scene::sdf_domain::elongate_2d_correction`: the non-positive interior
/// correction `min(max(|p.x| - h.x, |p.y| - h.y), 0)`.
fn elongate_2d_correction_host(point: [f32; 2], half_extent: [f32; 2]) -> f32 {
    let qx = point[0].abs() - half_extent[0];
    let qy = point[1].abs() - half_extent[1];
    qx.max(qy).min(0.0)
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Builds one query at `point` with the default per-shape scalars so every twin
/// output is exercised on every dispatch.
fn q_at(point: [f32; 2]) -> Sdf2dOpsQuery {
    Sdf2dOpsQuery {
        point,
        circle_radius: CIRCLE_RADIUS,
        box_half_extent: BOX_HALF_EXTENT,
        box_radii: BOX_RADII,
        elongate_half_extent: ELONGATE_HALF_EXTENT,
    }
}

/// Dispatches `queries` and asserts every output matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuSdf2dOps, queries: &[Sdf2dOpsQuery]) {
    let got: Vec<Sdf2dOpsResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_circle = circle_2d_host(q.point, q.circle_radius);
        let golden_box = rounded_box_2d_host(q.point, q.box_half_extent, q.box_radii);
        let golden_corr = elongate_2d_correction_host(q.point, q.elongate_half_extent);
        let golden_elong = elongate_2d_host(q.point, q.elongate_half_extent);
        assert!(
            close(r.circle, golden_circle),
            "circle mismatch: gpu={} golden={} (point={:?} radius={})",
            r.circle,
            golden_circle,
            q.point,
            q.circle_radius
        );
        assert!(
            close(r.rounded_box, golden_box),
            "rounded_box mismatch: gpu={} golden={} (point={:?} half={:?} radii={:?})",
            r.rounded_box,
            golden_box,
            q.point,
            q.box_half_extent,
            q.box_radii
        );
        assert!(
            close(r.correction, golden_corr),
            "correction mismatch: gpu={} golden={} (point={:?} half={:?})",
            r.correction,
            golden_corr,
            q.point,
            q.elongate_half_extent
        );
        assert!(
            close(r.elongate[0], golden_elong[0]) && close(r.elongate[1], golden_elong[1]),
            "elongate mismatch: gpu={:?} golden={:?} (point={:?} half={:?})",
            r.elongate,
            golden_elong,
            q.point,
            q.elongate_half_extent
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_2d_ops parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn circle_interior_surface_exterior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    // Centre (negative), near the rim, and well outside (positive).
    let queries = [
        q_at([0.0, 0.0]),
        q_at([0.4, 0.0]),
        q_at([0.0, 0.45]),
        q_at([1.3, 1.1]),
        q_at([-1.4, 0.2]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn rounded_box_all_four_quadrants() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    // One point per open quadrant so each corner radius selection is exercised,
    // interior and exterior.
    let queries = [
        q_at([0.3, 0.2]),
        q_at([0.3, -0.2]),
        q_at([-0.3, 0.2]),
        q_at([-0.3, -0.2]),
        q_at([0.9, 0.8]),
        q_at([-1.0, -0.7]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn elongate_inside_and_outside_core() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    // Inside the [-h, h] core (zero displacement, negative correction), on one
    // axis only, and fully outside the core (nonzero displacement, zero
    // correction).
    let queries = [
        q_at([0.1, 0.05]),
        q_at([0.5, 0.1]),
        q_at([0.1, 0.5]),
        q_at([0.8, 0.7]),
        q_at([-0.9, -0.6]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    // A heterogeneous batch with varied per-shape parameters in one dispatch.
    let queries = [
        Sdf2dOpsQuery {
            point: [0.3, 0.5],
            circle_radius: 0.7,
            box_half_extent: [0.5, 0.3],
            box_radii: [0.25, 0.2, 0.15, 0.1],
            elongate_half_extent: [0.2, 0.4],
        },
        Sdf2dOpsQuery {
            point: [-0.8, 0.9],
            circle_radius: 0.3,
            box_half_extent: [0.7, 0.6],
            box_radii: [0.05, 0.1, 0.2, 0.15],
            elongate_half_extent: [0.5, 0.1],
        },
        Sdf2dOpsQuery {
            point: [1.4, -1.1],
            circle_radius: 1.0,
            box_half_extent: [0.4, 0.9],
            box_radii: [0.3, 0.1, 0.05, 0.2],
            elongate_half_extent: [0.3, 0.3],
        },
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdf2dOps::new(&ctx);
    let mut state: u64 = 0x51ed_270b_6a3c_19f4;
    // Margin kept away from the coordinate axes (where the rounded-box quadrant
    // radius selection switches) and from each surface zero. Those loci are
    // continuous (the branches agree there), so this is extra safety only.
    let margin = 0.03_f32;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let point = [draw(&mut state, -1.8, 1.8), draw(&mut state, -1.8, 1.8)];
        // Reject near the coordinate axes so the quadrant radius pick is
        // unambiguous between host and device.
        if point[0].abs() < margin || point[1].abs() < margin {
            continue;
        }

        let circle_radius = draw(&mut state, 0.2, 1.0);
        let box_half_extent = [draw(&mut state, 0.3, 0.9), draw(&mut state, 0.3, 0.9)];
        // Keep each corner radius below the smaller half-extent so the shape
        // stays a well-formed rounded rectangle.
        let r_cap = box_half_extent[0].min(box_half_extent[1]) * 0.9;
        let box_radii = [
            draw(&mut state, 0.02, r_cap),
            draw(&mut state, 0.02, r_cap),
            draw(&mut state, 0.02, r_cap),
            draw(&mut state, 0.02, r_cap),
        ];
        let elongate_half_extent = [draw(&mut state, 0.1, 0.6), draw(&mut state, 0.1, 0.6)];

        let q = Sdf2dOpsQuery {
            point,
            circle_radius,
            box_half_extent,
            box_radii,
            elongate_half_extent,
        };
        // Reject near each surface zero where a divide/sqrt could amplify a
        // last-place difference (the field is continuous there regardless).
        if circle_2d_host(point, circle_radius).abs() < margin
            || rounded_box_2d_host(point, box_half_extent, box_radii).abs() < margin
        {
            continue;
        }
        queries.push(q);
    }
    check_batch(&ctx, &gpu, &queries);
}
