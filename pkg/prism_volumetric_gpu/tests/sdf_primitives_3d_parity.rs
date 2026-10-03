//! Real-device parity for the 3D-primitive signed-distance twin:
//! [`GpuSdfPrimitives3d`](prism_volumetric_gpu::sdf_primitives_3d::GpuSdfPrimitives3d)
//! must reproduce the three closed-form fields of the host-side independent
//! reimplementations
//! [`box_frame_sdf`](prism_volumetric_gpu::sdf_primitives_3d::box_frame_sdf),
//! [`vertical_capsule_sdf`](prism_volumetric_gpu::sdf_primitives_3d::vertical_capsule_sdf)
//! and
//! [`segment_3d_sdf`](prism_volumetric_gpu::sdf_primitives_3d::segment_3d_sdf)
//! across box-frame interior and exterior points, the vertical capsule's
//! cap-end hot-spots, segment points exterior to both endpoints, and a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected distances come from the module's own host-side independent
//! reimplementations; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! evaluates the same distance.
//!
//! # Parity criterion
//!
//! Each distance is a *continuous* quantity threaded through `sqrt` and
//! products, so every assertion compares with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`). All three fields are
//! continuous everywhere (there is no discrete classification), so the only
//! genuine degeneracy is a zero-length `segment_3d` (`a == b`, where the
//! projection division is guarded). The fixtures and the randomized sweep keep
//! the segment endpoints a clear margin apart so the guarded branch never sits
//! on its threshold.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_primitives_3d::{
    box_frame_sdf, segment_3d_sdf, vertical_capsule_sdf, GpuSdfPrimitives3d, SdfPrimitives3dQuery,
    SdfPrimitives3dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous distance comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous distance comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;
/// Minimum segment length kept by the fixtures and the sweep so the guarded
/// zero-length `segment_3d` projection branch never sits on its threshold.
const SEG_MARGIN: f32 = 0.3;

/// Standard box-frame half-extent and bar thickness for the named tests.
const FRAME: [f32; 4] = [1.0, 0.8, 0.6, 0.1];
/// Standard vertical-capsule height and radius for the named tests.
const CAP: [f32; 2] = [1.5, 0.4];
/// Standard segment endpoints for the named tests; clearly longer than
/// `SEG_MARGIN` so the projection branch stays well-conditioned.
const SEG: [f32; 6] = [-0.8, -0.5, -0.3, 0.9, 0.6, 0.4];

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Computes the three expected distances from the host-side reimplementations.
fn oracle(q: &SdfPrimitives3dQuery) -> [f32; 3] {
    [
        box_frame_sdf(
            [q.px, q.py, q.pz],
            [q.fr_hx, q.fr_hy, q.fr_hz],
            q.fr_thickness,
        ),
        vertical_capsule_sdf([q.px, q.py, q.pz], q.cap_height, q.cap_radius),
        segment_3d_sdf(
            [q.px, q.py, q.pz],
            [q.seg_ax, q.seg_ay, q.seg_az],
            [q.seg_bx, q.seg_by, q.seg_bz],
        ),
    ]
}

/// Dispatches every query and pins each `GPU` distance against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfPrimitives3d, queries: &[SdfPrimitives3dQuery]) {
    let got: Vec<SdfPrimitives3dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.dist_box_frame, want[0])
                && close(result.dist_capsule, want[1])
                && close(result.dist_segment, want[2]),
            "query {idx}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            result.dist_box_frame,
            result.dist_capsule,
            result.dist_segment,
            want[0],
            want[1],
            want[2],
        );
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// Rejects queries whose `segment_3d` endpoints sit closer than `SEG_MARGIN`,
/// the lone genuine degeneracy (a zero-length segment pins the guarded
/// projection parameter). The three fields are otherwise continuous
/// everywhere, so near-zero magnitudes stay inside the absolute tolerance.
fn well_conditioned(q: &SdfPrimitives3dQuery) -> bool {
    let dx = q.seg_bx - q.seg_ax;
    let dy = q.seg_by - q.seg_ay;
    let dz = q.seg_bz - q.seg_az;
    dx * dx + dy * dy + dz * dz >= SEG_MARGIN * SEG_MARGIN
}

/// Builds a query at `point` with the standard shape parameters.
fn standard(px: f32, py: f32, pz: f32) -> SdfPrimitives3dQuery {
    SdfPrimitives3dQuery::new(
        px, py, pz, FRAME[0], FRAME[1], FRAME[2], FRAME[3], CAP[0], CAP[1], SEG[0], SEG[1], SEG[2],
        SEG[3], SEG[4], SEG[5],
    )
}

/// The deterministic fixtures: box-frame interior and exterior points, capsule
/// cap-end points and segment points exterior to both endpoints, across
/// varied-but-well-formed shapes, all keeping the segment endpoints a clear
/// margin apart.
fn edge_fixtures() -> Vec<SdfPrimitives3dQuery> {
    vec![
        // Inside a corner bar of the standard frame.
        standard(0.95, 0.75, 0.0),
        // Far exterior of every shape.
        standard(3.0, 3.0, 3.0),
        // Near the top cap of the capsule, exterior.
        standard(0.0, 2.1, 0.0),
        // Near the bottom cap of the capsule, exterior.
        standard(0.0, -0.6, 0.0),
        // Beyond the segment `b` endpoint along its axis.
        standard(1.6, 1.1, 0.9),
        // Beyond the segment `a` endpoint on the far side.
        standard(-1.6, -1.1, -0.8),
        // Off-axis exterior with a thicker frame and a taller capsule.
        SdfPrimitives3dQuery::new(
            0.4, 0.3, 1.4, 1.2, 0.9, 0.7, 0.2, 1.8, 0.5, -0.7, -0.6, -0.4, 0.8, 0.7, 0.5,
        ),
        // Interior-adjacent with a slim capsule and a shifted segment.
        SdfPrimitives3dQuery::new(
            -0.3, 0.5, 0.2, 0.9, 1.1, 0.5, 0.15, 1.2, 0.3, -0.5, 0.2, -0.6, 0.9, 1.0, 0.6,
        ),
        // Lower corner exterior with a wide frame and a long segment.
        SdfPrimitives3dQuery::new(
            -1.4, -1.2, 0.9, 1.3, 1.0, 0.8, 0.25, 1.4, 0.6, -1.0, -0.8, -0.6, 1.0, 0.9, 0.7,
        ),
        // Mixed with a tall capsule and a near-vertical segment.
        SdfPrimitives3dQuery::new(
            0.2, 1.3, -0.4, 1.0, 1.2, 0.6, 0.12, 2.0, 0.45, 0.1, -0.9, 0.2, 0.2, 0.9, 0.3,
        ),
    ]
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "a skipped device test should report why on hosts without a GPU"
        )]
        {
            eprintln!("skipping sdf_primitives_3d parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn box_frame_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let q = standard(0.95, 0.75, 0.0);
    let want = oracle(&q);
    assert!(
        want[0] < 0.0,
        "a point inside a frame bar has negative distance"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn box_frame_exterior_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let q = standard(3.0, 3.0, 3.0);
    let want = oracle(&q);
    assert!(want[0] > 0.0, "a far point outside the frame is positive");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn capsule_interior_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let q = standard(0.0, 0.75, 0.0);
    let want = oracle(&q);
    assert!(
        want[1] < 0.0,
        "a point on the capsule axis is inside the pill"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn capsule_cap_end_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let q = standard(0.0, 2.1, 0.0);
    let want = oracle(&q);
    assert!(
        want[1] > 0.0,
        "a point beyond the top cap is outside the pill"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn segment_distance_is_non_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let q = standard(1.6, 1.1, 0.9);
    let want = oracle(&q);
    assert!(
        want[2] >= 0.0,
        "an unsigned segment distance is never negative"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let fixtures = edge_fixtures();
    for q in &fixtures {
        assert!(
            well_conditioned(q),
            "edge fixtures must keep the segment endpoints a clear margin apart"
        );
    }
    check(&ctx, &gpu, &fixtures);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfPrimitives3d::new(&ctx);
    let mut state: u64 = 0x1357_9BDF_2468_ACE0;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let px = uniform(&mut state, -2.5, 2.5);
        let py = uniform(&mut state, -2.5, 2.5);
        let pz = uniform(&mut state, -2.5, 2.5);
        let fr_hx = uniform(&mut state, 0.5, 1.5);
        let fr_hy = uniform(&mut state, 0.5, 1.5);
        let fr_hz = uniform(&mut state, 0.5, 1.5);
        let fr_thickness = uniform(&mut state, 0.05, 0.3);
        let cap_height = uniform(&mut state, 0.5, 2.0);
        let cap_radius = uniform(&mut state, 0.2, 0.8);
        let seg_ax = uniform(&mut state, -1.5, 1.5);
        let seg_ay = uniform(&mut state, -1.5, 1.5);
        let seg_az = uniform(&mut state, -1.5, 1.5);
        let seg_bx = uniform(&mut state, -1.5, 1.5);
        let seg_by = uniform(&mut state, -1.5, 1.5);
        let seg_bz = uniform(&mut state, -1.5, 1.5);
        let q = SdfPrimitives3dQuery::new(
            px,
            py,
            pz,
            fr_hx,
            fr_hy,
            fr_hz,
            fr_thickness,
            cap_height,
            cap_radius,
            seg_ax,
            seg_ay,
            seg_az,
            seg_bx,
            seg_by,
            seg_bz,
        );
        // Reject near-degenerate (zero-length) segments so the guarded
        // projection branch stays clear of its threshold.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
