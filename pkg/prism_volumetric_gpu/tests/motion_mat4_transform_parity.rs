//! Real-device parity for the homogeneous `4x4` matrix-vector transform twin:
//! [`GpuMotionMat4Transform`](prism_volumetric_gpu::motion_mat4_transform::GpuMotionMat4Transform)
//! must reproduce the column-major product of the `CPU` golden
//! [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4) across
//! the identity, translation, scale, a projection-like transform, and a
//! randomized batch compared component-for-component.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4)
//! is `pub`, so each `GPU` result is pinned directly against the golden run on
//! the same input: a column-major matrix built with
//! [`Mat4::from_cols`](prism_render_architecture::motion::Mat4::from_cols) times
//! a [`Vec4`](prism_render_architecture::motion::Vec4).
//!
//! # Parity criterion
//!
//! Every output component is a sum of four products — no `sqrt`, no
//! transcendental — so each coordinate is a continuous `f32` asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The product has no truncation or branch on the input, so there is no
//! half-step tie or discrete threshold to straddle. Fixture magnitudes are kept
//! moderate so the summed products stay comfortably inside the relative bound.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::Mat4::mul_vec4`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::{Mat4, Vec4};
use prism_volumetric_gpu::motion_mat4_transform::{
    GpuMotionMat4Transform, MotionMat4TransformQuery, MotionMat4TransformResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a transformed component.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &MotionMat4TransformQuery) -> MotionMat4TransformResult {
    let c = q.columns;
    let m = Mat4::from_cols(
        Vec4::new(c[0][0], c[0][1], c[0][2], c[0][3]),
        Vec4::new(c[1][0], c[1][1], c[1][2], c[1][3]),
        Vec4::new(c[2][0], c[2][1], c[2][2], c[2][3]),
        Vec4::new(c[3][0], c[3][1], c[3][2], c[3][3]),
    );
    let v = Vec4::new(q.vector[0], q.vector[1], q.vector[2], q.vector[3]);
    let r = m.mul_vec4(v);
    MotionMat4TransformResult {
        transformed: [r.x, r.y, r.z, r.w],
    }
}

/// Pins one `GPU` result against the golden oracle: all four continuous
/// components within tolerance.
fn assert_result(idx: usize, got: &MotionMat4TransformResult, want: &MotionMat4TransformResult) {
    let g = got.transformed;
    let w = want.transformed;
    assert!(
        close(g[0], w[0]) && close(g[1], w[1]) && close(g[2], w[2]) && close(g[3], w[3]),
        "result {idx} transform: gpu {g:?} vs cpu {w:?}"
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[MotionMat4TransformQuery]) {
    let gpu = GpuMotionMat4Transform::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
    }
}

/// Column-major identity matrix columns.
fn identity_columns() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Column-major translation matrix columns (translation in the last column).
fn translation_columns(tx: f32, ty: f32, tz: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx, ty, tz, 1.0],
    ]
}

/// Column-major non-uniform scale matrix columns.
fn scale_columns(sx: f32, sy: f32, sz: f32) -> [[f32; 4]; 4] {
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_mat4_transform parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionMat4Transform::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Identity transform returns the input unchanged.
    let cols = identity_columns();
    let queries: Vec<MotionMat4TransformQuery> = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0, 1.0],
        [-4.0, 5.0, -6.0, 1.0],
        [0.5, -0.25, 7.0, 2.0],
    ]
    .into_iter()
    .map(|vector| MotionMat4TransformQuery::new(cols, vector))
    .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn translation_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let cols = translation_columns(3.0, -2.0, 5.0);
    // Points (w = 1) pick up the translation; directions (w = 0) do not.
    let queries = vec![
        MotionMat4TransformQuery::new(cols, [0.0, 0.0, 0.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [1.0, 1.0, 1.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [2.0, -3.0, 4.0, 0.0]),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn scale_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let cols = scale_columns(2.0, -0.5, 3.0);
    let queries = vec![
        MotionMat4TransformQuery::new(cols, [1.0, 2.0, 3.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [-4.0, 8.0, -1.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [0.0, 0.0, 0.0, 1.0]),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn projection_like_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A perspective-style column-major matrix with a -1 in the w row of the z
    // column, so the output w carries -z.
    let cols = [
        [1.3, 0.0, 0.0, 0.0],
        [0.0, 1.7, 0.0, 0.0],
        [0.0, 0.0, -1.002, -1.0],
        [0.0, 0.0, -0.2, 0.0],
    ];
    let queries = vec![
        MotionMat4TransformQuery::new(cols, [0.0, 0.0, -5.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [2.0, -1.0, -10.0, 1.0]),
        MotionMat4TransformQuery::new(cols, [-3.0, 4.0, -2.5, 1.0]),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x51a3_9f0c_7e26_84b1_u64;

    let mut queries: Vec<MotionMat4TransformQuery> = Vec::new();
    while queries.len() < 256 {
        let mut columns = [[0.0_f32; 4]; 4];
        for col in &mut columns {
            for comp in col.iter_mut() {
                *comp = ranged(&mut state, -8.0, 8.0);
            }
        }
        let vector = [
            ranged(&mut state, -8.0, 8.0),
            ranged(&mut state, -8.0, 8.0),
            ranged(&mut state, -8.0, 8.0),
            ranged(&mut state, -8.0, 8.0),
        ];
        queries.push(MotionMat4TransformQuery::new(columns, vector));
    }
    run_and_check(&ctx, &queries);
}
