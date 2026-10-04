//! Real-device parity for the 2x2 matrix twin:
//! [`GpuMat2`](prism_volumetric_gpu::mat2::GpuMat2) must reproduce the `CPU`
//! golden `prism_math::mat::Mat2`'s four operators `determinant`, `transpose`,
//! `mul_vec2` and `inverse`, selected by an integer `op_id`.
//!
//! `Mat2` is column-major, so a matrix is flattened as
//! `m = [x_axis.x, x_axis.y, y_axis.x, y_axis.y]`, laid out as
//!
//! ```text
//! | m0  m2 |
//! | m1  m3 |
//! ```
//!
//! For `op_id == 0` (`determinant`) the result is the scalar `m0 * m3 - m2 * m1`
//! in `out[0]`. For `op_id == 1` (`transpose`) the columns swap rows, giving
//! `out = [m0, m2, m1, m3]`. For `op_id == 2` (`mul_vec2`) the result is the
//! column combination `[m0 * v.x + m2 * v.y, m1 * v.x + m3 * v.y, 0, 0]`. For
//! `op_id == 3` (`inverse`), with `inv = 1 / det`, the adjugate layout is
//! `out = [m3 * inv, -m1 * inv, -m2 * inv, m0 * inv]`; a singular matrix
//! (`abs(det) <= 1e-20`) yields `valid = 0`. Any `op_id > 3` also yields
//! `valid = 0` with a cleared `out`.
//!
//! The oracle here is an independent re-implementation of those four closed
//! forms, written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core`, `prism_math` or `glam`.
//!
//! # Parity criterion
//!
//! Because the kernel evaluates the operators natively on the device while the
//! golden uses the host, `CPU` and `GPU` are not bit-exact. Each valid `out`
//! entry is compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly. The random sweep keeps
//! `abs(det)` well away from the singularity knee so the `inverse` validity
//! decision cannot flip under round-off.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_math::mat::Mat2`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mat2::{GpuMat2, Mat2Query, Mat2Result};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Singularity knee for `inverse`: at or below this `abs(det)` the matrix is
/// treated as singular and the result is invalid.
const DET_KNEE: f32 = 1.0e-20;

/// Independent host oracle: reproduces the four golden `Mat2` operators in pure
/// `f32`, returning `(out, valid)`. An `op_id > 3` is invalid, as is a singular
/// matrix under `inverse`; both clear the output.
fn oracle(q: &Mat2Query) -> ([f32; 4], u32) {
    let [m0, m1, m2, m3] = q.m;
    let [vx, vy] = q.v;
    match q.op_id {
        0 => {
            let det = m0 * m3 - m2 * m1;
            ([det, 0.0, 0.0, 0.0], 1u32)
        }
        1 => ([m0, m2, m1, m3], 1u32),
        2 => ([m0 * vx + m2 * vy, m1 * vx + m3 * vy, 0.0, 0.0], 1u32),
        3 => {
            let det = m0 * m3 - m2 * m1;
            if det.abs() > DET_KNEE {
                let inv = 1.0 / det;
                ([m3 * inv, -m1 * inv, -m2 * inv, m0 * inv], 1u32)
            } else {
                ([0.0, 0.0, 0.0, 0.0], 0u32)
            }
        }
        _ => ([0.0, 0.0, 0.0, 0.0], 0u32),
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, and each `out` entry to tolerance when valid, else
/// zero.
fn assert_parity(gpu: &Mat2Result, q: &Mat2Query, label: &str) {
    let (out, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1u32 {
        for k in 0..4 {
            assert!(
                close(gpu.out[k], out[k]),
                "{label}: out[{k}] mismatch gpu={} oracle={}",
                gpu.out[k],
                out[k]
            );
        }
    } else {
        for k in 0..4 {
            assert_eq!(gpu.out[k], 0.0, "{label}: invalid out[{k}] should be zero");
        }
    }
}

#[test]
fn determinant_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    // Identity det = 1; a known general matrix det = 2*5 - 4*3 = -2.
    let queries = vec![
        Mat2Query::new(0, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
        Mat2Query::new(0, [2.0, 3.0, 4.0, 5.0], [0.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].out[0], 1.0));
    assert!(close(out[1].out[0], -2.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "det[{i}] should be valid");
        assert_parity(res, q, &format!("det[{i}]"));
    }
}

#[test]
fn transpose_swaps_off_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    // Non-symmetric matrix: transpose must swap m1 and m2.
    let m = [2.0, 3.0, 4.0, 5.0];
    let queries = vec![Mat2Query::new(1, m, [0.0, 0.0])];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert!(close(out[0].out[0], 2.0));
    assert!(close(out[0].out[1], 4.0));
    assert!(close(out[0].out[2], 3.0));
    assert!(close(out[0].out[3], 5.0));
    assert_parity(&out[0], &queries[0], "transpose");
}

#[test]
fn mul_vec2_combines_columns() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    // | 2 4 |   | 1 |   | 2*1 + 4*(-1) | = | -2 |
    // | 3 5 | * |-1 | = | 3*1 + 5*(-1) |   | -2 |
    let queries = vec![Mat2Query::new(2, [2.0, 3.0, 4.0, 5.0], [1.0, -1.0])];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert!(close(out[0].out[0], -2.0));
    assert!(close(out[0].out[1], -2.0));
    assert_parity(&out[0], &queries[0], "mul_vec2");
}

#[test]
fn inverse_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    // det = 2*5 - 4*3 = -2, inv = -0.5. Adjugate: [5, -3, -4, 2] * inv.
    let m = [2.0, 3.0, 4.0, 5.0];
    let queries = vec![Mat2Query::new(3, m, [0.0, 0.0])];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    let inv = -0.5_f32;
    assert!(close(out[0].out[0], 5.0 * inv));
    assert!(close(out[0].out[1], -3.0 * inv));
    assert!(close(out[0].out[2], -4.0 * inv));
    assert!(close(out[0].out[3], 2.0 * inv));
    assert_parity(&out[0], &queries[0], "inverse");
}

#[test]
fn singular_inverse_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    // Zero matrix and a rank-1 matrix (columns parallel) both have det = 0.
    let queries = vec![
        Mat2Query::new(3, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Mat2Query::new(3, [1.0, 2.0, 2.0, 4.0], [0.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0u32, "singular[{i}] should be invalid");
        for k in 0..4 {
            assert_eq!(res.out[k], 0.0, "singular[{i}] out[{k}] should be zero");
        }
        assert_parity(res, q, &format!("singular[{i}]"));
    }
}

#[test]
fn out_of_range_op_id_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    let m = [2.0, 3.0, 4.0, 5.0];
    let queries = vec![
        Mat2Query::new(4, m, [1.0, 1.0]),
        Mat2Query::new(9, m, [1.0, 1.0]),
        Mat2Query::new(100, m, [1.0, 1.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0u32, "oob[{i}] should be invalid");
        for k in 0..4 {
            assert_eq!(res.out[k], 0.0, "oob[{i}] out[{k}] should be zero");
        }
        assert_parity(res, q, &format!("oob[{i}]"));
    }
}

#[test]
fn batch_mixes_ops_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    let m = [2.0, 3.0, 4.0, 5.0];
    let queries = vec![
        Mat2Query::new(0, m, [0.0, 0.0]),
        Mat2Query::new(5, m, [0.0, 0.0]),
        Mat2Query::new(1, m, [0.0, 0.0]),
        Mat2Query::new(3, m, [0.0, 0.0]),
        Mat2Query::new(2, m, [1.0, -1.0]),
        Mat2Query::new(3, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 0u32);
    assert_eq!(out[2].valid, 1u32);
    assert_eq!(out[3].valid, 1u32);
    assert_eq!(out[4].valid, 1u32);
    assert_eq!(out[5].valid, 0u32);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMat2::new(&ctx);
    let mut lcg = Lcg::new(0x0A2B_11CE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let op_id = lcg.next_u32() % 4;
        let m = [
            lcg.next_range(-4.0, 4.0),
            lcg.next_range(-4.0, 4.0),
            lcg.next_range(-4.0, 4.0),
            lcg.next_range(-4.0, 4.0),
        ];
        let v = [lcg.next_range(-3.0, 3.0), lcg.next_range(-3.0, 3.0)];
        // Keep every matrix comfortably non-singular so the inverse validity
        // decision and the branch cannot flip under round-off between host and
        // device.
        let det = m[0] * m[3] - m[2] * m[1];
        if det.abs() < 0.1 {
            continue;
        }
        queries.push(Mat2Query::new(op_id, m, v));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "sweep[{i}] should be valid");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
