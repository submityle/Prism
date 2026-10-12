//! Real-device parity for the spherical-harmonics basis twin:
//! [`GpuSphericalHarmonicsBasis`](prism_volumetric_gpu::spherical_harmonics_basis::GpuSphericalHarmonicsBasis)
//! must reproduce the `CPU` golden `prism_math::spherical` basis family —
//! `basis3`, `basis2`, `Sh3::eval` and `Sh3::convolve_cosine` — selected by an
//! integer `op_id`.
//!
//! * `op_id == 0` (`basis3`): the 16 real `SH` basis values for bands `0..=3`.
//! * `op_id == 1` (`basis2`): the first nine basis values, rest zero.
//! * `op_id == 2` (`eval`): the scalar `dot(coeffs, basis3(dir))` in `out[0]`.
//! * `op_id == 3` (`convolve_cosine`): per-band clamped-cosine scaling, with
//!   band 3 exactly zero.
//!
//! Any `op_id > 3` yields `valid = 0` with a cleared `out`.
//!
//! The oracle here is an independent re-implementation of those closed forms,
//! written out directly with the exact golden constants and ordering so the
//! test never imports `prism_math`, `glam`, `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! # Parity criterion
//!
//! Because the kernel evaluates the operators natively on the device while the
//! golden uses the host, `CPU` and `GPU` are not bit-exact. Each valid `out`
//! entry is compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly. Band 3 of the convolution is
//! exactly zero and is asserted as such.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_math::spherical`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::spherical_harmonics_basis::{
    GpuSphericalHarmonicsBasis, SphericalHarmonicsBasisQuery, SphericalHarmonicsBasisResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

// Golden real-`SH` normalization constants, byte-matching
// `prism_math::spherical`.
const K0: f32 = 0.282_094_79;
const K1: f32 = 0.488_602_51;
const K2A: f32 = 1.092_548_4;
const K2B: f32 = 0.315_391_57;
const K2C: f32 = 0.546_274_2;
const K3A: f32 = 0.590_043_6;
const K3B: f32 = 2.890_611_4;
const K3C: f32 = 0.457_045_8;
const K3D: f32 = 0.373_176_3;
// Clamped-cosine convolution per-band multipliers.
const COS_A0: f32 = core::f32::consts::PI;
const COS_A1: f32 = 2.094_395_1;
const COS_A2: f32 = core::f32::consts::FRAC_PI_4;

/// Independent host oracle: the 16 real `SH` basis values for `dir`, in exact
/// golden order.
fn basis3(dir: [f32; 3]) -> [f32; 16] {
    let (x, y, z) = (dir[0], dir[1], dir[2]);
    let (x2, y2, z2) = (x * x, y * y, z * z);
    [
        K0,
        K1 * y,
        K1 * z,
        K1 * x,
        K2A * x * y,
        K2A * y * z,
        K2B * (3.0 * z2 - 1.0),
        K2A * x * z,
        K2C * (x2 - y2),
        K3A * y * (3.0 * x2 - y2),
        K3B * x * y * z,
        K3C * y * (5.0 * z2 - 1.0),
        K3D * z * (5.0 * z2 - 3.0),
        K3C * x * (5.0 * z2 - 1.0),
        0.5 * K3B * z * (x2 - y2),
        K3A * x * (x2 - 3.0 * y2),
    ]
}

/// Independent host oracle: reproduces the selected golden operator in pure
/// `f32`, returning `(out, valid)`. An `op_id > 3` is invalid and clears the
/// output.
fn oracle(q: &SphericalHarmonicsBasisQuery) -> ([f32; 16], u32) {
    let b = basis3(q.dir);
    match q.op_id {
        0 => (b, 1u32),
        1 => {
            let mut out = [0.0f32; 16];
            out[..9].copy_from_slice(&b[..9]);
            (out, 1u32)
        }
        2 => {
            // Sequential dot product, matching the golden iterator-sum order.
            let mut acc = 0.0f32;
            for i in 0..16 {
                acc += q.coeffs[i] * b[i];
            }
            let mut out = [0.0f32; 16];
            out[0] = acc;
            (out, 1u32)
        }
        3 => {
            let c = q.coeffs;
            (
                [
                    c[0] * COS_A0,
                    c[1] * COS_A1,
                    c[2] * COS_A1,
                    c[3] * COS_A1,
                    c[4] * COS_A2,
                    c[5] * COS_A2,
                    c[6] * COS_A2,
                    c[7] * COS_A2,
                    c[8] * COS_A2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                1u32,
            )
        }
        _ => ([0.0f32; 16], 0u32),
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
fn assert_parity(
    gpu: &SphericalHarmonicsBasisResult,
    q: &SphericalHarmonicsBasisQuery,
    label: &str,
) {
    let (out, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1u32 {
        for k in 0..16 {
            assert!(
                close(gpu.out[k], out[k]),
                "{label}: out[{k}] mismatch gpu={} oracle={}",
                gpu.out[k],
                out[k]
            );
        }
    } else {
        for k in 0..16 {
            assert_eq!(gpu.out[k], 0.0, "{label}: invalid out[{k}] should be zero");
        }
    }
}

/// Normalizes a vector to unit length, returning `None` for a near-zero input.
/// `sqrt` is permitted in host test code; only the `WGSL` kernel is
/// transcendental-free.
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < 1.0e-6 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

const ZERO_COEFFS: [f32; 16] = [0.0; 16];

#[test]
fn basis3_matches_oracle_on_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let axes = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let queries: Vec<SphericalHarmonicsBasisQuery> = axes
        .iter()
        .map(|d| SphericalHarmonicsBasisQuery::new(0, *d, ZERO_COEFFS))
        .collect();
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "axis[{i}] should be valid");
        // The band-0 term is the constant K0 for every direction.
        assert!(close(res.out[0], K0), "axis[{i}] band0 should equal K0");
        assert_parity(res, q, &format!("basis3_axis[{i}]"));
    }
}

#[test]
fn basis2_clears_high_bands() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let dir = normalize([0.3, -0.5, 0.8]).expect("non-zero direction");
    let queries = vec![SphericalHarmonicsBasisQuery::new(1, dir, ZERO_COEFFS)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    // Bands 0..=2 match basis3; band 3 (entries 9..16) is cleared.
    let b = basis3(dir);
    for k in 0..9 {
        assert!(close(out[0].out[k], b[k]), "basis2 out[{k}] mismatch");
    }
    for k in 9..16 {
        assert_eq!(out[0].out[k], 0.0, "basis2 high band[{k}] should be zero");
    }
    assert_parity(&out[0], &queries[0], "basis2");
}

#[test]
fn eval_matches_dot_product() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let dir = normalize([0.6, 0.2, -0.4]).expect("non-zero direction");
    let coeffs = [
        0.5, -0.25, 0.75, 1.0, -0.5, 0.3, 0.9, -0.1, 0.4, 0.2, -0.7, 0.6, 0.15, -0.35, 0.8, -0.2,
    ];
    let queries = vec![SphericalHarmonicsBasisQuery::new(2, dir, coeffs)];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    // The scalar reconstruction equals the host dot product; the rest zero.
    let (want, _) = oracle(&queries[0]);
    assert!(close(out[0].out[0], want[0]), "eval scalar mismatch");
    for k in 1..16 {
        assert_eq!(out[0].out[k], 0.0, "eval out[{k}] should be zero");
    }
    assert_parity(&out[0], &queries[0], "eval");
}

#[test]
fn convolve_cosine_scales_bands() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let coeffs = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let queries = vec![SphericalHarmonicsBasisQuery::new(
        3,
        [0.0, 0.0, 1.0],
        coeffs,
    )];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    // Band-0 multiplier is PI, band 1 is COS_A1, band 2 is COS_A2.
    assert!(close(out[0].out[0], coeffs[0] * COS_A0), "band0 scale");
    assert!(close(out[0].out[1], coeffs[1] * COS_A1), "band1 scale");
    assert!(close(out[0].out[4], coeffs[4] * COS_A2), "band2 scale");
    // Band 3 (entries 9..16) is exactly zero.
    for k in 9..16 {
        assert_eq!(
            out[0].out[k], 0.0,
            "convolve band3[{k}] should be exactly zero"
        );
    }
    assert_parity(&out[0], &queries[0], "convolve");
}

#[test]
fn out_of_range_op_id_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let dir = normalize([0.1, 0.2, 0.3]).expect("non-zero direction");
    let queries = vec![
        SphericalHarmonicsBasisQuery::new(4, dir, ZERO_COEFFS),
        SphericalHarmonicsBasisQuery::new(42, dir, ZERO_COEFFS),
        SphericalHarmonicsBasisQuery::new(1000, dir, ZERO_COEFFS),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0u32, "oob[{i}] should be invalid");
        for k in 0..16 {
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
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let dir = normalize([0.4, -0.7, 0.3]).expect("non-zero direction");
    let coeffs = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    ];
    let queries = vec![
        SphericalHarmonicsBasisQuery::new(0, dir, coeffs),
        SphericalHarmonicsBasisQuery::new(7, dir, coeffs),
        SphericalHarmonicsBasisQuery::new(1, dir, coeffs),
        SphericalHarmonicsBasisQuery::new(2, dir, coeffs),
        SphericalHarmonicsBasisQuery::new(3, dir, coeffs),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 0u32);
    assert_eq!(out[2].valid, 1u32);
    assert_eq!(out[3].valid, 1u32);
    assert_eq!(out[4].valid, 1u32);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsBasis::new(&ctx);
    let mut lcg = Lcg::new(0x5A17_3B29);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let op_id = lcg.next_u32() % 4;
        let raw = [
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-1.0, 1.0),
        ];
        let Some(dir) = normalize(raw) else {
            continue;
        };
        let mut coeffs = [0.0f32; 16];
        for c in coeffs.iter_mut() {
            *c = lcg.next_range(-2.0, 2.0);
        }
        queries.push(SphericalHarmonicsBasisQuery::new(op_id, dir, coeffs));
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
