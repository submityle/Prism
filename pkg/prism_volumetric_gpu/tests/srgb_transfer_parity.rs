//! Real-device parity for the sRGB transfer-function twin:
//! [`GpuSrgbTransfer`](prism_volumetric_gpu::srgb_transfer::GpuSrgbTransfer)
//! must reproduce the `CPU` golden `prism_math::color::transfer`'s four
//! functions `srgb_to_linear`, `linear_to_srgb`, `fast_srgb_to_linear` and
//! `fast_linear_to_srgb`, selected by an integer `func_id`.
//!
//! The exact pair is the IEC 61966-2-1 piecewise curve; the `fast_*` pair is
//! the single-`pow` gamma-2.2 approximation. For `func_id == 0`
//! (`srgb_to_linear`) a component at or below `0.040_448_237` maps to
//! `c / 12.92`, else to `pow((c + 0.055) / 1.055, 2.4)`. For `func_id == 1`
//! (`linear_to_srgb`) a component at or below `0.003_130_8` maps to
//! `c * 12.92`, else to `1.055 * pow(c, 1.0 / 2.4) - 0.055`. For `func_id == 2`
//! and `func_id == 3` the result is `pow(c, 2.2)` and `pow(c, 1.0 / 2.2)`
//! respectively. Any `func_id > 3` yields `valid = 0` with a cleared `value`.
//!
//! The oracle here is an independent re-implementation of those four closed
//! forms, written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core` or `prism_math`.
//!
//! # Parity criterion
//!
//! Because the kernel evaluates `pow` natively on the device while the golden
//! uses the host `pow`, `CPU` and `GPU` are not bit-exact. The valid `value`
//! scalar is compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`),
//! which absorbs that difference; the discrete `valid` flag is compared
//! exactly. The random sweep keeps the component away from the two piecewise
//! knees (`0.003_130_8` and `0.040_448_237`) so the branch decision cannot flip
//! under round-off.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_math::color::transfer`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::srgb_transfer::{GpuSrgbTransfer, SrgbTransferQuery, SrgbTransferResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Lower piecewise knee of `linear_to_srgb` (`func_id == 1`).
const KNEE_LINEAR_TO_SRGB: f32 = 0.003_130_8;
/// Lower piecewise knee of `srgb_to_linear` (`func_id == 0`).
const KNEE_SRGB_TO_LINEAR: f32 = 0.040_448_237;

/// Independent host oracle: reproduces the four golden transfer functions in
/// pure `f32`, returning `(value, valid)`. A `func_id > 3` is invalid and
/// clears the value.
fn oracle(q: &SrgbTransferQuery) -> (f32, u32) {
    let c = q.c;
    let value = match q.func_id {
        0 => {
            if c <= KNEE_SRGB_TO_LINEAR {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        1 => {
            if c <= KNEE_LINEAR_TO_SRGB {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        }
        2 => c.powf(2.2),
        3 => c.powf(1.0 / 2.2),
        _ => return (0.0, 0u32),
    };
    (value, 1u32)
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
/// `valid` flag exactly, and the `value` scalar to tolerance when valid, else
/// zero.
fn assert_parity(gpu: &SrgbTransferResult, q: &SrgbTransferQuery, label: &str) {
    let (value, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1u32 {
        assert!(
            close(gpu.value, value),
            "{label}: value mismatch gpu={} oracle={}",
            gpu.value,
            value
        );
    } else {
        assert_eq!(gpu.value, 0.0, "{label}: invalid value should be zero");
    }
}

#[test]
fn srgb_to_linear_both_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    // Below the knee -> linear segment c/12.92; above it -> the power segment.
    let queries = vec![
        SrgbTransferQuery::new(0, 0.02),
        SrgbTransferQuery::new(0, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 1u32);
    assert!(close(out[0].value, 0.02 / 12.92));
    assert_parity(&out[0], &queries[0], "s2l_below");
    assert_parity(&out[1], &queries[1], "s2l_above");
}

#[test]
fn linear_to_srgb_both_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    // Below the knee -> linear segment c*12.92; above it -> the power segment.
    let queries = vec![
        SrgbTransferQuery::new(1, 0.001),
        SrgbTransferQuery::new(1, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1u32);
    assert_eq!(out[1].valid, 1u32);
    assert!(close(out[0].value, 0.001 * 12.92));
    assert_parity(&out[0], &queries[0], "l2s_below");
    assert_parity(&out[1], &queries[1], "l2s_above");
}

#[test]
fn fast_variants_roundtrip_fixture() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    // fast_srgb_to_linear then fast_linear_to_srgb at a mid value and 1.0.
    let queries = vec![
        SrgbTransferQuery::new(2, 0.5),
        SrgbTransferQuery::new(3, 0.5),
        SrgbTransferQuery::new(2, 1.0),
        SrgbTransferQuery::new(3, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "fast[{i}] should be valid");
        assert_parity(res, q, &format!("fast[{i}]"));
    }
    // pow(1.0, k) == 1.0 for both gamma directions.
    assert!(close(out[2].value, 1.0));
    assert!(close(out[3].value, 1.0));
}

#[test]
fn hdr_components_extrapolate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    // Values above 1.0 stay well-defined and monotonic in the golden.
    let queries = vec![
        SrgbTransferQuery::new(0, 1.5),
        SrgbTransferQuery::new(1, 1.5),
        SrgbTransferQuery::new(2, 2.0),
        SrgbTransferQuery::new(3, 2.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1u32, "hdr[{i}] should be valid");
        assert_parity(res, q, &format!("hdr[{i}]"));
    }
}

#[test]
fn out_of_range_func_id_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    let queries = vec![
        SrgbTransferQuery::new(4, 0.5),
        SrgbTransferQuery::new(7, 0.25),
        SrgbTransferQuery::new(100, 0.9),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0u32, "oob[{i}] should be invalid");
        assert_eq!(res.value, 0.0, "oob[{i}] value should be zero");
        assert_parity(res, q, &format!("oob[{i}]"));
    }
}

#[test]
fn batch_mixes_known_and_unknown_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    let queries = vec![
        SrgbTransferQuery::new(0, 0.3),
        SrgbTransferQuery::new(5, 0.3),
        SrgbTransferQuery::new(1, 0.6),
        SrgbTransferQuery::new(2, 0.7),
        SrgbTransferQuery::new(3, 0.8),
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
    let gpu = GpuSrgbTransfer::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSrgbTransfer::new(&ctx);
    let mut lcg = Lcg::new(0x5A17_C0DE);
    let mut queries = Vec::with_capacity(512);
    // Margin around each knee so the piecewise branch cannot flip under
    // round-off between the host oracle and the device.
    let margin = 0.01_f32;
    while queries.len() < 512 {
        let func_id = lcg.next_u32() % 4;
        // Draw the component in a non-negative band (so the power branches never
        // see a negative base) plus a sprinkling of HDR values above 1.0.
        let c = if (lcg.next_u32() & 3) == 0 {
            lcg.next_range(1.0, 3.0)
        } else {
            lcg.next_range(0.0, 1.0)
        };
        // Reject samples near either knee regardless of func_id so every id is
        // branch-stable.
        if (c - KNEE_SRGB_TO_LINEAR).abs() < margin || (c - KNEE_LINEAR_TO_SRGB).abs() < margin {
            continue;
        }
        queries.push(SrgbTransferQuery::new(func_id, c));
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
