//! Real-device parity tests for the novel attribute-compression codec twin
//! ([`compression`](prism_volumetric_gpu::compression)).
//!
//! Each test uploads a batch of [`CompressionQuery`] values, runs the single
//! op-dispatch `solve` kernel on a real `wgpu` device, and compares every lane
//! against the `CPU` golden [`cpu_reference`], which delegates straight to the
//! published routines of
//! [`compression`](prism_render_architecture::particle::compression). The
//! fixtures cover: octahedral encode across the `pz >= 0` and folded `pz < 0`
//! hemispheres; octahedral decode including the `z < 0` re-fold; the `snorm16`
//! oct packing (exact integer codes); the `snorm16` oct unpacking; relative
//! quantization and dequantization across several bit widths including the
//! `bits >= 32` and degenerate-range edges; a mixed-variant batch that pins
//! input ordering; and a large pseudo-random batch over the tolerance-safe
//! operations compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous octahedral and dequantized outputs are fixed closed-form
//! algebra: `CPU` and `GPU` evaluate the same expression but may differ by a few
//! units in the last place because a `GPU` can fuse a multiply-add the scalar
//! reference leaves separate. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on those values. The integer `snorm16` codes and
//! quantizer codes must match exactly; their fixtures stay a half-step clear of
//! the rounding boundary so no legal `ULP` slack can flip the rounded integer,
//! and the `bits = 32` quantizer fixtures keep the code well below `2^31` so the
//! `f32`-to-`u32` conversion never saturates.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::compression`
//! octahedral 与 relative-quantization 闭式编解码；纯整数/无超越数学，无需外部
//! 数学库，无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::compression::{
    cpu_reference, CompressionQuery, CompressionResult, GpuCompression,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on continuous values. A `GPU` may fuse a multiply-add
/// the scalar reference leaves separate, perturbing the low mantissa bits by a
/// few units in the last place; `1e-4` admits that legal slack while still
/// failing a genuinely wrong port.
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

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden: the result variant must match, continuous values must agree within
/// tolerance, and integer codes must match exactly.
fn assert_lane(lane: usize, got: &CompressionResult, want: &CompressionResult) {
    match (got, want) {
        (
            CompressionResult::OctPair { x: xg, y: yg },
            CompressionResult::OctPair { x: xc, y: yc },
        ) => {
            assert!(close(*xg, *xc), "lane {lane}: oct.x gpu {xg} vs cpu {xc}");
            assert!(close(*yg, *yc), "lane {lane}: oct.y gpu {yg} vs cpu {yc}");
        }
        (CompressionResult::Vector { v: vg }, CompressionResult::Vector { v: vc }) => {
            for ch in 0..3 {
                assert!(
                    close(vg[ch], vc[ch]),
                    "lane {lane}: vector[{ch}] gpu {vg:?} vs cpu {vc:?}"
                );
            }
        }
        (
            CompressionResult::Snorm16Pair { codes: cg },
            CompressionResult::Snorm16Pair { codes: cc },
        ) => {
            assert_eq!(
                cg, cc,
                "lane {lane}: snorm16 codes gpu {cg:?} vs cpu {cc:?}"
            );
        }
        (CompressionResult::Code { code: cg }, CompressionResult::Code { code: cc }) => {
            assert_eq!(cg, cc, "lane {lane}: quantized code gpu {cg} vs cpu {cc}");
        }
        (CompressionResult::Scalar { value: sg }, CompressionResult::Scalar { value: sc }) => {
            assert!(close(*sg, *sc), "lane {lane}: scalar gpu {sg} vs cpu {sc}");
        }
        _ => panic!("lane {lane}: result variant mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs the `GPU` dispatch and asserts strict parity against the `CPU` golden
/// for every lane; returns the `GPU` verdicts for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuCompression,
    queries: &[CompressionQuery],
) -> Vec<CompressionResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_lane(lane, g, &cpu_reference(q));
    }
    got
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

/// A pseudo-random `f32` in `[lo, hi)`, derived from the integer `lcg`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

/// A pseudo-random integer in `[lo, hi]`, derived from the integer `lcg` bits.
fn ranged_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let span = (hi - lo + 1) as u64;
    lo + ((*state >> 33) % span) as i32
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn oct_encode_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let queries = [
        // Cardinal axes (upper hemisphere, pz >= 0).
        CompressionQuery::OctEncode {
            direction: [1.0, 0.0, 0.0],
        },
        CompressionQuery::OctEncode {
            direction: [0.0, 1.0, 0.0],
        },
        CompressionQuery::OctEncode {
            direction: [0.0, 0.0, 1.0],
        },
        // Lower hemisphere (pz < 0) exercises the fold branch.
        CompressionQuery::OctEncode {
            direction: [0.0, 0.0, -1.0],
        },
        CompressionQuery::OctEncode {
            direction: [1.0, 1.0, 1.0],
        },
        CompressionQuery::OctEncode {
            direction: [-1.0, 2.0, 3.0],
        },
        CompressionQuery::OctEncode {
            direction: [0.5, -0.5, 0.7],
        },
        CompressionQuery::OctEncode {
            direction: [2.0, -3.0, -1.0],
        },
        CompressionQuery::OctEncode {
            direction: [-1.0, -1.0, -0.2],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn oct_decode_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let queries = [
        CompressionQuery::OctDecode {
            encoded: [0.0, 0.0],
        },
        CompressionQuery::OctDecode {
            encoded: [0.3, 0.4],
        },
        // z = 1 - 1.2 < 0 triggers the re-fold.
        CompressionQuery::OctDecode {
            encoded: [0.6, 0.6],
        },
        CompressionQuery::OctDecode {
            encoded: [-0.5, 0.2],
        },
        CompressionQuery::OctDecode {
            encoded: [0.9, -0.05],
        },
        // z < 0 fold with both components negative.
        CompressionQuery::OctDecode {
            encoded: [-0.7, -0.7],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn oct_encode_snorm16_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    // Directions whose exact octahedral components times 32767 land a clear
    // half-step away from a rounding boundary, so the integer codes are exact.
    let queries = [
        CompressionQuery::OctEncodeSnorm16 {
            direction: [0.0, 0.0, 1.0],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [1.0, 0.0, 0.0],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [0.0, 1.0, 0.0],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [1.0, 1.0, 1.0],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [2.0, 1.0, 2.0],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [0.5, 0.3, -0.4],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn oct_decode_snorm16_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let queries = [
        CompressionQuery::OctDecodeSnorm16 { code: [0, 0] },
        CompressionQuery::OctDecodeSnorm16 {
            code: [16384, -8192],
        },
        CompressionQuery::OctDecodeSnorm16 { code: [32767, 0] },
        CompressionQuery::OctDecodeSnorm16 {
            code: [-20000, 12000],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn quantize_relative_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    // Values chosen so `t * levels` is a clear half-step from a rounding
    // boundary, so the integer code is exact.
    let queries = [
        CompressionQuery::QuantizeRelative {
            value: 0.2,
            min: 0.0,
            max: 1.0,
            bits: 8,
        },
        CompressionQuery::QuantizeRelative {
            value: 0.4,
            min: 0.0,
            max: 1.0,
            bits: 10,
        },
        CompressionQuery::QuantizeRelative {
            value: 0.5,
            min: -2.0,
            max: 2.0,
            bits: 12,
        },
        CompressionQuery::QuantizeRelative {
            value: 0.123,
            min: 0.0,
            max: 1.0,
            bits: 16,
        },
        CompressionQuery::QuantizeRelative {
            value: 0.4,
            min: 0.0,
            max: 1.0,
            bits: 4,
        },
        // `bits = 32` takes the `u32::MAX` level-count branch; the small `t`
        // keeps the code well below `2^31` so no `f32`-to-`u32` saturation.
        CompressionQuery::QuantizeRelative {
            value: 0.0001,
            min: 0.0,
            max: 1.0,
            bits: 32,
        },
        // Degenerate zero-width quantizer returns code 0.
        CompressionQuery::QuantizeRelative {
            value: 0.5,
            min: 0.0,
            max: 1.0,
            bits: 0,
        },
        // Degenerate range (max <= min) returns code 0.
        CompressionQuery::QuantizeRelative {
            value: 0.5,
            min: 5.0,
            max: 3.0,
            bits: 8,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn dequantize_relative_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let queries = [
        CompressionQuery::DequantizeRelative {
            code: 51,
            min: 0.0,
            max: 1.0,
            bits: 8,
        },
        CompressionQuery::DequantizeRelative {
            code: 409,
            min: 0.0,
            max: 1.0,
            bits: 10,
        },
        CompressionQuery::DequantizeRelative {
            code: 2559,
            min: -2.0,
            max: 2.0,
            bits: 12,
        },
        // `bits = 0` returns the range minimum.
        CompressionQuery::DequantizeRelative {
            code: 100,
            min: 7.0,
            max: 9.0,
            bits: 0,
        },
        // Code beyond the level count clamps to the maximum.
        CompressionQuery::DequantizeRelative {
            code: 100_000,
            min: 0.0,
            max: 1.0,
            bits: 16,
        },
        CompressionQuery::DequantizeRelative {
            code: 429_497,
            min: 0.0,
            max: 1.0,
            bits: 32,
        },
        // `bits = 40` clamps to 32 bits (same `u32::MAX` level count).
        CompressionQuery::DequantizeRelative {
            code: 1000,
            min: -1.0,
            max: 1.0,
            bits: 40,
        },
        // Degenerate range yields the minimum for every code.
        CompressionQuery::DequantizeRelative {
            code: 100,
            min: 2.0,
            max: 2.0,
            bits: 8,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_variant_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    // One of each operation, interleaved, to pin the per-lane input ordering.
    let queries = [
        CompressionQuery::OctEncode {
            direction: [1.0, 1.0, 1.0],
        },
        CompressionQuery::QuantizeRelative {
            value: 0.2,
            min: 0.0,
            max: 1.0,
            bits: 8,
        },
        CompressionQuery::OctDecode {
            encoded: [0.3, 0.4],
        },
        CompressionQuery::OctEncodeSnorm16 {
            direction: [2.0, 1.0, 2.0],
        },
        CompressionQuery::DequantizeRelative {
            code: 51,
            min: 0.0,
            max: 1.0,
            bits: 8,
        },
        CompressionQuery::OctDecodeSnorm16 {
            code: [16384, -8192],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompression::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_0001;
    let mut queries = Vec::new();
    // Only the tolerance-safe operations are randomized: the exact-integer
    // `snorm16` and quantize codes are pinned by their deterministic fixtures to
    // keep every random lane a half-step clear of a rounding boundary.
    for _ in 0..160 {
        match ranged_int(&mut state, 0, 3) {
            0 => {
                // A direction with a guaranteed nonzero L1 length.
                let sign = if ranged(&mut state, -1.0, 1.0) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                queries.push(CompressionQuery::OctEncode {
                    direction: [
                        sign * ranged(&mut state, 0.6, 2.0),
                        ranged(&mut state, -2.0, 2.0),
                        ranged(&mut state, -2.0, 2.0),
                    ],
                });
            }
            1 => {
                queries.push(CompressionQuery::OctDecode {
                    encoded: [
                        ranged(&mut state, -0.95, 0.95),
                        ranged(&mut state, -0.95, 0.95),
                    ],
                });
            }
            2 => {
                queries.push(CompressionQuery::OctDecodeSnorm16 {
                    code: [
                        ranged_int(&mut state, -32767, 32767),
                        ranged_int(&mut state, -32767, 32767),
                    ],
                });
            }
            _ => {
                let min = ranged(&mut state, -5.0, -1.0);
                let max = ranged(&mut state, 1.0, 5.0);
                let bits = ranged_int(&mut state, 4, 20) as u32;
                let code = ranged_int(&mut state, 0, 1_000_000) as u32;
                queries.push(CompressionQuery::DequantizeRelative {
                    code,
                    min,
                    max,
                    bits,
                });
            }
        }
    }
    check(&ctx, &gpu, &queries);
}
