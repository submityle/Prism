//! Real-device parity for the motion-vector quantization twin:
//! [`GpuMotionVectorQuantize`](prism_volumetric_gpu::motion_vector_quantize::GpuMotionVectorQuantize)
//! must reproduce the numeric core of the `CPU` golden
//! [`encode`](prism_render_architecture::motion::encode) scalar codecs — the
//! `snorm16` and `unorm8` encode/decode pair and the
//! [`VelocityEncoding`](prism_render_architecture::motion::encode::VelocityEncoding)
//! velocity round-trip and quantization step — across saturation endpoints,
//! zero, out-of-range clamp, half-step-safe interior values, and a randomized
//! batch mixing all seven operations compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden functions
//! [`encode_snorm16`](prism_render_architecture::motion::encode::encode_snorm16),
//! [`decode_snorm16`](prism_render_architecture::motion::encode::decode_snorm16),
//! [`encode_unorm8`](prism_render_architecture::motion::encode::encode_unorm8),
//! and
//! [`decode_unorm8`](prism_render_architecture::motion::encode::decode_unorm8),
//! together with
//! [`VelocityEncoding`](prism_render_architecture::motion::encode::VelocityEncoding),
//! are `pub`, so each `GPU` result is pinned directly against the golden run on
//! the same input. Every velocity fixture carries a `max_velocity_pixels` that
//! has already passed through
//! [`VelocityEncoding::new`](prism_render_architecture::motion::encode::VelocityEncoding::new),
//! so the host-sanitized full-scale fed to the device matches the oracle's.
//!
//! # Parity criterion
//!
//! The encoded `snorm16` / `unorm8` codes are integers built from a clamp, a
//! multiply, a half-step bias, and a truncating cast, so for fixtures clear of a
//! half-step tie the `CPU` and `GPU` land on the same integer and the codes are
//! asserted with exact `==`. The decoded values thread through a divide and a
//! clamp only (no `sqrt`, no transcendental), so they are asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Each encode fixture keeps its scaled value away from a half-step tie (where
//! the `CPU` `as`-truncate and the `GPU` `i32`-truncate could straddle different
//! integers once a divide disagrees by a unit in the last place); the random
//! sweep rejects any interior encode input within `0.05` of a tie. Saturated and
//! zero inputs land exactly on an integer and so are tie-free by construction.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::encode`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::encode::{
    decode_snorm16, decode_unorm8, encode_snorm16, encode_unorm8, VelocityEncoding,
};
use prism_render_architecture::motion::Vec2;
use prism_volumetric_gpu::motion_vector_quantize::{
    GpuMotionVectorQuantize, MotionVectorQuantizeQuery, MotionVectorQuantizeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a decoded value.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Full-scale of a signed 16-bit channel, mirroring the golden `SNORM16_SCALE`.
const SNORM16_SCALE: f32 = 32767.0;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &MotionVectorQuantizeQuery) -> MotionVectorQuantizeResult {
    match *q {
        MotionVectorQuantizeQuery::EncodeSnorm16 { normalized } => {
            MotionVectorQuantizeResult::EncodeSnorm16 {
                encoded: encode_snorm16(normalized),
            }
        }
        MotionVectorQuantizeQuery::DecodeSnorm16 { encoded } => {
            MotionVectorQuantizeResult::DecodeSnorm16 {
                normalized: decode_snorm16(encoded),
            }
        }
        MotionVectorQuantizeQuery::EncodeUnorm8 { value } => {
            MotionVectorQuantizeResult::EncodeUnorm8 {
                encoded: encode_unorm8(value),
            }
        }
        MotionVectorQuantizeQuery::DecodeUnorm8 { value } => {
            MotionVectorQuantizeResult::DecodeUnorm8 {
                value: decode_unorm8(value),
            }
        }
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity,
            max_velocity_pixels,
        } => {
            let codec = VelocityEncoding::new(max_velocity_pixels);
            MotionVectorQuantizeResult::VelocityEncode {
                encoded: codec.encode(Vec2::new(velocity[0], velocity[1])),
            }
        }
        MotionVectorQuantizeQuery::VelocityDecode {
            encoded,
            max_velocity_pixels,
        } => {
            let codec = VelocityEncoding::new(max_velocity_pixels);
            let v = codec.decode(encoded);
            MotionVectorQuantizeResult::VelocityDecode {
                velocity: [v.x, v.y],
            }
        }
        MotionVectorQuantizeQuery::QuantizationStep {
            max_velocity_pixels,
        } => MotionVectorQuantizeResult::QuantizationStep {
            step_pixels: VelocityEncoding::new(max_velocity_pixels).quantization_step_pixels(),
        },
    }
}

/// Pins one `GPU` result against the golden oracle: encoded codes exactly,
/// decoded values within tolerance.
fn assert_result(idx: usize, got: &MotionVectorQuantizeResult, want: &MotionVectorQuantizeResult) {
    match (*got, *want) {
        (
            MotionVectorQuantizeResult::EncodeSnorm16 { encoded: g },
            MotionVectorQuantizeResult::EncodeSnorm16 { encoded: w },
        ) => assert_eq!(g, w, "result {idx} snorm16 code: gpu {g} vs cpu {w}"),
        (
            MotionVectorQuantizeResult::DecodeSnorm16 { normalized: g },
            MotionVectorQuantizeResult::DecodeSnorm16 { normalized: w },
        ) => assert!(
            close(g, w),
            "result {idx} snorm16 decode: gpu {g} vs cpu {w}"
        ),
        (
            MotionVectorQuantizeResult::EncodeUnorm8 { encoded: g },
            MotionVectorQuantizeResult::EncodeUnorm8 { encoded: w },
        ) => assert_eq!(g, w, "result {idx} unorm8 code: gpu {g} vs cpu {w}"),
        (
            MotionVectorQuantizeResult::DecodeUnorm8 { value: g },
            MotionVectorQuantizeResult::DecodeUnorm8 { value: w },
        ) => assert!(
            close(g, w),
            "result {idx} unorm8 decode: gpu {g} vs cpu {w}"
        ),
        (
            MotionVectorQuantizeResult::VelocityEncode { encoded: g },
            MotionVectorQuantizeResult::VelocityEncode { encoded: w },
        ) => assert_eq!(g, w, "result {idx} velocity codes: gpu {g:?} vs cpu {w:?}"),
        (
            MotionVectorQuantizeResult::VelocityDecode { velocity: g },
            MotionVectorQuantizeResult::VelocityDecode { velocity: w },
        ) => {
            assert!(
                close(g[0], w[0]) && close(g[1], w[1]),
                "result {idx} velocity decode: gpu {g:?} vs cpu {w:?}"
            );
        }
        (
            MotionVectorQuantizeResult::QuantizationStep { step_pixels: g },
            MotionVectorQuantizeResult::QuantizationStep { step_pixels: w },
        ) => assert!(close(g, w), "result {idx} quant step: gpu {g} vs cpu {w}"),
        _ => panic!("result {idx} variant mismatch: gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[MotionVectorQuantizeQuery]) {
    let gpu = GpuMotionVectorQuantize::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

/// Whether a `snorm16` encode input is within `0.05` of a half-step tie, where
/// the `CPU` and `GPU` truncation could disagree.
fn snorm_tie(value: f32) -> bool {
    let scaled = value.clamp(-1.0, 1.0) * SNORM16_SCALE;
    (scaled.abs().fract() - 0.5).abs() < 0.05
}

/// Whether a `unorm8` encode input is within `0.05` of a half-step tie.
fn unorm_tie(value: f32) -> bool {
    let scaled = value.clamp(0.0, 1.0) * 255.0;
    (scaled.fract() - 0.5).abs() < 0.05
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_vector_quantize parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionVectorQuantize::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn encode_snorm16_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Zero, both saturation endpoints, out-of-range clamp both ways, and three
    // interior values whose scaled magnitude is clear of a half-step tie.
    let queries: Vec<MotionVectorQuantizeQuery> = [0.0, 1.0, -1.0, 5.0, -5.0, 0.25, -0.6, 0.123]
        .into_iter()
        .map(|normalized| MotionVectorQuantizeQuery::EncodeSnorm16 { normalized })
        .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn decode_snorm16_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries: Vec<MotionVectorQuantizeQuery> = [0i16, 32767, -32767, 16384, -8000, 12345]
        .into_iter()
        .map(|encoded| MotionVectorQuantizeQuery::DecodeSnorm16 { encoded })
        .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn encode_unorm8_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Zero, full scale, out-of-range clamp, and three interior values clear of a
    // half-step tie.
    let queries: Vec<MotionVectorQuantizeQuery> = [0.0, 1.0, 2.0, 0.12, 0.33, 0.77]
        .into_iter()
        .map(|value| MotionVectorQuantizeQuery::EncodeUnorm8 { value })
        .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn decode_unorm8_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries: Vec<MotionVectorQuantizeQuery> = [0u8, 255, 128, 64, 200, 17]
        .into_iter()
        .map(|value| MotionVectorQuantizeQuery::DecodeUnorm8 { value })
        .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn velocity_round_trip_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // The full-scale is sanitized once through the golden constructor so the
    // device query carries the same value the oracle normalizes against.
    let max_256 = VelocityEncoding::new(256.0).max_velocity_pixels;
    let max_100 = VelocityEncoding::new(100.0).max_velocity_pixels;
    let queries = vec![
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity: [10.0, -20.0],
            max_velocity_pixels: max_256,
        },
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity: [255.0, -255.0],
            max_velocity_pixels: max_256,
        },
        // Saturating faster-than-full-scale motion maps to the endpoints.
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity: [-256.0, 256.0],
            max_velocity_pixels: max_256,
        },
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity: [1000.0, -1000.0],
            max_velocity_pixels: max_100,
        },
        MotionVectorQuantizeQuery::VelocityDecode {
            encoded: [16000, -8000],
            max_velocity_pixels: max_256,
        },
        MotionVectorQuantizeQuery::VelocityDecode {
            encoded: [32767, -32767],
            max_velocity_pixels: max_100,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn quantization_step_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries: Vec<MotionVectorQuantizeQuery> = [1.0, 100.0, 256.0, 540.0]
        .into_iter()
        .map(|raw| MotionVectorQuantizeQuery::QuantizationStep {
            max_velocity_pixels: VelocityEncoding::new(raw).max_velocity_pixels,
        })
        .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x2f6e_1c84_bb17_90a5_u64;

    let mut queries: Vec<MotionVectorQuantizeQuery> = Vec::new();
    let mut guard = 0u32;
    while queries.len() < 256 && guard < 100_000 {
        guard += 1;
        let op = lcg(&mut state) % 7;
        let q = match op {
            0 => {
                let normalized = ranged(&mut state, -1.5, 1.5);
                if snorm_tie(normalized) {
                    continue;
                }
                MotionVectorQuantizeQuery::EncodeSnorm16 { normalized }
            }
            1 => {
                let encoded = (lcg(&mut state) % 65_535) as i32 - 32_767;
                MotionVectorQuantizeQuery::DecodeSnorm16 {
                    encoded: encoded as i16,
                }
            }
            2 => {
                let value = ranged(&mut state, -0.25, 1.25);
                if unorm_tie(value) {
                    continue;
                }
                MotionVectorQuantizeQuery::EncodeUnorm8 { value }
            }
            3 => {
                let value = (lcg(&mut state) % 256) as u8;
                MotionVectorQuantizeQuery::DecodeUnorm8 { value }
            }
            4 => {
                let max = VelocityEncoding::new(ranged(&mut state, 4.0, 600.0)).max_velocity_pixels;
                let vx = ranged(&mut state, -2.0 * max, 2.0 * max);
                let vy = ranged(&mut state, -2.0 * max, 2.0 * max);
                if snorm_tie(vx / max) || snorm_tie(vy / max) {
                    continue;
                }
                MotionVectorQuantizeQuery::VelocityEncode {
                    velocity: [vx, vy],
                    max_velocity_pixels: max,
                }
            }
            5 => {
                let max = VelocityEncoding::new(ranged(&mut state, 4.0, 600.0)).max_velocity_pixels;
                let ex = (lcg(&mut state) % 65_535) as i32 - 32_767;
                let ey = (lcg(&mut state) % 65_535) as i32 - 32_767;
                MotionVectorQuantizeQuery::VelocityDecode {
                    encoded: [ex as i16, ey as i16],
                    max_velocity_pixels: max,
                }
            }
            _ => {
                let max = VelocityEncoding::new(ranged(&mut state, 4.0, 600.0)).max_velocity_pixels;
                MotionVectorQuantizeQuery::QuantizationStep {
                    max_velocity_pixels: max,
                }
            }
        };
        queries.push(q);
    }
    assert!(
        queries.len() >= 200,
        "rejection sampling yielded too few fixtures"
    );
    run_and_check(&ctx, &queries);
}
