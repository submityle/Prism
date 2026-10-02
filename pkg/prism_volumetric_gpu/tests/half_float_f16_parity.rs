//! Real-device parity for the half-float twin:
//! [`GpuHalfFloatF16`](prism_volumetric_gpu::half_float_f16::GpuHalfFloatF16)
//! must reproduce the `CPU` golden
//! [`particle::half_float_f16`](prism_render_architecture::particle::half_float_f16)
//! in both directions — the `RNE` forward narrowing
//! [`f32_to_f16_bits`](prism_render_architecture::particle::half_float_f16::f32_to_f16_bits)
//! and the reverse widening
//! [`f16_bits_to_f32`](prism_render_architecture::particle::half_float_f16::f16_bits_to_f32)
//! — across curated edge cases, the exhaustive set of all `65536` binary16
//! patterns, and a large random batch of arbitrary `f32` bit patterns.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Both converters are pure integer bit manipulation — no `f32` arithmetic or
//! comparison — so there is no rounding slack and no critical region to avoid.
//! The forward binary16 word and the reverse `f32` bit pattern are therefore
//! asserted bit-identical with exact `==`, covering the `NaN` quiet bit, every
//! subnormal, both signed zeros and the infinities.
//!
//! Provenance: textbook `IEEE` 754 binary16 <-> `f32` bit manipulation; no
//! Unreal Engine source or derived code.

use prism_render_architecture::particle::half_float_f16::{
    f16_bits_to_f32, f16_is_nan, f32_to_f16_bits,
};
use prism_volumetric_gpu::half_float_f16::{GpuHalfFloatF16, HalfFloatQuery, HalfFloatResult};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears in the fixture. Returns the raw stepped state.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A spread of interesting `f32` forward inputs: representable magnitudes,
/// signed zeros, the infinities, the smallest normal and subnormal binary16
/// values, the two round-to-nearest-even tie fixtures, and the overflow and
/// underflow thresholds. Specials are built from raw bits so the exact pattern
/// is uploaded unchanged.
fn forward_values() -> Vec<f32> {
    vec![
        1.0,
        2.0,
        0.5,
        0.25,
        -2.0,
        65504.0,
        -65504.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        // Smallest positive normal binary16 value, 2^-14.
        f32::from_bits(0x3880_0000),
        // Smallest positive subnormal binary16 value, 2^-24.
        f32::from_bits(0x3380_0000),
        // 2^-25: exactly halfway to 2^-24, ties to the even value (zero).
        f32::from_bits(0x3300_0000),
        // Just above 2^-25: rounds up to the smallest subnormal.
        f32::from_bits(0x3300_0001),
        // 2^-26: far below the smallest subnormal, flushes to zero.
        f32::from_bits(0x3280_0000),
        // Finite magnitudes beyond the binary16 range overflow to infinity.
        70000.0,
        -70000.0,
    ]
}

/// A spread of interesting binary16 reverse inputs: both signed zeros, the
/// smallest and largest subnormals, the smallest normal, several finite
/// magnitudes, the max finite value, both infinities and a quiet `NaN`.
fn reverse_halves() -> Vec<u16> {
    vec![
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0x4000, 0x7bff, 0xc000, 0x7c00,
        0xfc00, 0x7e00,
    ]
}

/// Runs the twin over `queries` and asserts both directions are bit-identical
/// to the `CPU` reference for every element, returning the `GPU` results.
fn check(
    ctx: &GpuContext,
    gpu: &GpuHalfFloatF16,
    queries: &[HalfFloatQuery],
) -> Vec<HalfFloatResult> {
    let results = gpu.eval(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (idx, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_eq!(
            r.forward_bits,
            f32_to_f16_bits(q.value),
            "forward narrowing must be bit-identical at element {idx} (value bits {:#010x})",
            q.value.to_bits()
        );
        assert_eq!(
            r.reverse_value.to_bits(),
            f16_bits_to_f32(q.half_bits).to_bits(),
            "reverse widening must be bit-identical at element {idx} (half {:#06x})",
            q.half_bits
        );
    }
    results
}

/// Builds queries covering every forward value and every reverse half,
/// pairing the two lists independently (each element is checked in both
/// directions regardless of its partner).
fn curated_queries() -> Vec<HalfFloatQuery> {
    let values = forward_values();
    let halves = reverse_halves();
    let n = values.len().max(halves.len());
    let mut queries = Vec::with_capacity(n);
    for i in 0..n {
        queries.push(HalfFloatQuery {
            value: values[i % values.len()],
            half_bits: halves[i % halves.len()],
        });
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_curated_edge_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);
    let queries = curated_queries();
    let results = check(&ctx, &gpu, &queries);

    // A degenerate all-zero kernel could not pass: some narrowings and some
    // widenings are non-zero.
    assert!(
        results.iter().any(|r| r.forward_bits != 0),
        "some forward narrowing should be non-zero"
    );
    assert!(
        results.iter().any(|r| r.reverse_value.to_bits() != 0),
        "some reverse widening should be non-zero"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_all_binary16_patterns() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);

    // Exhaustively widen every one of the 65536 binary16 patterns, covering the
    // entire subnormal range, both signed zeros, every normal, the infinities
    // and all NaN payloads. The forward slot carries a benign representable
    // value so it is exercised too.
    let mut queries = Vec::with_capacity(0x1_0000);
    for h in 0u32..=0xffff {
        let half = h as u16;
        queries.push(HalfFloatQuery {
            value: f16_bits_to_f32(half),
            half_bits: half,
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);

    // Several thousand arbitrary f32 bit patterns (including subnormals, NaNs
    // and infinities) narrowed, paired with arbitrary binary16 patterns widened.
    // Because both converters are pure integer work, every pattern — however
    // exotic — is compared bit-exact with no rejection sampling.
    let mut state = 0x1234_5678_9abc_def0u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        let value_bits = (lcg(&mut state) >> 24) as u32;
        let half_bits = (lcg(&mut state) >> 40) as u16;
        queries.push(HalfFloatQuery {
            value: f32::from_bits(value_bits),
            half_bits,
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn forward_nan_stays_quiet_on_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);

    // A canonical quiet NaN and a signalling-looking payload both narrow to a
    // quiet binary16 NaN, matching the reference exactly.
    let queries = vec![
        HalfFloatQuery {
            value: f32::NAN,
            half_bits: 0x7e00,
        },
        HalfFloatQuery {
            value: f32::from_bits(0x7f80_0001),
            half_bits: 0x7c01,
        },
    ];
    let results = check(&ctx, &gpu, &queries);
    for (idx, r) in results.iter().enumerate() {
        assert!(
            f16_is_nan(r.forward_bits),
            "forward NaN must stay a NaN at element {idx}"
        );
        assert_ne!(
            r.forward_bits & 0x0200,
            0,
            "forward NaN must keep the quiet bit set at element {idx}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn representable_halves_round_trip_through_device() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);

    // Widen every non-NaN binary16 pattern on device, then narrow the device's
    // own widened f32 back on device, and assert the pattern survives exactly.
    let mut halves: Vec<u16> = Vec::new();
    for h in 0u32..=0xffff {
        let half = h as u16;
        if !f16_is_nan(half) {
            halves.push(half);
        }
    }

    let widen: Vec<HalfFloatQuery> = halves
        .iter()
        .map(|&half| HalfFloatQuery {
            value: 1.0,
            half_bits: half,
        })
        .collect();
    let widened = check(&ctx, &gpu, &widen);

    let narrow: Vec<HalfFloatQuery> = widened
        .iter()
        .map(|r| HalfFloatQuery {
            value: r.reverse_value,
            half_bits: 0x0000,
        })
        .collect();
    let narrowed = gpu.eval(&ctx, &narrow);
    assert_eq!(narrowed.len(), halves.len());
    for (idx, (&half, r)) in halves.iter().zip(narrowed.iter()).enumerate() {
        assert_eq!(
            r.forward_bits, half,
            "half -> f32 -> half must round trip at element {idx}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_yields_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping half-float parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfFloatF16::new(&ctx);
    let results = gpu.eval(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield an empty result");
}
