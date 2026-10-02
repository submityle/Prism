//! Real-device parity for the `YCbCr` colour-transform twin:
//! [`GpuYcbcrBt709`](prism_volumetric_gpu::ycbcr_bt709::GpuYcbcrBt709) must
//! reproduce the `CPU` golden
//! [`ycbcr_bt709`](prism_render_architecture::particle::ycbcr_bt709) across the
//! forward `RGB` -> `YCbCr` matrix, the inverse `YCbCr` -> `RGB` matrix, the
//! `8`-bit quantiser and the `8`-bit dequantiser, under both `luma`-weight
//! standards and both signal ranges, plus a randomized batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The matrix transforms and the dequantiser thread through multiplies, adds
//! and guarded divisions, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous channel.
//! The quantiser emits integer codewords, compared with an exact `==`.
//!
//! # Conditioning
//!
//! Every quantiser fixture is kept clear of the half-codeword rounding boundary
//! by rejection sampling: a sample is accepted only when every `digital + 0.5`
//! is comfortably away from an integer, so a fused multiply-add in
//! `chroma * 255 + 128` cannot tip a tie to a different byte on one device. The
//! matrix transforms and the dequantiser have no classification to straddle, so
//! their fixtures are drawn freely.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::ycbcr_bt709::{Coefficients, Range, Rgb, YCbCr};
use prism_volumetric_gpu::ycbcr_bt709::{golden, GpuYcbcrBt709, YcbcrOp, YcbcrQuery, YcbcrResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// The two `luma`-weight standards exercised by every structural test.
const COEFFS: [Coefficients; 2] = [Coefficients::Bt601, Coefficients::Bt709];

/// The two signal ranges exercised by every structural test.
const RANGES: [Range; 2] = [Range::Full, Range::Limited];

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random linear-`RGB` triple with each channel in `[0, 1)`.
fn rand_rgb(state: &mut u64) -> Rgb {
    Rgb::new(lcg(state), lcg(state), lcg(state))
}

/// A pseudo-random `YCbCr` triple in a range-appropriate band: full swing
/// centres `chroma` on zero, limited swing keeps every channel positive.
fn rand_ycbcr(state: &mut u64, range: Range) -> YCbCr {
    match range {
        Range::Full => YCbCr::new(lcg(state), signed(state, 0.5), signed(state, 0.5)),
        Range::Limited => YCbCr::new(
            ranged(state, 0.1, 0.9),
            ranged(state, 0.2, 0.8),
            ranged(state, 0.2, 0.8),
        ),
    }
}

/// Three pseudo-random `8`-bit codewords in `[0, 255]`.
fn rand_codes(state: &mut u64) -> [u8; 3] {
    [byte(state), byte(state), byte(state)]
}

/// A single pseudo-random codeword in `[0, 255]`.
fn byte(state: &mut u64) -> u8 {
    let v = (lcg(state) * 256.0) as u32;
    u8::try_from(v.min(255)).unwrap_or(255)
}

/// Returns whether a digital-scale value is clear of the half-codeword rounding
/// boundary: `digital + 0.5` must be at least `0.05` away from an integer, so
/// `floor(digital + 0.5)` lands on the same byte regardless of a few units in
/// the last place of fused-multiply-add slack. `floor` is not transcendental.
fn stable_byte(digital: f32) -> bool {
    let x = digital + 0.5;
    let frac = x - x.floor();
    frac.min(1.0 - frac) >= 0.05
}

/// Draws a quantiser query by rejection sampling so every channel's rounding is
/// stable across devices (see [`stable_byte`]).
fn rand_quantize(state: &mut u64, range: Range) -> YcbcrQuery {
    loop {
        let y = lcg(state);
        let (cb, cr) = match range {
            Range::Full => (signed(state, 0.5), signed(state, 0.5)),
            Range::Limited => (lcg(state), lcg(state)),
        };
        let (d0, d1, d2) = match range {
            Range::Full => (y * 255.0, cb * 255.0 + 128.0, cr * 255.0 + 128.0),
            Range::Limited => (y * 255.0, cb * 255.0, cr * 255.0),
        };
        if stable_byte(d0) && stable_byte(d1) && stable_byte(d2) {
            return YcbcrQuery::quantize(YCbCr::new(y, cb, cr), range);
        }
    }
}

/// Draws a random query of a random routine under a random standard and range.
fn rand_query(state: &mut u64) -> YcbcrQuery {
    let coeff = COEFFS[(lcg(state) * 2.0) as usize % 2];
    let range = RANGES[(lcg(state) * 2.0) as usize % 2];
    match (lcg(state) * 4.0) as u32 {
        0 => YcbcrQuery::forward(rand_rgb(state), coeff, range),
        1 => YcbcrQuery::inverse(rand_ycbcr(state, range), coeff, range),
        2 => rand_quantize(state, range),
        _ => YcbcrQuery::dequantize(rand_codes(state), range),
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the quantiser is
/// compared byte-exact, every other routine channel within the documented
/// tolerance.
fn pin(idx: usize, query: &YcbcrQuery, got: &YcbcrResult) {
    let want = golden(query);
    match query.op {
        YcbcrOp::Quantize => assert_eq!(
            got.bytes, want.bytes,
            "query {idx} bytes: gpu {:?} vs cpu {:?}",
            got.bytes, want.bytes
        ),
        _ => {
            for ch in 0..3 {
                assert!(
                    close(got.values[ch], want.values[ch]),
                    "query {idx} channel {ch}: gpu {} vs cpu {}",
                    got.values[ch],
                    want.values[ch]
                );
            }
        }
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuYcbcrBt709, queries: &[YcbcrQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn forward_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    // Primaries, greys and off-axis colours across both standards and ranges;
    // the forward matrix is pinned channel-for-channel within tolerance.
    let samples = [
        Rgb::new(0.0, 0.0, 0.0),
        Rgb::new(1.0, 1.0, 1.0),
        Rgb::new(0.5, 0.5, 0.5),
        Rgb::new(1.0, 0.0, 0.0),
        Rgb::new(0.0, 1.0, 0.0),
        Rgb::new(0.0, 0.0, 1.0),
        Rgb::new(0.2, 0.4, 0.6),
        Rgb::new(0.9, 0.1, 0.3),
    ];
    let mut queries = Vec::new();
    for &coeff in &COEFFS {
        for &range in &RANGES {
            for &rgb in &samples {
                queries.push(YcbcrQuery::forward(rgb, coeff, range));
            }
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn inverse_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    // The inverse matrix is pinned on YCbCr samples produced by the forward
    // transform, so each round trip is a realistic in-range colour.
    let rgbs = [
        Rgb::new(0.0, 0.0, 0.0),
        Rgb::new(1.0, 1.0, 1.0),
        Rgb::new(0.2, 0.4, 0.6),
        Rgb::new(0.9, 0.1, 0.3),
        Rgb::new(0.05, 0.95, 0.5),
        Rgb::new(0.7, 0.7, 0.2),
    ];
    let mut queries = Vec::new();
    for &coeff in &COEFFS {
        for &range in &RANGES {
            for &rgb in &rgbs {
                let yc = golden(&YcbcrQuery::forward(rgb, coeff, range));
                let sample = YCbCr::new(yc.values[0], yc.values[1], yc.values[2]);
                queries.push(YcbcrQuery::inverse(sample, coeff, range));
            }
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn quantize_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    // Many boundary-clear quantiser samples per range, compared byte-exact.
    let mut state = 0x5157_1a2b_3c4d_5e6f_u64;
    let mut queries = Vec::new();
    for &range in &RANGES {
        for _ in 0..64 {
            queries.push(rand_quantize(&mut state, range));
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn dequantize_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    // Codeword corners and midpoints across both ranges; the dequantiser is a
    // pure divide, pinned within tolerance.
    let codes = [
        [0u8, 0, 0],
        [255, 255, 255],
        [16, 128, 128],
        [235, 240, 16],
        [128, 64, 192],
        [73, 201, 37],
    ];
    let mut queries = Vec::new();
    for &range in &RANGES {
        for &c in &codes {
            queries.push(YcbcrQuery::dequantize(c, range));
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries of every
    // routine, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        YcbcrQuery::forward(Rgb::new(0.2, 0.4, 0.6), Coefficients::Bt709, Range::Full),
        YcbcrQuery::inverse(YCbCr::new(0.5, 0.1, -0.2), Coefficients::Bt601, Range::Full),
        YcbcrQuery::dequantize([73, 201, 37], Range::Limited),
    ];
    for _ in 0..61 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuYcbcrBt709::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every routine across many
    // random colours.
    let queries: Vec<YcbcrQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
