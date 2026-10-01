//! Real-device **bit-exact** parity for the isolated hair dither-`alpha` twin:
//! [`GpuHairDitherAlpha`] must reproduce the `CPU` golden
//! [`reference_dither_alpha_map`](prism_hair_gpu::dither_alpha::reference_dither_alpha_map)
//! (built on
//! [`dither_threshold`](prism_render_architecture::hair::reactive_mask::dither_threshold)
//! and
//! [`dither_alpha`](prism_render_architecture::hair::reactive_mask::dither_alpha))
//! for a batch of per-pixel sub-pixel coverages, mapping each pixel to a hard
//! draw decision from a deterministic blue-noise threshold field. The suite
//! drives full coverage (always drawn), zero coverage, a half-coverage sweep
//! that must both draw and skip, a higher-coverage-draws-more comparison,
//! non-finite coverages sanitising to the golden decision, a frame rotation
//! changing the pattern, a tile origin near `u32::MAX` exercising the index
//! wrap, the empty no-op, and a large multi-workgroup batch that crosses the
//! 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Unlike every prior twin this kernel is checked **bit-exact**: the threshold
//! is pure 32-bit integer arithmetic (`wrapping` multiply, xor-shift) finalised
//! by an exact `u32`->`f32` divide by `2^32`, and the `alpha` is a hard draw, so
//! `WGSL`'s wrapping unsigned integer path must reproduce Rust's
//! `wrapping_mul` / `wrapping_add` to the bit. Every output is therefore
//! compared by its raw bit pattern ([`f32::to_bits`]) rather than an approximate
//! difference, and each is asserted to be exactly `0.0` or `1.0`. This test file
//! never uses a float `==`/`!=` or `sin`/`cos`; all comparisons go through the
//! bit pattern and all inputs are explicit literals or integer-derived
//! fractions.
//!
//! Provenance: ordered-dither / blue-noise sub-pixel fallback plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::dither_alpha::{reference_dither_alpha_map, GpuHairDitherAlpha};
use prism_hair_gpu::GpuContext;

/// The raw bit patterns of the only two legal dither outputs.
const DRAW_BITS: u32 = 1.0f32.to_bits();
const SKIP_BITS: u32 = 0.0f32.to_bits();

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, coverages: &[f32], base_x: u32, base_y: u32, frame: u32) -> Vec<f32> {
    GpuHairDitherAlpha::new(ctx).eval(ctx, coverages, base_x, base_y, frame)
}

/// Asserts a whole batch matches the `CPU` golden **bit-for-bit** and that every
/// output is exactly `0.0` or `1.0`.
fn assert_batch_exact(got: &[f32], coverages: &[f32], base_x: u32, base_y: u32, frame: u32) {
    let want = reference_dither_alpha_map(coverages, base_x, base_y, frame);
    assert_eq!(
        got.len(),
        want.len(),
        "one decision per pixel (got {}, want {})",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let gb = g.to_bits();
        let wb = w.to_bits();
        assert_eq!(
            gb, wb,
            "pixel {i}: device bits {gb:#010x} must equal golden bits {wb:#010x}"
        );
        assert!(
            gb == DRAW_BITS || gb == SKIP_BITS,
            "pixel {i}: dither alpha must be exactly 0.0 or 1.0, got bits {gb:#010x}"
        );
    }
}

/// Counts how many pixels were drawn, by bit pattern (no float compare).
fn draw_count(alpha: &[f32]) -> usize {
    alpha.iter().filter(|v| v.to_bits() == DRAW_BITS).count()
}

#[test]
fn full_coverage_always_draws() {
    let Some(ctx) = context_or_skip("full_coverage_always_draws") else {
        return;
    };
    // coverage == 1.0 is at or above every threshold in [0, 1), so every pixel
    // draws regardless of the threshold field.
    let coverages = [1.0f32; 48];
    let got = run(&ctx, &coverages, 10, 20, 0);
    assert_batch_exact(&got, &coverages, 10, 20, 0);
    assert_eq!(draw_count(&got), coverages.len(), "full coverage draws all");
}

#[test]
fn zero_coverage_matches_golden() {
    let Some(ctx) = context_or_skip("zero_coverage_matches_golden") else {
        return;
    };
    // coverage == 0.0 only draws where the threshold is exactly 0.0 (coverage >=
    // threshold). The device must agree with the golden pixel for pixel, whatever
    // that count is.
    let coverages = [0.0f32; 48];
    let got = run(&ctx, &coverages, 7, 3, 5);
    assert_batch_exact(&got, &coverages, 7, 3, 5);
}

#[test]
fn half_coverage_dithers() {
    let Some(ctx) = context_or_skip("half_coverage_dithers") else {
        return;
    };
    // A full 64-pixel scanline at coverage 0.5 must straddle the threshold field:
    // some pixels draw, some skip. A constant decision would mean the threshold
    // field is degenerate, so assert both outcomes appear.
    let coverages = [0.5f32; 64];
    let got = run(&ctx, &coverages, 0, 0, 0);
    assert_batch_exact(&got, &coverages, 0, 0, 0);
    let drawn = draw_count(&got);
    assert!(
        drawn > 0 && drawn < coverages.len(),
        "half coverage must both draw and skip, got {drawn}/{} drawn",
        coverages.len()
    );
}

#[test]
fn higher_coverage_draws_more() {
    let Some(ctx) = context_or_skip("higher_coverage_draws_more") else {
        return;
    };
    // Over the identical threshold field, a higher constant coverage can only draw
    // a superset of pixels (coverage >= threshold is monotone in coverage), so the
    // 0.7 sweep draws at least as many pixels as the 0.3 sweep.
    let low = [0.3f32; 128];
    let high = [0.7f32; 128];
    let got_low = run(&ctx, &low, 500, 11, 2);
    let got_high = run(&ctx, &high, 500, 11, 2);
    assert_batch_exact(&got_low, &low, 500, 11, 2);
    assert_batch_exact(&got_high, &high, 500, 11, 2);
    assert!(
        draw_count(&got_high) >= draw_count(&got_low),
        "higher coverage must draw at least as many: {} vs {}",
        draw_count(&got_high),
        draw_count(&got_low)
    );
    // The two sweeps should not be identical over 128 pixels, confirming the
    // threshold field actually discriminates between 0.3 and 0.7.
    assert!(
        draw_count(&got_high) > draw_count(&got_low),
        "a 0.3->0.7 jump should newly draw some pixels"
    );
}

#[test]
fn non_finite_coverage_sanitizes() {
    let Some(ctx) = context_or_skip("non_finite_coverage_sanitizes") else {
        return;
    };
    // NaN/+inf/-inf and an out-of-range coverage each sanitise exactly like the
    // golden (non-finite -> 0, finite -> clamp [0, 1]) and still yield a valid
    // 0.0/1.0 decision matching the golden bit for bit.
    let coverages = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.5,
        2.0,
        0.5,
        1.0,
        0.0,
    ];
    let got = run(&ctx, &coverages, 42, 99, 7);
    assert_batch_exact(&got, &coverages, 42, 99, 7);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "decisions must stay finite for non-finite coverages, got {got:?}"
    );
}

#[test]
fn frame_rotation_changes_pattern() {
    let Some(ctx) = context_or_skip("frame_rotation_changes_pattern") else {
        return;
    };
    // `frame` rotates the threshold sequence in time, so a mid-coverage sweep must
    // produce a different draw pattern on frame 0 versus frame 1 (while each still
    // matches its own golden exactly).
    let coverages = [0.5f32; 96];
    let got0 = run(&ctx, &coverages, 123, 45, 0);
    let got1 = run(&ctx, &coverages, 123, 45, 1);
    assert_batch_exact(&got0, &coverages, 123, 45, 0);
    assert_batch_exact(&got1, &coverages, 123, 45, 1);
    let differs = got0
        .iter()
        .zip(got1.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differs,
        "the dither pattern must change between frames 0 and 1"
    );
}

#[test]
fn large_base_x_wraps() {
    let Some(ctx) = context_or_skip("large_base_x_wraps") else {
        return;
    };
    // base_x near u32::MAX forces the per-pixel index `base_x + i` to wrap through
    // zero; WGSL unsigned addition must wrap exactly like the golden's
    // wrapping_add, so the device still matches the golden bit for bit.
    let coverages = [0.5f32; 8];
    let base_x = u32::MAX - 1;
    let got = run(&ctx, &coverages, base_x, 17, 3);
    assert_batch_exact(&got, &coverages, base_x, 17, 3);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[], 0, 0, 0);
    assert!(got.is_empty(), "empty batch yields no decisions");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 pixels span three 64-wide workgroups over a deterministic sweep of
    // coverage derived from integer slots (with every 13th slot forced non-finite
    // to exercise the sanitizer across the dispatch boundary). The decision must
    // match the golden bit for bit through the boundary.
    let mut coverages = Vec::new();
    for k in 0u32..130 {
        if k % 13 == 0 {
            coverages.push(f32::NAN);
        } else {
            coverages.push((k % 101) as f32 / 100.0);
        }
    }
    let got = run(&ctx, &coverages, 1000, 7, 9);
    assert_batch_exact(&got, &coverages, 1000, 7, 9);
}
