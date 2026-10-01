//! Real-device parity for the isolated kept-strand-count twin:
//! [`GpuHairKeptStrandCount`] must reproduce the `CPU` golden
//! [`reference_kept_strand_count`](prism_hair_gpu::kept_strand_count::reference_kept_strand_count)
//! (which forwards to
//! [`kept_strand_count`](prism_render_architecture::hair::cluster::kept_strand_count))
//! for a batch of `(total, ratio)` pairs.
//!
//! # Parity criterion
//!
//! The map is integer-exact for member counts in the f32-exact integer range
//! (strand counts, well under `2^24`): the ratio clamp, the single multiply and
//! the hand-written round-half-away-from-zero (`floor(scaled + 0.5)`, which
//! matches Rust's [`f32::round`] rather than WGSL's round-to-even built-in) all
//! agree bit-for-bit, so counts are asserted with integer equality (not a
//! tolerance). Every count must also stay inside `[0, total]`.
//!
//! The suite drives several ratios across several member counts, the exact
//! half-ratio cases that distinguish away-from-zero rounding from round-to-even
//! (`5 * 0.5 = 2.5 -> 3`, `9 * 0.5 = 4.5 -> 5`), the saturating `0`/`1` ratios,
//! non-finite and out-of-range ratios that sanitise, a zero member count, the
//! empty no-op batch, and a large multi-workgroup batch crossing the 64-wide
//! dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: screen-footprint strand decimation member count plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::kept_strand_count::{reference_kept_strand_count, GpuHairKeptStrandCount};
use prism_hair_gpu::GpuContext;

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

/// Dispatches one batch of `(total, ratio)` pairs through the device twin.
fn run(ctx: &GpuContext, totals: &[u32], ratios: &[f32]) -> Vec<u32> {
    GpuHairKeptStrandCount::new(ctx).eval(ctx, totals, ratios)
}

/// Asserts a whole batch matches the `CPU` golden with integer equality and
/// stays inside the `[0, total]` invariant.
fn assert_batch_exact(got: &[u32], totals: &[u32], ratios: &[f32]) {
    assert_eq!(
        got.len(),
        totals.len(),
        "one kept count per pair (got {}, want {})",
        got.len(),
        totals.len()
    );
    for (i, (&g, (&total, &ratio))) in got.iter().zip(totals.iter().zip(ratios.iter())).enumerate()
    {
        let want = reference_kept_strand_count(total, ratio);
        assert_eq!(
            g, want,
            "pair {i} (total {total}, ratio {ratio}): device count {g} must match golden {want}"
        );
        assert!(
            g <= total,
            "pair {i} (total {total}): kept count {g} must not exceed the member count"
        );
    }
}

#[test]
fn ratios_across_member_counts() {
    let Some(ctx) = context_or_skip("ratios_across_member_counts") else {
        return;
    };
    let totals = [100u32, 100, 100, 100, 64, 37, 1000, 7];
    let ratios = [0.05f32, 0.25, 0.5, 0.75, 0.1, 0.3, 0.123, 0.4];
    let got = run(&ctx, &totals, &ratios);
    assert_batch_exact(&got, &totals, &ratios);
}

#[test]
fn exact_half_rounds_away_from_zero() {
    let Some(ctx) = context_or_skip("exact_half_rounds_away_from_zero") else {
        return;
    };
    // 5 * 0.5 = 2.5 -> 3 and 9 * 0.5 = 4.5 -> 5 under round-half-away-from-zero;
    // WGSL's round-to-even built-in would give 2 and 4, so these cases prove the
    // hand-written rounding matches the Rust golden.
    let totals = [5u32, 9, 3, 7, 11];
    let ratios = [0.5f32, 0.5, 0.5, 0.5, 0.5];
    let got = run(&ctx, &totals, &ratios);
    assert_batch_exact(&got, &totals, &ratios);
    assert_eq!(got[0], 3, "5 * 0.5 rounds to 3 (away from zero)");
    assert_eq!(got[1], 5, "9 * 0.5 rounds to 5 (away from zero)");
}

#[test]
fn saturating_ratios() {
    let Some(ctx) = context_or_skip("saturating_ratios") else {
        return;
    };
    // Ratio 0 keeps none; ratio 1 keeps all; above 1 clamps to all.
    let totals = [250u32, 250, 250, 0];
    let ratios = [0.0f32, 1.0, 4.0, 1.0];
    let got = run(&ctx, &totals, &ratios);
    assert_batch_exact(&got, &totals, &ratios);
    assert_eq!(got[0], 0, "ratio 0 keeps no strands");
    assert_eq!(got[1], 250, "ratio 1 keeps every strand");
    assert_eq!(got[2], 250, "an above-1 ratio clamps to every strand");
}

#[test]
fn non_finite_and_negative_ratios_sanitise() {
    let Some(ctx) = context_or_skip("non_finite_and_negative_ratios_sanitise") else {
        return;
    };
    // NaN / +inf / -inf collapse to 0 kept; a negative ratio clamps to 0 kept.
    let totals = [500u32, 500, 500, 500];
    let ratios = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25];
    let got = run(&ctx, &totals, &ratios);
    assert_batch_exact(&got, &totals, &ratios);
    for &g in &got {
        assert_eq!(g, 0, "a non-finite or negative ratio keeps no strands");
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[], &[]);
    assert!(got.is_empty(), "an empty batch yields no kept counts");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 pairs > 3 full 64-wide workgroups: every pair index must map to its
    // own count independent of the dispatch tiling.
    let mut totals = Vec::with_capacity(200);
    let mut ratios = Vec::with_capacity(200);
    for i in 0..200u32 {
        totals.push(i % 97 + 1);
        ratios.push((i as f32) / 199.0);
    }
    let got = run(&ctx, &totals, &ratios);
    assert_batch_exact(&got, &totals, &ratios);
}
