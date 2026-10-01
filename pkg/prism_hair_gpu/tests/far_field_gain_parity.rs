//! Real-device parity for the isolated far-field roughness-gain twin:
//! [`GpuHairFarFieldGain`] must reproduce the `CPU` golden
//! [`reference_far_field_gain`](prism_hair_gpu::far_field_gain::reference_far_field_gain)
//! (which forwards to
//! [`far_field_roughness_gain`](prism_render_architecture::hair::scatter_lod::far_field_roughness_gain))
//! for a batch of `(blend, max_gain)` pairs.
//!
//! # Parity criterion
//!
//! The gain is a single `1 + blend * max_gain` multiply-add the `GPU` may fuse
//! where the scalar reference leaves it separate, so each returned gain is
//! compared against a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not by
//! raw bit pattern.
//!
//! The suite drives regular `(blend, max_gain)` pairs, blend saturation at the
//! `0`/`1` ends, negative/non-finite `max_gain` (which sanitise to `0`, gain
//! `1`), non-finite blend (which clamps to `0`, gain `1`), the empty no-op
//! batch, and a large multi-workgroup batch that crosses the 64-wide dispatch
//! boundary. It also asserts the structural invariant: the gain is always
//! `>= 1`.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: `d'Eon` 2011 near-/far-field hair scattering split plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::far_field_gain::{reference_far_field_gain, GpuHairFarFieldGain};
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

/// Dispatches one batch of `(blend, max_gain)` pairs through the device twin.
fn run(ctx: &GpuContext, pairs: &[(f32, f32)]) -> Vec<f32> {
    GpuHairFarFieldGain::new(ctx).eval(ctx, pairs)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a whole batch matches the `CPU` golden within tolerance.
fn assert_batch_close(got: &[f32], pairs: &[(f32, f32)]) {
    assert_eq!(
        got.len(),
        pairs.len(),
        "one gain per pair (got {}, want {})",
        got.len(),
        pairs.len()
    );
    for (i, (&g, &(blend, max_gain))) in got.iter().zip(pairs.iter()).enumerate() {
        let want = reference_far_field_gain(blend, max_gain);
        assert!(
            close(g, want),
            "pair {i} (blend {blend}, max_gain {max_gain}): device {g} must match golden {want}"
        );
    }
}

#[test]
fn regular_pairs_match_golden() {
    let Some(ctx) = context_or_skip("regular_pairs_match_golden") else {
        return;
    };
    // Mix of blends across [0,1] and non-negative widening budgets.
    let pairs = [
        (0.0, 2.0),
        (0.25, 2.0),
        (0.5, 2.0),
        (0.75, 1.5),
        (1.0, 1.5),
        (0.5, 0.0),
        (0.3, 4.0),
    ];
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);

    // Spot-check the known endpoints: blend 0 -> no widening, blend 1 -> full.
    assert!(close(got[0], 1.0), "zero blend -> no widening");
    assert!(close(got[2], 2.0), "half blend, max 2 -> 1 + 0.5*2 = 2");
    assert!(close(got[4], 2.5), "full blend, max 1.5 -> 1 + 1.5 = 2.5");
    assert!(close(got[5], 1.0), "zero max_gain -> no widening");
}

#[test]
fn gain_is_at_least_one() {
    let Some(ctx) = context_or_skip("gain_is_at_least_one") else {
        return;
    };
    let mut pairs = Vec::with_capacity(40);
    for i in 0..40u32 {
        let blend = i as f32 * 0.025;
        let max_gain = (i as f32).mul_add(0.1, 0.5);
        pairs.push((blend, max_gain));
    }
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);
    for g in &got {
        assert!(*g >= 1.0 - 1.0e-4, "gain never widens below 1: {g}");
    }
    // Larger blend at fixed max_gain yields a non-smaller gain.
    for pair in got.windows(2) {
        assert!(
            pair[1] >= pair[0] - 1.0e-4,
            "gain is non-decreasing along this ramp: {} then {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn blend_saturates_at_the_ends() {
    let Some(ctx) = context_or_skip("blend_saturates_at_the_ends") else {
        return;
    };
    // Blends outside [0,1] clamp: below 0 -> gain 1, above 1 -> full widening.
    let pairs = [(-2.0, 3.0), (-0.1, 3.0), (1.1, 3.0), (5.0, 3.0)];
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);
    assert!(close(got[0], 1.0), "blend below 0 clamps -> gain 1");
    assert!(close(got[1], 1.0), "blend below 0 clamps -> gain 1");
    assert!(close(got[2], 4.0), "blend above 1 clamps -> 1 + 3 = 4");
    assert!(close(got[3], 4.0), "blend above 1 clamps -> 1 + 3 = 4");
}

#[test]
fn nonfinite_max_gain_sanitises_to_no_widening() {
    let Some(ctx) = context_or_skip("nonfinite_max_gain_sanitises_to_no_widening") else {
        return;
    };
    // Negative / NaN / +inf / -inf max_gain all collapse to 0 -> gain 1.
    let pairs = [
        (0.7, -1.0),
        (0.7, f32::NAN),
        (0.7, f32::INFINITY),
        (0.7, f32::NEG_INFINITY),
    ];
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);
    for g in &got {
        assert!(
            close(*g, 1.0),
            "negative/non-finite max_gain -> gain 1: {g}"
        );
    }
}

#[test]
fn nonfinite_blend_sanitises_to_no_widening() {
    let Some(ctx) = context_or_skip("nonfinite_blend_sanitises_to_no_widening") else {
        return;
    };
    // NaN / +inf / -inf blend collapse to 0 -> gain 1 regardless of max_gain.
    let pairs = [
        (f32::NAN, 3.0),
        (f32::INFINITY, 3.0),
        (f32::NEG_INFINITY, 3.0),
    ];
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);
    for g in &got {
        assert!(close(*g, 1.0), "non-finite blend -> gain 1: {g}");
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no gains");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 pairs > 3 full 64-wide workgroups: every pair index must map to its
    // own gain independent of the dispatch tiling.
    let mut pairs = Vec::with_capacity(200);
    for i in 0..200u32 {
        let blend = (i as f32 * 0.005).min(1.0);
        let max_gain = (i % 5) as f32;
        pairs.push((blend, max_gain));
    }
    let got = run(&ctx, &pairs);
    assert_batch_close(&got, &pairs);
}
