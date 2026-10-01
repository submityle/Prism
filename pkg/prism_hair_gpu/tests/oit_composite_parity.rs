//! Real-device parity for the isolated `OIT` `k-layer` composite twin:
//! [`GpuHairOitComposite`] must reproduce the `CPU` goldens
//! [`composite_transmittance`](prism_render_architecture::hair::oit_frontend::composite_transmittance)
//! and
//! [`composite_coverage`](prism_render_architecture::hair::oit_frontend::composite_coverage)
//! (bundled by
//! [`reference_composite`](prism_hair_gpu::oit_composite::reference_composite))
//! for a batch of resolved `per-pixel` fragment stacks.
//!
//! # Parity criterion
//!
//! The composite is a plain product chain of `(1 - alpha)` terms times the
//! `(1 - tail_alpha)` tail, so the device result is effectively bit-identical to
//! the scalar reference; each returned `(transmittance, coverage)` is
//! nonetheless compared against a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`), not by raw bit pattern. The suite also asserts the
//! structural invariant that coverage is the exact complement of transmittance
//! and both stay in `[0, 1]`.
//!
//! The suite drives an empty pixel (full transmittance), a fully opaque layer
//! (zero transmittance), a partially-covered stack, a tail-only stack
//! (`k == 0`), a stack deeper than `MAX_OIT_LAYERS` (overflow folded into the
//! tail), the empty no-op batch, and a large multi-workgroup batch that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: `d'Eon`/`TressFX` `k-layer` `MLAB` `alpha` compositing plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::oit_composite::{reference_composite, GpuHairOitComposite};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::oit_frontend::{
    sort_and_clip, HairFragment, LayeredFragments, MAX_OIT_LAYERS,
};

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
fn run(ctx: &GpuContext, pixels: &[LayeredFragments]) -> Vec<(f32, f32)> {
    GpuHairOitComposite::new(ctx).eval(ctx, pixels)
}

/// True when `got` matches `want` within the composite tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Builds a resolved pixel stack from raw `(depth, alpha)` fragments and a
/// requested layer count `k`, exactly as the `PPLL` resolve front-end does.
fn pixel(fragments: &[(f32, f32)], k: usize) -> LayeredFragments {
    let frags: Vec<HairFragment> = fragments
        .iter()
        .map(|&(depth, alpha)| HairFragment::new(depth, alpha, 1.0))
        .collect();
    sort_and_clip(&frags, k)
}

/// Asserts a whole batch matches both goldens within tolerance and respects the
/// coverage/transmittance complement invariant.
fn assert_batch_close(got: &[(f32, f32)], pixels: &[LayeredFragments]) {
    assert_eq!(
        got.len(),
        pixels.len(),
        "one (transmittance, coverage) pair per pixel (got {}, want {})",
        got.len(),
        pixels.len()
    );
    for (i, (&(gt, gc), layered)) in got.iter().zip(pixels.iter()).enumerate() {
        let (wt, wc) = reference_composite(layered);
        assert!(
            close(gt, wt),
            "pixel {i}: device transmittance {gt} must match golden {wt}"
        );
        assert!(
            close(gc, wc),
            "pixel {i}: device coverage {gc} must match golden {wc}"
        );
        assert!(
            (0.0..=1.0).contains(&gt) && (0.0..=1.0).contains(&gc),
            "pixel {i}: outputs stay in [0,1]: ({gt}, {gc})"
        );
        assert!(
            close(gc, 1.0 - gt),
            "pixel {i}: coverage {gc} is the complement of transmittance {gt}"
        );
    }
}

#[test]
fn regular_stacks_match_golden() {
    let Some(ctx) = context_or_skip("regular_stacks_match_golden") else {
        return;
    };
    let pixels = [
        // Empty pixel: no fragments -> full transmittance.
        pixel(&[], 8),
        // Fully opaque near layer -> zero transmittance.
        pixel(&[(0.1, 1.0), (0.2, 0.5)], 8),
        // Two half-covered layers -> 0.25 transmittance, 0.75 coverage.
        pixel(&[(0.1, 0.5), (0.2, 0.5)], 8),
        // Mixed coverage, deeper stack within k.
        pixel(&[(0.3, 0.2), (0.1, 0.4), (0.2, 0.6), (0.4, 0.1)], 8),
    ];
    let got = run(&ctx, &pixels);
    assert_batch_close(&got, &pixels);

    // Spot-check the known endpoints.
    assert!(close(got[0].0, 1.0), "empty pixel has full transmittance");
    assert!(close(got[0].1, 0.0), "empty pixel has zero coverage");
    assert!(close(got[1].0, 0.0), "opaque layer zeroes transmittance");
    assert!(
        close(got[2].0, 0.25),
        "two half layers -> 0.25 transmittance"
    );
    assert!(close(got[2].1, 0.75), "two half layers -> 0.75 coverage");
}

#[test]
fn tail_only_stack_matches_golden() {
    let Some(ctx) = context_or_skip("tail_only_stack_matches_golden") else {
        return;
    };
    // k == 0 sends the whole stack into the tail: 1 - 0.5*0.5 = 0.75 tail,
    // transmittance 0.25.
    let pixels = [pixel(&[(0.1, 0.5), (0.2, 0.5)], 0)];
    let got = run(&ctx, &pixels);
    assert_batch_close(&got, &pixels);
    assert!(close(got[0].0, 0.25), "tail-only transmittance is 0.25");
    assert!(close(got[0].1, 0.75), "tail-only coverage is 0.75");
}

#[test]
fn overflow_stack_folds_into_tail() {
    let Some(ctx) = context_or_skip("overflow_stack_folds_into_tail") else {
        return;
    };
    // More fragments than MAX_OIT_LAYERS: the nearest 8 resolve, the rest fold
    // into the tail. The composite must still match the golden exactly.
    let mut frags = Vec::with_capacity(MAX_OIT_LAYERS + 5);
    for i in 0..(MAX_OIT_LAYERS + 5) {
        frags.push(((i as f32) * 0.1, 0.2));
    }
    let pixels = [pixel(&frags, MAX_OIT_LAYERS)];
    assert_eq!(
        pixels[0].layers.len(),
        MAX_OIT_LAYERS,
        "overflow stack resolves exactly MAX_OIT_LAYERS layers"
    );
    let got = run(&ctx, &pixels);
    assert_batch_close(&got, &pixels);
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no composite pairs");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 pixels > 3 full 64-wide workgroups: every pixel index must map to its
    // own composite independent of the dispatch tiling. Each pixel gets a stack
    // whose depth count and alphas vary with the index.
    let mut pixels = Vec::with_capacity(200);
    for i in 0..200u32 {
        let depth_count = (i % 11) as usize;
        let mut frags = Vec::with_capacity(depth_count);
        for j in 0..depth_count {
            let depth = (j as f32) * 0.07 + (i as f32) * 0.001;
            let alpha = ((i + j as u32) % 7) as f32 * 0.1 + 0.05;
            frags.push((depth, alpha));
        }
        pixels.push(pixel(&frags, MAX_OIT_LAYERS));
    }
    let got = run(&ctx, &pixels);
    assert_batch_close(&got, &pixels);
}
