//! Real-device parity for the density-LOD binding-importance twin:
//! [`GpuHairBindingImportance`] must reproduce the `CPU` golden
//! [`binding_importances`](prism_render_architecture::hair::density_lod::binding_importances)
//! for a batch of render-strand bindings. The suite covers a general weighted
//! batch, the guide-order-independent weighted gather, out-of-range guide
//! indices and non-positive weights (both skipped), the authored-priority clamp,
//! curvature-dominant custom weights, the degenerate `weight_sum <= 0` and empty
//! guide-set cases (both exact `0`), a large multi-workgroup batch and the empty
//! no-op.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The two groom-level maxima, the weight sum and both normalizations are plain
//! IEEE operations both sides evaluate identically; only the weighted gather
//! sums and the final blend are multiply-add chains a `GPU` may fuse, so parity
//! is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough
//! to fail a genuinely wrong port (a swapped weight, a dropped guide, a missing
//! clamp), loose enough to admit the fma contraction. Degenerate cases collapse
//! to exact `0` and are asserted exactly. A spread guard rejects a kernel that
//! ignores its inputs. No `sin`/`cos` appears anywhere.
//!
//! Provenance: standard guide->render metric propagation plus importance-weighted
//! density-LOD fold; no Unreal Engine source or derived code.

use prism_hair_gpu::binding_importance::GpuHairBindingImportance;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::decimation::ImportanceWeights;
use prism_render_architecture::hair::density_lod::{binding_importances, GuideMetrics};
use prism_render_architecture::hair::interpolation::RenderStrandBinding;

/// Asserts a single importance matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

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

/// Builds a binding from four `(guide, weight)` pairs (unused slots pass a
/// zero weight so they are ignored on both sides).
fn binding(pairs: [(u32, f32); 4]) -> RenderStrandBinding {
    RenderStrandBinding {
        guides: [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
        weights: [pairs[0].1, pairs[1].1, pairs[2].1, pairs[3].1],
        root_uv: (0.0, 0.0),
        seed: 0,
    }
}

/// Runs the twin and the golden on the same inputs and asserts value-for-value
/// parity, returning the `GPU` vector for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuHairBindingImportance,
    bindings: &[RenderStrandBinding],
    metrics: &GuideMetrics,
    weights: ImportanceWeights,
) -> Vec<f32> {
    let gpu = kernel.eval(ctx, bindings, metrics, weights);
    let cpu = binding_importances(bindings, metrics, weights);
    assert_eq!(gpu.len(), cpu.len(), "importance vector length");
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        assert_close(*g, *c, &format!("importance[{i}]"));
    }
    gpu
}

/// Five guides with distinct metric spreads so no single guide dominates the
/// blend and the normalization is genuinely exercised.
fn sample_metrics() -> GuideMetrics {
    GuideMetrics {
        lengths: vec![1.0, 3.5, 0.75, 6.0, 2.25],
        curvatures: vec![0.2, 1.4, 0.05, 0.9, 2.1],
        authored: vec![0.1, 0.8, 0.5, 0.3, 1.0],
    }
}

#[test]
fn matches_cpu_within_tolerance() {
    let Some(ctx) = context_or_skip("binding-importance weighted-batch parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();

    // A spread of bindings mixing one-, two-, three- and four-guide skinning,
    // each with re-normalized positive weights.
    let bindings = [
        binding([(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]),
        binding([(1, 0.6), (3, 0.4), (0, 0.0), (0, 0.0)]),
        binding([(2, 0.5), (4, 0.3), (1, 0.2), (0, 0.0)]),
        binding([(0, 0.25), (1, 0.25), (3, 0.25), (4, 0.25)]),
        binding([(4, 0.7), (2, 0.3), (0, 0.0), (0, 0.0)]),
    ];

    let gpu = assert_parity(
        &ctx,
        &kernel,
        &bindings,
        &metrics,
        ImportanceWeights::default(),
    );

    for (i, v) in gpu.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(v),
            "importance[{i}] = {v} escaped [0, 1]"
        );
    }
    // Spread guard: a kernel that ignored its inputs would emit a constant.
    let lo = gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi - lo > 0.05,
        "importances are implausibly flat: {lo}..{hi}"
    );
}

#[test]
fn gather_is_guide_order_independent() {
    let Some(ctx) = context_or_skip("binding-importance guide-order parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();

    // The same three-guide blend written in two guide orders: the weighted sum
    // is commutative, so both must fold to the same importance.
    let ordered = [binding([(1, 0.5), (3, 0.3), (4, 0.2), (0, 0.0)])];
    let shuffled = [binding([(4, 0.2), (1, 0.5), (3, 0.3), (0, 0.0)])];

    let a = assert_parity(
        &ctx,
        &kernel,
        &ordered,
        &metrics,
        ImportanceWeights::default(),
    );
    let b = assert_parity(
        &ctx,
        &kernel,
        &shuffled,
        &metrics,
        ImportanceWeights::default(),
    );
    assert_close(a[0], b[0], "guide order independence");
}

#[test]
fn out_of_range_guides_and_nonpositive_weights_skip() {
    let Some(ctx) = context_or_skip("binding-importance skip parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics(); // guide_count == 5

    // Guide index 9 is out of range; a zero and a negative weight are muted.
    // Only guide 2 with weight 0.5 should contribute on both sides.
    let bindings = [
        binding([(9, 0.8), (2, 0.5), (1, 0.0), (0, -0.4)]),
        // A binding whose only in-range guide has a negative weight contributes
        // nothing: the blended metrics are all zero.
        binding([(3, -1.0), (7, 0.9), (0, 0.0), (0, 0.0)]),
    ];

    assert_parity(
        &ctx,
        &kernel,
        &bindings,
        &metrics,
        ImportanceWeights::default(),
    );
}

#[test]
fn curvature_dominant_weights_match() {
    let Some(ctx) = context_or_skip("binding-importance curvature-weight parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();
    let weights = ImportanceWeights {
        length: 0.1,
        curvature: 0.85,
        authored: 0.05,
    };

    let bindings = [
        binding([(1, 0.5), (4, 0.5), (0, 0.0), (0, 0.0)]),
        binding([(3, 0.7), (2, 0.3), (0, 0.0), (0, 0.0)]),
        binding([(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]),
    ];

    assert_parity(&ctx, &kernel, &bindings, &metrics, weights);
}

#[test]
fn authored_blend_is_clamped() {
    let Some(ctx) = context_or_skip("binding-importance authored-clamp parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    // Every guide already at the maximum authored priority; weights summing above
    // one push the blended authored value past 1, exercising the clamp on both
    // sides.
    let metrics = GuideMetrics {
        lengths: vec![0.0, 0.0],
        curvatures: vec![0.0, 0.0],
        authored: vec![1.0, 1.0],
    };
    let weights = ImportanceWeights {
        length: 0.0,
        curvature: 0.0,
        authored: 1.0,
    };
    let bindings = [binding([(0, 0.9), (1, 0.9), (0, 0.0), (0, 0.0)])];

    let gpu = assert_parity(&ctx, &kernel, &bindings, &metrics, weights);
    assert_close(gpu[0], 1.0, "authored blend clamps to 1");
}

#[test]
fn nonpositive_weight_sum_is_zero() {
    let Some(ctx) = context_or_skip("binding-importance zero-weight-sum parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();
    let weights = ImportanceWeights {
        length: 0.0,
        curvature: 0.0,
        authored: 0.0,
    };
    let bindings = [
        binding([(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]),
        binding([(1, 0.5), (3, 0.5), (0, 0.0), (0, 0.0)]),
    ];

    let gpu = assert_parity(&ctx, &kernel, &bindings, &metrics, weights);
    for (i, v) in gpu.iter().enumerate() {
        assert_eq!(*v, 0.0, "muted weights force exact 0 at {i}");
    }
}

#[test]
fn empty_guide_set_yields_zero_importances() {
    let Some(ctx) = context_or_skip("binding-importance empty-guides parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    // No guides measured: every gather reads nothing, so every binding folds to
    // an exact 0 — but the vector still has one entry per binding.
    let metrics = GuideMetrics::default();
    let bindings = [
        binding([(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]),
        binding([(2, 0.5), (1, 0.5), (0, 0.0), (0, 0.0)]),
        binding([(0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]),
    ];

    let gpu = assert_parity(
        &ctx,
        &kernel,
        &bindings,
        &metrics,
        ImportanceWeights::default(),
    );
    assert_eq!(
        gpu.len(),
        3,
        "one importance per binding even with no guides"
    );
    for (i, v) in gpu.iter().enumerate() {
        assert_eq!(*v, 0.0, "no guides force exact 0 at {i}");
    }
}

#[test]
fn large_batch_spans_workgroups() {
    let Some(ctx) = context_or_skip("binding-importance large-batch parity") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();
    let guide_count = metrics.len() as u32;

    // 300 bindings (> 4 workgroups of 64) with deterministic, non-constant guide
    // choices and weights so the whole grid is exercised.
    let bindings: Vec<RenderStrandBinding> = (0..300u32)
        .map(|i| {
            let g0 = i % guide_count;
            let g1 = (i * 7 + 1) % guide_count;
            let w0 = 0.3 + ((i % 5) as f32) * 0.1;
            let w1 = 1.0 - w0;
            binding([(g0, w0), (g1, w1), (0, 0.0), (0, 0.0)])
        })
        .collect();

    let gpu = assert_parity(
        &ctx,
        &kernel,
        &bindings,
        &metrics,
        ImportanceWeights::default(),
    );
    assert_eq!(
        gpu.len(),
        300,
        "one importance per binding across workgroups"
    );
}

#[test]
fn empty_bindings_are_a_noop() {
    let Some(ctx) = context_or_skip("binding-importance empty-bindings no-op") else {
        return;
    };
    let kernel = GpuHairBindingImportance::new(&ctx);
    let metrics = sample_metrics();
    let gpu = kernel.eval(&ctx, &[], &metrics, ImportanceWeights::default());
    assert!(gpu.is_empty(), "empty bindings must yield an empty vector");
}
