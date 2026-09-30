//! Real-device parity for the density-LOD importance-fold twin:
//! [`GpuHairImportance`] must reproduce the `CPU` golden
//! [`compute_importance`](prism_render_architecture::hair::decimation::compute_importance)
//! for a batch of strands. The suite covers a general weighted batch, the
//! authored-priority clamp, the degenerate `weight_sum <= 0` and zero-maxima
//! cases (both exact `0`), a large multi-workgroup batch and the empty /
//! length-mismatch no-ops.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The two groom-level maxima, the weight sum and both normalizations are plain
//! IEEE operations both sides evaluate identically; only the final blend
//! `w_l*len_n + w_c*curv_n + w_a*auth` is a multiply-add chain a `GPU` may fuse,
//! so parity is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` —
//! tight enough to fail a genuinely wrong fold (a swapped weight, a dropped
//! normalization, a missing clamp), loose enough to admit the fma contraction.
//! Degenerate cases collapse to exact `0` and are asserted exactly. A spread
//! guard rejects a kernel that ignores its inputs. No `sin`/`cos` appears
//! anywhere.
//!
//! Provenance: standard importance-weighted density-LOD metric fold; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::importance::GpuHairImportance;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::decimation::{compute_importance, ImportanceWeights};

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

#[test]
fn matches_cpu_within_tolerance() {
    let Some(ctx) = context_or_skip("importance weighted-batch parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    let n = 512usize;
    // Distinct, non-monotonic metric spreads so no single strand dominates and
    // the normalization / blend is genuinely exercised.
    let lengths: Vec<f32> = (0..n).map(|i| 0.25 + ((i * 5 + 1) % 37) as f32).collect();
    let curvatures: Vec<f32> = (0..n).map(|i| ((i * 3 + 2) % 19) as f32 * 0.5).collect();
    let authored: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) / 10.0).collect();
    let weights = ImportanceWeights::default();
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    assert_eq!(gpu.len(), n, "one importance per strand");
    assert_eq!(cpu.len(), n, "golden emits one importance per strand");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} importance"));
        assert!(
            (0.0..=1.0 + 1e-4).contains(&g),
            "strand {i} importance {g} must stay in [0, 1]"
        );
    }
    // A kernel that ignored its inputs (constant output) must not pass: the
    // folded importances span a non-trivial range.
    let min = gpu.iter().copied().fold(f32::INFINITY, f32::min);
    let max = gpu.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max - min > 0.1,
        "importances must vary across strands (min {min}, max {max})"
    );
}

#[test]
fn custom_weights_match_cpu() {
    let Some(ctx) = context_or_skip("importance custom-weights parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    let n = 300usize;
    let lengths: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 53) as f32).collect();
    let curvatures: Vec<f32> = (0..n).map(|i| ((i * 2 + 5) % 29) as f32).collect();
    let authored: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) / 6.0).collect();
    // Curvature-dominant weights, distinct from the default, to catch a kernel
    // that hard-codes the default blend.
    let weights = ImportanceWeights {
        length: 0.3,
        curvature: 2.0,
        authored: 0.7,
    };
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} importance"));
    }
}

#[test]
fn authored_priority_is_clamped() {
    let Some(ctx) = context_or_skip("importance authored-clamp parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    // Authored priorities intentionally out of range (negative and > 1); the
    // fold must clamp them to [0, 1] exactly as the golden does. Length and
    // curvature weights are zeroed so only the authored term drives importance.
    let lengths = vec![1.0f32, 2.0, 3.0, 4.0];
    let curvatures = vec![0.5f32, 1.0, 1.5, 2.0];
    let authored = vec![-0.5f32, 0.25, 0.75, 1.5];
    let weights = ImportanceWeights {
        length: 0.0,
        curvature: 0.0,
        authored: 1.0,
    };
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} importance"));
    }
    // Only the authored term survives, so clamped authored is the importance:
    // -0.5 -> 0, 1.5 -> 1.
    assert!(
        gpu[0].abs() <= f32::EPSILON,
        "negative authored clamps to 0"
    );
    assert!(
        (gpu[3] - 1.0).abs() <= 1e-4,
        "authored > 1 clamps to 1 (got {})",
        gpu[3]
    );
}

#[test]
fn non_positive_weight_sum_yields_zero() {
    let Some(ctx) = context_or_skip("importance zero-weight parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    let lengths = vec![1.0f32, 5.0, 9.0];
    let curvatures = vec![2.0f32, 4.0, 6.0];
    let authored = vec![0.2f32, 0.6, 1.0];
    let weights = ImportanceWeights {
        length: 0.0,
        curvature: 0.0,
        authored: 0.0,
    };
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            g.abs() <= f32::EPSILON && c.abs() <= f32::EPSILON,
            "strand {i}: a non-positive weight sum yields exact 0 (gpu {g}, cpu {c})"
        );
    }
}

#[test]
fn zero_maxima_yield_zero() {
    let Some(ctx) = context_or_skip("importance zero-maxima parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    // Every length and curvature is zero (so both maxima are zero) and authored
    // is zero, so the whole fold collapses to exact 0 on both sides.
    let n = 8usize;
    let lengths = vec![0.0f32; n];
    let curvatures = vec![0.0f32; n];
    let authored = vec![0.0f32; n];
    let weights = ImportanceWeights::default();
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    assert_eq!(gpu.len(), n, "one importance per strand");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            g.abs() <= f32::EPSILON && c.abs() <= f32::EPSILON,
            "strand {i}: zero maxima yield exact 0 (gpu {g}, cpu {c})"
        );
    }
}

#[test]
fn large_batch_matches_across_workgroups() {
    let Some(ctx) = context_or_skip("importance large-batch parity") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    let n = 4096usize;
    let lengths: Vec<f32> = (0..n).map(|i| ((i * 13 + 1) % 251) as f32).collect();
    let curvatures: Vec<f32> = (0..n).map(|i| ((i * 17 + 4) % 97) as f32).collect();
    let authored: Vec<f32> = (0..n).map(|i| ((i % 101) as f32) / 100.0).collect();
    let weights = ImportanceWeights::default();
    let gpu = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    let cpu = compute_importance(&lengths, &curvatures, &authored, weights);
    assert_eq!(gpu.len(), n, "one importance per strand across workgroups");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} importance"));
    }
}

#[test]
fn empty_and_mismatched_inputs_are_noops() {
    let Some(ctx) = context_or_skip("importance no-op") else {
        return;
    };
    let kernel = GpuHairImportance::new(&ctx);
    let weights = ImportanceWeights::default();
    let empty = kernel.eval(&ctx, &[], &[], &[], weights);
    assert!(empty.is_empty(), "empty batch yields an empty result");
    let lengths = vec![1.0f32, 2.0, 3.0, 4.0];
    let curvatures = vec![0.5f32, 1.0, 1.5];
    let authored = vec![0.1f32, 0.2, 0.3, 0.4];
    let mismatch = kernel.eval(&ctx, &lengths, &curvatures, &authored, weights);
    assert!(
        mismatch.is_empty(),
        "length mismatch yields an empty result without a dispatch"
    );
}
