//! Real-device parity for the should-fallback twin:
//! [`GpuShouldFallback`] must reproduce the `CPU` golden
//! [`should_fallback`](prism_render_architecture::volumetric::temporal::should_fallback)
//! across every combination of history validity and a deterministic spread of
//! disocclusion measures against a fixed threshold, including the boundary
//! (`disocclusion == max_disocclusion`, which keeps history).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The decision is a boolean expression (a comparison and a logical or), so the
//! parity test asserts every decision matches the `CPU` golden bit for bit —
//! there is no floating-point slack. A degenerate kernel (dropped validity
//! term, `>=` instead of `>`) could not pass.
//!
//! Provenance: standard TAA disocclusion / validity fallback rule; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::temporal::should_fallback;
use prism_volumetric_gpu::{GpuContext, GpuShouldFallback, ShouldFallbackQuery};

/// Asserts every `gpu` decision matches the `CPU` golden exactly.
fn assert_parity(queries: &[ShouldFallbackQuery], gpu: &[bool]) {
    assert_eq!(gpu.len(), queries.len(), "one decision per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = should_fallback(q.history_valid, q.disocclusion, q.max_disocclusion);
        assert_eq!(
            gpu[i], exp,
            "should-fallback mismatch for query {i} (valid {}, disocclusion {}, max {}): \
             gpu {}, cpu {exp}",
            q.history_valid, q.disocclusion, q.max_disocclusion, gpu[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_should_fallback_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping should-fallback parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuShouldFallback::new(&ctx);

    // Hand-picked scenes: invalid history always falls back; valid history falls
    // back only above the threshold; the boundary keeps history.
    let max = 0.5_f32;
    let mut queries: Vec<ShouldFallbackQuery> = vec![
        // Invalid history: always fall back, regardless of disocclusion.
        ShouldFallbackQuery {
            history_valid: false,
            disocclusion: 0.0,
            max_disocclusion: max,
        },
        ShouldFallbackQuery {
            history_valid: false,
            disocclusion: 10.0,
            max_disocclusion: max,
        },
        // Valid history below the threshold: keep it.
        ShouldFallbackQuery {
            history_valid: true,
            disocclusion: 0.1,
            max_disocclusion: max,
        },
        // Valid history at the boundary: keep it (`>` is strict).
        ShouldFallbackQuery {
            history_valid: true,
            disocclusion: max,
            max_disocclusion: max,
        },
        // Valid history above the threshold: fall back.
        ShouldFallbackQuery {
            history_valid: true,
            disocclusion: 0.9,
            max_disocclusion: max,
        },
    ];

    // A deterministic sweep of disocclusion values, for both validity states,
    // crossing the threshold.
    for k in 0..=200 {
        let disocclusion = (k as f32) * 0.005;
        for &history_valid in &[false, true] {
            queries.push(ShouldFallbackQuery {
                history_valid,
                disocclusion,
                max_disocclusion: max,
            });
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Invalid history always falls back.
    assert!(gpu[0] && gpu[1], "invalid history must always fall back");
    // Valid history below / at the threshold keeps history.
    assert!(
        !gpu[2] && !gpu[3],
        "valid history at or below the threshold keeps history"
    );
    // Valid history above the threshold falls back.
    assert!(gpu[4], "valid history above the threshold falls back");
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuShouldFallback::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
