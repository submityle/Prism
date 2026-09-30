//! Real-device parity for the clamp-history twin:
//! [`GpuClampHistory`] must reproduce the `CPU` golden
//! [`clamp_history`](prism_render_architecture::volumetric::temporal::clamp_history)
//! across a deterministic spread of history samples and neighbourhood windows,
//! including inverted `[min, max]` pairs that must be ordered first and the
//! degenerate `min == max` hard-clamp case.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `clamp` is expanded to the same branch form the CPU `math` module uses,
//! so `CPU` and `GPU` evaluate the same closed-form algebra. Values are asserted
//! to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a
//! wrong port (a dropped bound ordering, a swapped edge). The scenes also assert
//! every result lands inside the ordered window `[min(a,b), max(a,b)]`, so a
//! degenerate kernel could not pass.
//!
//! Provenance: standard TAA neighbourhood history clamp; no Unreal Engine source
//! or derived code.

use prism_render_architecture::volumetric::temporal::clamp_history;
use prism_volumetric_gpu::{ClampHistoryQuery, GpuClampHistory, GpuContext};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and lands inside the ordered clip window.
fn assert_parity(queries: &[ClampHistoryQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = clamp_history(q.history_sample, q.history_min, q.history_max);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "clamp-history mismatch for query {i} (sample {}, min {}, max {}): \
             gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})",
            q.history_sample,
            q.history_min,
            q.history_max
        );

        let lo = q.history_min.min(q.history_max);
        let hi = q.history_min.max(q.history_max);
        assert!(
            got >= lo - 1e-6 && got <= hi + 1e-6,
            "clamp-history result for query {i} must land in [{lo}, {hi}]: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_clamp_history_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping clamp-history parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuClampHistory::new(&ctx);

    // Hand-picked scenes: sample below/inside/above the window, an inverted
    // window that must be ordered first, and a degenerate min == max hard clamp.
    let mut queries: Vec<ClampHistoryQuery> = vec![
        // Below the window: clamped up to the low bound.
        ClampHistoryQuery {
            history_sample: -3.0,
            history_min: 0.2,
            history_max: 0.8,
        },
        // Inside the window: passes through unchanged.
        ClampHistoryQuery {
            history_sample: 0.5,
            history_min: 0.2,
            history_max: 0.8,
        },
        // Above the window: clamped down to the high bound.
        ClampHistoryQuery {
            history_sample: 5.0,
            history_min: 0.2,
            history_max: 0.8,
        },
        // Inverted bounds: ordered first, then clamped.
        ClampHistoryQuery {
            history_sample: -3.0,
            history_min: 0.8,
            history_max: 0.2,
        },
        ClampHistoryQuery {
            history_sample: 5.0,
            history_min: 0.8,
            history_max: 0.2,
        },
        // Degenerate window: hard clamp onto the single value.
        ClampHistoryQuery {
            history_sample: 100.0,
            history_min: 0.42,
            history_max: 0.42,
        },
        ClampHistoryQuery {
            history_sample: -100.0,
            history_min: 0.42,
            history_max: 0.42,
        },
    ];

    // A deterministic sweep of samples across a fixed window.
    let history_min = -1.0_f32;
    let history_max = 1.0_f32;
    for k in 0..=200 {
        let history_sample = -2.5 + (k as f32) * 0.025;
        queries.push(ClampHistoryQuery {
            history_sample,
            history_min,
            history_max,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The sample inside the window passes through unchanged.
    assert!(
        (gpu[1] - 0.5).abs() < 1e-6,
        "sample inside the window passes through: {}",
        gpu[1]
    );
    // The inverted window is ordered before clamping, so it behaves like the
    // ordered one.
    assert!(
        (gpu[3] - 0.2).abs() < 1e-6,
        "inverted window clamps below to the low bound: {}",
        gpu[3]
    );
    assert!(
        (gpu[4] - 0.8).abs() < 1e-6,
        "inverted window clamps above to the high bound: {}",
        gpu[4]
    );
    // The degenerate window collapses onto its single value.
    assert!(
        (gpu[5] - 0.42).abs() < 1e-6 && (gpu[6] - 0.42).abs() < 1e-6,
        "degenerate window clamps onto the single value: {}, {}",
        gpu[5],
        gpu[6]
    );

    // The swept segment (indices 7..) is monotone non-decreasing in the sample.
    let swept = &gpu[7..];
    let mut prev = f32::NEG_INFINITY;
    for (offset, &v) in swept.iter().enumerate() {
        assert!(
            v >= prev - 1e-6,
            "clamp history must be monotone non-decreasing in the sample: {v} \
             (swept index {offset}) after {prev}"
        );
        prev = v;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuClampHistory::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
