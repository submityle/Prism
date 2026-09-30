//! Real-device parity for the variance-clip twin:
//! [`GpuVarianceClip`] must reproduce the `CPU` golden
//! [`variance_clip`](prism_render_architecture::volumetric::temporal::variance_clip)
//! across a deterministic spread of history samples, means, standard deviations
//! and gamma window scales, including the degenerate negative `std_dev`/`gamma`
//! cases that floor the window to a hard clamp onto the mean.
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
//! wrong port (a dropped floor, a swapped bound). The scenes also assert every
//! result lands inside the floored clip window `[mean - half, mean + half]`, and
//! that a negative `std_dev` or `gamma` collapses the window so the result is
//! exactly the mean, so a degenerate kernel could not pass.
//!
//! Provenance: standard TAA variance box clip; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::temporal::variance_clip;
use prism_volumetric_gpu::{GpuContext, GpuVarianceClip, VarianceClipQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and lands inside the floored clip window.
fn assert_parity(queries: &[VarianceClipQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = variance_clip(q.history_sample, q.mean, q.std_dev, q.gamma);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "variance-clip mismatch for query {i} (history {}, mean {}, std_dev {}, gamma {}): \
             gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})",
            q.history_sample,
            q.mean,
            q.std_dev,
            q.gamma
        );

        // The result must land inside the floored clip window.
        let sd = q.std_dev.max(0.0);
        let g = q.gamma.max(0.0);
        let half = sd * g;
        let lo = q.mean - half;
        let hi = q.mean + half;
        assert!(
            got >= lo - 1e-6 && got <= hi + 1e-6,
            "variance-clip result for query {i} must land in [{lo}, {hi}]: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_variance_clip_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping variance-clip parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuVarianceClip::new(&ctx);

    // A deterministic sweep of history samples across a fixed window, plus a few
    // hand-picked scenes: sample inside/below/above the window, and the
    // degenerate negative std_dev / gamma cases that floor the window to a hard
    // clamp onto the mean.
    let mut queries: Vec<VarianceClipQuery> = vec![
        // Sample below the window: clamped up to the low bound.
        VarianceClipQuery {
            history_sample: -5.0,
            mean: 0.5,
            std_dev: 0.1,
            gamma: 1.0,
        },
        // Sample inside the window: passes through unchanged.
        VarianceClipQuery {
            history_sample: 0.55,
            mean: 0.5,
            std_dev: 0.1,
            gamma: 1.0,
        },
        // Sample above the window: clamped down to the high bound.
        VarianceClipQuery {
            history_sample: 9.0,
            mean: 0.5,
            std_dev: 0.1,
            gamma: 1.0,
        },
        // Negative std_dev: floored to zero, so the window collapses to the mean.
        VarianceClipQuery {
            history_sample: 3.0,
            mean: 0.42,
            std_dev: -1.0,
            gamma: 2.0,
        },
        // Negative gamma: floored to zero, so the window collapses to the mean.
        VarianceClipQuery {
            history_sample: -3.0,
            mean: -0.17,
            std_dev: 0.3,
            gamma: -4.0,
        },
        // Wider window with a negative mean.
        VarianceClipQuery {
            history_sample: 0.0,
            mean: -2.0,
            std_dev: 0.5,
            gamma: 3.0,
        },
    ];

    // A deterministic 2-D sweep: history samples across a fixed mean and window.
    let mean = 0.5_f32;
    let std_dev = 0.2_f32;
    let gamma = 1.5_f32;
    for k in 0..=200 {
        let history_sample = -2.0 + (k as f32) * 0.02;
        queries.push(VarianceClipQuery {
            history_sample,
            mean,
            std_dev,
            gamma,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The degenerate negative std_dev / gamma cases collapse the window onto the
    // mean exactly.
    assert!(
        (gpu[3] - 0.42).abs() < 1e-6,
        "negative std_dev floors the window onto the mean: {}",
        gpu[3]
    );
    assert!(
        (gpu[4] - (-0.17)).abs() < 1e-6,
        "negative gamma floors the window onto the mean: {}",
        gpu[4]
    );

    // The sample inside the window passes through unchanged.
    assert!(
        (gpu[1] - 0.55).abs() < 1e-6,
        "sample inside the window passes through: {}",
        gpu[1]
    );

    // The swept segment (indices 6..) is monotone non-decreasing in the history
    // sample (clamp of a monotone input against a fixed window).
    let swept = &gpu[6..];
    let mut prev = f32::NEG_INFINITY;
    for (offset, &v) in swept.iter().enumerate() {
        assert!(
            v >= prev - 1e-6,
            "variance clip must be monotone non-decreasing in the history sample: {v} \
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
    let gpu_kernel = GpuVarianceClip::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
