//! Real-device parity for the imposter-fade twin:
//! [`GpuImposterFade`] must reproduce the `CPU` golden
//! [`imposter_fade`](prism_render_architecture::volumetric::cloud_lod::imposter_fade)
//! across a deterministic spread of distances and fade windows, including the
//! degenerate `start == end` hard-step case.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `smoothstep` is expanded to the same closed form the CPU `math` module
//! uses, so `CPU` and `GPU` evaluate the same closed-form algebra. Values are
//! asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to
//! fail a wrong port (a dropped saturate, a swapped edge order). The scenes also
//! assert the weight stays in `0..=1` and is monotone non-decreasing in distance
//! (a farther cloud always fades at least as much toward the imposter), so a
//! degenerate kernel could not pass.
//!
//! Provenance: standard smoothstep LOD cross-fade; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::cloud_lod::imposter_fade;
use prism_volumetric_gpu::{GpuContext, GpuImposterFade, ImposterFadeQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[ImposterFadeQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = imposter_fade(q.distance, q.start, q.end);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "imposter-fade mismatch for query {i} (distance {}, start {}, end {}): \
             gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})",
            q.distance,
            q.start,
            q.end
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "imposter-fade weight for query {i} must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_imposter_fade_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping imposter-fade parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuImposterFade::new(&ctx);

    // A deterministic sweep of distances across a fixed fade window, plus a few
    // hand-picked scenes: before/after the window, and the degenerate
    // start == end hard step.
    let start = 800.0_f32;
    let end = 1200.0_f32;
    let mut queries: Vec<ImposterFadeQuery> = vec![
        ImposterFadeQuery {
            distance: 0.0,
            start,
            end,
        },
        ImposterFadeQuery {
            distance: 800.0,
            start,
            end,
        },
        ImposterFadeQuery {
            distance: 1000.0,
            start,
            end,
        },
        ImposterFadeQuery {
            distance: 1200.0,
            start,
            end,
        },
        ImposterFadeQuery {
            distance: 5000.0,
            start,
            end,
        },
        // Degenerate window: a hard step at 1000.
        ImposterFadeQuery {
            distance: 999.0,
            start: 1000.0,
            end: 1000.0,
        },
        ImposterFadeQuery {
            distance: 1001.0,
            start: 1000.0,
            end: 1000.0,
        },
    ];
    for k in 0..=200 {
        let distance = (k as f32) * 10.0;
        queries.push(ImposterFadeQuery {
            distance,
            start,
            end,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The swept segment (indices 7..) is monotone non-decreasing in distance.
    let swept = &gpu[7..];
    let mut prev = f32::NEG_INFINITY;
    for (offset, &w) in swept.iter().enumerate() {
        assert!(
            w >= prev - 1e-6,
            "imposter fade must be monotone non-decreasing in distance: {w} \
             (swept index {offset}) after {prev}"
        );
        prev = w;
    }

    // Before the window is a full raymarch (0); past the window is a full
    // imposter (1); the degenerate window is a hard step.
    assert!(
        gpu[0].abs() < 1e-6,
        "distance below start is a full raymarch: {}",
        gpu[0]
    );
    assert!(
        (gpu[4] - 1.0).abs() < 1e-6,
        "distance past end is a full imposter: {}",
        gpu[4]
    );
    assert!(
        gpu[5].abs() < 1e-6,
        "hard step below the edge is 0: {}",
        gpu[5]
    );
    assert!(
        (gpu[6] - 1.0).abs() < 1e-6,
        "hard step above the edge is 1: {}",
        gpu[6]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuImposterFade::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
