//! Real-device parity for the overshooting-bump twin: [`GpuOvershootingBump`]
//! must reproduce the `CPU` golden
//! [`overshooting_bump`](prism_render_architecture::volumetric::storm::overshooting_bump)
//! across the full height/strength grid, including heights above the band top
//! and saturating out-of-range strengths.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The exponential uses the same hand-rolled `exp_approx` the CPU golden uses
//! (base-two range reduction with a fractional polynomial), not the
//! device-native `exp`, so `CPU` and `GPU` evaluate the same closed-form
//! algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! wrong sharpness, the native `exp`). The scenes also assert the value peaks at
//! `height_fraction == 1`, decays with `|height_fraction - 1|`, scales with the
//! dome strength and stays in `0..=1`, so a degenerate kernel could not pass.
//!
//! Provenance: standard cumulonimbus overshooting-top dome; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::storm::overshooting_bump;
use prism_volumetric_gpu::{GpuContext, GpuOvershootingBump, OvershootingBumpQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[OvershootingBumpQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = overshooting_bump(q.height_fraction, q.overshooting_top);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "overshooting bump mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu overshooting bump must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_overshooting_bump_matches_cpu_golden_across_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping overshooting bump parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuOvershootingBump::new(&ctx);

    // A deterministic height/strength grid. Heights run past 1 so the dome can
    // bulge above the band top, plus out-of-range strengths that must saturate.
    let mut queries: Vec<OvershootingBumpQuery> = Vec::new();
    for hi in 0..=40 {
        for ti in 0..=20 {
            queries.push(OvershootingBumpQuery {
                height_fraction: hi as f32 / 20.0,
                overshooting_top: ti as f32 / 20.0,
            });
        }
    }
    queries.push(OvershootingBumpQuery {
        height_fraction: -0.5,
        overshooting_top: 1.5,
    });
    queries.push(OvershootingBumpQuery {
        height_fraction: 2.5,
        overshooting_top: -0.3,
    });

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_overshooting_bump_peaks_at_band_top_and_scales_with_strength() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuOvershootingBump::new(&ctx);

    // A height sweep at full strength must peak at height_fraction == 1 and
    // decay symmetrically as |height_fraction - 1| grows.
    let height_sweep: Vec<OvershootingBumpQuery> = (0..=40)
        .map(|h| OvershootingBumpQuery {
            height_fraction: h as f32 / 20.0,
            overshooting_top: 1.0,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &height_sweep);
    assert_parity(&height_sweep, &gpu);
    // The peak sits at index 20 (height_fraction == 1.0).
    let peak = gpu[20];
    for (i, &v) in gpu.iter().enumerate() {
        assert!(
            v <= peak + 1e-6,
            "overshooting bump must peak at the band top: idx {i} {v} > peak {peak}"
        );
    }
    // Rising into the peak, then falling away from it.
    for w in 0..20 {
        assert!(
            gpu[w + 1] >= gpu[w] - 1e-6,
            "must rise toward the band top: {} then {}",
            gpu[w],
            gpu[w + 1]
        );
    }
    for w in 20..gpu.len() - 1 {
        assert!(
            gpu[w + 1] <= gpu[w] + 1e-6,
            "must fall past the band top: {} then {}",
            gpu[w],
            gpu[w + 1]
        );
    }

    // At the band top the dome weight equals the (saturated) strength, so a
    // strength sweep is the identity there and rises monotonically.
    let strength_sweep: Vec<OvershootingBumpQuery> = (0..=40)
        .map(|t| OvershootingBumpQuery {
            height_fraction: 1.0,
            overshooting_top: t as f32 / 40.0,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &strength_sweep);
    assert_parity(&strength_sweep, &gpu);
    let mut prev = f32::NEG_INFINITY;
    for &v in &gpu {
        assert!(
            v >= prev - 1e-6,
            "dome weight must rise with strength: {prev} then {v}"
        );
        prev = v;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuOvershootingBump::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
