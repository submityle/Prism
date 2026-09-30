//! Real-device parity for the gravity-wave twin: [`GpuGravityWave`] must
//! reproduce the `CPU` golden
//! [`gravity_wave`](prism_render_architecture::volumetric::storm::gravity_wave)
//! across a phase/offset grid, including large phases that exercise the angle
//! reduction.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The sine uses the same hand-rolled `sin_approx` the CPU golden uses (reduce
//! to `[-PI, PI]` with `round`-halves-away-from-zero, fold to `[-PI/2, PI/2]`,
//! seventh-order Taylor polynomial), not the device-native `sin`, so `CPU` and
//! `GPU` evaluate the same closed-form algebra. Values are asserted to within
//! `abs_diff < 1e-5` or `rel_diff < 1e-5` — tight enough to fail a wrong port (a
//! wrong constant, the native `sin`, a to-even rounding). The scenes also assert
//! the value stays within roughly `[-1, 1]` and is periodic in phase, so a
//! degenerate kernel could not pass.
//!
//! Provenance: standard stable-layer gravity-wave ripple; no Unreal Engine
//! source or derived code.

use core::f32::consts::PI;
use prism_render_architecture::volumetric::storm::gravity_wave;
use prism_volumetric_gpu::{GpuContext, GpuGravityWave, GravityWaveQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays within roughly `[-1, 1]`.
fn assert_parity(queries: &[GravityWaveQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = gravity_wave(q.phase, q.x);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-5,
            "gravity wave mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (-1.01..=1.01).contains(&got),
            "gpu gravity wave must stay within roughly [-1, 1]: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_gravity_wave_matches_cpu_golden_across_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping gravity wave parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuGravityWave::new(&ctx);

    // A deterministic phase/offset grid. Phases run past several cycles (both
    // signs) to exercise the wrap-to-[-PI, PI] reduction, and offsets span a
    // full turn.
    let mut queries: Vec<GravityWaveQuery> = Vec::new();
    for pi in -30..=30 {
        for xi in -12..=12 {
            queries.push(GravityWaveQuery {
                phase: pi as f32 / 10.0,
                x: xi as f32 * (PI / 12.0),
            });
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_gravity_wave_is_periodic_in_phase() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuGravityWave::new(&ctx);

    // The base cycle and the same cycle shifted by whole phases must agree:
    // gravity_wave is periodic with period 1 in phase.
    let base: Vec<GravityWaveQuery> = (0..=40)
        .map(|k| GravityWaveQuery {
            phase: k as f32 / 40.0,
            x: 0.2,
        })
        .collect();
    let shifted: Vec<GravityWaveQuery> = base
        .iter()
        .map(|q| GravityWaveQuery {
            phase: q.phase + 5.0,
            x: q.x,
        })
        .collect();

    let gpu_base = gpu_kernel.eval(&ctx, &base);
    let gpu_shifted = gpu_kernel.eval(&ctx, &shifted);
    assert_parity(&base, &gpu_base);
    assert_parity(&shifted, &gpu_shifted);
    for (a, b) in gpu_base.iter().zip(gpu_shifted.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "gravity wave must be periodic in phase: {a} then {b}"
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuGravityWave::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
