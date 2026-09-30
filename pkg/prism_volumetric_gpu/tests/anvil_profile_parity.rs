//! Real-device parity for the anvil-profile twin: [`GpuAnvilProfile`] must
//! reproduce the `CPU` golden
//! [`anvil_profile`](prism_render_architecture::volumetric::storm::anvil_profile)
//! across the full height/spread grid, including saturating out-of-range
//! inputs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `lerp` and `smoothstep` are expanded to the same closed forms the CPU
//! `math` module uses, and the kernel contains no transcendental call —
//! saturate and a multiply-add — so `CPU` and `GPU` evaluate the same
//! closed-form algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! wrong base). The scenes also assert the value stays in `0..=1` and rises
//! monotonically with both height and spread, so a degenerate kernel could not
//! pass.
//!
//! Provenance: standard cumulonimbus anvil-spread profile; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::storm::anvil_profile;
use prism_volumetric_gpu::{AnvilProfileQuery, GpuAnvilProfile, GpuContext};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[AnvilProfileQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = anvil_profile(q.height_fraction, q.spread);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "anvil profile mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu anvil profile must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_anvil_profile_matches_cpu_golden_across_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping anvil profile parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuAnvilProfile::new(&ctx);

    // A deterministic height/spread grid, plus out-of-range inputs that must
    // saturate into 0..=1.
    let mut queries: Vec<AnvilProfileQuery> = Vec::new();
    for hi in 0..=20 {
        for si in 0..=20 {
            queries.push(AnvilProfileQuery {
                height_fraction: hi as f32 / 20.0,
                spread: si as f32 / 20.0,
            });
        }
    }
    queries.push(AnvilProfileQuery {
        height_fraction: -0.5,
        spread: 0.7,
    });
    queries.push(AnvilProfileQuery {
        height_fraction: 1.5,
        spread: 1.5,
    });

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_anvil_profile_rises_with_height_and_spread() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuAnvilProfile::new(&ctx);

    // At a fixed high altitude, a larger spread yields at least as much anvil.
    let spread_sweep: Vec<AnvilProfileQuery> = (0..=40)
        .map(|s| AnvilProfileQuery {
            height_fraction: 0.95,
            spread: s as f32 / 40.0,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &spread_sweep);
    assert_parity(&spread_sweep, &gpu);
    let mut prev = f32::NEG_INFINITY;
    for &v in &gpu {
        assert!(
            v >= prev - 1e-6,
            "anvil weight must rise with spread: {prev} then {v}"
        );
        prev = v;
    }

    // At a fixed mature spread, a greater height yields at least as much anvil.
    let height_sweep: Vec<AnvilProfileQuery> = (0..=40)
        .map(|h| AnvilProfileQuery {
            height_fraction: h as f32 / 40.0,
            spread: 0.8,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &height_sweep);
    assert_parity(&height_sweep, &gpu);
    let mut prev = f32::NEG_INFINITY;
    for &v in &gpu {
        assert!(
            v >= prev - 1e-6,
            "anvil weight must rise with height: {prev} then {v}"
        );
        prev = v;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuAnvilProfile::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
