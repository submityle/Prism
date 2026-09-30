//! Real-device parity for the storm vertical-profile twin:
//! [`GpuStormVerticalProfile`] must reproduce the `CPU` golden
//! [`StormState::vertical_profile`](prism_render_architecture::volumetric::storm::StormState::vertical_profile)
//! across the full height/driver grid, including heights above the band top and
//! saturating out-of-range drivers.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `lerp`/`smoothstep` are expanded to the same closed forms and the dome's
//! exponential uses the same hand-rolled `exp_approx` the CPU golden uses (not
//! the device-native `exp`), so `CPU` and `GPU` evaluate the same closed-form
//! algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! wrong OR fold, the native `exp`). The scenes also assert the profile stays in
//! `0..=1` and is monotonically non-decreasing in each of the three driving
//! fields, so a degenerate kernel could not pass.
//!
//! Provenance: standard cumulonimbus vertical-development composite; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::storm::StormState;
use prism_volumetric_gpu::{GpuContext, GpuStormVerticalProfile, StormVerticalProfileQuery};

/// Evaluates the `CPU` golden for one query by folding the drivers into a
/// `StormState` and calling its `vertical_profile` seam.
fn cpu_golden(q: &StormVerticalProfileQuery) -> f32 {
    StormState {
        anvil_spread: q.anvil_spread,
        overshooting_top: q.overshooting_top,
        pyrocumulus: q.pyrocumulus,
        ..StormState::default()
    }
    .vertical_profile(q.height_fraction)
}

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[StormVerticalProfileQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = cpu_golden(q);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "storm vertical profile mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu storm vertical profile must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_storm_vertical_profile_matches_cpu_golden_across_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping storm vertical profile parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuStormVerticalProfile::new(&ctx);

    // A deterministic height/driver grid. Heights run past 1 so the dome can
    // bulge above the band top; the three drivers sweep their unit range.
    let mut queries: Vec<StormVerticalProfileQuery> = Vec::new();
    for hi in 0..=24 {
        for ai in 0..=4 {
            for oi in 0..=4 {
                for pi in 0..=4 {
                    queries.push(StormVerticalProfileQuery {
                        height_fraction: hi as f32 / 20.0,
                        anvil_spread: ai as f32 / 4.0,
                        overshooting_top: oi as f32 / 4.0,
                        pyrocumulus: pi as f32 / 4.0,
                    });
                }
            }
        }
    }
    // Out-of-range inputs that must saturate rather than escape 0..=1.
    queries.push(StormVerticalProfileQuery {
        height_fraction: -0.5,
        anvil_spread: 1.5,
        overshooting_top: 1.5,
        pyrocumulus: 1.5,
    });
    queries.push(StormVerticalProfileQuery {
        height_fraction: 2.5,
        anvil_spread: -0.3,
        overshooting_top: -0.3,
        pyrocumulus: -0.3,
    });

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_storm_vertical_profile_is_monotone_in_each_driver() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuStormVerticalProfile::new(&ctx);

    // At a fixed high band a sweep of each driver (others held mid-range) must
    // be monotonically non-decreasing: a maturing storm only ever adds
    // vertical development.
    let height = 0.95_f32;
    let base = 0.5_f32;

    let anvil_sweep: Vec<StormVerticalProfileQuery> = (0..=40)
        .map(|s| StormVerticalProfileQuery {
            height_fraction: height,
            anvil_spread: s as f32 / 40.0,
            overshooting_top: base,
            pyrocumulus: base,
        })
        .collect();
    let overshoot_sweep: Vec<StormVerticalProfileQuery> = (0..=40)
        .map(|s| StormVerticalProfileQuery {
            height_fraction: height,
            anvil_spread: base,
            overshooting_top: s as f32 / 40.0,
            pyrocumulus: base,
        })
        .collect();
    let pyro_sweep: Vec<StormVerticalProfileQuery> = (0..=40)
        .map(|s| StormVerticalProfileQuery {
            height_fraction: height,
            anvil_spread: base,
            overshooting_top: base,
            pyrocumulus: s as f32 / 40.0,
        })
        .collect();

    for sweep in [&anvil_sweep, &overshoot_sweep, &pyro_sweep] {
        let gpu = gpu_kernel.eval(&ctx, sweep);
        assert_parity(sweep, &gpu);
        let mut prev = f32::NEG_INFINITY;
        for &v in &gpu {
            assert!(
                v >= prev - 1e-6,
                "vertical profile must not decrease as a driver rises: {prev} then {v}"
            );
            prev = v;
        }
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuStormVerticalProfile::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
