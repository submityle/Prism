//! Real-device parity for the pyrocumulus-buoyancy twin:
//! [`GpuPyrocumulusBuoyancy`] must reproduce the `CPU` golden
//! [`pyrocumulus_buoyancy`](prism_render_architecture::volumetric::storm::pyrocumulus_buoyancy)
//! across the full heat range, including negative heat that must floor at zero.
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
//! device-native `exp`, and the gain is the same `PYRO_BUOYANCY_GAIN`, so `CPU`
//! and `GPU` evaluate the same closed-form algebra. Values are asserted to
//! within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong
//! port (a dropped saturate, a wrong gain, the native `exp`). The scenes also
//! assert the value stays in `0..=1`, rises monotonically with heat and floors
//! negative heat at zero, so a degenerate kernel could not pass.
//!
//! Provenance: standard fire-fed pyrocumulus buoyancy ramp; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::storm::pyrocumulus_buoyancy;
use prism_volumetric_gpu::{GpuContext, GpuPyrocumulusBuoyancy, PyrocumulusBuoyancyQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[PyrocumulusBuoyancyQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = pyrocumulus_buoyancy(q.heat);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "pyrocumulus buoyancy mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu pyrocumulus buoyancy must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_pyrocumulus_buoyancy_matches_cpu_golden_across_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pyrocumulus buoyancy parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuPyrocumulusBuoyancy::new(&ctx);

    // A deterministic heat sweep from negative (must floor at zero) through the
    // saturating tail.
    let mut queries: Vec<PyrocumulusBuoyancyQuery> = Vec::new();
    for hi in -10..=80 {
        queries.push(PyrocumulusBuoyancyQuery {
            heat: hi as f32 / 20.0,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_pyrocumulus_buoyancy_rises_with_heat_and_floors_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuPyrocumulusBuoyancy::new(&ctx);

    // Non-positive heat must all yield zero lift.
    let floored: Vec<PyrocumulusBuoyancyQuery> = (-20..=0)
        .map(|h| PyrocumulusBuoyancyQuery {
            heat: h as f32 / 10.0,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &floored);
    assert_parity(&floored, &gpu);
    for &v in &gpu {
        assert!(v.abs() < 1e-6, "non-positive heat must floor at zero: {v}");
    }

    // Positive heat rises monotonically with diminishing returns.
    let sweep: Vec<PyrocumulusBuoyancyQuery> = (0..=60)
        .map(|h| PyrocumulusBuoyancyQuery {
            heat: h as f32 / 20.0,
        })
        .collect();
    let gpu = gpu_kernel.eval(&ctx, &sweep);
    assert_parity(&sweep, &gpu);
    let mut prev = f32::NEG_INFINITY;
    for &v in &gpu {
        assert!(
            v >= prev - 1e-6,
            "buoyancy must rise with heat: {prev} then {v}"
        );
        prev = v;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuPyrocumulusBuoyancy::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
