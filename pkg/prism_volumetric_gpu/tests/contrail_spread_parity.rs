//! Real-device parity for the contrail-spread twin:
//! [`GpuContrailSpread`] must reproduce the `CPU` golden
//! [`contrail_spread`](prism_render_architecture::volumetric::fog::contrail_spread)
//! across a deterministic spread of ages, including negative ages that must be
//! floored at zero.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel is a single `max` plus a multiply-add using the same constants as
//! the CPU reference, so `CPU` and `GPU` evaluate the same closed-form algebra.
//! Values are asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight
//! enough to fail a wrong port (a dropped `max`, a swapped constant). The scenes
//! also assert the spread never drops below the base sigma and is monotone
//! non-decreasing in age (an older contrail always spreads at least as much), so
//! a degenerate kernel could not pass.
//!
//! Provenance: standard linear diffusion spread; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::fog::contrail_spread;
use prism_volumetric_gpu::{ContrailSpreadQuery, GpuContext, GpuContrailSpread};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance.
fn assert_parity(queries: &[ContrailSpreadQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one spread per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = contrail_spread(q.age);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "contrail spread mismatch for query {i} (age {}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})",
            q.age
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_contrail_spread_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping contrail-spread parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuContrailSpread::new(&ctx);

    // A deterministic sweep of ages, including negative ages that must floor at
    // zero and large ages that keep growing.
    let mut queries: Vec<ContrailSpreadQuery> = vec![
        ContrailSpreadQuery { age: -100.0 },
        ContrailSpreadQuery { age: -1.0 },
        ContrailSpreadQuery { age: 0.0 },
    ];
    for k in 0..=200 {
        let age = (k as f32) * 0.5 - 5.0;
        queries.push(ContrailSpreadQuery { age });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The spread never drops below the base sigma and is monotone non-decreasing
    // in age.
    let base = gpu[0];
    let mut prev = f32::NEG_INFINITY;
    let mut sorted: Vec<(f32, f32)> = queries
        .iter()
        .map(|q| q.age)
        .zip(gpu.iter().copied())
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite ages"));
    for (age, spread) in sorted {
        assert!(
            spread >= base - 1e-6,
            "spread {spread} at age {age} must not drop below the base sigma {base}"
        );
        assert!(
            spread >= prev - 1e-6,
            "spread must be monotone non-decreasing in age: {spread} at age {age} \
             after {prev}"
        );
        prev = spread;
    }

    // Negative ages all collapse to the same base sigma.
    assert!(
        (gpu[0] - gpu[1]).abs() < 1e-6 && (gpu[1] - gpu[2]).abs() < 1e-6,
        "all non-positive ages share the base sigma: {}, {}, {}",
        gpu[0],
        gpu[1],
        gpu[2]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuContrailSpread::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
