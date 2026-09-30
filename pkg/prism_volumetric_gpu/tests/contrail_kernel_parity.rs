//! Real-device parity for the contrail diffusion-kernel twin:
//! [`GpuContrailKernel`] must reproduce the `CPU` golden
//! [`contrail_kernel`](prism_render_architecture::volumetric::fog::contrail_kernel)
//! across a spread of cross-section offsets and trail ages / widths /
//! diffusion coefficients.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The exponential is evaluated with the *same* hand-rolled `exp_approx` the
//! reference uses, and the normalization constant uses the native `sqrt` both
//! sides share, so `CPU` and `GPU` evaluate the same closed-form algebra.
//! Values are asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight
//! enough to fail a wrong port (a dropped normalization, a swapped sigma). The
//! scenes also assert the kernel stays non-negative, peaks at `offset = 0`, is
//! symmetric in `offset`, and that a freshly formed trail is narrower (taller
//! peak) than an aged one, so a degenerate kernel could not pass.
//!
//! Provenance: standard normalized-Gaussian condensation kernel; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::fog::{contrail_kernel, Contrail};
use prism_volumetric_gpu::{ContrailKernelQuery, GpuContext, GpuContrailKernel};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays non-negative.
fn assert_parity(queries: &[ContrailKernelQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one kernel value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = contrail_kernel(
            q.offset,
            Contrail {
                age: q.age,
                width: q.width,
                diffusion: q.diffusion,
            },
        );
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "contrail kernel mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            got >= 0.0,
            "gpu contrail kernel must stay non-negative: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_contrail_kernel_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping contrail kernel parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuContrailKernel::new(&ctx);

    // A deterministic spread: on-axis for a fresh trail, off-axis for the same
    // trail, an aged/diffused trail, a degenerate zero-width fresh trail (sigma
    // floored at EPS), and negative trail inputs that must floor to zero.
    let mut queries: Vec<ContrailKernelQuery> = vec![
        ContrailKernelQuery {
            offset: 0.0,
            age: 0.0,
            width: 2.0,
            diffusion: 0.5,
        },
        ContrailKernelQuery {
            offset: 1.5,
            age: 0.0,
            width: 2.0,
            diffusion: 0.5,
        },
        ContrailKernelQuery {
            offset: 1.5,
            age: 10.0,
            width: 2.0,
            diffusion: 0.5,
        },
        ContrailKernelQuery {
            offset: 0.0,
            age: 0.0,
            width: 0.0,
            diffusion: 0.0,
        },
        ContrailKernelQuery {
            offset: 0.5,
            age: -3.0,
            width: -1.0,
            diffusion: -2.0,
        },
    ];
    // A deterministic offset sweep at a fixed trail, plus an age sweep at a
    // fixed offset — both exercise the exp mirror across a range of arguments.
    for k in 0..48 {
        queries.push(ContrailKernelQuery {
            offset: -6.0 + (k as f32) * 0.25,
            age: 4.0,
            width: 3.0,
            diffusion: 0.4,
        });
    }
    for k in 0..48 {
        queries.push(ContrailKernelQuery {
            offset: 1.0,
            age: (k as f32) * 0.5,
            width: 1.5,
            diffusion: 0.6,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // On-axis is the peak; the same offset off-axis is smaller; an aged trail
    // is broader so its off-axis value is closer to the (lower) peak but still
    // matches the golden — parity already covers that. Here we assert the two
    // shape facts a degenerate kernel would violate.
    assert!(
        gpu[0] > gpu[1],
        "on-axis value must exceed the off-axis value: {} vs {}",
        gpu[0],
        gpu[1]
    );
    // The degenerate zero-width fresh trail has sigma == EPS, so its on-axis
    // value is the very large 1/(EPS*sqrt(2*pi)); it must be finite and huge.
    assert!(
        gpu[3].is_finite() && gpu[3] > gpu[0],
        "degenerate trail peaks higher than a finite-width trail: {}",
        gpu[3]
    );
}

#[test]
fn gpu_contrail_kernel_decays_with_offset_and_is_symmetric() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuContrailKernel::new(&ctx);

    // At a fixed trail the kernel decays monotonically away from the axis. We
    // sweep non-negative offsets and, for each, also query its mirror so we can
    // assert symmetry.
    let (age, width, diffusion) = (5.0f32, 2.0f32, 0.5f32);
    let mut queries: Vec<ContrailKernelQuery> = Vec::new();
    for k in 0..=40 {
        let offset = (k as f32) * 0.2;
        queries.push(ContrailKernelQuery {
            offset,
            age,
            width,
            diffusion,
        });
        queries.push(ContrailKernelQuery {
            offset: -offset,
            age,
            width,
            diffusion,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Even indices are the +offset samples in increasing |offset| order.
    let mut prev = f32::INFINITY;
    for k in 0..=40 {
        let pos = gpu[2 * k];
        let neg = gpu[2 * k + 1];
        assert!(
            (pos - neg).abs() < 1e-6,
            "kernel must be symmetric in offset: {pos} vs {neg}"
        );
        assert!(
            pos <= prev + 1e-6,
            "kernel must decay monotonically with |offset|: {prev} then {pos}"
        );
        prev = pos;
    }
}

#[test]
fn gpu_contrail_kernel_peak_shrinks_as_the_trail_ages() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuContrailKernel::new(&ctx);

    // On-axis, a widening (aging) trail has a larger sigma and therefore a
    // lower normalized peak — mass is conserved, so spreading lowers the peak.
    let (width, diffusion) = (2.0f32, 0.5f32);
    let queries: Vec<ContrailKernelQuery> = (0..=40)
        .map(|k| ContrailKernelQuery {
            offset: 0.0,
            age: (k as f32) * 0.5,
            width,
            diffusion,
        })
        .collect();

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "on-axis peak must not grow as the trail ages: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        gpu[gpu.len() - 1] < gpu[0],
        "an aged trail has a strictly lower on-axis peak than a fresh one"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuContrailKernel::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
