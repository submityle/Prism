//! Real-device parity for the early-termination twin:
//! [`GpuShouldEarlyTerminate`] must reproduce the `CPU` golden
//! [`should_early_terminate`](prism_render_architecture::volumetric::raymarch::should_early_terminate)
//! across a deterministic grid of transmittances and config cutoffs, including
//! the boundary where `transmittance == transmittance_cutoff` (strict `<`).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The predicate is a single float comparison, so every decision is asserted to
//! match the `CPU` golden bit for bit, including the boundary case.
//!
//! Provenance: standard transmittance-cutoff early termination; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::raymarch::{should_early_terminate, RaymarchConfig};
use prism_volumetric_gpu::{GpuContext, GpuShouldEarlyTerminate, ShouldEarlyTerminateQuery};

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_should_early_terminate_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping early-termination parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuShouldEarlyTerminate::new(&ctx);

    // Configs with distinct cutoffs, including the default.
    let cfgs = [
        RaymarchConfig::default(),
        RaymarchConfig {
            transmittance_cutoff: 0.05,
            ..RaymarchConfig::default()
        },
        RaymarchConfig {
            transmittance_cutoff: 0.5,
            ..RaymarchConfig::default()
        },
    ];

    // A transmittance sweep that straddles each cutoff exactly (boundary), just
    // below and just above, plus the full 0..=1 endpoints.
    let transmittances = [
        0.0_f32, 0.009, 0.01, 0.011, 0.049, 0.05, 0.051, 0.25, 0.499, 0.5, 0.501, 0.75, 1.0,
    ];

    let mut queries: Vec<ShouldEarlyTerminateQuery> = Vec::new();
    for &cfg in &cfgs {
        for &transmittance in &transmittances {
            queries.push(ShouldEarlyTerminateQuery { transmittance, cfg });
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");

    for (i, q) in queries.iter().enumerate() {
        let exp = should_early_terminate(q.transmittance, q.cfg);
        assert_eq!(
            gpu[i], exp,
            "early-termination mismatch for query {i} (transmittance {}, cutoff {}): \
             gpu {}, cpu {exp}",
            q.transmittance, q.cfg.transmittance_cutoff, gpu[i]
        );
    }

    // Boundary: transmittance exactly equal to the cutoff must NOT terminate
    // (strict `<`).
    let boundary = ShouldEarlyTerminateQuery {
        transmittance: 0.5,
        cfg: RaymarchConfig {
            transmittance_cutoff: 0.5,
            ..RaymarchConfig::default()
        },
    };
    let out = gpu_kernel.eval(&ctx, &[boundary]);
    assert!(
        !out[0],
        "transmittance == cutoff must not terminate (strict <)"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuShouldEarlyTerminate::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
