//! Real-device parity for the storm virga-veil twin: [`GpuVirgaVeil`] must
//! reproduce the `CPU` golden
//! [`StormState::virga_veil`](prism_render_architecture::volumetric::storm::StormState::virga_veil)
//! across the full veil-strength/veil-fraction grid, including saturating
//! out-of-range inputs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `smoothstep` is expanded to the same closed form the CPU golden uses and
//! the kernel has no transcendental call, so `CPU` and `GPU` evaluate the same
//! algebra. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — tight enough to fail a wrong port (a dropped saturate, a
//! missing veil scale). The scenes also assert the veil stays in `0..=1` and is
//! monotonically non-decreasing in both the veil strength and the veil
//! fraction, so a degenerate kernel could not pass.
//!
//! Provenance: standard virga precipitation-veil falloff; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::storm::StormState;
use prism_volumetric_gpu::{GpuContext, GpuVirgaVeil, VirgaVeilQuery};

/// Evaluates the `CPU` golden for one query by folding the veil strength into a
/// `StormState` and calling its `virga_veil` seam.
fn cpu_golden(q: &VirgaVeilQuery) -> f32 {
    StormState {
        virga: q.virga,
        ..StormState::default()
    }
    .virga_veil(q.veil_fraction)
}

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays in `0..=1`.
fn assert_parity(queries: &[VirgaVeilQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = cpu_golden(q);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "virga veil mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu virga veil must stay in 0..=1: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_virga_veil_matches_cpu_golden_across_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping virga veil parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuVirgaVeil::new(&ctx);

    // A deterministic veil-strength/veil-fraction grid.
    let mut queries: Vec<VirgaVeilQuery> = Vec::new();
    for vi in 0..=40 {
        for fi in 0..=40 {
            queries.push(VirgaVeilQuery {
                virga: vi as f32 / 40.0,
                veil_fraction: fi as f32 / 40.0,
            });
        }
    }
    // Out-of-range inputs that must saturate rather than escape 0..=1.
    queries.push(VirgaVeilQuery {
        virga: 1.5,
        veil_fraction: 1.5,
    });
    queries.push(VirgaVeilQuery {
        virga: -0.3,
        veil_fraction: -0.3,
    });

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn gpu_virga_veil_is_monotone_in_both_drivers() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuVirgaVeil::new(&ctx);

    // A veil-fraction sweep at full strength must rise monotonically from the
    // trailing tip toward the cloud base.
    let fraction_sweep: Vec<VirgaVeilQuery> = (0..=40)
        .map(|f| VirgaVeilQuery {
            virga: 1.0,
            veil_fraction: f as f32 / 40.0,
        })
        .collect();
    // A veil-strength sweep at a fixed mid height must also rise monotonically.
    let strength_sweep: Vec<VirgaVeilQuery> = (0..=40)
        .map(|v| VirgaVeilQuery {
            virga: v as f32 / 40.0,
            veil_fraction: 0.5,
        })
        .collect();

    for sweep in [&fraction_sweep, &strength_sweep] {
        let gpu = gpu_kernel.eval(&ctx, sweep);
        assert_parity(sweep, &gpu);
        let mut prev = f32::NEG_INFINITY;
        for &v in &gpu {
            assert!(
                v >= prev - 1e-6,
                "virga veil must not decrease as a driver rises: {prev} then {v}"
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
    let gpu_kernel = GpuVirgaVeil::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
