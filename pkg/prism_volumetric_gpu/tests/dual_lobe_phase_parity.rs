//! Real-device parity for the non-`Draine` dual-lobe cloud-phase twin:
//! [`GpuDualLobePhase`] must reproduce the `CPU` golden
//! [`dual_lobe_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_phase)
//! for every query across a scattering-angle sweep and several anisotropy /
//! blend variants.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The phase functions contain no transcendental call, so the `CPU` and `GPU`
//! evaluate the same closed-form algebra and diverge only through legal
//! fused-multiply-add contraction. Each value is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped lobe, a missing normalization factor, a sign error in
//! the anisotropy), loose enough to admit fma contraction. The sweep also
//! asserts the physical shape (a forward peak well above isotropic and a softer
//! backward lobe) so a degenerate all-constant kernel could not pass.
//!
//! Provenance: standard Henyey-Greenstein dual-lobe cloud phase; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::scatter::{
    dual_lobe_phase, isotropic_phase, DEFAULT_BACKWARD_G, DEFAULT_FORWARD_G, DEFAULT_LOBE_BLEND,
};
use prism_volumetric_gpu::{DualLobePhaseQuery, GpuContext, GpuDualLobePhase};

/// A non-`Draine` dual-lobe phase query built from the cloud defaults at
/// scattering cosine `cos_theta`.
fn default_query(cos_theta: f32) -> DualLobePhaseQuery {
    DualLobePhaseQuery {
        cos_theta,
        g_forward: DEFAULT_FORWARD_G,
        g_backward: DEFAULT_BACKWARD_G,
        blend: DEFAULT_LOBE_BLEND,
    }
}

/// Asserts `gpu` matches the `CPU` golden for every query to within the
/// documented fma tolerance, returning the `CPU` reference values for further
/// physical-shape assertions.
fn assert_parity(queries: &[DualLobePhaseQuery], gpu: &[f32]) -> Vec<f32> {
    assert_eq!(gpu.len(), queries.len(), "one phase value per query");
    let mut cpu = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let expected = dual_lobe_phase(q.cos_theta, q.g_forward, q.g_backward, q.blend);
        let got = gpu[i];
        let abs_diff = (got - expected).abs();
        let rel_diff = abs_diff / expected.abs().max(1e-6);
        assert!(
            abs_diff < 1e-4 || rel_diff < 1e-3,
            "phase mismatch for query {q:?}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
        );
        cpu.push(expected);
    }
    cpu
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_dual_lobe_phase_matches_cpu_golden_across_angle_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dual-lobe phase parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuDualLobePhase::new(&ctx);

    // Sweep the scattering cosine from pure back-scatter to pure forward-scatter
    // with the cloud defaults.
    let queries: Vec<DualLobePhaseQuery> = (0..=40)
        .map(|k| {
            let cos_theta = -1.0 + (k as f32) * 0.05;
            default_query(cos_theta)
        })
        .collect();

    let gpu = evaluator.eval(&ctx, &queries);
    let cpu = assert_parity(&queries, &gpu);

    // Physical shape: the last sample (cos_theta = +1, pure forward) must be the
    // sharp forward peak and sit well above the isotropic reference, while the
    // first sample (cos_theta = -1, pure back) is the softer backward lobe.
    let iso = isotropic_phase();
    let forward_peak = *cpu.last().expect("non-empty sweep");
    let back = cpu[0];
    assert!(
        forward_peak > iso * 4.0,
        "forward peak {forward_peak} should tower over isotropic {iso}"
    );
    assert!(
        forward_peak > back,
        "forward peak {forward_peak} should exceed the backward lobe {back}"
    );
    assert!(back > 0.0, "backward lobe {back} must stay positive");
    // The GPU forward peak itself (not just the CPU reference) must clear
    // isotropic, proving the running kernel produced a real anisotropic phase.
    assert!(
        gpu.last().copied().expect("non-empty sweep") > iso * 4.0,
        "gpu forward peak must also tower over isotropic"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_dual_lobe_phase_matches_cpu_golden_across_parameter_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dual-lobe phase parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuDualLobePhase::new(&ctx);

    // A grid over the four inputs, including edge anisotropies (clamped by both
    // lobes), a zero-g isotropic lobe, blend endpoints selecting each pure lobe,
    // and out-of-range cos_theta / blend / g values that must saturate exactly
    // as the CPU golden clamps them.
    let cosines = [-1.5f32, -1.0, -0.5, 0.0, 0.3, 0.85, 1.0, 1.5];
    let forwards = [0.0f32, 0.3, 0.8, 0.999, 1.2];
    let backwards = [-1.2f32, -0.999, -0.3, 0.0];
    let blends = [-0.25f32, 0.0, 0.25, 0.6, 1.0, 1.4];

    let mut queries = Vec::new();
    for &cos_theta in &cosines {
        for &g_forward in &forwards {
            for &g_backward in &backwards {
                for &blend in &blends {
                    queries.push(DualLobePhaseQuery {
                        cos_theta,
                        g_forward,
                        g_backward,
                        blend,
                    });
                }
            }
        }
    }

    let gpu = evaluator.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_dual_lobe_phase_handles_empty_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dual-lobe phase parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuDualLobePhase::new(&ctx);
    assert!(evaluator.eval(&ctx, &[]).is_empty(), "empty in, empty out");
}
