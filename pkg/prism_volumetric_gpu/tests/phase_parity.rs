//! Real-device parity for the dual-lobe cloud-phase twin: [`GpuPhaseEvaluator`]
//! must reproduce the `CPU` golden
//! [`dual_lobe_draine_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_draine_phase)
//! for every query across a scattering-angle sweep and several anisotropy /
//! `Draine` / blend variants.
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
//! Provenance: standard Henyey-Greenstein / Draine dual-lobe cloud phase; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::scatter::{
    dual_lobe_draine_phase, isotropic_phase, DEFAULT_BACKWARD_G, DEFAULT_DRAINE_ALPHA,
    DEFAULT_FORWARD_G, DEFAULT_LOBE_BLEND,
};
use prism_volumetric_gpu::{GpuContext, GpuPhaseEvaluator, PhaseQuery};

/// A dual-lobe phase query built from the cloud defaults at scattering cosine
/// `cos_theta`, with an explicit `Draine` weight.
fn default_query(cos_theta: f32, draine_weight: f32) -> PhaseQuery {
    PhaseQuery {
        cos_theta,
        g_forward: DEFAULT_FORWARD_G,
        g_backward: DEFAULT_BACKWARD_G,
        alpha: DEFAULT_DRAINE_ALPHA,
        draine_weight,
        blend: DEFAULT_LOBE_BLEND,
    }
}

/// Asserts `gpu` matches the `CPU` golden for every query to within the
/// documented fma tolerance, returning the `CPU` reference values for further
/// physical-shape assertions.
fn assert_parity(queries: &[PhaseQuery], gpu: &[f32]) -> Vec<f32> {
    assert_eq!(gpu.len(), queries.len(), "one phase value per query");
    let mut cpu = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let expected = dual_lobe_draine_phase(
            q.cos_theta,
            q.g_forward,
            q.g_backward,
            q.alpha,
            q.draine_weight,
            q.blend,
        );
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
fn gpu_phase_matches_cpu_golden_across_angle_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping phase parity: no wgpu adapter on this host");
        return;
    };
    let evaluator = GpuPhaseEvaluator::new(&ctx);

    // Sweep the scattering cosine from pure back-scatter to pure forward-scatter
    // with the cloud defaults; a Draine weight of 0.5 exercises both the HG and
    // Draine terms of the forward lobe.
    let queries: Vec<PhaseQuery> = (0..=40)
        .map(|k| {
            let cos_theta = -1.0 + (k as f32) * 0.05;
            default_query(cos_theta, 0.5)
        })
        .collect();

    let gpu = evaluator.eval(&ctx, &queries);
    let cpu = assert_parity(&queries, &gpu);

    // Physical shape: the last sample (cos_theta = +1, pure forward) must be the
    // sharp silver-lining peak and sit well above the isotropic reference, while
    // the first sample (cos_theta = -1, pure back) is the softer backward lobe.
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
fn gpu_phase_matches_cpu_golden_across_parameter_variants() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let evaluator = GpuPhaseEvaluator::new(&ctx);

    // A spread of anisotropy, Draine shape, weight and blend values at several
    // scattering angles, including out-of-range arguments the reference clamps
    // (g beyond MAX_ABS_G, negative alpha, blend/draine_weight outside [0, 1]).
    let queries: Vec<PhaseQuery> = vec![
        default_query(0.0, 0.0),
        default_query(0.0, 1.0),
        default_query(0.7, 0.25),
        default_query(-0.5, 0.75),
        PhaseQuery {
            cos_theta: 0.9,
            g_forward: 0.95,
            g_backward: -0.6,
            alpha: 3.0,
            draine_weight: 0.9,
            blend: 0.8,
        },
        PhaseQuery {
            cos_theta: -0.2,
            g_forward: 0.5,
            g_backward: -0.1,
            alpha: 0.0,
            draine_weight: 0.3,
            blend: 0.4,
        },
        // Out-of-range: clamps must be reproduced identically on both sides.
        PhaseQuery {
            cos_theta: 1.5,
            g_forward: 1.4,
            g_backward: -1.4,
            alpha: -2.0,
            draine_weight: 1.7,
            blend: -0.3,
        },
    ];

    let gpu = evaluator.eval(&ctx, &queries);
    let _ = assert_parity(&queries, &gpu);
}

#[test]
fn empty_queries_yields_no_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let evaluator = GpuPhaseEvaluator::new(&ctx);
    let out = evaluator.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
