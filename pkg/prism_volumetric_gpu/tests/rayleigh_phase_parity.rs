//! Real-device parity for the Rayleigh phase-function twin:
//! [`GpuRayleighPhase`] must reproduce the `CPU` golden
//! [`rayleigh_phase`](prism_render_architecture::volumetric::spectral::rayleigh_phase)
//! across the full range of scattering-angle cosines, including out-of-range
//! inputs that must clamp.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The normalization constant is folded the same way the reference folds it,
//! and the kernel contains no transcendental call — a clamp and a multiply-add
//! — so `CPU` and `GPU` evaluate the same closed-form algebra. Values are
//! asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to
//! fail a wrong port (a dropped clamp, a wrong normalization). The scenes also
//! assert the value stays non-negative, is symmetric in `cos_theta`, peaks at
//! the forward/backward directions and troughs at right angles (a `2:1` ratio),
//! and that out-of-range inputs clamp, so a degenerate kernel could not pass.
//!
//! Provenance: standard Rayleigh scattering phase function; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::spectral::rayleigh_phase;
use prism_volumetric_gpu::{GpuContext, GpuRayleighPhase, RayleighPhaseQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays non-negative.
fn assert_parity(queries: &[RayleighPhaseQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one phase value per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = rayleigh_phase(q.cos_theta);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "rayleigh phase mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            got >= 0.0,
            "gpu rayleigh phase must stay non-negative: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_rayleigh_phase_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping rayleigh phase parity: no wgpu adapter on this host");
        return;
    };
    let gpu_phase = GpuRayleighPhase::new(&ctx);

    // A deterministic spread: forward scatter, right angle, back scatter, and
    // out-of-range cosines that must clamp to the endpoints.
    let mut queries: Vec<RayleighPhaseQuery> = vec![
        RayleighPhaseQuery { cos_theta: 1.0 },
        RayleighPhaseQuery { cos_theta: 0.0 },
        RayleighPhaseQuery { cos_theta: -1.0 },
        RayleighPhaseQuery { cos_theta: 2.5 },
        RayleighPhaseQuery { cos_theta: -2.5 },
    ];
    // A deterministic cosine sweep across the full valid domain.
    for k in 0..=100 {
        queries.push(RayleighPhaseQuery {
            cos_theta: -1.0 + (k as f32) / 50.0,
        });
    }

    let gpu = gpu_phase.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Forward and back scatter are equal peaks; the right angle is the trough
    // at exactly half the peak; the out-of-range cosines clamp to the peaks.
    assert!(
        (gpu[0] - gpu[2]).abs() < 1e-6,
        "forward and back scatter must be equal: {} vs {}",
        gpu[0],
        gpu[2]
    );
    assert!(
        gpu[0] > gpu[1],
        "the forward peak must exceed the right-angle trough: {} vs {}",
        gpu[0],
        gpu[1]
    );
    assert!(
        (gpu[0] - 2.0 * gpu[1]).abs() < 1e-6,
        "the peak is exactly twice the trough: {} vs {}",
        gpu[0],
        gpu[1]
    );
    assert!(
        (gpu[3] - gpu[0]).abs() < 1e-6,
        "a cosine above one clamps to the forward peak: {}",
        gpu[3]
    );
    assert!(
        (gpu[4] - gpu[2]).abs() < 1e-6,
        "a cosine below minus one clamps to the back peak: {}",
        gpu[4]
    );
}

#[test]
fn gpu_rayleigh_phase_is_symmetric_in_cos_theta() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_phase = GpuRayleighPhase::new(&ctx);

    // Each positive cosine is paired with its negation; the phase must be equal
    // for the pair and rise monotonically as |cos_theta| grows.
    let mut queries: Vec<RayleighPhaseQuery> = Vec::new();
    for k in 0..=50 {
        let c = (k as f32) / 50.0;
        queries.push(RayleighPhaseQuery { cos_theta: c });
        queries.push(RayleighPhaseQuery { cos_theta: -c });
    }

    let gpu = gpu_phase.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    let mut prev = f32::NEG_INFINITY;
    for k in 0..=50 {
        let pos = gpu[2 * k];
        let neg = gpu[2 * k + 1];
        assert!(
            (pos - neg).abs() < 1e-6,
            "phase must be symmetric in cos_theta: {pos} vs {neg}"
        );
        assert!(
            pos >= prev - 1e-6,
            "phase must rise monotonically with |cos_theta|: {prev} then {pos}"
        );
        prev = pos;
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_phase = GpuRayleighPhase::new(&ctx);
    let out = gpu_phase.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
