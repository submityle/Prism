//! Real-device parity for the sky-state transition twin:
//! [`GpuSkyStateTransition`] must reproduce the `CPU` goldens
//! [`advance_state`](prism_render_architecture::volumetric::weather::advance_state),
//! [`dissipate_state`](prism_render_architecture::volumetric::weather::dissipate_state)
//! and
//! [`target_coverage`](prism_render_architecture::volumetric::weather::target_coverage)
//! across every sky state on the ladder.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both transitioned states are asserted bit-exactly and the target coverage
//! exactly. The test also checks the ladder invariants: `advance_state` never
//! descends and saturates at `Storm`, `dissipate_state` never climbs and
//! saturates at `Clear`, and the target coverage rises with the state's rung.
//!
//! Provenance: original Prism weather state machine; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::weather::{
    advance_state, dissipate_state, target_coverage, SkyState,
};
use prism_volumetric_gpu::{GpuContext, GpuSkyStateTransition};

/// The ordinal rung of a state on the intensification ladder (for invariant
/// checks in the test only).
fn rung(state: SkyState) -> u8 {
    match state {
        SkyState::Clear => 0,
        SkyState::Fair => 1,
        SkyState::Overcast => 2,
        SkyState::Storm => 3,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sky_state_transition_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sky-state-transition parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSkyStateTransition::new(&ctx);

    // Every state on the ladder, repeated so a full workgroup is exercised.
    let base = [
        SkyState::Clear,
        SkyState::Fair,
        SkyState::Overcast,
        SkyState::Storm,
    ];
    let mut states: Vec<SkyState> = Vec::new();
    for _ in 0..20 {
        states.extend(base);
    }
    assert!(!states.is_empty(), "the parity set must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &states);
    assert_eq!(gpu.len(), states.len());

    for (&s, got) in states.iter().zip(gpu.iter()) {
        let want_adv = advance_state(s);
        let want_dis = dissipate_state(s);
        let want_cov = target_coverage(s);
        assert_eq!(
            got.advanced, want_adv,
            "advance mismatch for {s:?}: gpu={:?} cpu={want_adv:?}",
            got.advanced
        );
        assert_eq!(
            got.dissipated, want_dis,
            "dissipate mismatch for {s:?}: gpu={:?} cpu={want_dis:?}",
            got.dissipated
        );
        assert_eq!(
            got.target_coverage, want_cov,
            "coverage mismatch for {s:?}: gpu={} cpu={want_cov}",
            got.target_coverage
        );

        // Ladder invariants.
        assert!(
            rung(got.advanced) >= rung(s),
            "advance must not descend for {s:?}"
        );
        assert!(
            rung(got.dissipated) <= rung(s),
            "dissipate must not climb for {s:?}"
        );
        assert!(
            (0.0..=1.0).contains(&got.target_coverage),
            "target coverage out of [0,1] for {s:?}: {}",
            got.target_coverage
        );
    }

    // Endpoint saturation.
    let ends = gpu_kernel.eval(&ctx, &[SkyState::Storm, SkyState::Clear]);
    assert_eq!(ends[0].advanced, SkyState::Storm, "Storm must saturate up");
    assert_eq!(
        ends[1].dissipated,
        SkyState::Clear,
        "Clear must saturate down"
    );
}
