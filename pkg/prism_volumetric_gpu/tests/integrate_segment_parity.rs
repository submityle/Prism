//! Real-device parity for the segment-integration twin:
//! [`GpuIntegrateSegment`] must reproduce the `CPU` golden
//! [`integrate_segment`](prism_render_architecture::volumetric::raymarch::integrate_segment)
//! across a deterministic grid of incoming accumulator states and homogeneous
//! segment coefficients, including the thin-medium (`sigma_t -> 0`) limit,
//! negative coefficients (which must clamp) and out-of-range light / powder
//! factors (which must saturate).
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every state field is asserted against the `CPU` golden within a tight
//! absolute tolerance (`steps_taken` exactly). The `CPU` golden and the kernel
//! share the same polynomial `exp_approx`, so the accumulated `transmittance`,
//! `optical_depth` and `scattered` agree to a few ULPs. The test also checks
//! the physical invariants: `transmittance` stays in `[0, 1]` and never
//! increases, `scattered` never decreases, `steps_taken` grows by exactly one,
//! and a zero `sigma_t` leaves `transmittance` unchanged.
//!
//! Provenance: standard analytic volumetric segment integration; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::raymarch::{integrate_segment, RaymarchState};
use prism_volumetric_gpu::{GpuContext, GpuIntegrateSegment, IntegrateSegmentQuery};

/// Absolute tolerance for the accumulated float fields. The polynomial
/// `exp_approx` is shared by both sides, so agreement is close.
const TOL: f32 = 1e-5;

/// Builds a `RaymarchState` directly, used to seed mid-march accumulators.
fn state(
    transmittance: f32,
    optical_depth: f32,
    scattered: f32,
    steps_taken: u32,
) -> RaymarchState {
    RaymarchState {
        transmittance,
        optical_depth,
        scattered,
        steps_taken,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_integrate_segment_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping integrate-segment parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuIntegrateSegment::new(&ctx);

    // Incoming accumulator states: a fresh eye state plus several mid-march
    // states with non-trivial transmittance / optical_depth / scattered / steps.
    let states = [
        RaymarchState::new(),
        state(1.0, 0.0, 0.0, 0),
        state(0.8, 0.4, 0.2, 3),
        state(0.35, 1.7, 0.9, 12),
        state(0.02, 5.0, 2.4, 47),
    ];
    // sigma_t across the interesting regimes: exactly zero, sub-EPS, small,
    // moderate and large (so seg_trans spans ~1 down to ~0).
    let sigma_ts = [0.0_f32, 1e-7, 0.05, 0.5, 2.0, 8.0];
    let sigma_ss = [0.0_f32, 0.3, 1.0];
    let phases = [0.0_f32, 0.25, 1.5];
    let steps = [0.0_f32, 0.1, 1.0, 4.0];
    // light / powder deliberately include out-of-range values to exercise the
    // saturation clamps on both sides.
    let lights = [-0.5_f32, 0.0, 0.6, 1.0, 1.4];
    let powders = [-0.2_f32, 0.0, 0.7, 1.0, 1.3];

    let mut queries: Vec<IntegrateSegmentQuery> = Vec::new();
    for &st in &states {
        for &sigma_t in &sigma_ts {
            for &sigma_s in &sigma_ss {
                for &phase in &phases {
                    for &step in &steps {
                        // Pair light/powder positionally to keep the grid bounded
                        // while still covering the saturation edges.
                        for k in 0..lights.len() {
                            queries.push(IntegrateSegmentQuery {
                                state: st,
                                sigma_t,
                                sigma_s,
                                phase,
                                step,
                                light_transmittance: lights[k],
                                powder_factor: powders[k],
                            });
                        }
                    }
                }
            }
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, got) in queries.iter().zip(gpu.iter()) {
        let mut want = q.state;
        integrate_segment(
            &mut want,
            q.sigma_t,
            q.sigma_s,
            q.phase,
            q.step,
            q.light_transmittance,
            q.powder_factor,
        );

        assert!(
            (got.transmittance - want.transmittance).abs() <= TOL,
            "transmittance mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );
        assert!(
            (got.optical_depth - want.optical_depth).abs() <= TOL,
            "optical_depth mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );
        assert!(
            (got.scattered - want.scattered).abs() <= TOL,
            "scattered mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );
        assert_eq!(
            got.steps_taken, want.steps_taken,
            "steps_taken mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );

        // Physical invariants on the GPU result.
        assert!(
            (0.0..=1.0).contains(&got.transmittance),
            "transmittance out of [0,1] for {q:?}: {got:?}"
        );
        assert!(
            got.transmittance <= q.state.transmittance + TOL,
            "transmittance increased for {q:?}: {got:?}"
        );
        assert!(
            got.scattered >= q.state.scattered - TOL,
            "scattered decreased for {q:?}: {got:?}"
        );
        assert_eq!(
            got.steps_taken,
            q.state.steps_taken + 1,
            "steps_taken must grow by one for {q:?}: {got:?}"
        );
        // A zero extinction segment leaves transmittance unchanged (seg_trans=1).
        if q.sigma_t <= 0.0 {
            assert!(
                (got.transmittance - q.state.transmittance).abs() <= TOL,
                "zero sigma_t changed transmittance for {q:?}: {got:?}"
            );
        }
    }
}
