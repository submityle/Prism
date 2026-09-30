//! Real-device parity for the coverage-relaxation twin:
//! [`GpuRelaxCoverage`] must reproduce the `CPU` golden
//! [`relax_coverage`](prism_render_architecture::volumetric::weather::relax_coverage)
//! across a deterministic grid of current/target coverages, rates and
//! timesteps, including the degenerate inputs (zero rate, zero dt, negative
//! rate/dt clamped to zero).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors the `CPU` golden's hand-rolled polynomial `exp_approx`
//! (rather than the WGSL `exp` builtin), so agreement is asserted within a tight
//! absolute tolerance. The suite also checks the invariants that make the
//! kernel correct: the result never overshoots `target` and, for in-range
//! inputs, stays in `0..=1`.
//!
//! Provenance: standard exponential relaxation / weather coverage coupling; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::weather::relax_coverage;
use prism_volumetric_gpu::{GpuContext, GpuRelaxCoverage, RelaxCoverageQuery};

/// Absolute tolerance: the mirrored polynomial exp differs from the `CPU`
/// golden's own polynomial exp only in the last few ULPs, well under this.
const TOL: f32 = 1e-5;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_relax_coverage_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping relax-coverage parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuRelaxCoverage::new(&ctx);

    // A deterministic grid over current, target, rate and dt, including the
    // degenerate inputs (zero and negative rate/dt).
    let currents = [0.0_f32, 0.2, 0.5, 0.8, 1.0];
    let targets = [0.0_f32, 0.3, 0.6, 1.0];
    let rates = [-1.0_f32, 0.0, 0.5, 2.0, 10.0];
    let dts = [-0.5_f32, 0.0, 0.016, 0.1, 1.0];

    let mut queries: Vec<RelaxCoverageQuery> = Vec::new();
    for &current in &currents {
        for &target in &targets {
            for &rate in &rates {
                for &dt in &dts {
                    queries.push(RelaxCoverageQuery {
                        current,
                        target,
                        rate,
                        dt,
                    });
                }
            }
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");

    for (i, q) in queries.iter().enumerate() {
        let exp = relax_coverage(q.current, q.target, q.rate, q.dt);
        assert!(
            (gpu[i] - exp).abs() <= TOL,
            "relax-coverage mismatch for query {i} (current {}, target {}, rate {}, dt {}): \
             gpu {}, cpu {exp}, |diff| {}",
            q.current,
            q.target,
            q.rate,
            q.dt,
            gpu[i],
            (gpu[i] - exp).abs()
        );

        // The result never overshoots the target: it lies between current and
        // target (within tolerance).
        let lo = q.current.min(q.target) - TOL;
        let hi = q.current.max(q.target) + TOL;
        assert!(
            gpu[i] >= lo && gpu[i] <= hi,
            "relax-coverage overshoot for query {i}: gpu {} outside [{lo}, {hi}]",
            gpu[i]
        );

        // For in-range inputs the result stays in `0..=1`.
        assert!(
            gpu[i] >= -TOL && gpu[i] <= 1.0 + TOL,
            "relax-coverage out of [0,1] for query {i}: gpu {}",
            gpu[i]
        );
    }

    // Degenerate: zero rate or zero dt means no movement (weight 0), so the
    // result equals `current`. Query index 0 is current 0.0 / target 0.0, so
    // find a case with distinct current/target and zero dt.
    let no_move = RelaxCoverageQuery {
        current: 0.2,
        target: 0.9,
        rate: 5.0,
        dt: 0.0,
    };
    let out = gpu_kernel.eval(&ctx, &[no_move]);
    assert!(
        (out[0] - no_move.current).abs() <= TOL,
        "zero dt must not move coverage: gpu {}, expected {}",
        out[0],
        no_move.current
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuRelaxCoverage::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
