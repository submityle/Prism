//! Real-device parity for the wind-velocity twin: [`GpuVelocityAt`] must
//! reproduce the `CPU` golden
//! [`velocity_at`](prism_render_architecture::volumetric::weather::WindField::velocity_at)
//! across a deterministic grid of wind fields, sample positions and per-cell
//! disturbances, including out-of-range disturbances (which must saturate) and
//! positions large enough to exercise the trig range reduction.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both velocity components are asserted against the `CPU` golden within a tight
//! absolute tolerance. The `CPU` golden and the kernel share the same
//! `sin_approx` / `cos_approx` range reduction and polynomial, so agreement is
//! to a few ULPs of the accumulated arithmetic.
//!
//! Provenance: standard divergence-free curl-noise wind advection; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::math::Vec2;
use prism_render_architecture::volumetric::weather::WindField;
use prism_volumetric_gpu::{GpuContext, GpuVelocityAt, VelocityAtQuery};

/// Absolute tolerance for the velocity components. Both sides share the same
/// polynomial trig; the gust scaling only amplifies float rounding modestly.
const TOL: f32 = 2e-4;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_velocity_at_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping velocity-at parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuVelocityAt::new(&ctx);

    // A spread of wind directions (including a degenerate zero that normalizes
    // to zero), speeds and curl strengths (including a pure-uniform zero curl).
    let directions = [
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(0.7, -0.7),
        Vec2::new(0.0, 0.0),
    ];
    let speeds = [0.0_f32, 1.0, 3.5, 10.0];
    let curls = [0.0_f32, 0.5, 2.0];
    // Positions across and well beyond one curl period (2*PI / 0.15 ~= 41.9),
    // plus negatives, to exercise the trig range reduction on both sides.
    let positions = [
        Vec2::new(0.0, 0.0),
        Vec2::new(3.2, -1.7),
        Vec2::new(20.0, 20.0),
        Vec2::new(-55.0, 130.0),
        Vec2::new(500.0, -777.0),
    ];
    // Disturbances straddling the 0..=1 saturation range.
    let disturbances = [-0.5_f32, 0.0, 0.4, 1.0, 1.8];

    let mut queries: Vec<VelocityAtQuery> = Vec::new();
    for &dir in &directions {
        for &speed in &speeds {
            for &curl in &curls {
                let wind = WindField::new(dir, speed, curl);
                for &pos in &positions {
                    for &local_disturbance in &disturbances {
                        queries.push(VelocityAtQuery {
                            wind,
                            pos,
                            local_disturbance,
                        });
                    }
                }
            }
        }
    }
    assert!(!queries.is_empty(), "the parity grid must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len());

    for (q, got) in queries.iter().zip(gpu.iter()) {
        let want = q.wind.velocity_at(q.pos, q.local_disturbance);
        assert!(
            (got.x - want.x).abs() <= TOL,
            "velocity.x mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );
        assert!(
            (got.y - want.y).abs() <= TOL,
            "velocity.y mismatch for {q:?}: gpu={got:?} cpu={want:?}"
        );
        // A zero curl strength leaves only the base flow direction * speed.
        if q.wind.curl_strength() == 0.0 {
            let base = q.wind.direction().scale(q.wind.speed());
            assert!(
                (got.x - base.x).abs() <= TOL && (got.y - base.y).abs() <= TOL,
                "zero-curl wind must be pure base flow for {q:?}: {got:?}"
            );
        }
    }
}
