//! Real-device parity test for the trigonometry twin.
//!
//! Confirms the `wgpu` [`GpuTrigApprox`] kernel reproduces the CPU golden
//! [`sin_approx`](prism_render_architecture::volumetric::math::sin_approx) and
//! [`cos_approx`](prism_render_architecture::volumetric::math::cos_approx)
//! across a wide angle sweep: several positive and negative periods, the
//! near-tie angles where `round_ties_away` matters (odd multiples of `PI`, the
//! `±0.5 * TWO_PI` reduction boundaries, and `n*PI + PI/2` fold edges), and a
//! few large magnitudes that stress the range reduction. Both `sin` and `cos`
//! must agree within a conditioning-aware tolerance `1e-6 + 2*f32::EPSILON *
//! |angle|` that tracks the range-reduction cancellation growing with the
//! angle magnitude, so a wrong reduction branch or a dropped polynomial term
//! would still fail. An empty query slice must yield an empty result.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::math::{cos_approx, sin_approx};
use prism_volumetric_gpu::{GpuContext, GpuTrigApprox, TrigApproxQuery};

/// Builds the wide-angle sweep exercised by the parity check.
fn sweep_angles() -> Vec<f32> {
    use core::f32::consts::{FRAC_PI_2, PI};
    let two_pi = 2.0 * PI;

    let mut angles = Vec::new();

    // A dense march across several full periods, positive and negative.
    let mut a = -4.0 * two_pi;
    while a <= 4.0 * two_pi {
        angles.push(a);
        a += 0.05;
    }

    // Near-tie and fold-boundary angles where the reduction branches matter.
    for n in -6i32..=6 {
        let k = n as f32;
        angles.push(k * PI); // odd multiples: sin ~ 0, tie in round.
        angles.push(k * PI + FRAC_PI_2); // fold edges of [-PI/2, PI/2].
        angles.push(k * two_pi + PI); // exact half period.
        angles.push(k * two_pi - PI);
        angles.push(k * two_pi + 0.5 * two_pi); // reduction boundary.
        angles.push(k * two_pi - 0.5 * two_pi);
    }

    // Large magnitudes to stress the range reduction.
    for &big in &[
        100.0f32, -100.0, 50.0, -50.0, 256.5, -256.5, 1000.0, -1000.0,
    ] {
        angles.push(big);
    }

    angles
}

#[test]
fn trig_approx_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping trig_approx_parity: no GPU adapter available");
        return;
    };

    let angles = sweep_angles();
    let queries: Vec<TrigApproxQuery> = angles
        .iter()
        .map(|&angle| TrigApproxQuery { angle })
        .collect();

    let kernel = GpuTrigApprox::new(&ctx);
    let results = kernel.eval(&ctx, &queries);
    assert_eq!(results.len(), queries.len());

    for (&angle, gpu) in angles.iter().zip(results.iter()) {
        let sin_cpu = sin_approx(angle);
        let cos_cpu = cos_approx(angle);
        // Conditioning-aware tolerance. `wrap_pi` reduces the angle by
        // subtracting `k * TWO_PI`; the `GPU` fuses that `x - k*TWO_PI` into a
        // single-rounding `FMA` while the `CPU` rounds the product first, so the
        // reduced angle can differ by up to a couple of `ULP` of the subtracted
        // multiple, i.e. `~f32::EPSILON * |angle|`. Near a zero crossing (odd
        // multiples of `PI`, where the folded angle is a catastrophic
        // cancellation to `~0`) that reduction spread is the whole signal, so a
        // fixed `1e-6` floor is too tight for large `|angle|`. The linear term
        // tracks the genuine range-reduction conditioning and stays far below
        // any real port error (a dropped polynomial term shifts the value by
        // `>= ~5e-3` at the fold edge).
        let tol = 1e-6 + 2.0 * f32::EPSILON * angle.abs();
        assert!(
            (gpu.sin - sin_cpu).abs() < tol,
            "sin mismatch at angle {angle}: gpu {} vs cpu {sin_cpu} (tol {tol})",
            gpu.sin
        );
        assert!(
            (gpu.cos - cos_cpu).abs() < tol,
            "cos mismatch at angle {angle}: gpu {} vs cpu {cos_cpu} (tol {tol})",
            gpu.cos
        );
    }
}

#[test]
fn trig_approx_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping trig_approx_parity empty case: no GPU adapter available");
        return;
    };
    let kernel = GpuTrigApprox::new(&ctx);
    let results = kernel.eval(&ctx, &[]);
    assert!(results.is_empty());
}
