//! Real-device parity test for the transcendental twin.
//!
//! Confirms the `wgpu` [`GpuTranscendental`] kernel reproduces the CPU goldens
//! [`exp_approx`](prism_render_architecture::volumetric::math::exp_approx),
//! [`ln_approx`](prism_render_architecture::volumetric::math::ln_approx) and
//! [`pow_approx`](prism_render_architecture::volumetric::math::pow_approx)
//! across a wide domain: `exp` from strongly negative (flush to `0`) through
//! large positive (saturate to `+inf`), `ln` from below the smallest normal
//! (clamped) through huge values crossing the `SQRT_2` mantissa-centring
//! branch, and `pow` with positive, zero and negative bases against varied
//! exponents. Comparison uses a combined absolute/relative tolerance, with
//! exact equality required for infinities. An empty query slice must yield an
//! empty result.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::math::{exp_approx, ln_approx, pow_approx};
use prism_volumetric_gpu::{GpuContext, GpuTranscendental, TranscendentalQuery};

/// Combined absolute/relative closeness, treating infinities as equal only when
/// both sides are the same infinity.
fn close(a: f32, b: f32) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= 1e-4 + 1e-5 * scale
}

/// Builds the wide-domain sweep exercised by the parity check.
fn sweep() -> Vec<TranscendentalQuery> {
    // Independent 1-D sweeps for each function, zipped into shared queries by
    // cycling the shorter axes so every listed value is exercised.
    let exp_x: Vec<f32> = {
        let mut v = Vec::new();
        let mut x = -30.0f32;
        while x <= 30.0 {
            v.push(x);
            x += 0.25;
        }
        // Saturation extremes: large negative flushes to 0, large positive to +inf.
        v.extend_from_slice(&[-200.0, -100.0, 88.0, 89.0, 100.0, 200.0]);
        v
    };
    let ln_x: Vec<f32> = {
        let mut v = Vec::new();
        // Small (including sub-normal-clamp) through large, log-spaced-ish.
        for &x in &[
            1e-40f32,
            1e-38,
            1e-20,
            1e-6,
            0.001,
            0.1,
            0.5,
            1.0,
            core::f32::consts::SQRT_2,
            2.0,
            3.0,
            10.0,
            100.0,
            1e6,
            1e12,
            1e30,
        ] {
            v.push(x);
        }
        // Non-positive inputs are clamped to the smallest positive normal.
        v.extend_from_slice(&[0.0, -1.0, -100.0]);
        v
    };
    let pow: Vec<(f32, f32)> = vec![
        (2.0, 0.5),
        (2.0, 2.0),
        (2.0, -1.0),
        (10.0, 3.0),
        (0.5, 4.0),
        (3.0, 0.0),
        (1.0, 100.0),
        (0.1, -2.0),
        (5.0, 1.5),
        // Guarded regimes: zero and negative bases return 0.
        (0.0, 2.0),
        (-2.0, 3.0),
        (-1.0, 0.5),
    ];

    let n = exp_x.len().max(ln_x.len()).max(pow.len());
    (0..n)
        .map(|i| {
            let (pb, pe) = pow[i % pow.len()];
            TranscendentalQuery {
                exp_x: exp_x[i % exp_x.len()],
                ln_x: ln_x[i % ln_x.len()],
                pow_base: pb,
                pow_exp: pe,
            }
        })
        .collect()
}

#[test]
fn transcendental_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping transcendental_approx_parity: no GPU adapter available");
        return;
    };

    let queries = sweep();
    let kernel = GpuTranscendental::new(&ctx);
    let results = kernel.eval(&ctx, &queries);
    assert_eq!(results.len(), queries.len());

    for (q, gpu) in queries.iter().zip(results.iter()) {
        let exp_cpu = exp_approx(q.exp_x);
        let ln_cpu = ln_approx(q.ln_x);
        let pow_cpu = pow_approx(q.pow_base, q.pow_exp);
        assert!(
            close(gpu.exp, exp_cpu),
            "exp mismatch at {}: gpu {} vs cpu {exp_cpu}",
            q.exp_x,
            gpu.exp
        );
        assert!(
            close(gpu.ln, ln_cpu),
            "ln mismatch at {}: gpu {} vs cpu {ln_cpu}",
            q.ln_x,
            gpu.ln
        );
        assert!(
            close(gpu.pow, pow_cpu),
            "pow mismatch at ({}, {}): gpu {} vs cpu {pow_cpu}",
            q.pow_base,
            q.pow_exp,
            gpu.pow
        );
    }
}

#[test]
fn transcendental_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping transcendental_approx_parity empty case: no GPU adapter available");
        return;
    };
    let kernel = GpuTranscendental::new(&ctx);
    let results = kernel.eval(&ctx, &[]);
    assert!(results.is_empty());
}
