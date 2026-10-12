//! Real-device parity for the §24.6 GPU SH clamped-cosine convolution mirror.
//!
//! Each test builds a batch of radiance SH coefficient sets, convolves them on
//! the CPU with [`prism_math::Sh2::convolve_cosine`] /
//! [`prism_math::Sh3::convolve_cosine`], runs the same convolution on a real
//! `GPU` from the single-sourced
//! [`WGSL_SH_COSINE`](prism_math::shader_mirror::WGSL_SH_COSINE) fragment, and
//! asserts the irradiance coefficients agree.
//!
//! Each coefficient is a single multiply by the per-band factor `π` / `2π/3` /
//! `π/4` / `0`; the parity suite uses a tight tolerance that still rejects a
//! wrong factor or band split. Band-3 inputs are included to confirm they vanish.
//! It skips gracefully when no adapter is available so it still passes on a
//! device-less CI image while running the full dispatch on a real `GPU`.

use prism_math::{Sh2, Sh3};
use prism_math_gpu::{GpuContext, GpuShCosine};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness for the single-multiply factors.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-6 + 1.0e-6 * a.abs().max(b.abs())
}

/// A spread of order-2 radiance coefficient sets.
fn sh2_sets() -> [[f32; 9]; 3] {
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.2, 0.3, -0.4, 0.1, 0.6, -0.3, 0.2, -0.5],
        [2.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.75, -1.25, 1.5],
    ]
}

/// A spread of order-3 radiance coefficient sets; the tail is non-zero so the
/// band-3 vanishing is exercised.
fn sh3_sets() -> [[f32; 16]; 3] {
    [
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.5, -0.2, 0.3, -0.4, 0.1, 0.6, -0.3, 0.2, -0.5, 0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3,
        ],
        [
            2.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.75, -1.25, 1.5, 1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7,
        ],
    ]
}

#[test]
fn sh2_convolve_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShCosine::new(&ctx);
    let sets = sh2_sets();
    let got = kernel.convolve2(&ctx, &sets);
    assert_eq!(got.len(), sets.len());
    for (i, set) in sets.iter().enumerate() {
        let expected = Sh2::from_coeffs(*set).convolve_cosine().c;
        for (j, (g, c)) in got[i].iter().zip(expected).enumerate() {
            assert!(close(*g, c), "sh2[{i}][{j}] gpu={g} cpu={c}");
        }
    }
}

#[test]
fn sh3_convolve_matches_cpu_and_zeroes_band3() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShCosine::new(&ctx);
    let sets = sh3_sets();
    let got = kernel.convolve3(&ctx, &sets);
    assert_eq!(got.len(), sets.len());
    for (i, set) in sets.iter().enumerate() {
        let expected = Sh3::from_coeffs(*set).convolve_cosine().c;
        for (j, (g, c)) in got[i].iter().zip(expected).enumerate() {
            assert!(close(*g, c), "sh3[{i}][{j}] gpu={g} cpu={c}");
        }
        // Band 3 (indices 9..16) vanishes under the clamped-cosine kernel.
        assert!(
            got[i][SH3_BAND3_START..].iter().all(|&v| v == 0.0),
            "sh3[{i}] band-3 not zeroed: {:?}",
            &got[i][SH3_BAND3_START..]
        );
    }
}

/// First index of band 3 in an order-3 coefficient array.
const SH3_BAND3_START: usize = 9;

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuShCosine::new(&ctx);
    assert!(kernel.convolve2(&ctx, &[]).is_empty());
    assert!(kernel.convolve3(&ctx, &[]).is_empty());
}
