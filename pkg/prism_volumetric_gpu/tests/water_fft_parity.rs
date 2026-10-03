//! Real-device parity for the radix-2 butterfly `FFT` twin:
//! [`GpuWaterFft`](prism_volumetric_gpu::water_fft::GpuWaterFft) must reproduce
//! the dependency-free `CPU` golden
//! [`fft`](prism_render_architecture::water::fft::fft) /
//! [`ifft`](prism_render_architecture::water::fft::ifft) — the separable
//! `Cooley-Tukey` transform at the heart of the spectral ocean's height-field
//! synthesis — across hand-chosen fixtures, boundary cases, and a randomized
//! sweep over several batched transforms.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The goldens `fft` and `ifft` are public and pure, so the expected transform
//! is built in-host by calling them directly, array by array. A `GPU == oracle`
//! pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each output threads through `log2(n)` butterfly stages of range-reduced
//! Taylor `sin`/`cos` and complex multiply/add. The `CPU` and `GPU` share the
//! polynomial and stage structure, so that approximation is common-mode and
//! cancels; the residual is only a `GPU` fused multiply-add's last-place slack,
//! accumulated over the stages. Each component is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough to fail a wrong port —
//! a dropped bit-reversal, a flipped twiddle sign, a missing inverse scale — yet
//! loose enough to admit the accumulated slack even on the large-magnitude `DC`
//! bins.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::fft`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::fft::{fft, ifft};
use prism_render_architecture::water::spectrum::Complex;
use prism_volumetric_gpu::water_fft::{GpuWaterFft, WaterFftComplex};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a transformed component. A `GPU` fused multiply-add
/// may land a few units in the last place from the scalar reference, summed
/// over `log2(n)` stages; `1e-4` admits that while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (such as the `DC` bin,
/// which carries `N` times the mean) where a few units in the last place exceed
/// the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Converts a host complex array to the `GPU` input element type.
fn to_gpu(xs: &[Complex]) -> Vec<WaterFftComplex> {
    xs.iter()
        .map(|c| WaterFftComplex::new(c.re, c.im))
        .collect()
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A reproducible `f32` in `[-1, 1)` from the generator.
fn rand_unit(state: &mut u64) -> f32 {
    let bits = lcg(state);
    (f32::from(bits as u16) / 32_768.0) - 1.0
}

/// A reproducible complex sample with both parts in `[-1, 1)`.
fn rand_complex(state: &mut u64) -> Complex {
    Complex::new(rand_unit(state), rand_unit(state))
}

/// Pins one batched `GPU` transform against the per-array `CPU` golden.
///
/// `input` holds `count` length-`n` arrays back to back; the oracle transforms
/// each array independently and the two layouts are compared element by
/// element.
fn check(ctx: &GpuContext, gpu: &GpuWaterFft, arrays: &[Vec<Complex>], n: usize, inverse: bool) {
    let mut flat: Vec<Complex> = Vec::new();
    for a in arrays {
        assert_eq!(a.len(), n, "every array in a batch shares the length n");
        flat.extend_from_slice(a);
    }
    let got = gpu.evaluate(ctx, &to_gpu(&flat), n as u32, inverse);
    assert_eq!(
        got.len(),
        flat.len(),
        "result count must match the input count"
    );

    for (ai, a) in arrays.iter().enumerate() {
        let want = if inverse { ifft(a) } else { fft(a) };
        for (k, w) in want.iter().enumerate() {
            let g = got[ai * n + k];
            assert!(
                close(g.re, w.re),
                "array {ai} bin {k} re: gpu {} vs cpu {}",
                g.re,
                w.re
            );
            assert!(
                close(g.im, w.im),
                "array {ai} bin {k} im: gpu {} vs cpu {}",
                g.im,
                w.im
            );
        }
    }
}

/// A deterministic ramp `x[i] = (i, -i/2)` of length `n`.
fn ramp(n: usize) -> Vec<Complex> {
    (0..n)
        .map(|i| Complex::new(i as f32, -(i as f32) * 0.5))
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_fft parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    let got = gpu.evaluate(&ctx, &[], 8, false);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn forward_matches_golden_over_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    for &n in &[1usize, 2, 4, 8, 16, 32, 64] {
        check(&ctx, &gpu, &[ramp(n)], n, false);
    }
}

#[test]
fn inverse_matches_golden_over_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    for &n in &[1usize, 2, 4, 8, 16, 32, 64] {
        // Feed the forward spectrum of a ramp so the inverse has structure.
        let spectrum = fft(&ramp(n));
        check(&ctx, &gpu, &[spectrum], n, true);
    }
}

#[test]
fn round_trip_matches_golden_round_trip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    for &n in &[2usize, 4, 8, 16, 32, 64] {
        let x = ramp(n);
        // GPU forward, then GPU inverse, pinned against the CPU golden composed
        // the same way (`ifft(fft(x))`). Both share the Taylor `sin`/`cos`, so
        // that approximation is common-mode and cancels; this isolates the
        // GPU-vs-CPU last-place slack of the composed transform. (Comparing the
        // round trip to the raw input instead would conflate the goldens' own
        // Taylor round-trip error, which the twin is not meant to measure.)
        let spectrum = gpu.evaluate(&ctx, &to_gpu(&x), n as u32, false);
        let recovered = gpu.evaluate(&ctx, &spectrum, n as u32, true);
        assert_eq!(recovered.len(), n);
        let want = ifft(&fft(&x));
        for (i, w) in want.iter().enumerate() {
            assert!(close(recovered[i].re, w.re), "roundtrip {i} re");
            assert!(close(recovered[i].im, w.im), "roundtrip {i} im");
        }
    }
}

#[test]
fn delta_transforms_to_flat_spectrum() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    let n = 8usize;
    let mut x = vec![Complex::ZERO; n];
    x[0] = Complex::new(1.0, 0.0);
    check(&ctx, &gpu, &[x], n, false);
}

#[test]
fn constant_concentrates_at_dc() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    let n = 8usize;
    let x = vec![Complex::new(2.0, 0.0); n];
    check(&ctx, &gpu, &[x], n, false);
}

#[test]
fn non_power_of_two_is_identity_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    // 3 and 6 are not powers of two: the golden (and the twin) return the input
    // verbatim, for both the forward and inverse paths.
    for &n in &[3usize, 6] {
        let x = ramp(n);
        check(&ctx, &gpu, std::slice::from_ref(&x), n, false);
        check(&ctx, &gpu, &[x], n, true);
    }
}

#[test]
fn batched_arrays_transform_independently() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    let n = 16usize;
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    // A handful of distinct arrays dispatched as one batch, each pinned to its
    // own golden transform.
    let arrays: Vec<Vec<Complex>> = (0..5)
        .map(|_| (0..n).map(|_| rand_complex(&mut state)).collect())
        .collect();
    check(&ctx, &gpu, &arrays, n, false);
    check(&ctx, &gpu, &arrays, n, true);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFft::new(&ctx);
    let mut state = 0x1a2b_3c4d_5e6f_7081_u64;
    // A wide span of random arrays across every supported power-of-two length,
    // batched and pinned both ways.
    for &n in &[2usize, 4, 8, 16, 32, 64] {
        let arrays: Vec<Vec<Complex>> = (0..24)
            .map(|_| (0..n).map(|_| rand_complex(&mut state)).collect())
            .collect();
        check(&ctx, &gpu, &arrays, n, false);
        check(&ctx, &gpu, &arrays, n, true);
    }
}
