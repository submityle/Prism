//! Radix-2 `Cooley-Tukey` butterfly `FFT` — the production inverse transform
//! for the ocean spectrum.
//!
//! The spectral ocean (`Tessendorf`) synthesises a tiled height field by
//! inverse-transforming a grid of time-advanced complex amplitudes `h(k, t)`
//! into spatial displacement. Evaluated naively that inverse transform is a
//! direct summation over every wave vector for every output texel — `O(N^4)`
//! for an `N*N` patch, which the reference compute shader spells out only as a
//! compilable ground truth. Every shipping ocean (`WaveWorks`, `Crest`, UE5's
//! Water) instead uses a separable `Cooley-Tukey` butterfly `FFT`: `O(N log N)`
//! per row/column, `O(N^2 log N)` for the whole patch. At `N = 256` that is the
//! difference between ~4.3 billion and ~0.5 million complex products per field.
//!
//! This module is the dependency-free `CPU` golden of that butterfly transform.
//! It is the numerical reference a `GPU` ping-pong butterfly pass is validated
//! against, and it lets the spectral driver synthesise height fields on the
//! `CPU` for tests and tooling without the quartic direct sum. Only `sqrt` is
//! used among the float intrinsics (via nothing here — the transform needs no
//! roots); the twiddle phasors go through the crate's hand-rolled
//! [`sin_approx`](crate::water::sin_approx) / [`cos_approx`](crate::water::cos_approx),
//! so results stay deterministic and `libm`-free.
//!
//! # Conventions
//!
//! * Forward transform `fft`: `X[k] = sum_n x[n] e^{-i 2*PI k n / N}` (no scale).
//! * Inverse transform `ifft`: `x[n] = (1/N) sum_k X[k] e^{+i 2*PI k n / N}`.
//! * 2D variants are separable: transform every row, then every column, with the
//!   inverse normalised once by `1/(N*N)`.
//! * Non-power-of-two lengths (and mis-sized 2D grids) are returned unchanged, a
//!   deterministic no-op rather than a panic, matching the crate's "skip, do not
//!   crash" contract for out-of-contract inputs.

use alloc::vec::Vec;

use crate::water::spectrum::Complex;
use crate::water::{cos_approx, sin_approx, TWO_PI};

/// Returns `true` when `n` is a positive power of two, the only lengths the
/// radix-2 butterfly can transform in place.
#[must_use]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Reverses the low `bits` bits of `x` (the bit-reversal permutation index used
/// to reorder a decimation-in-time butterfly's input).
#[must_use]
fn reverse_bits(mut x: usize, bits: u32) -> usize {
    let mut reversed = 0usize;
    let mut b = 0u32;
    while b < bits {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
        b += 1;
    }
    reversed
}

/// In-place radix-2 `Cooley-Tukey` transform of a power-of-two buffer.
///
/// `inverse` flips the twiddle sign (`+` for the inverse transform); no
/// normalisation is applied here, so callers divide by the length. Buffers whose
/// length is not a power of two are left untouched.
fn transform(buf: &mut [Complex], inverse: bool) {
    let n = buf.len();
    if n <= 1 || !is_power_of_two(n) {
        return;
    }

    // Decimation-in-time reorder: place each sample at its bit-reversed index.
    let bits = n.trailing_zeros();
    let mut i = 1usize;
    while i < n {
        let j = reverse_bits(i, bits);
        if j > i {
            buf.swap(i, j);
        }
        i += 1;
    }

    // Butterfly stages: combine sub-transforms of length `len/2` into length
    // `len`, doubling until the whole buffer is one transform.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let step = sign * TWO_PI / (len as f32);
        let mut start = 0usize;
        while start < n {
            let mut k = 0usize;
            while k < half {
                let theta = step * (k as f32);
                let twiddle = Complex::new(cos_approx(theta), sin_approx(theta));
                let top = buf[start + k];
                let bottom = buf[start + k + half].mul(twiddle);
                buf[start + k] = top.add(bottom);
                buf[start + k + half] = top.add(Complex::new(-bottom.re, -bottom.im));
                k += 1;
            }
            start += len;
        }
        len *= 2;
    }
}

/// Forward `FFT`: `X[k] = sum_n x[n] e^{-i 2*PI k n / N}`.
///
/// Returns the input unchanged when its length is not a power of two.
#[must_use]
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let mut buf = input.to_vec();
    transform(&mut buf, false);
    buf
}

/// Inverse `FFT`: `x[n] = (1/N) sum_k X[k] e^{+i 2*PI k n / N}`.
///
/// Returns the input unchanged when its length is not a power of two.
#[must_use]
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let mut buf = input.to_vec();
    transform(&mut buf, true);
    let n = buf.len();
    if n > 1 && is_power_of_two(n) {
        let inv = 1.0 / (n as f32);
        for c in &mut buf {
            c.re *= inv;
            c.im *= inv;
        }
    }
    buf
}

/// Separable 2D transform of a row-major `n*n` grid; `inverse` selects the
/// inverse (normalised by `1/(N*N)`). Mis-sized or non-power-of-two grids are
/// returned unchanged.
fn transform2(grid: &[Complex], n: usize, inverse: bool) -> Vec<Complex> {
    if n == 0 || grid.len() != n * n || !is_power_of_two(n) {
        return grid.to_vec();
    }

    let mut data = grid.to_vec();

    // Transform every row in place.
    let mut row = 0usize;
    while row < n {
        let start = row * n;
        transform(&mut data[start..start + n], inverse);
        row += 1;
    }

    // Gather each column, transform it, and scatter it back.
    let mut col = 0usize;
    while col < n {
        let mut column = Vec::with_capacity(n);
        let mut r = 0usize;
        while r < n {
            column.push(data[r * n + col]);
            r += 1;
        }
        transform(&mut column, inverse);
        let mut r = 0usize;
        while r < n {
            data[r * n + col] = column[r];
            r += 1;
        }
        col += 1;
    }

    if inverse {
        let inv = 1.0 / ((n * n) as f32);
        for c in &mut data {
            c.re *= inv;
            c.im *= inv;
        }
    }

    data
}

/// Forward separable 2D `FFT` of a row-major `n*n` grid.
///
/// Returns the input unchanged when the grid is mis-sized or `n` is not a power
/// of two.
#[must_use]
pub fn fft2(grid: &[Complex], n: usize) -> Vec<Complex> {
    transform2(grid, n, false)
}

/// Inverse separable 2D `FFT` of a row-major `n*n` grid (normalised by
/// `1/(N*N)`), the transform that turns a spectral amplitude grid into a spatial
/// ocean field.
///
/// Returns the input unchanged when the grid is mis-sized or `n` is not a power
/// of two.
#[must_use]
pub fn ifft2(grid: &[Complex], n: usize) -> Vec<Complex> {
    transform2(grid, n, true)
}

#[cfg(test)]
mod tests {
    use super::{fft, fft2, ifft, ifft2, is_power_of_two};
    use crate::water::spectrum::Complex;
    use crate::water::{cos_approx, sin_approx, TWO_PI};
    use alloc::vec::Vec;

    /// Loose tolerance: both the butterfly and the naive reference share the
    /// crate's polynomial `sin_approx`/`cos_approx` (error ~2e-4), so the only
    /// gap between them is float summation order; round trips additionally
    /// accumulate that trig error across `N` terms.
    const FFT_EPS: f32 = 1.0e-2;

    /// Direct-summation `DFT`/`iDFT` used as the golden reference. Shares the
    /// crate trig with the butterfly so the comparison isolates the algorithm.
    fn naive_dft(input: &[Complex], inverse: bool) -> Vec<Complex> {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut out = Vec::with_capacity(n);
        let mut k = 0usize;
        while k < n {
            let mut acc = Complex::ZERO;
            let mut j = 0usize;
            while j < n {
                let theta = sign * TWO_PI * (k as f32) * (j as f32) / (n as f32);
                let w = Complex::new(cos_approx(theta), sin_approx(theta));
                acc = acc.add(input[j].mul(w));
                j += 1;
            }
            if inverse {
                acc.re /= n as f32;
                acc.im /= n as f32;
            }
            out.push(acc);
            k += 1;
        }
        out
    }

    fn ramp(n: usize) -> Vec<Complex> {
        let mut v = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let fi = i as f32;
            v.push(Complex::new(0.5 - 0.03 * fi, 0.2 * cos_approx(0.7 * fi)));
            i += 1;
        }
        v
    }

    fn assert_close(a: &[Complex], b: &[Complex], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        let mut i = 0usize;
        while i < a.len() {
            assert!(
                (a[i].re - b[i].re).abs() <= eps && (a[i].im - b[i].im).abs() <= eps,
                "index {i}: ({}, {}) vs ({}, {})",
                a[i].re,
                a[i].im,
                b[i].re,
                b[i].im
            );
            i += 1;
        }
    }

    #[test]
    fn power_of_two_classifies_correctly() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(4));
        assert!(!is_power_of_two(6));
        assert!(is_power_of_two(256));
        assert!(!is_power_of_two(255));
    }

    #[test]
    fn fft_matches_naive_dft_for_small_sizes() {
        for &n in &[2usize, 4, 8, 16, 32] {
            let x = ramp(n);
            assert_close(&fft(&x), &naive_dft(&x, false), FFT_EPS);
        }
    }

    #[test]
    fn ifft_matches_naive_inverse() {
        for &n in &[2usize, 4, 8, 16] {
            let x = ramp(n);
            let spectrum = fft(&x);
            assert_close(&ifft(&spectrum), &naive_dft(&spectrum, true), FFT_EPS);
        }
    }

    #[test]
    fn round_trip_recovers_input() {
        for &n in &[2usize, 4, 8, 16, 32, 64] {
            let x = ramp(n);
            assert_close(&ifft(&fft(&x)), &x, FFT_EPS);
        }
    }

    #[test]
    fn fft_of_delta_is_a_flat_spectrum() {
        let n = 8usize;
        let mut x = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            x.push(if i == 0 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::ZERO
            });
            i += 1;
        }
        let spectrum = fft(&x);
        // A unit impulse at the origin transforms to an all-ones spectrum.
        let mut k = 0usize;
        while k < n {
            assert!((spectrum[k].re - 1.0).abs() < FFT_EPS && spectrum[k].im.abs() < FFT_EPS);
            k += 1;
        }
    }

    #[test]
    fn fft_of_constant_concentrates_at_dc() {
        let n = 8usize;
        let x = alloc::vec![Complex::new(2.0, 0.0); n];
        let spectrum = fft(&x);
        // A constant signal puts all energy in the DC bin (= N * value).
        assert!((spectrum[0].re - 2.0 * n as f32).abs() < FFT_EPS);
        assert!(spectrum[0].im.abs() < FFT_EPS);
        let mut k = 1usize;
        while k < n {
            assert!(spectrum[k].re.abs() < FFT_EPS && spectrum[k].im.abs() < FFT_EPS);
            k += 1;
        }
    }

    #[test]
    fn transform_is_linear() {
        let n = 16usize;
        let a = ramp(n);
        let mut b = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let fi = i as f32;
            b.push(Complex::new(0.1 * sin_approx(0.5 * fi), 0.4 - 0.02 * fi));
            i += 1;
        }
        let mut sum = Vec::with_capacity(n);
        i = 0;
        while i < n {
            sum.push(a[i].add(b[i]));
            i += 1;
        }
        let fa = fft(&a);
        let fb = fft(&b);
        let fsum = fft(&sum);
        let mut combined = Vec::with_capacity(n);
        i = 0;
        while i < n {
            combined.push(fa[i].add(fb[i]));
            i += 1;
        }
        assert_close(&fsum, &combined, FFT_EPS);
    }

    #[test]
    fn parseval_energy_is_conserved() {
        let n = 16usize;
        let x = ramp(n);
        let spectrum = fft(&x);
        let mut time_energy = 0.0f32;
        let mut i = 0usize;
        while i < n {
            time_energy += x[i].norm_squared();
            i += 1;
        }
        let mut freq_energy = 0.0f32;
        i = 0;
        while i < n {
            freq_energy += spectrum[i].norm_squared();
            i += 1;
        }
        freq_energy /= n as f32;
        // Parseval: sum |x|^2 == (1/N) sum |X|^2.
        assert!((time_energy - freq_energy).abs() < FFT_EPS * (1.0 + time_energy));
    }

    #[test]
    fn non_power_of_two_is_identity_noop() {
        let x = ramp(3);
        assert_close(&fft(&x), &x, 0.0);
        assert_close(&ifft(&x), &x, 0.0);
    }

    #[test]
    fn ifft2_round_trips_a_grid() {
        let n = 8usize;
        let mut grid = Vec::with_capacity(n * n);
        let mut r = 0usize;
        while r < n {
            let mut c = 0usize;
            while c < n {
                let fr = r as f32;
                let fc = c as f32;
                grid.push(Complex::new(
                    0.3 * cos_approx(0.4 * fr + 0.2 * fc),
                    0.1 * fr - 0.05 * fc,
                ));
                c += 1;
            }
            r += 1;
        }
        let recovered = ifft2(&fft2(&grid, n), n);
        assert_close(&recovered, &grid, FFT_EPS);
    }

    #[test]
    fn mis_sized_2d_grid_is_identity_noop() {
        let grid = ramp(6);
        // 6 is not 4*4 and not a power of two: returned verbatim.
        assert_close(&fft2(&grid, 4), &grid, 0.0);
        assert_close(&ifft2(&grid, 3), &grid, 0.0);
    }
}
