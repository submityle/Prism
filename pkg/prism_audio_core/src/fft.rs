//! Reusable in-place radix-2 complex fast Fourier transform.
//!
//! A short-time Fourier transform (`STFT`) sits at the heart of every spectral
//! audio tool -- metering, spectral gating, phase-vocoder pitch shifting -- and
//! each of those tools needs the same forward/inverse discrete Fourier
//! transform (`DFT`). This module factors that transform out of any single node
//! into one shared, well-tested primitive so every spectral node reuses the
//! identical bit-reversal table, twiddle-factor table, and butterfly schedule
//! instead of copying them.
//!
//! The transform is the iterative, decimation-in-time (`DIT`) radix-2
//! Cooley-Tukey algorithm. The input length must be a power of two; [`Fft::new`]
//! rounds any requested length up to the next power of two (never below
//! [`MIN_FFT_SIZE`]) so the hot path never has to validate it. The forward
//! transform applies no scaling and the inverse divides by the length `N`, so
//! `inverse(forward(x)) == x` to within floating-point rounding.
//!
//! The complex twiddle stored at index `k` is the forward `DFT` root of unity
//! `W_N^k = exp(-j * 2 * pi * k / N) = cos(theta) - j * sin(theta)` with
//! `theta = 2 * pi * k / N`; the inverse path negates the imaginary part to use
//! the conjugate roots and then scales by `1 / N`.
//!
//! # Real-time contract
//!
//! All tables (bit-reversal permutation, cosine/sine twiddles) are built once in
//! [`Fft::new`]. [`Fft::forward`] and [`Fft::inverse`] borrow `&self`, operate
//! in place on caller-owned scratch slices, perform no allocation, take no
//! locks, and cannot panic in release builds (the length match is a
//! `debug_assert`). All transcendental math routes through [`bevy_math::ops`],
//! so the twiddle tables are bit-reproducible across platforms.
//!
//! # Provenance
//!
//! The iterative radix-2 decimation-in-time Cooley-Tukey `FFT`, the bit-reversal
//! permutation, and the precomputed twiddle-factor table are textbook
//! signal-processing constructions described in every digital-signal-processing
//! reference (for example Oppenheim and Schafer). This module reuses only this
//! crate's own [`Sample`] scalar and [`bevy_math::ops`]. It is pure classic DSP
//! with no AI or ML and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from those publicly documented algorithms.
//!
//! # Relationship
//!
//! This primitive is intended to back the frequency-domain nodes in
//! [`nodes::analysis`](crate::nodes::analysis) and
//! [`nodes::effects`](crate::nodes::effects) -- the spectrum analyzer, the
//! spectral gate, and the phase-vocoder pitch shifter -- so they share one
//! transform rather than each carrying a private copy. It is the
//! frequency-domain counterpart to the time-domain
//! [`Oversampler`](crate::oversampler): one resamples a signal, this transforms
//! one between the time and frequency domains.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::math::Sample;

/// Smallest supported transform length (one radix-2 butterfly stage).
pub const MIN_FFT_SIZE: usize = 2;

/// Rounds `requested` up to the next power of two, never below [`MIN_FFT_SIZE`].
#[must_use]
pub fn power_of_two_at_least(requested: usize) -> usize {
    let mut size = MIN_FFT_SIZE;
    while size < requested {
        size <<= 1;
    }
    size
}

/// Reverses the low `bits` of `value` (the bit-reversal permutation index).
fn reverse_low_bits(mut value: usize, bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    result
}

/// Precomputed radix-2 complex `FFT` plan for a fixed power-of-two length.
///
/// Construct once with [`Fft::new`], then call [`Fft::forward`] and
/// [`Fft::inverse`] on scratch buffers whose length equals [`Fft::size`].
///
/// ```
/// use prism_audio_core::fft::Fft;
///
/// let fft = Fft::new(8);
/// let mut re = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let mut im = [0.0f32; 8];
/// // The transform of a unit impulse is flat (all ones).
/// fft.forward(&mut re, &mut im);
/// assert!(re.iter().all(|&v| (v - 1.0).abs() < 1e-5));
/// // Round-trip returns the original impulse.
/// fft.inverse(&mut re, &mut im);
/// assert!((re[0] - 1.0).abs() < 1e-5);
/// assert!(re[1..].iter().all(|&v| v.abs() < 1e-5));
/// ```
#[derive(Clone, Debug)]
pub struct Fft {
    /// Transform length (a power of two).
    size: usize,
    /// Bit-reversal permutation, one entry per sample.
    rev: Vec<usize>,
    /// Real part of the forward twiddles `cos(2*pi*k/N)`, length `size / 2`.
    tw_re: Vec<Sample>,
    /// Imaginary part of the forward twiddles `-sin(2*pi*k/N)`, length `size / 2`.
    tw_im: Vec<Sample>,
    /// Reciprocal `1 / size` applied by the inverse transform.
    inv_size: Sample,
}

impl Fft {
    /// Builds a plan for a transform of length `power_of_two_at_least(size)`.
    ///
    /// The requested length is rounded up to the next power of two (never below
    /// [`MIN_FFT_SIZE`]) so callers never have to pre-validate it. Query the
    /// resulting length with [`Fft::size`].
    #[must_use]
    pub fn new(size: usize) -> Self {
        let size = power_of_two_at_least(size);
        let half = size / 2;
        let tw_re: Vec<Sample> = (0..half)
            .map(|k| ops::cos(-TAU * k as Sample / size as Sample))
            .collect();
        let tw_im: Vec<Sample> = (0..half)
            .map(|k| ops::sin(-TAU * k as Sample / size as Sample))
            .collect();
        let bits = size.trailing_zeros();
        let rev: Vec<usize> = (0..size).map(|i| reverse_low_bits(i, bits)).collect();
        Self {
            size,
            rev,
            tw_re,
            tw_im,
            inv_size: 1.0 / size as Sample,
        }
    }

    /// The transform length in samples (always a power of two).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Forward transform in place. `re` and `im` must each hold [`Fft::size`]
    /// samples; no scaling is applied.
    pub fn forward(&self, re: &mut [Sample], im: &mut [Sample]) {
        self.transform(re, im, false);
    }

    /// Inverse transform in place. `re` and `im` must each hold [`Fft::size`]
    /// samples; the result is scaled by `1 / size` so that
    /// `inverse(forward(x)) == x`.
    pub fn inverse(&self, re: &mut [Sample], im: &mut [Sample]) {
        self.transform(re, im, true);
    }

    /// Shared in-place radix-2 decimation-in-time butterfly schedule. The
    /// inverse path conjugates the twiddles and scales the result by `1 / size`.
    fn transform(&self, re: &mut [Sample], im: &mut [Sample], inverse: bool) {
        debug_assert_eq!(re.len(), self.size, "real buffer length must equal size");
        debug_assert_eq!(im.len(), self.size, "imag buffer length must equal size");
        let size = self.size;

        // Bit-reversal permutation into transform order.
        for (i, &j) in self.rev.iter().enumerate() {
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // Iterative decimation-in-time radix-2 butterflies.
        let mut len = 2;
        while len <= size {
            let half = len / 2;
            let step = size / len;
            let mut base = 0;
            while base < size {
                for k in 0..half {
                    let tw = k * step;
                    let wr = self.tw_re[tw];
                    let wi = if inverse { -self.tw_im[tw] } else { self.tw_im[tw] };
                    let a = base + k;
                    let b = base + k + half;
                    let tr = wr * re[b] - wi * im[b];
                    let ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
                base += len;
            }
            len <<= 1;
        }

        if inverse {
            let inv = self.inv_size;
            for value in re.iter_mut() {
                *value *= inv;
            }
            for value in im.iter_mut() {
                *value *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::TAU;

    /// Naive O(N^2) reference DFT (forward) used to validate the fast path.
    fn naive_dft(re: &[Sample], im: &[Sample]) -> (Vec<Sample>, Vec<Sample>) {
        let n = re.len();
        let mut out_re = vec![0.0; n];
        let mut out_im = vec![0.0; n];
        for k in 0..n {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for t in 0..n {
                let angle = -TAU * (k as Sample) * (t as Sample) / (n as Sample);
                let (s, c) = ops::sin_cos(angle);
                acc_re += f64::from(re[t] * c - im[t] * s);
                acc_im += f64::from(re[t] * s + im[t] * c);
            }
            out_re[k] = acc_re as Sample;
            out_im[k] = acc_im as Sample;
        }
        (out_re, out_im)
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        assert_eq!(Fft::new(0).size(), MIN_FFT_SIZE);
        assert_eq!(Fft::new(1).size(), MIN_FFT_SIZE);
        assert_eq!(Fft::new(2).size(), 2);
        assert_eq!(Fft::new(3).size(), 4);
        assert_eq!(Fft::new(5).size(), 8);
        assert_eq!(Fft::new(1024).size(), 1024);
        assert_eq!(Fft::new(1025).size(), 2048);
    }

    #[test]
    fn impulse_transforms_to_flat_spectrum() {
        let fft = Fft::new(16);
        let mut re = vec![0.0; 16];
        let mut im = vec![0.0; 16];
        re[0] = 1.0;
        fft.forward(&mut re, &mut im);
        for &v in &re {
            assert!((v - 1.0).abs() < 1e-5, "re bin not unity: {v}");
        }
        for &v in &im {
            assert!(v.abs() < 1e-5, "im bin not zero: {v}");
        }
    }

    #[test]
    fn dc_input_transforms_to_single_bin() {
        let fft = Fft::new(8);
        let mut re = vec![1.0; 8];
        let mut im = vec![0.0; 8];
        fft.forward(&mut re, &mut im);
        assert!((re[0] - 8.0).abs() < 1e-4, "DC bin should equal N: {}", re[0]);
        for bin in 1..8 {
            assert!(re[bin].abs() < 1e-4, "non-DC bin leaked: {}", re[bin]);
            assert!(im[bin].abs() < 1e-4, "non-DC imag leaked: {}", im[bin]);
        }
    }

    #[test]
    fn pure_tone_lands_in_expected_bins() {
        // A real cosine at bin k produces conjugate-symmetric peaks at k and N-k.
        let n = 32usize;
        let fft = Fft::new(n);
        let k_tone = 3usize;
        let mut re: Vec<Sample> = (0..n)
            .map(|t| ops::cos(TAU * (k_tone as Sample) * (t as Sample) / (n as Sample)))
            .collect();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);
        for bin in 0..n {
            let mag = ops::sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
            if bin == k_tone || bin == n - k_tone {
                assert!((mag - (n as Sample) / 2.0).abs() < 1e-2, "bin {bin} mag {mag}");
            } else {
                assert!(mag < 1e-2, "bin {bin} should be empty, mag {mag}");
            }
        }
    }

    #[test]
    fn forward_matches_naive_dft() {
        let n = 16usize;
        let fft = Fft::new(n);
        let re0: Vec<Sample> = (0..n).map(|t| ops::sin(0.7 * t as Sample) + 0.3).collect();
        let im0: Vec<Sample> = (0..n).map(|t| ops::cos(0.21 * t as Sample)).collect();
        let (ref_re, ref_im) = naive_dft(&re0, &im0);
        let mut re = re0.clone();
        let mut im = im0.clone();
        fft.forward(&mut re, &mut im);
        for bin in 0..n {
            assert!((re[bin] - ref_re[bin]).abs() < 1e-3, "re bin {bin}");
            assert!((im[bin] - ref_im[bin]).abs() < 1e-3, "im bin {bin}");
        }
    }

    #[test]
    fn round_trip_recovers_input() {
        let n = 64usize;
        let fft = Fft::new(n);
        let re0: Vec<Sample> = (0..n)
            .map(|t| ops::sin(0.37 * t as Sample) + 0.5 * ops::cos(1.9 * t as Sample))
            .collect();
        let im0: Vec<Sample> = (0..n).map(|t| 0.1 * ops::sin(0.11 * t as Sample)).collect();
        let mut re = re0.clone();
        let mut im = im0.clone();
        fft.forward(&mut re, &mut im);
        fft.inverse(&mut re, &mut im);
        for t in 0..n {
            assert!((re[t] - re0[t]).abs() < 1e-4, "re[{t}] drifted");
            assert!((im[t] - im0[t]).abs() < 1e-4, "im[{t}] drifted");
        }
    }

    #[test]
    fn linearity_holds() {
        let n = 32usize;
        let fft = Fft::new(n);
        let a_re: Vec<Sample> = (0..n).map(|t| ops::sin(0.5 * t as Sample)).collect();
        let b_re: Vec<Sample> = (0..n).map(|t| ops::cos(0.3 * t as Sample)).collect();
        let zero = vec![0.0; n];

        let mut sum_re: Vec<Sample> = a_re.iter().zip(&b_re).map(|(a, b)| 2.0 * a + 3.0 * b).collect();
        let mut sum_im = zero.clone();
        fft.forward(&mut sum_re, &mut sum_im);

        let mut fa_re = a_re.clone();
        let mut fa_im = zero.clone();
        fft.forward(&mut fa_re, &mut fa_im);
        let mut fb_re = b_re.clone();
        let mut fb_im = zero.clone();
        fft.forward(&mut fb_re, &mut fb_im);

        for bin in 0..n {
            let expect_re = 2.0 * fa_re[bin] + 3.0 * fb_re[bin];
            let expect_im = 2.0 * fa_im[bin] + 3.0 * fb_im[bin];
            assert!((sum_re[bin] - expect_re).abs() < 1e-3, "re bin {bin}");
            assert!((sum_im[bin] - expect_im).abs() < 1e-3, "im bin {bin}");
        }
    }

    #[test]
    fn parseval_energy_is_conserved() {
        let n = 32usize;
        let fft = Fft::new(n);
        let re0: Vec<Sample> = (0..n).map(|t| ops::sin(0.9 * t as Sample) + 0.2).collect();
        let time_energy: f64 = re0.iter().map(|&v| f64::from(v * v)).sum();
        let mut re = re0.clone();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);
        let freq_energy: f64 = re
            .iter()
            .zip(&im)
            .map(|(&r, &i)| f64::from(r * r + i * i))
            .sum();
        // Parseval: sum|X|^2 = N * sum|x|^2 for this unnormalised forward.
        assert!(
            (freq_energy - (n as f64) * time_energy).abs() < 1e-1 * (n as f64) * time_energy,
            "energy mismatch: time {time_energy} freq {freq_energy}"
        );
    }

    #[test]
    fn minimum_size_transform_works() {
        let fft = Fft::new(2);
        let mut re = vec![1.0, 3.0];
        let mut im = vec![0.0, 0.0];
        fft.forward(&mut re, &mut im);
        // Length-2 DFT: X0 = x0 + x1, X1 = x0 - x1.
        assert!((re[0] - 4.0).abs() < 1e-6);
        assert!((re[1] - (-2.0)).abs() < 1e-6);
        fft.inverse(&mut re, &mut im);
        assert!((re[0] - 1.0).abs() < 1e-6);
        assert!((re[1] - 3.0).abs() < 1e-6);
    }
}
