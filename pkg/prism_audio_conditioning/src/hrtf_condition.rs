//! Offline conditioning of an `HRIR` dataset for runtime binaural rendering.
//!
//! A measured head-related impulse-response set is normalised in four classic
//! `DSP` stages so the runtime convolver sees a uniform, compact dataset:
//!
//! 1. each ear's `HRIR` is resampled to the target rate,
//! 2. a diffuse-field equaliser flattens the direction-independent colouration
//!    (the average magnitude response across all measurements is inverted and
//!    applied),
//! 3. the inter-aural time difference (`ITD`) is extracted per measurement by
//!    left/right cross-correlation and stored separately, and
//! 4. each `HRIR` is converted to its minimum-phase equivalent through the real
//!    cepstrum (log-magnitude, inverse transform, causal folding, forward
//!    transform, complex exponential).
//!
//! Keeping the `ITD` out of the impulse response lets the runtime apply a clean
//! fractional delay while convolving a short minimum-phase kernel.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//! Diffuse-field equalisation and real-cepstrum minimum-phasing are classic
//! public `DSP` techniques; only the ideas are reused.
//!
//! # Relationship
//! Implements the `HRIR`-conditioning leg of design section 51. It consumes the
//! [`HrtfDataset`] type of `prism_audio_hrtf`, reuses the `prism_audio_core`
//! `FFT`, and shares the resampling helper of [`crate::resample_offline`].

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::fft::Fft;
use prism_audio_core::math::Sample;
use prism_audio_hrtf::dataset::{HrtfDataset, Measurement};

use crate::config::HrtfConditionConfig;
use crate::resample_offline::resample_channel;

/// A conditioned `HRIR` dataset: measurement-major impulse responses at the
/// target rate plus a per-measurement [`ITD`](ConditionedHrtf::itd) in samples.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditionedHrtf {
    /// Sample rate of the conditioned `HRIR`s, in Hz.
    sample_rate: u32,
    /// Length of each conditioned `HRIR`, in samples.
    hrir_len: usize,
    /// Measurement grid, carried through unchanged.
    measurements: Vec<Measurement>,
    /// Left-ear `HRIR`s, measurement-major (`measurement m` at `m * hrir_len`).
    left: Vec<Sample>,
    /// Right-ear `HRIR`s, measurement-major.
    right: Vec<Sample>,
    /// Per-measurement inter-aural time difference in samples (positive when
    /// the right ear lags the left).
    itd: Vec<Sample>,
}

impl ConditionedHrtf {
    /// The conditioned sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The length of each conditioned `HRIR`, in samples.
    #[must_use]
    pub fn hrir_len(&self) -> usize {
        self.hrir_len
    }

    /// The number of measurements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns `true` when there are no measurements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Borrows the measurement grid.
    #[must_use]
    pub fn measurements(&self) -> &[Measurement] {
        &self.measurements
    }

    /// Borrows the left-ear `HRIR` for measurement `index`.
    #[must_use]
    pub fn left_hrir(&self, index: usize) -> &[Sample] {
        &self.left[index * self.hrir_len..(index + 1) * self.hrir_len]
    }

    /// Borrows the right-ear `HRIR` for measurement `index`.
    #[must_use]
    pub fn right_hrir(&self, index: usize) -> &[Sample] {
        &self.right[index * self.hrir_len..(index + 1) * self.hrir_len]
    }

    /// The inter-aural time difference of measurement `index`, in samples.
    #[must_use]
    pub fn itd(&self, index: usize) -> Sample {
        self.itd[index]
    }
}

/// Resamples one `HRIR` to the target rate, or copies it when already there.
fn condition_resample(buf: &[Sample], source_rate: u32, target_rate: u32) -> Vec<Sample> {
    if target_rate == 0 || source_rate == 0 || target_rate == source_rate {
        return buf.to_vec();
    }
    let ratio = target_rate as Sample / source_rate as Sample;
    resample_channel(buf, ratio, 256)
}

/// Accumulates the average magnitude spectrum across every supplied `HRIR`.
fn average_magnitude(buffers: &[&[Sample]], fft: &Fft) -> Vec<Sample> {
    let size = fft.size();
    let mut avg = vec![0.0 as Sample; size];
    if buffers.is_empty() {
        return avg;
    }
    let mut re = vec![0.0 as Sample; size];
    let mut im = vec![0.0 as Sample; size];
    for buf in buffers {
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for (dst, &src) in re.iter_mut().zip(buf.iter()) {
            *dst = src;
        }
        fft.forward(&mut re, &mut im);
        for (acc, (&r, &i)) in avg.iter_mut().zip(re.iter().zip(im.iter())) {
            *acc += ops::hypot(r, i);
        }
    }
    let count = buffers.len() as Sample;
    for value in &mut avg {
        *value /= count;
    }
    avg
}

/// Builds the zero-phase diffuse-field inverse-filter magnitude from an average
/// spectrum (unity mean gain, bounded boost/cut).
fn inverse_filter(avg: &[Sample]) -> Vec<Sample> {
    let eps = 1.0e-7 as Sample;
    let mean = if avg.is_empty() {
        0.0
    } else {
        avg.iter().copied().sum::<Sample>() / avg.len() as Sample
    };
    avg.iter()
        .map(|&a| (mean / a.max(eps)).clamp(0.25, 4.0))
        .collect()
}

/// Applies a real, zero-phase magnitude filter to one `HRIR`.
fn apply_filter(buf: &[Sample], filter: &[Sample], fft: &Fft, len: usize) -> Vec<Sample> {
    let size = fft.size();
    let mut re = vec![0.0 as Sample; size];
    let mut im = vec![0.0 as Sample; size];
    for (dst, &src) in re.iter_mut().zip(buf.iter()) {
        *dst = src;
    }
    fft.forward(&mut re, &mut im);
    for ((r, i), &g) in re.iter_mut().zip(im.iter_mut()).zip(filter.iter()) {
        *r *= g;
        *i *= g;
    }
    fft.inverse(&mut re, &mut im);
    re.truncate(len);
    re
}

/// Converts one `HRIR` to its minimum-phase equivalent via the real cepstrum.
fn minimum_phase(buf: &[Sample], fft: &Fft, len: usize) -> Vec<Sample> {
    let size = fft.size();
    let eps = 1.0e-7 as Sample;

    // Log-magnitude spectrum (zero imaginary part).
    let mut re = vec![0.0 as Sample; size];
    let mut im = vec![0.0 as Sample; size];
    for (dst, &src) in re.iter_mut().zip(buf.iter()) {
        *dst = src;
    }
    fft.forward(&mut re, &mut im);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = ops::ln(ops::hypot(*r, *i).max(eps));
        *i = 0.0;
    }

    // Real cepstrum, then causal folding to enforce minimum phase.
    fft.inverse(&mut re, &mut im);
    let half = size / 2;
    for n in 0..size {
        let weight = if n == 0 || n == half {
            1.0
        } else if n < half {
            2.0
        } else {
            0.0
        };
        re[n] *= weight;
        im[n] = 0.0;
    }

    // Forward transform yields the complex log of the minimum-phase spectrum.
    fft.forward(&mut re, &mut im);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        let magnitude = ops::exp(*r);
        let phase = *i;
        *r = magnitude * ops::cos(phase);
        *i = magnitude * ops::sin(phase);
    }

    fft.inverse(&mut re, &mut im);
    re.truncate(len);
    re
}

/// Extracts the inter-aural time difference between two ears by peak
/// cross-correlation over `+/- max_lag` samples.
fn extract_itd(left: &[Sample], right: &[Sample], max_lag: usize) -> Sample {
    let len = left.len().min(right.len());
    if len == 0 {
        return 0.0;
    }
    let max_lag = max_lag.min(len.saturating_sub(1));
    let mut best_lag = 0isize;
    let mut best_corr = Sample::MIN;
    let lag_lo = -(max_lag as isize);
    let lag_hi = max_lag as isize;
    let mut lag = lag_lo;
    while lag <= lag_hi {
        let mut corr = 0.0 as Sample;
        for (n, &ln) in left.iter().enumerate() {
            let j = n as isize + lag;
            if j >= 0 && (j as usize) < len {
                corr += ln * right[j as usize];
            }
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
        lag += 1;
    }
    best_lag as Sample
}

/// Conditions `dataset` into a [`ConditionedHrtf`] under `config`.
///
/// Diffuse-field equalisation and minimum-phasing are each applied only when
/// enabled in `config`. The measurement grid is preserved, so the output always
/// has the same number of measurements as the input.
#[must_use]
pub fn condition(dataset: &HrtfDataset, config: &HrtfConditionConfig) -> ConditionedHrtf {
    let source_rate = dataset.sample_rate();
    let target_rate = if config.target_sample_rate == 0 {
        source_rate
    } else {
        config.target_sample_rate
    };
    let count = dataset.len();

    // Stage 1: resample each ear of every measurement.
    let mut left_buffers: Vec<Vec<Sample>> = Vec::with_capacity(count);
    let mut right_buffers: Vec<Vec<Sample>> = Vec::with_capacity(count);
    let mut hrir_len = usize::MAX;
    for i in 0..count {
        let l = condition_resample(dataset.left_hrir(i), source_rate, target_rate);
        let r = condition_resample(dataset.right_hrir(i), source_rate, target_rate);
        hrir_len = hrir_len.min(l.len()).min(r.len());
        left_buffers.push(l);
        right_buffers.push(r);
    }
    if hrir_len == usize::MAX {
        hrir_len = 0;
    }
    for buf in left_buffers.iter_mut().chain(right_buffers.iter_mut()) {
        buf.truncate(hrir_len);
    }

    let fft = Fft::new(hrir_len.max(1));

    // Stage 2: diffuse-field equalisation (optional).
    if config.diffuse_field_eq && hrir_len > 0 {
        let mut pool: Vec<&[Sample]> = Vec::with_capacity(count * 2);
        for buf in left_buffers.iter().chain(right_buffers.iter()) {
            pool.push(buf.as_slice());
        }
        let avg = average_magnitude(&pool, &fft);
        let filter = inverse_filter(&avg);
        for buf in left_buffers.iter_mut().chain(right_buffers.iter_mut()) {
            *buf = apply_filter(buf, &filter, &fft, hrir_len);
        }
    }

    // Stage 3: extract the ITD before any phase is discarded.
    let max_lag = ((target_rate / 1000) as usize).max(1);
    let mut itd: Vec<Sample> = Vec::with_capacity(count);
    for i in 0..count {
        itd.push(extract_itd(&left_buffers[i], &right_buffers[i], max_lag));
    }

    // Stage 4: minimum-phase conversion (optional).
    if config.minimum_phase && hrir_len > 0 {
        for buf in left_buffers.iter_mut().chain(right_buffers.iter_mut()) {
            *buf = minimum_phase(buf, &fft, hrir_len);
        }
    }

    let mut left = Vec::with_capacity(count * hrir_len);
    let mut right = Vec::with_capacity(count * hrir_len);
    for i in 0..count {
        left.extend_from_slice(&left_buffers[i]);
        right.extend_from_slice(&right_buffers[i]);
    }

    ConditionedHrtf {
        sample_rate: target_rate,
        hrir_len,
        measurements: dataset.measurements().to_vec(),
        left,
        right,
        itd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;
    use prism_audio_hrtf::dataset::HrtfDataset;

    fn decaying_tone(len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| {
                let t = n as Sample;
                ops::exp(-t * 0.1) * ops::sin(TAU * 0.1 * t)
            })
            .collect()
    }

    fn shift_right(src: &[Sample], delay: usize) -> Vec<Sample> {
        let mut out = alloc::vec![0.0 as Sample; src.len()];
        for n in 0..src.len() {
            if n >= delay {
                out[n] = src[n - delay];
            }
        }
        out
    }

    fn build_dataset(rate: u32, hrir_len: usize) -> HrtfDataset {
        let base = decaying_tone(hrir_len);
        let delayed = shift_right(&base, 10);
        // Measurement 0: front (az = 0), identical ears.
        // Measurement 1: left side, right ear delayed by 10 samples.
        let measurements = alloc::vec![
            Measurement::new(0.0, 0.0, 1.0),
            Measurement::new(-1.0, 0.0, 1.0),
        ];
        let mut left = Vec::new();
        let mut right = Vec::new();
        left.extend_from_slice(&base);
        right.extend_from_slice(&base);
        left.extend_from_slice(&base);
        right.extend_from_slice(&delayed);
        HrtfDataset::from_samples(rate, hrir_len, measurements, left, right).unwrap()
    }

    #[test]
    fn preserves_measurement_count() {
        let dataset = build_dataset(48_000, 64);
        let config = HrtfConditionConfig {
            target_sample_rate: 48_000,
            diffuse_field_eq: true,
            minimum_phase: true,
        };
        let conditioned = condition(&dataset, &config);
        assert_eq!(conditioned.len(), dataset.len());
        assert_eq!(conditioned.sample_rate(), 48_000);
    }

    #[test]
    fn front_measurement_has_zero_itd() {
        let dataset = build_dataset(48_000, 64);
        let config = HrtfConditionConfig {
            target_sample_rate: 48_000,
            diffuse_field_eq: false,
            minimum_phase: false,
        };
        let conditioned = condition(&dataset, &config);
        assert!(conditioned.itd(0).abs() < 1.0e-3, "front ITD {}", conditioned.itd(0));
        // The delayed-right measurement recovers a positive ITD near 10.
        assert!(
            (conditioned.itd(1) - 10.0).abs() < 1.5,
            "side ITD {}",
            conditioned.itd(1)
        );
    }

    #[test]
    fn resample_changes_rate_but_not_count() {
        let dataset = build_dataset(44_100, 128);
        let config = HrtfConditionConfig {
            target_sample_rate: 48_000,
            diffuse_field_eq: true,
            minimum_phase: false,
        };
        let conditioned = condition(&dataset, &config);
        assert_eq!(conditioned.sample_rate(), 48_000);
        assert_eq!(conditioned.len(), 2);
        assert!(conditioned.hrir_len() > 0);
    }
}
