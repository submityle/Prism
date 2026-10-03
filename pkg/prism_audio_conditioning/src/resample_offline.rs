//! High-quality offline sample-rate conversion.
//!
//! Each channel is resampled independently with the long-kernel
//! [`HighOrderSincResampler`] from `prism_audio_resample`, driven to
//! completion in bounded blocks. The planar layout is preserved and the frame
//! count is recomputed from the produced output; an asset already at the target
//! rate is cloned unchanged.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! resampler itself lives in `prism_audio_resample` and is reused, not
//! reimplemented.
//!
//! # Relationship
//! Implements the offline resampling leg of design section 51, normalising
//! every decoded asset to the project rate before analysis and authoring.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;
use prism_audio_resample::{HighOrderSincResampler, Resampler};

use crate::config::ResampleConfig;
use crate::pcm::ConditionedPcm;

/// Resamples a single channel to completion using the long-kernel resampler.
pub(crate) fn resample_channel(
    input: &[Sample],
    ratio: Sample,
    chunk_frames: usize,
) -> Vec<Sample> {
    let mut resampler = HighOrderSincResampler::new();
    resampler.set_ratio(ratio);
    resampler.reset();

    let chunk = chunk_frames.max(1);
    let mut scratch = vec![0.0 as Sample; chunk];
    let mut output: Vec<Sample> = Vec::new();
    let mut offset = 0usize;

    loop {
        let progress = resampler.process(&input[offset..], &mut scratch);
        output.extend_from_slice(&scratch[..progress.produced]);
        offset += progress.consumed;
        if progress.produced == 0 && progress.consumed == 0 {
            break;
        }
        if offset >= input.len() && progress.produced == 0 {
            break;
        }
    }
    output
}

/// Resamples every channel of `pcm` to `target_rate`.
///
/// When `pcm` is already at `target_rate` (or either rate is zero) the input is
/// cloned. Otherwise every channel is converted with the same ratio and reset
/// state, so all output channels share one length and the planar invariant is
/// preserved. The returned container reports `target_rate`.
#[must_use]
pub fn resample_to(
    pcm: &ConditionedPcm,
    target_rate: u32,
    config: &ResampleConfig,
) -> ConditionedPcm {
    let source_rate = pcm.sample_rate();
    if target_rate == 0 || source_rate == 0 || target_rate == source_rate {
        return pcm.clone();
    }

    let ratio = target_rate as Sample / source_rate as Sample;
    let mut channels: Vec<Vec<Sample>> = Vec::with_capacity(pcm.channel_count());
    let mut min_len = usize::MAX;
    for ch in 0..pcm.channel_count() {
        let input = pcm.channel(ch).unwrap_or(&[]);
        let out = resample_channel(input, ratio, config.chunk_frames);
        min_len = min_len.min(out.len());
        channels.push(out);
    }

    // Guard the planar invariant: trim every channel to the shortest length
    // (equal-length inputs with equal ratios already agree; this only ever
    // trims a trailing partial tap).
    if min_len != usize::MAX {
        for channel in &mut channels {
            channel.truncate(min_len);
        }
    }

    ConditionedPcm::new(target_rate, pcm.layout(), channels)
        .unwrap_or_else(|_| ConditionedPcm::silence(target_rate, pcm.layout(), 0).unwrap_or_else(|_| pcm.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;
    use prism_audio_core::buffer::ChannelLayout;

    fn sine(freq: Sample, rate: u32, frames: usize) -> Vec<Sample> {
        (0..frames)
            .map(|i| 0.5 * ops::sin(TAU * freq * i as Sample / rate as Sample))
            .collect()
    }

    fn rms(x: &[Sample]) -> Sample {
        if x.is_empty() {
            return 0.0;
        }
        let sum: Sample = x.iter().map(|&v| v * v).sum();
        ops::sqrt(sum / x.len() as Sample)
    }

    #[test]
    fn same_rate_is_cloned() {
        let pcm =
            ConditionedPcm::new(48_000, ChannelLayout::Mono, vec![sine(1_000.0, 48_000, 1_000)])
                .unwrap();
        let out = resample_to(&pcm, 48_000, &ResampleConfig::default());
        assert_eq!(out, pcm);
    }

    #[test]
    fn upsampling_scales_length() {
        let frames = 4_000;
        let pcm = ConditionedPcm::new(
            24_000,
            ChannelLayout::Mono,
            vec![sine(1_000.0, 24_000, frames)],
        )
        .unwrap();
        let out = resample_to(&pcm, 48_000, &ResampleConfig::default());
        assert_eq!(out.sample_rate(), 48_000);
        // Roughly double the frames (allow for kernel edge effects).
        let expected = frames * 2;
        let diff = (out.frames() as isize - expected as isize).unsigned_abs();
        assert!(diff < 400, "frames {} vs ~{}", out.frames(), expected);
    }

    #[test]
    fn round_trip_preserves_energy() {
        let rate = 48_000;
        let frames = 8_000;
        let pcm =
            ConditionedPcm::new(rate, ChannelLayout::Mono, vec![sine(1_000.0, rate, frames)])
                .unwrap();
        let down = resample_to(&pcm, 24_000, &ResampleConfig::default());
        let up = resample_to(&down, rate, &ResampleConfig::default());

        // Compare interior energy (skip transient kernel edges).
        let original = pcm.channel(0).unwrap();
        let restored = up.channel(0).unwrap();
        let guard = 256usize;
        let common = original.len().min(restored.len());
        assert!(common > 2 * guard);
        let a = &original[guard..common - guard];
        let b = &restored[guard..common - guard];
        let rms_err = rms(
            &a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x - y)
                .collect::<Vec<Sample>>(),
        );
        assert!(rms_err < 0.1, "round-trip RMS error {rms_err}");
    }
}
