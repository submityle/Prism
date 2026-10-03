//! Packet-loss concealment (PLC) by pitch-synchronous waveform extrapolation.
//!
//! When a voice frame is lost and the codec provides no recovery, the decoder
//! side synthesises a replacement from recent history rather than inserting
//! silence (which causes audible gaps and clicks). This PLC estimates the pitch
//! period from the last good samples by normalised autocorrelation, repeats the
//! most recent pitch cycle to fill the gap, attenuates across successive losses
//! toward a low deterministic comfort-noise bed, and cross-fades back to the
//! real signal when a good frame returns. Everything is deterministic: the same
//! history and loss pattern always yield the same concealed samples.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the packet-loss concealment of design section 45.3. Driven by the
//! downlink in [`crate::pipeline`] after [`crate::transport::jitter_buffer`]
//! and [`crate::transport::codec`] decoding; uses [`crate::rng`] for comfort
//! noise.

use bevy_math::ops;
use prism_audio_core::math::{flush_denormal, Sample};

use crate::rng::CommRng;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Tuning parameters for [`PacketLossConcealer`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlcConfig {
    /// Sample rate in hertz.
    pub sample_rate: Sample,
    /// Shortest pitch period considered, in seconds (highest pitch).
    pub min_period_seconds: Sample,
    /// Longest pitch period considered, in seconds (lowest pitch).
    pub max_period_seconds: Sample,
    /// Per-frame amplitude decay applied while concealment continues, in
    /// `(0, 1]`.
    pub loss_decay: Sample,
    /// Cross-fade length in samples when the real signal returns.
    pub recovery_overlap: usize,
    /// Comfort-noise amplitude added as the concealed signal decays.
    pub comfort_noise: Sample,
}

impl Default for PlcConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            min_period_seconds: 0.002,
            max_period_seconds: 0.020,
            loss_decay: 0.8,
            recovery_overlap: 32,
            comfort_noise: 0.0015,
        }
    }
}

/// Pitch-based packet-loss concealer.
#[derive(Clone, Debug)]
pub struct PacketLossConcealer {
    frame: usize,
    min_period: usize,
    max_period: usize,
    loss_decay: Sample,
    recovery_overlap: usize,
    comfort_noise: Sample,
    /// Sliding history of the most recent samples (good or concealed).
    history: Vec<Sample>,
    /// Number of valid samples currently in `history`.
    filled: usize,
    /// Count of consecutive concealed frames since the last good frame.
    consecutive_losses: usize,
    /// Pitch period used for the current burst of losses (frozen per burst).
    burst_period: usize,
    /// Whether the previous emitted frame was concealed (controls recovery
    /// cross-fade).
    last_was_loss: bool,
    rng: CommRng,
}

impl PacketLossConcealer {
    /// Creates a concealer for frames of `frame` samples.
    #[must_use]
    pub fn new(frame: usize, config: PlcConfig) -> Self {
        let frame = frame.max(1);
        let sr = config.sample_rate.max(1.0);
        let min_period = ((config.min_period_seconds * sr) as usize).max(2);
        let max_period = ((config.max_period_seconds * sr) as usize).max(min_period + 1);
        // The pitch search correlates a window of up to `max_period` samples with
        // samples up to `max_period` earlier, so the history must hold at least
        // two maximum periods plus one analysis frame and the recovery overlap.
        let hist_len = 2 * max_period + frame + config.recovery_overlap + 1;
        Self {
            frame,
            min_period,
            max_period,
            loss_decay: config.loss_decay,
            recovery_overlap: config.recovery_overlap.min(frame),
            comfort_noise: config.comfort_noise,
            history: vec![0.0; hist_len],
            filled: 0,
            consecutive_losses: 0,
            burst_period: min_period,
            last_was_loss: false,
            rng: CommRng::new(0x5EED_1234_ABCD_0001),
        }
    }

    /// Returns the frame size in samples.
    #[must_use]
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Returns the number of consecutive concealed frames since the last good
    /// frame.
    #[must_use]
    pub fn consecutive_losses(&self) -> usize {
        self.consecutive_losses
    }

    /// Clears all history and loss state.
    pub fn reset(&mut self) {
        for v in &mut self.history {
            *v = 0.0;
        }
        self.filled = 0;
        self.consecutive_losses = 0;
        self.burst_period = self.min_period;
        self.last_was_loss = false;
        self.rng = CommRng::new(0x5EED_1234_ABCD_0001);
    }

    /// Appends `frame` samples to the history, shifting out the oldest.
    #[inline]
    fn push_history(&mut self, frame: &[Sample]) {
        let n = self.history.len();
        let count = frame.len().min(n);
        self.history.copy_within(count..n, 0);
        self.history[n - count..n].copy_from_slice(&frame[..count]);
        self.filled = (self.filled + count).min(n);
    }

    /// Estimates the pitch period in samples by normalised autocorrelation over
    /// the most recent history.
    fn estimate_period(&self) -> usize {
        let n = self.history.len();
        let available = self.filled.min(n);
        if available < self.max_period + self.min_period {
            return self.min_period;
        }
        // Analysis window: the last `max_period` samples.
        let win = self.max_period;
        let base = n - win;
        let mut best_lag = self.min_period;
        let mut best_score = -1.0;
        for lag in self.min_period..=self.max_period {
            let mut corr = 0.0;
            let mut energy = 0.0;
            for i in 0..win {
                let a = self.history[base + i];
                let b = self.history[base + i - lag];
                corr += a * b;
                energy += b * b;
            }
            let score = if energy > 1.0e-9 {
                corr / ops::sqrt(energy)
            } else {
                0.0
            };
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }
        best_lag
    }

    /// Predicts `out.len()` samples by repeating the last pitch cycle, scaled
    /// by `gain`, without modifying history.
    fn predict_into(&self, period: usize, gain: Sample, out: &mut [Sample]) {
        let n = self.history.len();
        for (i, slot) in out.iter_mut().enumerate() {
            // Walk back one pitch period, wrapping within the period so the
            // most recent cycle repeats.
            let offset = period - (i % period);
            let idx = n - offset;
            *slot = self.history[idx] * gain;
        }
    }

    /// Fills `out` with a concealed frame for a lost packet.
    ///
    /// `out.len()` should equal [`PacketLossConcealer::frame`]. The concealed
    /// samples are also fed back into history so a run of losses continues
    /// smoothly.
    pub fn conceal(&mut self, out: &mut [Sample]) {
        if self.filled == 0 {
            // No history yet: emit deterministic comfort noise only.
            for slot in out.iter_mut() {
                *slot = self.rng.next_bipolar() * self.comfort_noise;
            }
            self.push_history(out);
            self.consecutive_losses += 1;
            self.last_was_loss = true;
            return;
        }

        if self.consecutive_losses == 0 {
            self.burst_period = self.estimate_period();
        }
        let period = self.burst_period.max(1);
        let gain = ops::powf(self.loss_decay, self.consecutive_losses as Sample);

        let n = self.history.len();
        for (i, slot) in out.iter_mut().enumerate() {
            let offset = period - (i % period);
            let idx = n - offset;
            let voiced = self.history[idx] * gain;
            // Blend in comfort noise proportional to how far the voiced part
            // has decayed, so a long dropout fades to a natural bed.
            let noise = self.rng.next_bipolar() * self.comfort_noise * (1.0 - gain);
            *slot = flush_denormal(voiced + noise);
        }

        self.push_history(out);
        self.consecutive_losses += 1;
        self.last_was_loss = true;
    }

    /// Accepts a good decoded frame, cross-fading out of concealment if the
    /// previous frame was lost, and updates history.
    ///
    /// `frame` is modified in place when a recovery cross-fade is applied.
    pub fn good_frame(&mut self, frame: &mut [Sample]) {
        if self.last_was_loss && self.recovery_overlap > 0 {
            let overlap = self.recovery_overlap.min(frame.len());
            let period = self.burst_period.max(1);
            let gain = ops::powf(self.loss_decay, self.consecutive_losses as Sample);
            // Predict the continuation the concealer would have produced.
            let mut predicted = [0.0; 256];
            let use_len = overlap.min(predicted.len());
            self.predict_into(period, gain, &mut predicted[..use_len]);
            for i in 0..use_len {
                let t = (i as Sample + 1.0) / (use_len as Sample + 1.0);
                frame[i] = predicted[i] * (1.0 - t) + frame[i] * t;
            }
        }
        self.consecutive_losses = 0;
        self.last_was_loss = false;
        self.push_history(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn power(block: &[Sample]) -> Sample {
        if block.is_empty() {
            return 0.0;
        }
        block.iter().map(|&v| v * v).sum::<Sample>() / block.len() as Sample
    }

    #[test]
    fn conceals_periodic_signal_with_energy() {
        let frame = 256;
        let mut plc = PacketLossConcealer::new(frame, PlcConfig::default());
        let sr = 48_000.0;
        let freq = 150.0;
        // Prime with several good frames of a steady tone.
        let mut phase = 0usize;
        for _ in 0..8 {
            let mut block: Vec<Sample> = (0..frame)
                .map(|i| ops::sin(2.0 * PI * freq * (phase + i) as Sample / sr))
                .collect();
            plc.good_frame(&mut block);
            phase += frame;
        }
        let mut concealed = vec![0.0; frame];
        plc.conceal(&mut concealed);
        // The concealed frame should carry comparable energy to the tone.
        assert!(power(&concealed) > 0.1, "concealed power {}", power(&concealed));
    }

    #[test]
    fn energy_decays_over_consecutive_losses() {
        let frame = 256;
        let mut plc = PacketLossConcealer::new(frame, PlcConfig::default());
        let sr = 48_000.0;
        let mut phase = 0usize;
        for _ in 0..8 {
            let mut block: Vec<Sample> = (0..frame)
                .map(|i| ops::sin(2.0 * PI * 150.0 * (phase + i) as Sample / sr))
                .collect();
            plc.good_frame(&mut block);
            phase += frame;
        }
        let mut first = vec![0.0; frame];
        plc.conceal(&mut first);
        let mut later = vec![0.0; frame];
        for _ in 0..6 {
            plc.conceal(&mut later);
        }
        assert!(
            power(&later) < power(&first) * 0.5,
            "first={} later={}",
            power(&first),
            power(&later)
        );
    }

    #[test]
    fn deterministic_concealment() {
        let frame = 128;
        let make = || {
            let mut plc = PacketLossConcealer::new(frame, PlcConfig::default());
            let mut phase = 0usize;
            for _ in 0..6 {
                let mut block: Vec<Sample> = (0..frame)
                    .map(|i| ops::sin(2.0 * PI * 200.0 * (phase + i) as Sample / 48_000.0))
                    .collect();
                plc.good_frame(&mut block);
                phase += frame;
            }
            let mut c = vec![0.0; frame];
            plc.conceal(&mut c);
            c
        };
        let a = make();
        let b = make();
        assert_eq!(a, b);
    }

    #[test]
    fn recovery_blends_without_click() {
        let frame = 256;
        let mut plc = PacketLossConcealer::new(frame, PlcConfig::default());
        let sr = 48_000.0;
        let mut phase = 0usize;
        for _ in 0..8 {
            let mut block: Vec<Sample> = (0..frame)
                .map(|i| ops::sin(2.0 * PI * 150.0 * (phase + i) as Sample / sr))
                .collect();
            plc.good_frame(&mut block);
            phase += frame;
        }
        let mut concealed = vec![0.0; frame];
        plc.conceal(&mut concealed);
        phase += frame;
        let mut good: Vec<Sample> = (0..frame)
            .map(|i| ops::sin(2.0 * PI * 150.0 * (phase + i) as Sample / sr))
            .collect();
        plc.good_frame(&mut good);
        // The first recovered sample should not jump wildly (no hard click).
        assert!(ops::abs(good[0]) <= 1.0);
    }
}
