//! The pluggable haptic backend trait and three concrete implementations.
//!
//! A backend is the sink that turns a [`crate::waveform::HapticWaveform`] into
//! device motion. The engine renders one haptic signal and submits it to
//! whatever backend the active device exposes:
//!
//! * [`WideBandBackend`] consumes the wide-band waveform directly, as a
//!   high-fidelity voice-coil actuator would.
//! * [`DualMotorBackend`] collapses the low and high bands into two scalar
//!   motor intensities, as a classic two-motor rumble pad would.
//! * [`SilentBackend`] is the inert fallback for devices with no haptics; it
//!   accepts submissions and does nothing audible or physical.
//!
//! Every backend also reports its intrinsic latency and capabilities so the
//! [`crate::latency`] aligner and [`crate::governor`] can plan around it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the backend layer of design section 36. Latency reporting feeds
//! the plugin-delay-compensation contract of design section 29 via
//! [`crate::latency::LatencyAligner`].

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::waveform::HapticWaveform;

/// Static description of what a backend can reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HapticCapabilities {
    /// Whether the device reproduces a continuous wide-band waveform.
    pub wide_band: bool,
    /// Number of independently driven actuators.
    pub actuator_count: u8,
    /// Maximum haptic update rate the device accepts, in hertz.
    pub max_rate_hz: u32,
}

/// A sink that renders a haptic waveform on a physical device.
///
/// Implementations keep whatever state they need to drive hardware; the engine
/// only requires that a submitted waveform be consumed, that the device's fixed
/// processing latency be reported, and that its capabilities be queryable.
pub trait HapticBackend {
    /// Consumes one haptic waveform block and drives the device.
    fn submit(&mut self, waveform: &HapticWaveform);

    /// Returns the device's intrinsic latency in haptic-rate samples.
    fn intrinsic_latency_samples(&self) -> u32;

    /// Returns the device's static capabilities.
    fn capabilities(&self) -> HapticCapabilities;
}

/// Returns the mean rectified level of a channel, clamped to `[0, 1]`.
fn mean_abs(samples: &[Sample]) -> Sample {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: Sample = samples.iter().map(|s| ops::abs(*s)).sum();
    (sum / samples.len() as Sample).clamp(0.0, 1.0)
}

/// A wide-band backend that consumes the full haptic waveform.
///
/// It tracks the peak level and total frame count it has received so a host can
/// confirm the device is being driven.
#[derive(Debug, Clone)]
pub struct WideBandBackend {
    latency_samples: u32,
    max_rate_hz: u32,
    last_peak: Sample,
    frames_received: u64,
}

impl WideBandBackend {
    /// Builds a wide-band backend with the given latency and maximum rate.
    #[must_use]
    pub fn new(latency_samples: u32, max_rate_hz: u32) -> Self {
        Self {
            latency_samples,
            max_rate_hz: max_rate_hz.max(1),
            last_peak: 0.0,
            frames_received: 0,
        }
    }

    /// Returns the peak level of the most recently submitted waveform.
    #[inline]
    #[must_use]
    pub fn last_peak(&self) -> Sample {
        self.last_peak
    }

    /// Returns the total number of haptic frames submitted so far.
    #[inline]
    #[must_use]
    pub fn frames_received(&self) -> u64 {
        self.frames_received
    }
}

impl HapticBackend for WideBandBackend {
    fn submit(&mut self, waveform: &HapticWaveform) {
        let mut peak = 0.0;
        for ch in 0..waveform.channel_count() {
            for &s in waveform.channel(ch) {
                let a = ops::abs(s);
                if a > peak {
                    peak = a;
                }
            }
        }
        self.last_peak = peak;
        self.frames_received += waveform.len() as u64;
    }

    fn intrinsic_latency_samples(&self) -> u32 {
        self.latency_samples
    }

    fn capabilities(&self) -> HapticCapabilities {
        HapticCapabilities {
            wide_band: true,
            actuator_count: 1,
            max_rate_hz: self.max_rate_hz,
        }
    }
}

/// A dual-motor rumble backend driven by two band envelopes.
///
/// The low actuator channel drives the low-frequency motor and the high channel
/// drives the high-frequency motor; a mono waveform drives both motors from its
/// single channel. Each motor intensity is the mean rectified level of its band
/// over the submitted block.
#[derive(Debug, Clone)]
pub struct DualMotorBackend {
    latency_samples: u32,
    max_rate_hz: u32,
    low_motor: Sample,
    high_motor: Sample,
}

impl DualMotorBackend {
    /// Builds a dual-motor backend with the given latency and maximum rate.
    #[must_use]
    pub fn new(latency_samples: u32, max_rate_hz: u32) -> Self {
        Self {
            latency_samples,
            max_rate_hz: max_rate_hz.max(1),
            low_motor: 0.0,
            high_motor: 0.0,
        }
    }

    /// Returns the latest `(low_motor, high_motor)` intensities in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn motors(&self) -> (Sample, Sample) {
        (self.low_motor, self.high_motor)
    }
}

impl HapticBackend for DualMotorBackend {
    fn submit(&mut self, waveform: &HapticWaveform) {
        let channels = waveform.channel_count();
        if channels == 0 {
            self.low_motor = 0.0;
            self.high_motor = 0.0;
            return;
        }
        self.low_motor = mean_abs(waveform.channel(0));
        let high_index = if channels > 1 { 1 } else { 0 };
        self.high_motor = mean_abs(waveform.channel(high_index));
    }

    fn intrinsic_latency_samples(&self) -> u32 {
        self.latency_samples
    }

    fn capabilities(&self) -> HapticCapabilities {
        HapticCapabilities {
            wide_band: false,
            actuator_count: 2,
            max_rate_hz: self.max_rate_hz,
        }
    }
}

/// An inert fallback backend for devices without haptics.
///
/// It records how many submissions it has seen so callers can tell the fallback
/// is wired up, but it produces no motion.
#[derive(Debug, Clone, Default)]
pub struct SilentBackend {
    submissions: u64,
}

impl SilentBackend {
    /// Builds a silent fallback backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns how many waveforms have been submitted and discarded.
    #[inline]
    #[must_use]
    pub fn submissions(&self) -> u64 {
        self.submissions
    }
}

impl HapticBackend for SilentBackend {
    fn submit(&mut self, _waveform: &HapticWaveform) {
        self.submissions += 1;
    }

    fn intrinsic_latency_samples(&self) -> u32 {
        0
    }

    fn capabilities(&self) -> HapticCapabilities {
        HapticCapabilities {
            wide_band: false,
            actuator_count: 0,
            max_rate_hz: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::{ActuatorLayout, HapticWaveform};

    fn dual(low: Sample, high: Sample, frames: usize) -> HapticWaveform {
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Dual);
        for _ in 0..frames {
            wf.push_frame(&[low, high]);
        }
        wf
    }

    #[test]
    fn wide_band_tracks_peak_and_frames() {
        let mut be = WideBandBackend::new(32, 1_000);
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Mono);
        wf.push_frame(&[0.2]);
        wf.push_frame(&[0.8]);
        wf.push_frame(&[-0.5]);
        be.submit(&wf);
        assert!((be.last_peak() - 0.8).abs() < 1e-6);
        assert_eq!(be.frames_received(), 3);
        assert_eq!(be.intrinsic_latency_samples(), 32);
        assert!(be.capabilities().wide_band);
    }

    #[test]
    fn dual_motor_splits_bands() {
        let mut be = DualMotorBackend::new(16, 1_000);
        be.submit(&dual(0.4, 0.1, 10));
        let (low, high) = be.motors();
        assert!((low - 0.4).abs() < 1e-6);
        assert!((high - 0.1).abs() < 1e-6);
        assert_eq!(be.capabilities().actuator_count, 2);
    }

    #[test]
    fn dual_motor_mono_drives_both() {
        let mut be = DualMotorBackend::new(16, 1_000);
        let mut wf = HapticWaveform::new(1_000, ActuatorLayout::Mono);
        for _ in 0..8 {
            wf.push_frame(&[0.6]);
        }
        be.submit(&wf);
        let (low, high) = be.motors();
        assert!((low - high).abs() < 1e-6);
        assert!((low - 0.6).abs() < 1e-6);
    }

    #[test]
    fn dual_motor_clamps_to_unit() {
        let mut be = DualMotorBackend::new(0, 1_000);
        be.submit(&dual(5.0, 9.0, 4));
        let (low, high) = be.motors();
        assert!((low - 1.0).abs() < 1e-6);
        assert!((high - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silent_counts_but_does_nothing() {
        let mut be = SilentBackend::new();
        be.submit(&dual(1.0, 1.0, 4));
        be.submit(&dual(1.0, 1.0, 4));
        assert_eq!(be.submissions(), 2);
        assert_eq!(be.intrinsic_latency_samples(), 0);
        assert_eq!(be.capabilities().actuator_count, 0);
    }
}
