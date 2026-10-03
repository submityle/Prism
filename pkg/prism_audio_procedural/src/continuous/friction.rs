//! Friction noise source for sustained sliding contact.
//!
//! When two bodies grind against each other the audible result is broadband
//! noise shaped by the surfaces: faster relative motion is louder and brighter,
//! rougher surfaces spread the energy over a wider band, and the normal
//! pressure holding the bodies together scales the overall level. This source
//! realises that directly: it draws white noise from the deterministic RNG and
//! runs it through a band stage (a one-pole low-pass in series with a one-pole
//! high-pass) whose edges track a centroid and bandwidth derived from the
//! physical quantities. Its output is a per-sample drive meant to be fed into
//! the object's modal resonators so the body "colours" the raw friction, as the
//! design requires. All gain and edge changes are parameter-smoothed so a
//! change in contact state cannot click.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The filtered-noise
//! friction model and its speed/roughness mappings are standard, publicly
//! documented procedural-audio DSP.
//!
//! # Relationship
//! Implements the friction clause of design section 47.3; parameterised by
//! [`crate::material::FrictionTemplate`], excited through [`crate::rng`] and
//! [`crate::dsp::OnePole`], and summed into the modal bank by
//! [`crate::continuous::ContactVoice`].

use bevy_math::ops;

use crate::dsp::OnePole;
use crate::material::FrictionTemplate;
use crate::rng::ProceduralRng;
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

/// A deterministic filtered-noise friction generator.
#[derive(Clone, Debug)]
pub struct FrictionSource {
    rng: ProceduralRng,
    lowpass: OnePole,
    highpass: OnePole,
    gain: Smoothed,
    template: FrictionTemplate,
    sample_rate: u32,
    smoothing: Ramp,
}

impl FrictionSource {
    /// Creates a friction source seeded with `seed` using `template` at
    /// `sample_rate`, initially silent.
    #[must_use]
    pub fn new(seed: u64, template: FrictionTemplate, sample_rate: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        let centroid = template.centroid_hz.max(20.0);
        Self {
            rng: ProceduralRng::new(seed),
            lowpass: OnePole::new(centroid, sample_rate),
            highpass: OnePole::new(centroid * 0.5, sample_rate),
            gain: Smoothed::new(0.0),
            template,
            sample_rate,
            // ~8 ms level glide: fast enough to track motion, slow enough to
            // avoid zipper noise.
            smoothing: Ramp::linear_seconds(0.008, sample_rate),
        }
    }

    /// Replaces the material friction template (keeps the ringing filter state).
    #[inline]
    pub fn set_template(&mut self, template: FrictionTemplate) {
        self.template = template;
    }

    /// Updates the drive from the physical contact quantities.
    ///
    /// `tangential_speed` (m/s) raises the gain and brightness, `normal_pressure`
    /// scales the level, and `roughness` in `[0, 1]` widens the noise band. The
    /// mappings are compressive (via `tanh`) so runaway physics values cannot
    /// blow up the level.
    pub fn update(&mut self, tangential_speed: Sample, normal_pressure: Sample, roughness: Sample) {
        let speed = if tangential_speed.is_finite() {
            tangential_speed.max(0.0)
        } else {
            0.0
        };
        let pressure = if normal_pressure.is_finite() {
            normal_pressure.max(0.0)
        } else {
            0.0
        };
        let rough = if roughness.is_finite() {
            roughness.clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Compressive speed/pressure factors keep the level bounded.
        let speed_drive = ops::tanh(speed * 0.6);
        let pressure_drive = 0.3 + 0.7 * ops::tanh(pressure * 0.5);
        let target_gain = self.template.base_gain * speed_drive * pressure_drive;
        self.gain.set_target(target_gain, self.smoothing);

        // Faster motion brightens the centroid; roughness widens the band.
        let centroid = self.template.centroid_hz * (0.5 + 0.8 * speed_drive);
        let bandwidth = self.template.bandwidth_hz * (0.25 + 0.75 * rough);
        let upper = (centroid + 0.5 * bandwidth).max(40.0);
        let lower = (centroid - 0.5 * bandwidth).max(20.0);
        self.lowpass.set_cutoff(upper, self.sample_rate);
        self.highpass.set_cutoff(lower, self.sample_rate);
    }

    /// Immediately targets silence (used on separation); the gain still glides
    /// to zero so there is no click.
    #[inline]
    pub fn silence(&mut self) {
        self.gain.set_target(0.0, self.smoothing);
    }

    /// Returns the next friction drive sample.
    #[inline]
    pub fn tick(&mut self) -> Sample {
        let noise = self.rng.next_bipolar();
        let band = self.highpass.high(self.lowpass.low(noise));
        band * self.gain.next_sample()
    }

    /// Returns the current (smoothed) gain, for voice-liveness checks.
    #[inline]
    #[must_use]
    pub fn current_gain(&self) -> Sample {
        self.gain.current()
    }

    /// Clears the filter state and silences the gain immediately.
    pub fn reset(&mut self) {
        self.lowpass.reset();
        self.highpass.reset();
        self.gain.set_target(0.0, Ramp::Immediate);
        let _ = self.gain.next_sample();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template() -> FrictionTemplate {
        FrictionTemplate {
            base_gain: 0.5,
            centroid_hz: 2_000.0,
            bandwidth_hz: 1_200.0,
            rolling_density: 70.0,
        }
    }

    #[test]
    fn faster_is_louder() {
        let mut slow = FrictionSource::new(1, template(), 48_000);
        let mut fast = FrictionSource::new(1, template(), 48_000);
        slow.update(0.2, 1.0, 0.5);
        fast.update(5.0, 1.0, 0.5);
        let mut slow_peak = 0.0f32;
        let mut fast_peak = 0.0f32;
        for _ in 0..4_800 {
            slow_peak = slow_peak.max(slow.tick().abs());
            fast_peak = fast_peak.max(fast.tick().abs());
        }
        assert!(fast_peak > slow_peak, "slow={slow_peak} fast={fast_peak}");
    }

    #[test]
    fn silence_fades_to_zero() {
        let mut f = FrictionSource::new(2, template(), 48_000);
        f.update(4.0, 1.0, 0.6);
        for _ in 0..480 {
            f.tick();
        }
        f.silence();
        for _ in 0..4_800 {
            f.tick();
        }
        assert!(f.current_gain().abs() < 1e-4, "gain={}", f.current_gain());
    }

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = FrictionSource::new(9, template(), 48_000);
        let mut b = FrictionSource::new(9, template(), 48_000);
        a.update(2.0, 1.0, 0.5);
        b.update(2.0, 1.0, 0.5);
        for _ in 0..512 {
            assert_eq!(a.tick(), b.tick());
        }
    }

    #[test]
    fn output_is_finite() {
        let mut f = FrictionSource::new(3, template(), 48_000);
        f.update(f32::NAN, f32::INFINITY, 2.0);
        for _ in 0..1_000 {
            assert!(f.tick().is_finite());
        }
    }
}
