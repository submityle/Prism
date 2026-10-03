//! A parallel bank of modal resonators plus its mode table.
//!
//! A sounding object (or material pair) is modelled as a set of modes, each a
//! [`ModeResonator`]. The bank owns a fixed-capacity table of [`Mode`]
//! descriptors and a matching array of resonators, both allocated once at
//! construction. At run time it only looks up the table, injects impulses, and
//! sums the resonators per sample. An impact distributes its energy across the
//! modes using the contact-point mode-shape weights and a brightness tilt (from
//! the normal/tangential split), so where and how an object is struck changes
//! its timbre. A quality cap lets the host silence the highest modes for
//! distant or low-priority voices without rebuilding the table.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the modal bank of design section 47.2; the per-mode weights come
//! from [`crate::contact::energy_map`] and the active-mode cap mirrors the
//! quality governor of design section 32.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::contact::energy_map::contact_point_weight;
use crate::contact::event::ContactPoint;
use crate::dsp::lerp;
use crate::modal::resonator::ModeResonator;
use prism_audio_core::math::Sample;

/// Hard ceiling on the number of modes a bank can hold.
pub const MAX_MODES: usize = 48;

/// Descriptor of one mode: frequency, amplitude half-life, and output gain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Mode {
    /// Mode centre frequency in hertz.
    pub freq_hz: Sample,
    /// Amplitude half-life in seconds (longer rings longer).
    pub half_life_s: Sample,
    /// Linear output gain for the mode.
    pub gain: Sample,
}

impl Mode {
    /// Builds a mode descriptor.
    #[inline]
    #[must_use]
    pub fn new(freq_hz: Sample, half_life_s: Sample, gain: Sample) -> Self {
        Self {
            freq_hz,
            half_life_s,
            gain,
        }
    }
}

impl Default for Mode {
    #[inline]
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            half_life_s: 0.4,
            gain: 1.0,
        }
    }
}

/// A parallel bank of modal resonators.
#[derive(Clone, Debug)]
pub struct ModalBank {
    resonators: Vec<ModeResonator>,
    specs: Vec<Mode>,
    /// Number of modes loaded by the last [`ModalBank::configure`] call.
    configured: usize,
    /// Number of modes currently audible (`<= configured`), set by the quality
    /// cap.
    active: usize,
    sample_rate: u32,
}

impl ModalBank {
    /// Creates an empty bank with capacity for `capacity` modes (clamped to
    /// [`MAX_MODES`]) at `sample_rate`. All storage is preallocated.
    #[must_use]
    pub fn new(sample_rate: u32, capacity: usize) -> Self {
        let capacity = capacity.clamp(1, MAX_MODES);
        let mut resonators = Vec::with_capacity(capacity);
        let mut specs = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            resonators.push(ModeResonator::default());
            specs.push(Mode::default());
        }
        Self {
            resonators,
            specs,
            configured: 0,
            active: 0,
            sample_rate: sample_rate.max(1),
        }
    }

    /// Loads a mode table, tuning each resonator. Modes beyond the bank's
    /// capacity are ignored; the ringing state is cleared.
    pub fn configure(&mut self, modes: &[Mode]) {
        let n = modes.len().min(self.resonators.len());
        for (i, &m) in modes.iter().take(n).enumerate() {
            self.specs[i] = m;
            self.resonators[i].set_params(m.freq_hz, m.half_life_s, m.gain, self.sample_rate);
            self.resonators[i].reset();
        }
        self.configured = n;
        self.active = n;
    }

    /// Returns the capacity (maximum number of modes).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.resonators.len()
    }

    /// Returns the number of currently audible modes.
    #[inline]
    #[must_use]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Returns the number of configured modes.
    #[inline]
    #[must_use]
    pub fn configured(&self) -> usize {
        self.configured
    }

    /// Caps the number of audible modes to `n` (the quality governor hook).
    ///
    /// Clamped to the configured mode count; the silenced high modes keep their
    /// tuning so raising the cap again does not rebuild the table.
    #[inline]
    pub fn set_quality_cap(&mut self, n: usize) {
        self.active = n.min(self.configured);
    }

    /// Injects an impact, distributing `amplitude` across the active modes.
    ///
    /// Each mode `i` receives `amplitude * weight_i`, where `weight_i` combines
    /// the contact-point mode shape (rim strikes excite high modes) with a
    /// `brightness` tilt in `[0, 1]` (bright hits boost high modes, dark hits
    /// attenuate them).
    pub fn excite_impact(&mut self, amplitude: Sample, point: ContactPoint, brightness: Sample) {
        let active = self.active;
        if active == 0 {
            return;
        }
        let last = (active.saturating_sub(1)).max(1) as Sample;
        let bright = brightness.clamp(0.0, 1.0);
        for (i, res) in self.resonators.iter_mut().take(active).enumerate() {
            let p = i as Sample / last;
            let shape = contact_point_weight(i, point);
            // Dark hits roll off high modes; bright hits lift them.
            let tilt = lerp(1.0 - 0.8 * p, 0.2 + 0.8 * p, bright);
            res.excite(amplitude * shape * tilt);
        }
    }

    /// Advances every active mode one sample with the shared `input` drive and
    /// returns the summed output.
    #[inline]
    pub fn tick(&mut self, input: Sample) -> Sample {
        let mut sum = 0.0;
        for res in self.resonators.iter_mut().take(self.active) {
            sum += res.tick(input);
        }
        sum
    }

    /// Returns `true` while any active mode still rings above `threshold`.
    #[must_use]
    pub fn is_ringing(&self, threshold: Sample) -> bool {
        self.resonators
            .iter()
            .take(self.active)
            .any(|r| r.is_ringing(threshold))
    }

    /// Silences every resonator without discarding its tuning.
    pub fn reset(&mut self) {
        for res in &mut self.resonators {
            res.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_mode() -> [Mode; 3] {
        [
            Mode::new(220.0, 0.6, 1.0),
            Mode::new(440.0, 0.4, 0.7),
            Mode::new(880.0, 0.2, 0.5),
        ]
    }

    #[test]
    fn configure_sets_active() {
        let mut bank = ModalBank::new(48_000, 16);
        bank.configure(&tri_mode());
        assert_eq!(bank.active(), 3);
        assert_eq!(bank.configured(), 3);
    }

    #[test]
    fn quality_cap_clamps() {
        let mut bank = ModalBank::new(48_000, 16);
        bank.configure(&tri_mode());
        bank.set_quality_cap(1);
        assert_eq!(bank.active(), 1);
        bank.set_quality_cap(99);
        assert_eq!(bank.active(), 3);
    }

    #[test]
    fn impact_makes_sound() {
        let mut bank = ModalBank::new(48_000, 16);
        bank.configure(&tri_mode());
        bank.excite_impact(1.0, ContactPoint::new(0.5), 0.5);
        let mut peak = 0.0f32;
        for _ in 0..4_800 {
            peak = peak.max(bank.tick(0.0).abs());
        }
        assert!(peak > 0.01, "peak={peak}");
    }

    #[test]
    fn capped_to_zero_is_silent() {
        let mut bank = ModalBank::new(48_000, 16);
        bank.configure(&tri_mode());
        bank.set_quality_cap(0);
        bank.excite_impact(1.0, ContactPoint::new(0.5), 0.5);
        let mut peak = 0.0f32;
        for _ in 0..480 {
            peak = peak.max(bank.tick(0.0).abs());
        }
        assert_eq!(peak, 0.0);
    }

    #[test]
    fn over_capacity_modes_ignored() {
        let mut bank = ModalBank::new(48_000, 2);
        bank.configure(&tri_mode());
        assert_eq!(bank.active(), 2);
    }
}
