//! Continuous contact: friction, rolling, and sliding of sustained contact.
//!
//! Discrete strikes are only part of physical contact sound; two bodies that
//! stay in contact also grind, roll, and slide. This module turns the
//! sustained-contact facts from the bus into audio that shares the very same
//! modal bank the strikes use, exactly as the design requires. It is composed
//! of a filtered-noise [`friction`] source, a contact-pulse [`rolling`]
//! generator, and a per-contact [`state_machine`] that classifies the life
//! cycle and owns a click-free level fade. The [`ContactVoice`] here binds them
//! to a [`crate::modal::ModalSynth`]: friction noise and rolling impulses are
//! summed into a per-sample drive that excites (and is coloured by) the body's
//! resonators, while discrete impacts are scheduled sample-accurately on the
//! same synth. The render path performs no allocation.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 47.3 (continuous contact); driven by the contact
//! bus ([`crate::contact`]), parameterised by [`crate::material`], and renders
//! through [`crate::modal`].

pub mod friction;
pub mod rolling;
pub mod state_machine;

pub use friction::FrictionSource;
pub use rolling::RollingGenerator;
pub use state_machine::{ContactPhase, ContactStateMachine};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::contact::energy_map::{impulse_to_amplitude, normal_tangential_brightness};
use crate::contact::event::{ContactPoint, ImpactEvent, SustainEvent};
use crate::material::{FrictionTemplate, MaterialPairProfile};
use crate::modal::bank::Mode;
use crate::modal::synth::ModalSynth;
use prism_audio_core::math::Sample;

/// A complete physics-coupled contact voice: strikes plus sustained contact
/// sharing one modal bank.
#[derive(Clone, Debug)]
pub struct ContactVoice {
    synth: ModalSynth,
    friction: FrictionSource,
    rolling: RollingGenerator,
    state: ContactStateMachine,
    drive: Vec<Sample>,
    rolling_density: Sample,
    transient_len: u32,
    sample_rate: u32,
}

impl ContactVoice {
    /// Creates a contact voice.
    ///
    /// `mode_capacity` sizes the modal bank, `impact_budget` the per-block
    /// impact queue, `max_frames` the longest block the voice will render (its
    /// internal drive buffer is preallocated to this), and `seed` seeds every
    /// stochastic sub-stage so the whole voice is deterministic.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        mode_capacity: usize,
        impact_budget: usize,
        max_frames: usize,
        seed: u64,
    ) -> Self {
        let sample_rate = sample_rate.max(1);
        let max_frames = max_frames.max(1);
        let default_template = FrictionTemplate {
            base_gain: 0.5,
            centroid_hz: 2_600.0,
            bandwidth_hz: 1_560.0,
            rolling_density: 70.0,
        };
        // ~2 ms contact transient, at least 8 samples.
        let transient_len = (sample_rate / 500).max(8);
        Self {
            synth: ModalSynth::new(sample_rate, mode_capacity, impact_budget, seed),
            friction: FrictionSource::new(seed ^ 0x51ED_270B, default_template, sample_rate),
            rolling: RollingGenerator::new(seed ^ 0x27D4_EB2F, sample_rate),
            state: ContactStateMachine::new(sample_rate, 1.5),
            drive: vec![0.0; max_frames],
            rolling_density: default_template.rolling_density,
            transient_len,
            sample_rate,
        }
    }

    /// Loads the modal table and friction template from a material profile.
    pub fn configure(&mut self, profile: &MaterialPairProfile) {
        self.synth.configure(&profile.modes);
        self.friction.set_template(profile.friction);
        self.rolling_density = profile.friction.rolling_density;
    }

    /// Loads a modal table directly (bypassing the material library).
    pub fn configure_modes(&mut self, modes: &[Mode], friction: FrictionTemplate) {
        self.synth.configure(modes);
        self.friction.set_template(friction);
        self.rolling_density = friction.rolling_density;
    }

    /// Caps the number of audible modes (quality governor hook).
    #[inline]
    pub fn set_quality_cap(&mut self, modes: usize) {
        self.synth.set_quality_cap(modes);
    }

    /// Schedules a discrete impact for the next rendered block.
    pub fn trigger_impact(&mut self, impact: &ImpactEvent) {
        let amplitude = impulse_to_amplitude(impact.impulse);
        let brightness = normal_tangential_brightness(impact.normal, impact.tangential);
        self.synth.schedule_impact(
            impact.sample_offset,
            amplitude,
            impact.point,
            brightness,
            self.transient_len,
        );
        self.state.impact();
    }

    /// Schedules an impact from raw quantities (for callers not using the bus).
    pub fn trigger_impact_raw(
        &mut self,
        impulse: Sample,
        normal: Sample,
        tangential: Sample,
        point: ContactPoint,
        sample_offset: u32,
    ) {
        let amplitude = impulse_to_amplitude(impulse);
        let brightness = normal_tangential_brightness(normal, tangential);
        self.synth
            .schedule_impact(sample_offset, amplitude, point, brightness, self.transient_len);
        self.state.impact();
    }

    /// Updates the sustained-contact drive from a bus sustain event.
    pub fn update_contact(&mut self, sustain: &SustainEvent) {
        self.friction.update(
            sustain.tangential_speed,
            sustain.normal_pressure,
            sustain.roughness,
        );
        self.rolling.update(
            sustain.tangential_speed,
            sustain.normal_pressure,
            sustain.roughness,
            self.rolling_density,
        );
        self.state
            .update(sustain.tangential_speed, sustain.normal_pressure, false);
    }

    /// Marks the contact separated: the sustained drive fades out click-free.
    pub fn separate(&mut self) {
        self.friction.silence();
        self.rolling.silence();
        self.state.update(0.0, 0.0, true);
    }

    /// Renders `frames` samples into `out`.
    ///
    /// The sustained friction and rolling drive is gated by the state-machine
    /// level and summed into the modal bank, alongside any impacts scheduled for
    /// this block. No allocation is performed.
    pub fn render(&mut self, frames: usize, out: &mut [Sample]) {
        let frames = frames.min(out.len()).min(self.drive.len());
        if frames == 0 {
            return;
        }
        for slot in self.drive.iter_mut().take(frames) {
            let level = self.state.next_level();
            let fric = self.friction.tick();
            let roll = self.rolling.tick();
            *slot = level * (fric + roll);
        }
        let drive = &self.drive[..frames];
        self.synth.render(frames, Some(drive), out);
    }

    /// Returns the sample rate the voice renders at, in hertz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the current contact life-cycle phase.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> ContactPhase {
        self.state.phase()
    }

    /// Returns `true` while the voice still produces sound (ringing or active
    /// sustained contact).
    #[inline]
    #[must_use]
    pub fn is_active(&self, threshold: Sample) -> bool {
        self.synth.is_ringing(threshold) || self.state.is_active()
    }

    /// Silences the whole voice.
    pub fn reset(&mut self) {
        self.synth.reset();
        self.friction.reset();
        self.rolling.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{MaterialLibrary, MaterialPairId};

    fn voice() -> ContactVoice {
        let mut v = ContactVoice::new(48_000, 16, 8, 1024, 123);
        let lib = MaterialLibrary::default();
        v.configure(&lib.profile(MaterialPairId::new(1, 2)));
        v
    }

    #[test]
    fn impact_sounds() {
        let mut v = voice();
        v.trigger_impact_raw(5.0, 5.0, 0.0, ContactPoint::new(0.5), 0);
        let mut out = vec![0.0; 1024];
        v.render(1024, &mut out);
        let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.001, "peak={peak}");
    }

    #[test]
    fn sliding_sounds_and_separation_decays() {
        let mut v = voice();
        let sustain = SustainEvent::new(
            crate::contact::event::ContactId(1),
            MaterialPairId::new(1, 2),
            4.0,
            1.0,
            0.6,
        );
        v.update_contact(&sustain);
        assert_eq!(v.phase(), ContactPhase::Sliding);
        let mut out = vec![0.0; 1024];
        let mut sliding_peak = 0.0f32;
        for _ in 0..8 {
            v.update_contact(&sustain);
            v.render(1024, &mut out);
            sliding_peak = sliding_peak.max(out.iter().fold(0.0f32, |m, &s| m.max(s.abs())));
        }
        assert!(sliding_peak > 0.0001, "sliding_peak={sliding_peak}");

        v.separate();
        for _ in 0..40 {
            v.render(1024, &mut out);
        }
        let tail = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(tail < sliding_peak, "tail={tail} sliding={sliding_peak}");
    }

    #[test]
    fn deterministic_render() {
        let render = || {
            let mut v = voice();
            v.trigger_impact_raw(5.0, 4.0, 1.0, ContactPoint::new(0.4), 10);
            let sustain = SustainEvent::new(
                crate::contact::event::ContactId(1),
                MaterialPairId::new(1, 2),
                2.0,
                1.0,
                0.5,
            );
            v.update_contact(&sustain);
            let mut out = vec![0.0; 512];
            v.render(512, &mut out);
            out
        };
        assert_eq!(render(), render());
    }
}
