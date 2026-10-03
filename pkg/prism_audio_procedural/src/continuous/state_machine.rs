//! Per-contact life-cycle state machine with click-free level fading.
//!
//! A single tracked contact moves through a small life cycle: it begins with a
//! discrete impact, settles into rolling while the two bodies stay pressed and
//! move slowly, transitions to sliding once the tangential speed exceeds a
//! threshold, and ends in separation when the bodies part or all motion and
//! pressure vanish. This type is the deterministic classifier for that cycle.
//! Crucially it also owns a smoothed overall level that it drives to zero on
//! separation (and up from zero when a contact first becomes active), so a
//! contact that stops dead never produces a click: the audio tails off over a
//! short, fixed glide instead of truncating.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the per-contact state machine of design section 47.3; the level
//! envelope it produces gates the friction and rolling mix inside
//! [`crate::continuous::ContactVoice`].

use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

/// The life-cycle phase of a tracked contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ContactPhase {
    /// A fresh discrete strike; the modal bank is being impulse-excited.
    Impact,
    /// Pressed contact with low tangential speed: a contact-pulse train.
    Rolling,
    /// Pressed contact with high tangential speed: filtered-noise friction.
    Sliding,
    /// The bodies have parted (or all motion and pressure have ceased).
    Separated,
}

/// Classifier and level envelope for one contact's life cycle.
#[derive(Clone, Debug)]
pub struct ContactStateMachine {
    phase: ContactPhase,
    level: Smoothed,
    slide_speed: Sample,
    activity_floor: Sample,
    fade: Ramp,
}

impl ContactStateMachine {
    /// Creates a state machine at `sample_rate` with the given rolling/sliding
    /// speed boundary (`m/s`), starting separated and silent.
    #[must_use]
    pub fn new(sample_rate: u32, slide_speed: Sample) -> Self {
        let sample_rate = sample_rate.max(1);
        Self {
            phase: ContactPhase::Separated,
            level: Smoothed::new(0.0),
            slide_speed: slide_speed.max(0.0),
            activity_floor: 0.02,
            // ~12 ms fade: inaudible as a click, short enough to feel immediate.
            fade: Ramp::linear_seconds(0.012, sample_rate),
        }
    }

    /// Marks a discrete impact: the contact becomes active immediately.
    pub fn impact(&mut self) {
        self.phase = ContactPhase::Impact;
        self.level.set_target(1.0, self.fade);
    }

    /// Updates the phase from the latest sustained-contact quantities.
    ///
    /// `separated` forces the [`ContactPhase::Separated`] state and a fade to
    /// zero. Otherwise, if there is meaningful motion or pressure the contact is
    /// classified as rolling (slow) or sliding (fast) and faded up; if both fall
    /// below the activity floor the contact is treated as separated.
    pub fn update(&mut self, tangential_speed: Sample, normal_pressure: Sample, separated: bool) {
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

        let inactive = separated || (speed < self.activity_floor && pressure < self.activity_floor);
        if inactive {
            self.phase = ContactPhase::Separated;
            self.level.set_target(0.0, self.fade);
            return;
        }

        self.phase = if speed >= self.slide_speed {
            ContactPhase::Sliding
        } else {
            ContactPhase::Rolling
        };
        self.level.set_target(1.0, self.fade);
    }

    /// Advances the level envelope one sample and returns it in `[0, 1]`.
    #[inline]
    pub fn next_level(&mut self) -> Sample {
        self.level.next_sample()
    }

    /// Returns the current life-cycle phase.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> ContactPhase {
        self.phase
    }

    /// Returns the current smoothed level without advancing it.
    #[inline]
    #[must_use]
    pub fn level(&self) -> Sample {
        self.level.current()
    }

    /// Returns `true` while the contact is active or still fading out.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.phase != ContactPhase::Separated || self.level.current() > 1.0e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_roll_and_slide() {
        let mut sm = ContactStateMachine::new(48_000, 1.5);
        sm.update(0.5, 1.0, false);
        assert_eq!(sm.phase(), ContactPhase::Rolling);
        sm.update(3.0, 1.0, false);
        assert_eq!(sm.phase(), ContactPhase::Sliding);
    }

    #[test]
    fn separation_fades_out() {
        let mut sm = ContactStateMachine::new(48_000, 1.5);
        sm.update(2.0, 1.0, false);
        for _ in 0..2_000 {
            sm.next_level();
        }
        assert!(sm.level() > 0.5);
        sm.update(0.0, 0.0, true);
        for _ in 0..4_000 {
            sm.next_level();
        }
        assert!(sm.level() < 1.0e-3, "level={}", sm.level());
        assert_eq!(sm.phase(), ContactPhase::Separated);
    }

    #[test]
    fn impact_activates() {
        let mut sm = ContactStateMachine::new(48_000, 1.5);
        sm.impact();
        assert_eq!(sm.phase(), ContactPhase::Impact);
        for _ in 0..2_000 {
            sm.next_level();
        }
        assert!(sm.level() > 0.5);
    }

    #[test]
    fn tiny_motion_is_separated() {
        let mut sm = ContactStateMachine::new(48_000, 1.5);
        sm.update(0.0, 0.0, false);
        assert_eq!(sm.phase(), ContactPhase::Separated);
    }
}
