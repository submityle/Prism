//! The set of engine-side targets an authored value can drive, and the unit
//! conventions attached to each.
//!
//! RTPC bindings and container blend layers all ultimately steer one of these
//! targets. Keeping the target set in one enum lets the resolver emit a single
//! uniform [`crate::action::ResolvedAction::SetParameter`] stream that a lower
//! layer translates into concrete graph/command-ring mutations.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The target
//! taxonomy is original and expressed in standard engineering units. Pure
//! data, no AI/ML.
//!
//! # Relationship
//!
//! [`ParameterTarget`] is produced by [`crate::rtpc`] bindings and consumed by
//! the resolved-action stream in [`crate::action`]. The combination of a
//! target and a scalar is a [`ParameterSetting`].

use prism_audio_core::math::Sample;

/// An addressable, continuously-controllable engine parameter.
///
/// Values are expressed in the natural engineering unit for each target so a
/// downstream translation layer needs no per-target scaling table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ParameterTarget {
    /// Voice/bus gain, in decibels (0 dB = unity).
    VolumeDb,
    /// Playback pitch, in semitones (0 = unchanged).
    PitchSemitones,
    /// A low-pass filter cutoff applied to the voice, in hertz.
    LowpassCutoffHz,
    /// A high-pass filter cutoff applied to the voice, in hertz.
    HighpassCutoffHz,
    /// Equal-power stereo pan, in `[-1, 1]` (left to right).
    Pan,
    /// Auxiliary (reverb/environment) send level, in decibels.
    AuxSendDb,
    /// Blend position for a blend container, in `[0, 1]`.
    BlendPosition,
    /// A free-form user target distinguished by a small integer slot, letting
    /// authoring data drive engine parameters this enum does not name yet.
    User(u16),
}

impl ParameterTarget {
    /// Returns a short, stable identifier string for diagnostics/telemetry.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::VolumeDb => "volume_db",
            Self::PitchSemitones => "pitch_semitones",
            Self::LowpassCutoffHz => "lowpass_cutoff_hz",
            Self::HighpassCutoffHz => "highpass_cutoff_hz",
            Self::Pan => "pan",
            Self::AuxSendDb => "aux_send_db",
            Self::BlendPosition => "blend_position",
            Self::User(_) => "user",
        }
    }

    /// Clamps a raw value into the physically meaningful range for the target,
    /// also neutralising non-finite values to the target's identity.
    #[must_use]
    pub fn clamp_value(self, value: Sample) -> Sample {
        let v = if value.is_finite() { value } else { self.identity() };
        match self {
            Self::VolumeDb | Self::AuxSendDb => v.clamp(-120.0, 24.0),
            Self::PitchSemitones => v.clamp(-48.0, 48.0),
            Self::LowpassCutoffHz | Self::HighpassCutoffHz => v.clamp(10.0, 20_000.0),
            Self::Pan => v.clamp(-1.0, 1.0),
            Self::BlendPosition => v.clamp(0.0, 1.0),
            Self::User(_) => v,
        }
    }

    /// The no-op value for the target (the value that leaves the signal
    /// unchanged), used as the fallback for non-finite inputs.
    #[must_use]
    pub fn identity(self) -> Sample {
        match self {
            Self::VolumeDb
            | Self::PitchSemitones
            | Self::Pan
            | Self::BlendPosition
            | Self::User(_) => 0.0,
            Self::AuxSendDb => -120.0,
            Self::LowpassCutoffHz => 20_000.0,
            Self::HighpassCutoffHz => 10.0,
        }
    }
}

/// A target paired with a concrete value, i.e. one resolved parameter write.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterSetting {
    /// Which engine parameter to write.
    pub target: ParameterTarget,
    /// The value to write, already clamped to the target's valid range.
    pub value: Sample,
}

impl ParameterSetting {
    /// Builds a setting, clamping `value` to the target's valid range.
    #[must_use]
    pub fn new(target: ParameterTarget, value: Sample) -> Self {
        Self { target, value: target.clamp_value(value) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn volume_and_aux_clamp_to_db_window() {
        assert!(close(ParameterTarget::VolumeDb.clamp_value(100.0), 24.0));
        assert!(close(ParameterTarget::VolumeDb.clamp_value(-500.0), -120.0));
        assert!(close(ParameterTarget::AuxSendDb.clamp_value(100.0), 24.0));
        assert!(close(ParameterTarget::AuxSendDb.clamp_value(-500.0), -120.0));
    }

    #[test]
    fn pitch_clamps_to_semitone_window() {
        assert!(close(ParameterTarget::PitchSemitones.clamp_value(100.0), 48.0));
        assert!(close(ParameterTarget::PitchSemitones.clamp_value(-100.0), -48.0));
    }

    #[test]
    fn cutoffs_clamp_to_audio_band() {
        assert!(close(ParameterTarget::LowpassCutoffHz.clamp_value(0.0), 10.0));
        assert!(close(ParameterTarget::LowpassCutoffHz.clamp_value(1.0e6), 20_000.0));
        assert!(close(ParameterTarget::HighpassCutoffHz.clamp_value(0.0), 10.0));
        assert!(close(ParameterTarget::HighpassCutoffHz.clamp_value(1.0e6), 20_000.0));
    }

    #[test]
    fn pan_and_blend_clamp_to_normalised_windows() {
        assert!(close(ParameterTarget::Pan.clamp_value(5.0), 1.0));
        assert!(close(ParameterTarget::Pan.clamp_value(-5.0), -1.0));
        assert!(close(ParameterTarget::BlendPosition.clamp_value(2.0), 1.0));
        assert!(close(ParameterTarget::BlendPosition.clamp_value(-2.0), 0.0));
    }

    #[test]
    fn user_target_is_passed_through_unclamped() {
        assert!(close(ParameterTarget::User(3).clamp_value(123.0), 123.0));
        assert!(close(ParameterTarget::User(3).clamp_value(-999.0), -999.0));
    }

    #[test]
    fn non_finite_falls_back_to_identity() {
        assert!(close(ParameterTarget::VolumeDb.clamp_value(Sample::NAN), 0.0));
        assert!(close(ParameterTarget::AuxSendDb.clamp_value(Sample::INFINITY), -120.0));
        assert!(close(ParameterTarget::LowpassCutoffHz.clamp_value(Sample::NAN), 20_000.0));
        assert!(close(ParameterTarget::HighpassCutoffHz.clamp_value(Sample::NAN), 10.0));
    }

    #[test]
    fn labels_are_stable_strings() {
        assert_eq!(ParameterTarget::VolumeDb.label(), "volume_db");
        assert_eq!(ParameterTarget::PitchSemitones.label(), "pitch_semitones");
        assert_eq!(ParameterTarget::Pan.label(), "pan");
        assert_eq!(ParameterTarget::User(9).label(), "user");
    }

    #[test]
    fn setting_new_clamps_on_construction() {
        let s = ParameterSetting::new(ParameterTarget::VolumeDb, 999.0);
        assert_eq!(s.target, ParameterTarget::VolumeDb);
        assert!(close(s.value, 24.0));
    }
}
