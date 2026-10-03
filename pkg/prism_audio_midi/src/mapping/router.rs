//! Routing a voice's expression snapshot into sample-offset modulation writes.
//!
//! A [`VoiceExpression`] is the current expression state of one sounding voice:
//! its channel and per-note pitch bend, pressures, timbre, velocity, and any
//! per-voice controllers. An [`ExpressionRouter`] holds a list of
//! [`TargetMapping`]s and, for a voice at a given sample offset, emits one
//! [`ModulationWrite`] per mapping. The output is pure data describing what the
//! engine's modulation layer should write and when; this module never touches a
//! sample. Pitch bend is folded from the raw channel and per-note values into a
//! single semitone offset so a Pitch mapping needs no further scaling.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the router of design section 52 (expression to modulation). It
//! reads expression state tracked by [`crate::expression`] and emits writes for
//! the engine's section 12 modulation routing at offsets produced by section 8
//! event scheduling.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::expression::channel_state::{DEFAULT_PITCH_BEND_RANGE, normalized_bend};
use crate::mapping::target::{
    ExpressionDimension, ModulationTarget, TargetMapping, normalize_u16, normalize_u32,
};
use crate::ump::message::PITCH_BEND_CENTER_32;
use prism_audio_core::math::Sample;

/// The maximum number of per-voice controllers carried in a snapshot.
pub const MAX_VOICE_CONTROLLERS: usize = 8;

/// A snapshot of one voice's expression state.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceExpression {
    /// The engine voice identifier the writes are attributed to.
    pub voice_id: u32,
    /// The note number (`0`-`127`).
    pub note: u8,
    /// The raw 32-bit channel pitch bend, centred at `0x8000_0000`.
    pub channel_pitch_bend: u32,
    /// The raw 32-bit per-note pitch bend, centred at `0x8000_0000`.
    pub per_note_pitch_bend: u32,
    /// The shared pitch-bend range in semitones.
    pub pitch_bend_range_semitones: Sample,
    /// The raw 32-bit channel pressure.
    pub channel_pressure: u32,
    /// The raw 32-bit per-note pressure.
    pub per_note_pressure: u32,
    /// The raw 32-bit timbre (brightness) value.
    pub timbre: u32,
    /// The 16-bit note-on velocity.
    pub velocity: u16,
    controllers: [(u8, u32); MAX_VOICE_CONTROLLERS],
    controller_count: usize,
}

impl VoiceExpression {
    /// Creates a snapshot for `voice_id` playing `note` with every dimension at
    /// its neutral default: pitch bend centred, the default pitch-bend range,
    /// no pressure, no timbre, zero velocity, and no controllers.
    #[must_use]
    pub const fn new(voice_id: u32, note: u8) -> Self {
        Self {
            voice_id,
            note,
            channel_pitch_bend: PITCH_BEND_CENTER_32,
            per_note_pitch_bend: PITCH_BEND_CENTER_32,
            pitch_bend_range_semitones: DEFAULT_PITCH_BEND_RANGE,
            channel_pressure: 0,
            per_note_pressure: 0,
            timbre: 0,
            velocity: 0,
            controllers: [(0, 0); MAX_VOICE_CONTROLLERS],
            controller_count: 0,
        }
    }

    /// Sets a per-voice controller value by index. Updates an existing entry or
    /// inserts a new one while free slots remain. Returns `true` when stored.
    pub fn set_controller(&mut self, index: u8, value: u32) -> bool {
        for entry in &mut self.controllers[..self.controller_count] {
            if entry.0 == index {
                entry.1 = value;
                return true;
            }
        }
        if self.controller_count < MAX_VOICE_CONTROLLERS {
            self.controllers[self.controller_count] = (index, value);
            self.controller_count += 1;
            true
        } else {
            false
        }
    }

    /// Returns a per-voice controller value by index, or `None` when unset.
    #[must_use]
    pub fn controller(&self, index: u8) -> Option<u32> {
        self.controllers[..self.controller_count]
            .iter()
            .find(|entry| entry.0 == index)
            .map(|entry| entry.1)
    }

    /// Returns the combined channel-plus-per-note pitch bend in semitones.
    #[must_use]
    pub fn pitch_bend_semitones(&self) -> Sample {
        let combined = normalized_bend(self.channel_pitch_bend)
            + normalized_bend(self.per_note_pitch_bend);
        combined * self.pitch_bend_range_semitones
    }

    /// Returns the combined pressure in the unipolar `[0, 1]` range.
    #[must_use]
    pub fn pressure_unipolar(&self) -> Sample {
        let combined =
            normalize_u32(self.channel_pressure) + normalize_u32(self.per_note_pressure);
        combined.clamp(0.0, 1.0)
    }

    /// Returns the raw value of `dimension` for this voice, before any mapping
    /// curve, depth, or offset is applied.
    #[must_use]
    pub fn dimension_value(&self, dimension: ExpressionDimension) -> Sample {
        match dimension {
            ExpressionDimension::PitchBend => self.pitch_bend_semitones(),
            ExpressionDimension::Pressure => self.pressure_unipolar(),
            ExpressionDimension::Timbre => normalize_u32(self.timbre),
            ExpressionDimension::Velocity => normalize_u16(self.velocity),
            ExpressionDimension::Controller { index } => {
                normalize_u32(self.controller(index).unwrap_or(0))
            }
        }
    }
}

/// A single modulation write: a target, an amount, the voice it belongs to, and
/// the sample offset at which it takes effect.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ModulationWrite {
    /// The sample offset within the processing block.
    pub sample_offset: u32,
    /// The voice the write targets.
    pub voice_id: u32,
    /// The modulation destination.
    pub target: ModulationTarget,
    /// The modulation amount.
    pub value: Sample,
}

/// A router that turns voice expression into modulation writes using a fixed
/// list of mappings.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ExpressionRouter {
    mappings: Vec<TargetMapping>,
}

impl ExpressionRouter {
    /// Creates a router with no mappings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }

    /// Creates a router from a list of mappings.
    #[must_use]
    pub fn from_mappings(mappings: Vec<TargetMapping>) -> Self {
        Self { mappings }
    }

    /// Adds a mapping to the router.
    pub fn push(&mut self, mapping: TargetMapping) {
        self.mappings.push(mapping);
    }

    /// Returns the configured mappings.
    #[must_use]
    pub fn mappings(&self) -> &[TargetMapping] {
        &self.mappings
    }

    /// Appends one modulation write per mapping for `voice` at `sample_offset`
    /// to `out`. The relative order of writes matches the mapping order, so the
    /// output is deterministic.
    pub fn route(&self, voice: &VoiceExpression, sample_offset: u32, out: &mut Vec<ModulationWrite>) {
        for mapping in &self.mappings {
            let raw = voice.dimension_value(mapping.dimension);
            out.push(ModulationWrite {
                sample_offset,
                voice_id: voice.voice_id,
                target: mapping.target,
                value: mapping.apply(raw),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::{ExpressionRouter, ModulationWrite, VoiceExpression};
    use crate::mapping::target::{
        Curve, ExpressionDimension, ModulationTarget, TargetMapping,
    };
    use crate::ump::message::PITCH_BEND_CENTER_32;

    const EPS: f32 = 1.0e-4;

    #[test]
    fn centered_voice_has_zero_pitch_bend() {
        let voice = VoiceExpression::new(7, 60);
        assert!(voice.pitch_bend_semitones().abs() < EPS);
    }

    #[test]
    fn channel_and_per_note_bend_add() {
        let mut voice = VoiceExpression::new(1, 60);
        voice.pitch_bend_range_semitones = 2.0;
        // Full positive channel bend, centred per-note bend -> +2 semitones.
        voice.channel_pitch_bend = u32::MAX;
        voice.per_note_pitch_bend = PITCH_BEND_CENTER_32;
        assert!((voice.pitch_bend_semitones() - 2.0).abs() < 1.0e-3);
        // Add a full positive per-note bend as well -> about +4 semitones.
        voice.per_note_pitch_bend = u32::MAX;
        assert!((voice.pitch_bend_semitones() - 4.0).abs() < 1.0e-3);
    }

    #[test]
    fn pressure_combines_and_clamps() {
        let mut voice = VoiceExpression::new(1, 60);
        voice.channel_pressure = u32::MAX;
        voice.per_note_pressure = u32::MAX;
        // Both at maximum sum beyond one and clamp to exactly one.
        assert!((voice.pressure_unipolar() - 1.0).abs() < EPS);
    }

    #[test]
    fn router_emits_one_write_per_mapping() {
        let mut router = ExpressionRouter::new();
        router.push(TargetMapping::linear(
            ExpressionDimension::PitchBend,
            ModulationTarget::Pitch,
        ));
        router.push(TargetMapping::new(
            ExpressionDimension::Pressure,
            ModulationTarget::Gain,
            1.0,
            0.0,
            Curve::Linear,
        ));

        let mut voice = VoiceExpression::new(42, 64);
        voice.pitch_bend_range_semitones = 2.0;
        voice.channel_pitch_bend = u32::MAX;
        voice.per_note_pressure = u32::MAX;

        let mut writes: Vec<ModulationWrite> = Vec::new();
        router.route(&voice, 128, &mut writes);
        assert_eq!(writes.len(), 2);

        assert_eq!(writes[0].target, ModulationTarget::Pitch);
        assert_eq!(writes[0].voice_id, 42);
        assert_eq!(writes[0].sample_offset, 128);
        assert!((writes[0].value - 2.0).abs() < 1.0e-3);

        assert_eq!(writes[1].target, ModulationTarget::Gain);
        assert!((writes[1].value - 1.0).abs() < EPS);
    }

    #[test]
    fn controller_dimension_reads_voice_controller() {
        let mut router = ExpressionRouter::new();
        router.push(TargetMapping::linear(
            ExpressionDimension::Controller { index: 74 },
            ModulationTarget::Cutoff,
        ));
        let mut voice = VoiceExpression::new(1, 60);
        assert!(voice.set_controller(74, u32::MAX));

        let mut writes: Vec<ModulationWrite> = Vec::new();
        router.route(&voice, 0, &mut writes);
        assert_eq!(writes.len(), 1);
        assert!((writes[0].value - 1.0).abs() < EPS);
    }
}
