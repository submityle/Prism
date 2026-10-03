//! Per-voice status tracking with "why is this voice silent" diagnosis.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the voice-monitoring part of design section 26. A
//! [`VoiceMonitor`] holds a read-only snapshot of the voice pool exported by
//! the scheduler (design section 15) so tooling can list the active and
//! virtual voices and, for any voice that is producing no audible output,
//! report a single [`SilenceReason`]. Nothing here runs on the RT thread; the
//! control-rate observer fills the monitor from the scheduler's own state.

use alloc::string::String;
use alloc::vec::Vec;

use prism_audio_core::math::{linear_to_db, Sample};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Lifecycle state of a single voice, mirrored from the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum VoiceState {
    /// The voice is ramping up after being triggered.
    Starting,
    /// The voice is actively rendering.
    Playing,
    /// The voice is retained but culled; it advances its clock but emits no
    /// audio into the mix.
    Virtual,
    /// The voice is ramping down toward a stop.
    Stopping,
    /// The voice has finished and holds no slot.
    Stopped,
}

impl VoiceState {
    /// `true` when the voice is in a running state (starting or playing).
    #[must_use]
    #[inline]
    pub const fn is_running(self) -> bool {
        matches!(self, VoiceState::Starting | VoiceState::Playing)
    }
}

/// The single most relevant reason a voice is producing no audible output.
///
/// The ordering of the variants reflects diagnosis priority: the first
/// condition that applies is the one reported, so a stopped voice is never
/// also reported as "below the audibility floor".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum SilenceReason {
    /// The voice is stopping or stopped, so it is not expected to be audible.
    NotPlaying,
    /// The voice (or one of its parent buses) is muted.
    Muted,
    /// The voice is virtualized: retained but not mixed into the output.
    Virtualized,
    /// The voice is farther than its maximum audible distance.
    OutOfRange,
    /// The voice has no routing to an output bus.
    Unrouted,
    /// The voice's effective gain is at or below the audibility floor.
    BelowFloor,
}

/// A read-only snapshot of one voice's audibility-relevant state.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct VoiceStatus {
    /// Stable identifier of the voice within the pool.
    pub id: u64,
    /// Human-readable name (asset or event path) for display.
    pub name: String,
    /// Lifecycle state of the voice.
    pub state: VoiceState,
    /// `true` when the voice holds a physical rendering slot; `false` when it
    /// is virtual.
    pub physical: bool,
    /// `true` when the voice or a parent bus is muted.
    pub muted: bool,
    /// `true` when the voice is routed to at least one output bus.
    pub routed: bool,
    /// Effective linear gain applied to the voice (post attenuation).
    pub gain_linear: Sample,
    /// Distance from the listener, in metres. Zero for non-positional voices.
    pub distance_m: Sample,
    /// Maximum audible distance, in metres. Zero disables the range check.
    pub max_distance_m: Sample,
}

impl VoiceStatus {
    /// Create a non-positional, routed, playing voice at the given gain. The
    /// distance fields default to zero (range check disabled).
    #[must_use]
    pub fn playing(id: u64, name: impl Into<String>, gain_linear: Sample) -> Self {
        Self {
            id,
            name: name.into(),
            state: VoiceState::Playing,
            physical: true,
            muted: false,
            routed: true,
            gain_linear,
            distance_m: 0.0,
            max_distance_m: 0.0,
        }
    }

    /// Effective gain expressed in dBFS; silence maps to the dB floor.
    #[must_use]
    #[inline]
    pub fn gain_db(&self) -> Sample {
        linear_to_db(self.gain_linear)
    }

    /// Diagnose why the voice is silent, or `None` when it is audible.
    ///
    /// `floor_db` is the gain threshold below which a voice is treated as
    /// inaudible. The first applicable [`SilenceReason`], in variant order, is
    /// returned.
    #[must_use]
    pub fn silence_reason(&self, floor_db: Sample) -> Option<SilenceReason> {
        if matches!(self.state, VoiceState::Stopping | VoiceState::Stopped) {
            return Some(SilenceReason::NotPlaying);
        }
        if self.muted {
            return Some(SilenceReason::Muted);
        }
        if !self.physical || self.state == VoiceState::Virtual {
            return Some(SilenceReason::Virtualized);
        }
        if self.max_distance_m > 0.0 && self.distance_m > self.max_distance_m {
            return Some(SilenceReason::OutOfRange);
        }
        if !self.routed {
            return Some(SilenceReason::Unrouted);
        }
        if self.gain_db() <= floor_db {
            return Some(SilenceReason::BelowFloor);
        }
        None
    }

    /// `true` when the voice is audible at the given audibility floor.
    #[must_use]
    #[inline]
    pub fn is_audible(&self, floor_db: Sample) -> bool {
        self.silence_reason(floor_db).is_none()
    }
}

/// A read-only list of voice statuses with pool-wide queries.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct VoiceMonitor {
    voices: Vec<VoiceStatus>,
}

impl VoiceMonitor {
    /// Create an empty monitor.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self { voices: Vec::new() }
    }

    /// Number of tracked voices.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// `true` when no voices are tracked.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    /// Add a voice status to the monitor.
    pub fn push(&mut self, status: VoiceStatus) {
        self.voices.push(status);
    }

    /// Remove every tracked voice.
    pub fn clear(&mut self) {
        self.voices.clear();
    }

    /// Iterate every tracked voice.
    pub fn iter(&self) -> impl Iterator<Item = &VoiceStatus> {
        self.voices.iter()
    }

    /// Find a voice by identifier.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&VoiceStatus> {
        self.voices.iter().find(|voice| voice.id == id)
    }

    /// Iterate physical voices in a running state.
    pub fn physical_voices(&self) -> impl Iterator<Item = &VoiceStatus> {
        self.voices
            .iter()
            .filter(|voice| voice.physical && voice.state.is_running())
    }

    /// Iterate voices that are retained but virtual.
    pub fn virtual_voices(&self) -> impl Iterator<Item = &VoiceStatus> {
        self.voices
            .iter()
            .filter(|voice| !voice.physical || voice.state == VoiceState::Virtual)
    }

    /// Count voices that are audible at the given floor.
    #[must_use]
    pub fn audible_count(&self, floor_db: Sample) -> usize {
        self.voices
            .iter()
            .filter(|voice| voice.is_audible(floor_db))
            .count()
    }

    /// Collect each silent voice paired with its diagnosed reason.
    #[must_use]
    pub fn silent_voices(&self, floor_db: Sample) -> Vec<(u64, SilenceReason)> {
        self.voices
            .iter()
            .filter_map(|voice| voice.silence_reason(floor_db).map(|reason| (voice.id, reason)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audible_voice_has_no_silence_reason() {
        let voice = VoiceStatus::playing(1, "music", 0.5);
        assert_eq!(voice.silence_reason(-60.0), None);
        assert!(voice.is_audible(-60.0));
    }

    #[test]
    fn stopped_voice_reports_not_playing() {
        let mut voice = VoiceStatus::playing(1, "sfx", 1.0);
        voice.state = VoiceState::Stopped;
        assert_eq!(voice.silence_reason(-60.0), Some(SilenceReason::NotPlaying));
    }

    #[test]
    fn mute_takes_priority_over_virtualization() {
        let mut voice = VoiceStatus::playing(1, "sfx", 1.0);
        voice.muted = true;
        voice.physical = false;
        assert_eq!(voice.silence_reason(-60.0), Some(SilenceReason::Muted));
    }

    #[test]
    fn out_of_range_detected() {
        let mut voice = VoiceStatus::playing(1, "ambience", 1.0);
        voice.distance_m = 100.0;
        voice.max_distance_m = 50.0;
        assert_eq!(voice.silence_reason(-60.0), Some(SilenceReason::OutOfRange));
    }

    #[test]
    fn unrouted_detected() {
        let mut voice = VoiceStatus::playing(1, "sfx", 1.0);
        voice.routed = false;
        assert_eq!(voice.silence_reason(-60.0), Some(SilenceReason::Unrouted));
    }

    #[test]
    fn below_floor_detected() {
        let voice = VoiceStatus::playing(1, "quiet", 0.0001);
        assert_eq!(voice.silence_reason(-60.0), Some(SilenceReason::BelowFloor));
    }

    #[test]
    fn monitor_partitions_voices() {
        let mut monitor = VoiceMonitor::new();
        monitor.push(VoiceStatus::playing(1, "a", 1.0));
        let mut virt = VoiceStatus::playing(2, "b", 1.0);
        virt.physical = false;
        virt.state = VoiceState::Virtual;
        monitor.push(virt);
        let mut muted = VoiceStatus::playing(3, "c", 1.0);
        muted.muted = true;
        monitor.push(muted);

        assert_eq!(monitor.len(), 3);
        assert_eq!(monitor.physical_voices().count(), 2);
        assert_eq!(monitor.virtual_voices().count(), 1);
        assert_eq!(monitor.audible_count(-60.0), 1);
        assert_eq!(monitor.find(2).map(|v| v.id), Some(2));

        let silent = monitor.silent_voices(-60.0);
        assert_eq!(silent.len(), 2);
        assert!(silent.contains(&(2, SilenceReason::Virtualized)));
        assert!(silent.contains(&(3, SilenceReason::Muted)));
    }

    #[test]
    fn empty_monitor_is_empty() {
        let monitor = VoiceMonitor::new();
        assert!(monitor.is_empty());
        assert_eq!(monitor.audible_count(-60.0), 0);
        assert!(monitor.silent_voices(-60.0).is_empty());
    }
}
