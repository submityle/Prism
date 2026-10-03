//! The per-block telemetry data contract consumed by the profiler.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! [`AudioTelemetry`] mirrors, by value, the `prism_audio_rt` telemetry ring
//! frame (design section 21) so this analysis crate does not depend on the RT
//! transport. The engine's control-rate integration copies the RT frame into
//! this structure; see the crate-level `Relationship` note. Derived accessors
//! here add profiler-only conveniences (block duration, master levels in dB,
//! overrun classification) without changing the underlying contract.

use prism_audio_core::math::{linear_to_db, Sample};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A per-block snapshot of the audio runtime's state.
///
/// Every field is a plain scalar; the structure is `Copy` so it can be stored
/// in the profiler [session ring](crate::session::ProfilerSession) cheaply.
/// The field set and their meaning are the stable contract shared with the RT
/// telemetry ring.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AudioTelemetry {
    /// Monotonically increasing index of the processed block, starting at `0`.
    pub block_index: u64,
    /// Playhead position, in frames, after this block was rendered.
    pub playhead: u64,
    /// Number of frames rendered in this block.
    pub frames: u32,
    /// Physical (audible) voice count observed at the end of the block.
    pub physical_voices: u32,
    /// Virtual (culled but retained) voice count at the end of the block.
    pub virtual_voices: u32,
    /// Peak absolute sample magnitude of the master output this block.
    pub master_peak: Sample,
    /// Root-mean-square level of the master output this block.
    pub master_rms: Sample,
    /// Fraction of the block's wall-clock budget spent inside processing, where
    /// `1.0` means the block took exactly its real-time duration to render.
    /// Values above `1.0` indicate an overrun risk.
    pub cpu_load: Sample,
}

impl AudioTelemetry {
    /// A zeroed frame, useful as an initial value before the first block runs.
    #[must_use]
    #[inline]
    pub const fn empty() -> Self {
        Self {
            block_index: 0,
            playhead: 0,
            frames: 0,
            physical_voices: 0,
            virtual_voices: 0,
            master_peak: 0.0,
            master_rms: 0.0,
            cpu_load: 0.0,
        }
    }

    /// Total number of voices tracked this block (physical plus virtual).
    #[must_use]
    #[inline]
    pub const fn total_voices(&self) -> u32 {
        self.physical_voices.saturating_add(self.virtual_voices)
    }

    /// Duration of this block in seconds, given the output `sample_rate`.
    ///
    /// Returns `0.0` when the sample rate or frame count is zero so callers can
    /// divide safely.
    #[must_use]
    #[inline]
    pub fn block_duration_s(&self, sample_rate: u32) -> Sample {
        if sample_rate == 0 || self.frames == 0 {
            return 0.0;
        }
        self.frames as Sample / sample_rate as Sample
    }

    /// Master peak level expressed in dBFS. Silence maps to the floor value.
    #[must_use]
    #[inline]
    pub fn master_peak_db(&self) -> Sample {
        linear_to_db(self.master_peak)
    }

    /// Master RMS level expressed in dBFS. Silence maps to the floor value.
    #[must_use]
    #[inline]
    pub fn master_rms_db(&self) -> Sample {
        linear_to_db(self.master_rms)
    }

    /// `true` when the reported CPU load indicates the block overran its
    /// real-time budget (strictly greater than `1.0`).
    #[must_use]
    #[inline]
    pub fn is_overrun(&self) -> bool {
        self.cpu_load > 1.0
    }

    /// Remaining fraction of the real-time budget this block, clamped to
    /// `[0, 1]`. An overrunning block reports `0.0` headroom.
    #[must_use]
    #[inline]
    pub fn cpu_headroom(&self) -> Sample {
        let headroom = 1.0 - self.cpu_load;
        headroom.clamp(0.0, 1.0)
    }
}

impl Default for AudioTelemetry {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}
