//! The shared modulation-source abstraction.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Defines the `Modulator` trait and `ModContext` evaluation context used by
//! design section 12. Concrete sources (LFO, envelope, sample-and-hold, random)
//! implement this trait so the modulation matrix (see `super::matrix`) can
//! evaluate them uniformly at control rate.

use prism_audio_core::Sample;

/// Per-tick evaluation context handed to every [`Modulator`].
///
/// A control-rate tick typically corresponds to one audio block of `frames`
/// samples; sources that track wall-clock time use `dt_seconds` so their rate
/// is independent of the block size.
#[derive(Debug, Clone, Copy)]
pub struct ModContext {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio frames this control tick advances over.
    pub frames: u32,
}

impl ModContext {
    /// Builds a context for a control tick spanning `frames` at `sample_rate`.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` or `frames` is zero.
    #[inline]
    #[must_use]
    pub fn new(sample_rate: u32, frames: u32) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        assert!(frames > 0, "frames must be non-zero");
        Self {
            sample_rate,
            frames,
        }
    }

    /// Returns the duration of this control tick in seconds.
    #[inline]
    #[must_use]
    pub fn dt_seconds(&self) -> Sample {
        self.frames as Sample / self.sample_rate as Sample
    }
}

/// A control-rate modulation source producing a scalar per tick.
///
/// Implementations are deterministic and allocation-free so they can run on the
/// audio thread. Sources are advanced once per control tick via
/// [`Modulator::tick`]; stateful sources additionally honor [`Modulator::reset`]
/// when a voice is recycled.
pub trait Modulator: Send {
    /// Advances the source over one control tick and returns its new value.
    fn tick(&mut self, ctx: &ModContext) -> Sample;

    /// Returns the current value without advancing time.
    #[must_use]
    fn value(&self) -> Sample;

    /// Resets all internal state to its initial condition.
    fn reset(&mut self);
}
