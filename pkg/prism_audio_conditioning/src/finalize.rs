//! Program finalization: optional encoder-delay trim, loop-point rebasing, and
//! loop-seam crossfade baking folded into one config-gated stage.
//!
//! Decode records the encoder pre-roll/padding and loop detection runs on the
//! raw decoded timeline, but the *delivered* program is the trimmed, seam-baked
//! asset the engine streams at run time. This module is the stage that turns the
//! recorded metadata into that delivered program, composing the two pure
//! builders this crate already provides:
//!
//! - [`crate::delay_trim::trim_encoder_delay`] removes the leading pre-roll and
//!   trailing padding, yielding the exact authored program.
//! - [`crate::delay_trim::adjust_loop_points_to_program`] rebases loop points
//!   detected on the raw decode onto that trimmed program timeline.
//! - [`crate::loop_crossfade::apply_loop_crossfade`] bakes the equal-power (or
//!   linear) seam crossfade into the trimmed program so a forward loop plays
//!   click-free with zero run-time work.
//!
//! Every step is individually opt-in through [`FinalizeConfig`], whose
//! [`Default`] leaves *both* steps off so the stage is a transparent identity
//! unless a caller asks for it. That keeps the baseline pipeline byte-identical
//! to a run that skips finalization, while making the full gapless/seamless-loop
//! delivery contract available behind one flag per concern.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. Gapless
//! trimming (Ogg/Opus pre-skip, iTunes/MP4 gapless metadata) and equal-power
//! loop crossfades are public, widely documented container/authoring practices;
//! only the ideas are borrowed. This module adds no new DSP of its own: it only
//! sequences the two sibling builders in the correct order.
//!
//! # Relationship
//!
//! Implements the finalization leg of design section 51, binding together the
//! design-section-42 "codec pre-roll/padding trim calibration" work in
//! [`crate::delay_trim`] and the design-sections-10/20/51 seamless-loop bake in
//! [`crate::loop_crossfade`]. It is consumed by [`crate::pipeline`], which runs
//! it as the final stage before content hashing so the recorded hash keys the
//! delivered program.

use crate::delay_trim::{self, DelayTrimError};
use crate::loop_crossfade::{self, CrossfadeShape};
use crate::loop_point::LoopPoints;
use crate::pcm::{ConditionedPcm, EncoderDelay};

/// Which finalization steps to apply when turning the raw decoded program into
/// the delivered asset.
///
/// Both flags default to `false`, so a default-constructed config leaves the
/// program untouched. Enable [`FinalizeConfig::trim_encoder_delay`] to drop the
/// recorded encoder pre-roll/padding (and rebase any detected loop onto the
/// trimmed timeline) and [`FinalizeConfig::bake_loop_crossfade`] to bake the
/// loop seam. The two are independent: baking without trimming bakes the seam of
/// the raw-timeline loop, and trimming without baking delivers a trimmed program
/// whose loop seam is still crossfaded at run time by the player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FinalizeConfig {
    /// Remove the recorded encoder pre-roll/padding and rebase detected loop
    /// points onto the trimmed program timeline.
    pub trim_encoder_delay: bool,
    /// Bake the loop-seam crossfade into the delivered program (no-op when no
    /// forward loop was detected).
    pub bake_loop_crossfade: bool,
    /// Gain law used when [`FinalizeConfig::bake_loop_crossfade`] is set.
    pub crossfade_shape: CrossfadeShape,
}

impl FinalizeConfig {
    /// Returns `true` when at least one finalization step is enabled, so callers
    /// can skip the stage entirely (and its allocation) when it would be an
    /// identity transform.
    #[must_use]
    #[inline]
    pub const fn is_enabled(&self) -> bool {
        self.trim_encoder_delay || self.bake_loop_crossfade
    }
}

/// Error returned by [`finalize`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FinalizeError {
    /// The encoder-delay trim failed to rebuild its trimmed container. This is
    /// not expected for a valid source and is surfaced for completeness.
    Trim(DelayTrimError),
}

impl core::fmt::Display for FinalizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FinalizeError::Trim(e) => write!(f, "program finalization trim failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FinalizeError {}

impl From<DelayTrimError> for FinalizeError {
    fn from(error: DelayTrimError) -> Self {
        FinalizeError::Trim(error)
    }
}

/// The delivered program plus the loop points expressed in its timeline.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FinalizedProgram {
    /// The finalized `PCM`: trimmed and/or seam-baked per the config.
    pub pcm: ConditionedPcm,
    /// Loop points in the finalized timeline, rebased when trimming ran. `None`
    /// when no loop was detected.
    pub loop_points: Option<LoopPoints>,
}

/// Finalizes a decoded program into the delivered asset per `config`.
///
/// The steps run in a fixed, correct order:
///
/// 1. **Trim** (when [`FinalizeConfig::trim_encoder_delay`]): drop the encoder
///    pre-roll/padding, then rebase any `loop_points` from the raw timeline onto
///    the trimmed program with
///    [`crate::delay_trim::adjust_loop_points_to_program`].
/// 2. **Bake** (when [`FinalizeConfig::bake_loop_crossfade`]): crossfade the loop
///    seam of the (possibly rebased) loop into the program. This is a no-op when
///    there is no forward loop, so it is always safe to request.
///
/// When neither step is enabled the input is cloned through unchanged, so a
/// default config is a transparent identity.
///
/// # Errors
///
/// Returns [`FinalizeError::Trim`] when the encoder-delay trim fails to rebuild
/// its container (not expected for a valid source).
pub fn finalize(
    pcm: &ConditionedPcm,
    delay: EncoderDelay,
    loop_points: Option<LoopPoints>,
    config: &FinalizeConfig,
) -> Result<FinalizedProgram, FinalizeError> {
    let (program, loop_points) = if config.trim_encoder_delay {
        let trimmed = delay_trim::trim_encoder_delay(pcm, delay)?;
        let program_frames = trimmed.frames();
        let rebased = loop_points
            .map(|lp| delay_trim::adjust_loop_points_to_program(lp, delay, program_frames));
        (trimmed, rebased)
    } else {
        (pcm.clone(), loop_points)
    };

    let program = if config.bake_loop_crossfade {
        match loop_points.as_ref() {
            Some(lp) => loop_crossfade::apply_loop_crossfade(&program, lp, config.crossfade_shape),
            None => program,
        }
    } else {
        program
    };

    Ok(FinalizedProgram {
        pcm: program,
        loop_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_point::LoopMode;
    use prism_audio_core::buffer::ChannelLayout;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    /// Builds a mono program whose sample at frame `i` is `i as f32`, so trims
    /// and rebases can be checked by value.
    fn ramp(frames: usize) -> ConditionedPcm {
        let ch: Vec<f32> = (0..frames).map(|i| i as f32).collect();
        #[cfg(feature = "std")]
        let channels = vec![ch];
        #[cfg(not(feature = "std"))]
        let channels = alloc::vec![ch];
        ConditionedPcm::new(48_000, ChannelLayout::Mono, channels).expect("valid mono program")
    }

    fn delay(preroll: u32, padding: u32) -> EncoderDelay {
        EncoderDelay {
            preroll_frames: preroll,
            padding_frames: padding,
        }
    }

    #[test]
    fn default_config_is_disabled() {
        let c = FinalizeConfig::default();
        assert!(!c.trim_encoder_delay);
        assert!(!c.bake_loop_crossfade);
        assert!(!c.is_enabled());
    }

    #[test]
    fn disabled_finalize_is_identity() {
        let pcm = ramp(1000);
        let out = finalize(&pcm, delay(10, 20), None, &FinalizeConfig::default()).unwrap();
        assert_eq!(out.pcm, pcm);
        assert!(out.loop_points.is_none());
    }

    #[test]
    fn is_enabled_tracks_either_flag() {
        let trim_only = FinalizeConfig {
            trim_encoder_delay: true,
            ..FinalizeConfig::default()
        };
        let bake_only = FinalizeConfig {
            bake_loop_crossfade: true,
            ..FinalizeConfig::default()
        };
        assert!(trim_only.is_enabled());
        assert!(bake_only.is_enabled());
    }

    #[test]
    fn trim_drops_preroll_and_padding() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            trim_encoder_delay: true,
            ..FinalizeConfig::default()
        };
        let out = finalize(&pcm, delay(10, 20), None, &config).unwrap();
        assert_eq!(out.pcm.frames(), 970);
        // First kept frame is the raw frame 10 (value 10.0).
        assert_eq!(out.pcm.channel(0).unwrap()[0], 10.0);
        // Last kept frame is raw frame 979 (value 979.0).
        assert_eq!(out.pcm.channel(0).unwrap()[969], 979.0);
    }

    #[test]
    fn trim_rebases_loop_points() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            trim_encoder_delay: true,
            ..FinalizeConfig::default()
        };
        let lp = LoopPoints {
            start: 100,
            end: 900,
            crossfade_frames: 32,
            mode: LoopMode::Forward,
        };
        let out = finalize(&pcm, delay(10, 20), Some(lp), &config).unwrap();
        let rebased = out.loop_points.expect("loop preserved");
        // Both endpoints shift left by the 10-frame pre-roll.
        assert_eq!(rebased.start, 90);
        assert_eq!(rebased.end, 890);
        assert_eq!(rebased.crossfade_frames, 32);
    }

    #[test]
    fn disabled_trim_keeps_raw_loop_points() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            bake_loop_crossfade: false,
            trim_encoder_delay: false,
            ..FinalizeConfig::default()
        };
        let lp = LoopPoints {
            start: 100,
            end: 900,
            crossfade_frames: 0,
            mode: LoopMode::Forward,
        };
        let out = finalize(&pcm, delay(10, 20), Some(lp), &config).unwrap();
        assert_eq!(out.loop_points, Some(lp));
    }

    #[test]
    fn bake_without_loop_is_noop() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            bake_loop_crossfade: true,
            ..FinalizeConfig::default()
        };
        let out = finalize(&pcm, delay(0, 0), None, &config).unwrap();
        assert_eq!(out.pcm, pcm);
    }

    #[test]
    fn bake_crossfades_the_seam() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            bake_loop_crossfade: true,
            ..FinalizeConfig::default()
        };
        let lp = LoopPoints {
            start: 100,
            end: 900,
            crossfade_frames: 16,
            mode: LoopMode::Forward,
        };
        let out = finalize(&pcm, delay(0, 0), Some(lp), &config).unwrap();
        // The tail window [end-cf, end) is blended, so frames there no longer
        // equal their original ramp value.
        let baked = out.pcm.channel(0).unwrap();
        let mut changed = false;
        for k in 0..16 {
            if (baked[900 - 16 + k] - (900.0 - 16.0 + k as f32)).abs() > 1.0e-6 {
                changed = true;
            }
        }
        assert!(changed, "seam window should be blended");
        // Material outside the seam is untouched.
        assert_eq!(baked[0], 0.0);
        assert_eq!(baked[500], 500.0);
    }

    #[test]
    fn trim_then_bake_runs_both_steps() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            trim_encoder_delay: true,
            bake_loop_crossfade: true,
            crossfade_shape: CrossfadeShape::Linear,
        };
        let lp = LoopPoints {
            start: 100,
            end: 900,
            crossfade_frames: 16,
            mode: LoopMode::Forward,
        };
        let out = finalize(&pcm, delay(10, 20), Some(lp), &config).unwrap();
        assert_eq!(out.pcm.frames(), 970);
        let rebased = out.loop_points.expect("loop preserved");
        assert_eq!(rebased.start, 90);
        assert_eq!(rebased.end, 890);
    }

    #[test]
    fn finalize_is_deterministic() {
        let pcm = ramp(1000);
        let config = FinalizeConfig {
            trim_encoder_delay: true,
            bake_loop_crossfade: true,
            crossfade_shape: CrossfadeShape::EqualPower,
        };
        let lp = LoopPoints {
            start: 100,
            end: 900,
            crossfade_frames: 16,
            mode: LoopMode::Forward,
        };
        let a = finalize(&pcm, delay(10, 20), Some(lp), &config).unwrap();
        let b = finalize(&pcm, delay(10, 20), Some(lp), &config).unwrap();
        assert_eq!(a.pcm, b.pcm);
        assert_eq!(a.loop_points, b.loop_points);
    }

    #[test]
    fn trim_error_type_is_convertible() {
        // The From<DelayTrimError> bridge compiles and preserves the variant.
        use crate::pcm::PcmError;
        let e: FinalizeError = DelayTrimError::Rebuild(PcmError::ZeroSampleRate).into();
        assert!(matches!(e, FinalizeError::Trim(DelayTrimError::Rebuild(_))));
    }
}
