//! Viseme / phoneme timeline data carried alongside dialogue media.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the viseme / lip-sync data model of design section 35. A viseme
//! timeline (authored or exported from offline analysis) ships with the asset
//! and is surfaced at runtime through the telemetry ring to the animation
//! system; it carries no audio and spends no real-time budget. This module is a
//! pure data model with deterministic sampling and linear weight interpolation
//! (no AI / ML, no DSP).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

/// A viseme (visual mouth shape) identifier.
///
/// Values follow a small, engine-defined mouth-shape set; the concrete mapping
/// to an animation rig is the consumer's responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Viseme {
    /// Closed / resting mouth (silence).
    Silence,
    /// Bilabial closure (`p`, `b`, `m`).
    Pp,
    /// Labiodental (`f`, `v`).
    Ff,
    /// Dental / alveolar (`t`, `d`, `n`).
    Th,
    /// Open vowel (`a`).
    Aa,
    /// Front vowel (`e`, `i`).
    Ee,
    /// Rounded vowel (`o`, `u`).
    Oh,
}

/// A single keyframe on the viseme timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisemeKeyframe {
    /// Frame index (at the track's sample rate) of this keyframe.
    pub frame: u64,
    /// The viseme active from this keyframe until the next.
    pub viseme: Viseme,
    /// Target blend weight in `[0, 1]` for this viseme at this keyframe.
    pub weight: Sample,
}

impl VisemeKeyframe {
    /// Creates a keyframe, clamping `weight` into `[0, 1]`.
    #[must_use]
    pub fn new(frame: u64, viseme: Viseme, weight: Sample) -> Self {
        Self {
            frame,
            viseme,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

/// A sampled viseme pose: the active shape and its interpolated weight.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisemePose {
    /// The active viseme shape.
    pub viseme: Viseme,
    /// The interpolated blend weight in `[0, 1]`.
    pub weight: Sample,
}

/// A viseme timeline: ordered keyframes at a fixed sample rate.
///
/// Sampling interpolates the blend weight linearly between the surrounding
/// keyframes while holding the earlier keyframe's viseme shape, giving a smooth
/// open/close curve without changing shapes mid-interpolation.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VisemeTrack {
    /// Sample rate the keyframe timecodes are expressed in.
    pub sample_rate: u32,
    keyframes: Vec<VisemeKeyframe>,
}

impl VisemeTrack {
    /// Creates an empty track at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            keyframes: Vec::new(),
        }
    }

    /// Appends `keyframe`, keeping keyframes ordered by frame.
    pub fn push(&mut self, keyframe: VisemeKeyframe) {
        let position = self
            .keyframes
            .iter()
            .position(|existing| existing.frame > keyframe.frame)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(position, keyframe);
    }

    /// Returns the number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns `true` when the track has no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Returns the keyframes, in order.
    #[must_use]
    pub fn keyframes(&self) -> &[VisemeKeyframe] {
        &self.keyframes
    }

    /// Samples the pose at `frame`.
    ///
    /// Before the first keyframe the track reads as resting silence; after the
    /// last keyframe it holds that keyframe's pose. Between two keyframes the
    /// weight is linearly interpolated while the earlier shape is held.
    #[must_use]
    pub fn sample_at(&self, frame: u64) -> VisemePose {
        if self.keyframes.is_empty() {
            return VisemePose {
                viseme: Viseme::Silence,
                weight: 0.0,
            };
        }
        let first = &self.keyframes[0];
        if frame <= first.frame {
            return VisemePose {
                viseme: first.viseme,
                weight: first.weight,
            };
        }
        // Find the last keyframe at or before `frame`.
        let mut lower = 0usize;
        for (index, keyframe) in self.keyframes.iter().enumerate() {
            if keyframe.frame <= frame {
                lower = index;
            } else {
                break;
            }
        }
        let left = &self.keyframes[lower];
        match self.keyframes.get(lower + 1) {
            None => VisemePose {
                viseme: left.viseme,
                weight: left.weight,
            },
            Some(right) => {
                let span = right.frame - left.frame;
                let weight = if span == 0 {
                    right.weight
                } else {
                    let t = (frame - left.frame) as Sample / span as Sample;
                    left.weight + (right.weight - left.weight) * t
                };
                VisemePose {
                    viseme: left.viseme,
                    weight,
                }
            }
        }
    }
}
