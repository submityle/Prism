//! Membership crossfades: smooth per-voice gains that ramp in when a voice
//! joins a cluster and ramp out when it leaves, so re-clustering never clicks.
//!
//! Block-boundary re-clustering (see [`crate::clustering::assignment`]) can move
//! a voice between clusters, promote it into a representative source, or drop it
//! out entirely. Switching its contribution on or off instantly would inject a
//! discontinuity. Instead each voice carries a linear [`FadeRamp`] that walks
//! its gain toward the target over a fixed number of blocks, and
//! [`MembershipFades`] drives one ramp per currently-tracked voice from the
//! active set published each block.
//!
//! Gains are deterministic: a fixed sequence of active sets produces a fixed
//! gain trajectory, matching the smoothing discipline of design section 7.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Supports the source-clustering half of design section 33 and reuses the
//! click-free smoothing contract of design section 7.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

/// Smallest gain treated as fully silent when pruning finished fade-outs.
pub const REST_EPSILON: Sample = 1.0e-4;

/// A linear gain ramp that advances toward a target by a fixed step per block.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FadeRamp {
    gain: Sample,
    target: Sample,
    step: Sample,
}

impl FadeRamp {
    /// Creates a ramp starting at `initial`, with `fade_blocks` blocks to span
    /// the full `[0, 1]` range. A `fade_blocks` of zero means instantaneous.
    #[must_use]
    pub fn new(initial: Sample, fade_blocks: u32) -> Self {
        let step = step_for(fade_blocks);
        Self { gain: initial.clamp(0.0, 1.0), target: initial.clamp(0.0, 1.0), step }
    }

    /// Sets the gain the ramp walks toward, clamped to `[0, 1]`.
    pub fn set_target(&mut self, target: Sample) {
        self.target = target.clamp(0.0, 1.0);
    }

    /// Advances one block toward the target and returns the new gain.
    pub fn advance(&mut self) -> Sample {
        if self.gain < self.target {
            self.gain = (self.gain + self.step).min(self.target);
        } else if self.gain > self.target {
            self.gain = (self.gain - self.step).max(self.target);
        }
        self.gain
    }

    /// The current gain in `[0, 1]`.
    #[must_use]
    pub fn gain(&self) -> Sample {
        self.gain
    }

    /// The target gain in `[0, 1]`.
    #[must_use]
    pub fn target(&self) -> Sample {
        self.target
    }

    /// Returns `true` when the gain has reached its target.
    #[must_use]
    pub fn at_rest(&self) -> bool {
        (self.gain - self.target).abs() <= REST_EPSILON
    }
}

/// The per-block gain step for a given fade length (`1` means instantaneous).
fn step_for(fade_blocks: u32) -> Sample {
    if fade_blocks == 0 {
        1.0
    } else {
        1.0 / fade_blocks as Sample
    }
}

/// Tracks one [`FadeRamp`] per voice and drives them from the active-membership
/// set published each block.
///
/// Voices present in the active set fade toward `1`; voices that were tracked
/// but drop out of the active set fade toward `0` and are pruned once silent.
/// Entries are kept in an ordered map so iteration and pruning are
/// deterministic.
#[derive(Debug, Clone, Default)]
pub struct MembershipFades {
    fade_blocks: u32,
    ramps: BTreeMap<usize, FadeRamp>,
}

impl MembershipFades {
    /// Creates a tracker whose ramps span `fade_blocks` blocks edge to edge.
    #[must_use]
    pub fn new(fade_blocks: u32) -> Self {
        Self { fade_blocks, ramps: BTreeMap::new() }
    }

    /// Advances all ramps one block given the set of currently-active voices.
    ///
    /// New voices are inserted at gain `0` and begin fading in; absent voices
    /// fade out and are removed once their gain reaches zero.
    pub fn advance_block(&mut self, active: &[usize]) {
        // Retarget: active voices toward 1, everything else toward 0.
        for ramp in self.ramps.values_mut() {
            ramp.set_target(0.0);
        }
        for &voice in active {
            let fade_blocks = self.fade_blocks;
            let ramp = self.ramps.entry(voice).or_insert_with(|| FadeRamp::new(0.0, fade_blocks));
            ramp.set_target(1.0);
        }

        // Advance and prune silent fade-outs.
        let mut finished = Vec::new();
        for (&voice, ramp) in &mut self.ramps {
            ramp.advance();
            if ramp.target() <= 0.0 && ramp.gain() <= REST_EPSILON {
                finished.push(voice);
            }
        }
        for voice in finished {
            self.ramps.remove(&voice);
        }
    }

    /// Returns the current gain for `voice`, or `0` when it is not tracked.
    #[must_use]
    pub fn gain(&self, voice: usize) -> Sample {
        self.ramps.get(&voice).map_or(0.0, FadeRamp::gain)
    }

    /// Number of voices currently tracked (fading in, held, or fading out).
    #[must_use]
    pub fn len(&self) -> usize {
        self.ramps.len()
    }

    /// Returns `true` when no voices are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ramps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-5;

    #[test]
    fn ramp_fades_in_over_requested_blocks() {
        let mut r = FadeRamp::new(0.0, 4);
        r.set_target(1.0);
        for _ in 0..4 {
            r.advance();
        }
        assert!((r.gain() - 1.0).abs() < EPS);
        assert!(r.at_rest());
    }

    #[test]
    fn ramp_does_not_overshoot() {
        let mut r = FadeRamp::new(0.0, 4);
        r.set_target(1.0);
        for _ in 0..100 {
            r.advance();
        }
        assert!((r.gain() - 1.0).abs() < EPS);
    }

    #[test]
    fn ramp_fades_out_to_zero() {
        let mut r = FadeRamp::new(1.0, 2);
        r.set_target(0.0);
        r.advance();
        assert!((r.gain() - 0.5).abs() < EPS);
        r.advance();
        assert!(r.gain().abs() < EPS);
    }

    #[test]
    fn zero_fade_blocks_is_instant() {
        let mut r = FadeRamp::new(0.0, 0);
        r.set_target(1.0);
        r.advance();
        assert!((r.gain() - 1.0).abs() < EPS);
    }

    #[test]
    fn new_voice_fades_in_from_zero() {
        let mut fades = MembershipFades::new(2);
        fades.advance_block(&[7]);
        // After one block of a two-block fade, gain is halfway.
        assert!((fades.gain(7) - 0.5).abs() < EPS);
        fades.advance_block(&[7]);
        assert!((fades.gain(7) - 1.0).abs() < EPS);
    }

    #[test]
    fn departed_voice_fades_out_and_is_pruned() {
        let mut fades = MembershipFades::new(2);
        fades.advance_block(&[7]);
        fades.advance_block(&[7]);
        assert!((fades.gain(7) - 1.0).abs() < EPS);
        // Drop it: fade out over two blocks.
        fades.advance_block(&[]);
        assert!((fades.gain(7) - 0.5).abs() < EPS);
        assert_eq!(fades.len(), 1);
        fades.advance_block(&[]);
        assert!(fades.gain(7).abs() < EPS);
        // Pruned once silent.
        assert_eq!(fades.len(), 0);
    }

    #[test]
    fn untracked_voice_has_zero_gain() {
        let fades = MembershipFades::new(4);
        assert!(fades.gain(42).abs() < EPS);
        assert!(fades.is_empty());
    }

    #[test]
    fn concurrent_voices_fade_independently() {
        let mut fades = MembershipFades::new(4);
        fades.advance_block(&[1, 2]);
        fades.advance_block(&[1, 2]);
        // Both climbing together.
        assert!((fades.gain(1) - fades.gain(2)).abs() < EPS);
        // Drop voice 2 only.
        fades.advance_block(&[1]);
        assert!(fades.gain(1) > fades.gain(2));
    }

    #[test]
    fn membership_is_deterministic() {
        let script = [&[1usize, 2][..], &[2][..], &[2, 3][..], &[][..]];
        let run = || {
            let mut f = MembershipFades::new(3);
            let mut trace = Vec::new();
            for step in &script {
                f.advance_block(step);
                trace.push((f.gain(1), f.gain(2), f.gain(3)));
            }
            trace
        };
        let a = run();
        let b = run();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.0 - y.0).abs() < EPS);
            assert!((x.1 - y.1).abs() < EPS);
            assert!((x.2 - y.2).abs() < EPS);
        }
    }

    #[test]
    fn gain_stays_bounded() {
        let mut fades = MembershipFades::new(3);
        for _ in 0..10 {
            fades.advance_block(&[1]);
            let g = fades.gain(1);
            assert!((0.0..=1.0).contains(&g));
        }
    }
}
