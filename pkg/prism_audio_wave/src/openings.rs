//! Dynamic openings: perceptual parameters pre-baked at several door/window
//! states and blended by a continuous openness control at runtime.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "dynamic openings" of design section 43. A door or window is
//! solved offline at a handful of discrete states (shut, ajar, open); at
//! runtime the live openness in `[0, 1]` selects and interpolates between the
//! two bracketing states. Evaluation is allocation free and real-time safe.

use alloc::vec;
use alloc::vec::Vec;

use crate::encoding::PerceptualParams;

/// A multi-state opening: a set of `(openness, params)` keyframes blended by a
/// scalar openness value.
///
/// Keyframes are kept sorted by openness; [`MultiStateOpening::evaluate`] finds
/// the two surrounding keyframes and linearly interpolates between them.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiStateOpening {
    // Sorted ascending by openness; always non-empty.
    states: Vec<(f32, PerceptualParams)>,
}

impl MultiStateOpening {
    /// Builds an opening from `(openness, params)` keyframes.
    ///
    /// Openness values are clamped to `[0, 1]` and the set is sorted. An empty
    /// input yields a single fully-open keyframe so evaluation always has data.
    #[must_use]
    pub fn new(states: Vec<(f32, PerceptualParams)>) -> Self {
        let mut states = states;
        for s in &mut states {
            s.0 = s.0.clamp(0.0, 1.0);
        }
        states.sort_by(|a, b| a.0.total_cmp(&b.0));
        if states.is_empty() {
            states.push((1.0, PerceptualParams::OPEN));
        }
        Self { states }
    }

    /// A conventional shut/open opening: fully occluded when closed, fully open
    /// when open, interpolated in between.
    #[must_use]
    pub fn shut_open() -> Self {
        let states = vec![
            (0.0, PerceptualParams::OCCLUDED),
            (1.0, PerceptualParams::OPEN),
        ];
        Self { states }
    }

    /// The number of keyframes.
    #[must_use]
    #[inline]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Evaluates the opening at `openness` in `[0, 1]`.
    ///
    /// Values below the lowest keyframe clamp to it and values above the
    /// highest clamp to it; otherwise the two bracketing keyframes are blended.
    /// Allocation free and panic free.
    #[must_use]
    pub fn evaluate(&self, openness: f32) -> PerceptualParams {
        let o = openness.clamp(0.0, 1.0);
        let first = self.states[0];
        if o <= first.0 {
            return first.1;
        }
        let last = self.states[self.states.len() - 1];
        if o >= last.0 {
            return last.1;
        }
        // Find the bracketing pair.
        let mut lo = first;
        for &state in &self.states[1..] {
            if o <= state.0 {
                let span = state.0 - lo.0;
                let t = if span <= 0.0 { 0.0 } else { (o - lo.0) / span };
                return lo.1.lerp(&state.1, t);
            }
            lo = state;
        }
        last.1
    }
}

impl Default for MultiStateOpening {
    /// The default opening is a shut/open door.
    #[inline]
    fn default() -> Self {
        Self::shut_open()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn endpoints_match_keyframes() {
        let opening = MultiStateOpening::shut_open();
        assert!(approx(opening.evaluate(0.0).direct_gain, 0.0, 1e-6));
        assert!(approx(opening.evaluate(1.0).direct_gain, 1.0, 1e-6));
    }

    #[test]
    fn midpoint_blends_states() {
        let opening = MultiStateOpening::shut_open();
        let mid = opening.evaluate(0.5);
        assert!(approx(mid.direct_gain, 0.5, 1e-6));
        assert!(approx(mid.wet_gain, 0.5, 1e-6));
    }

    #[test]
    fn out_of_range_clamps_to_endpoints() {
        let opening = MultiStateOpening::shut_open();
        assert!(approx(opening.evaluate(-5.0).direct_gain, 0.0, 1e-6));
        assert!(approx(opening.evaluate(5.0).direct_gain, 1.0, 1e-6));
    }

    #[test]
    fn three_state_opening_brackets_correctly() {
        let ajar = PerceptualParams {
            direct_gain: 0.25,
            ..PerceptualParams::OCCLUDED
        };
        let opening = MultiStateOpening::new(vec![
            (1.0, PerceptualParams::OPEN),
            (0.0, PerceptualParams::OCCLUDED),
            (0.5, ajar),
        ]);
        assert_eq!(opening.state_count(), 3);
        // Exactly on the middle keyframe returns it.
        assert!(approx(opening.evaluate(0.5).direct_gain, 0.25, 1e-6));
        // Between middle and open blends 0.25 -> 1.0.
        let q = opening.evaluate(0.75);
        assert!(q.direct_gain > 0.25 && q.direct_gain < 1.0);
    }

    #[test]
    fn empty_input_is_open() {
        let opening = MultiStateOpening::new(Vec::new());
        assert_eq!(opening.state_count(), 1);
        assert!(approx(opening.evaluate(0.3).direct_gain, 1.0, 1e-6));
    }
}
