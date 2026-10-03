//! A timed, curve-shaped interpolation from one resolved map to another.
//!
//! A [`Transition`] moves every affected parameter from its `start` value
//! toward its `dest` value over `duration_secs`, shaping the normalized
//! progress with an interpolation curve and blending each parameter in its own
//! domain. The sampled map is the union of the start and destination keys: a
//! parameter present on only one side keeps that side's value (there is nothing
//! to blend against), so introducing or dropping a parameter never snaps an
//! unrelated value.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Interpolates [`crate::resolved::ResolvedParameters`] using
//! [`crate::blend::interpolate`] with per-parameter
//! [`crate::parameter::ParameterKind`], shaped by
//! `prism_audio_content::curve::Interpolation`. Driven by
//! [`crate::mixer::SnapshotMixer`].

use alloc::collections::BTreeMap;

use prism_audio_content::curve::Interpolation;
use prism_audio_core::math::Sample;

use crate::blend;
use crate::parameter::{ParameterId, ParameterKind};
use crate::resolved::ResolvedParameters;

/// An in-flight interpolation between two resolved parameter maps.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    /// Values the transition starts from.
    pub start: ResolvedParameters,
    /// Values the transition moves toward.
    pub dest: ResolvedParameters,
    /// Blending domain of each parameter, used while interpolating.
    pub kinds: BTreeMap<ParameterId, ParameterKind>,
    /// Seconds elapsed since the transition began.
    pub elapsed_secs: f32,
    /// Total duration of the transition, in seconds.
    pub duration_secs: f32,
    /// Curve shaping the normalized progress.
    pub interpolation: Interpolation,
}

impl Transition {
    /// Builds a transition from `start` to `dest` over `duration_secs`.
    ///
    /// `kinds` supplies the blending domain for parameters that appear on both
    /// sides; parameters present on only one side keep that side's value and
    /// do not consult `kinds`.
    #[must_use]
    pub fn new(
        start: ResolvedParameters,
        dest: ResolvedParameters,
        kinds: BTreeMap<ParameterId, ParameterKind>,
        duration_secs: f32,
        interpolation: Interpolation,
    ) -> Self {
        Self { start, dest, kinds, elapsed_secs: 0.0, duration_secs, interpolation }
    }

    /// Returns normalized progress in `[0, 1]`.
    ///
    /// A non-positive duration is treated as already complete and returns `1`.
    #[must_use]
    pub fn progress(&self) -> Sample {
        if self.duration_secs <= 0.0 {
            return 1.0;
        }
        (self.elapsed_secs / self.duration_secs).clamp(0.0, 1.0)
    }

    /// Advances the transition by `dt_secs` (negative steps are ignored) and
    /// returns `true` once it has completed.
    pub fn advance(&mut self, dt_secs: f32) -> bool {
        if dt_secs > 0.0 {
            self.elapsed_secs += dt_secs;
        }
        self.is_complete()
    }

    /// Returns `true` once progress has reached the destination.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.progress() >= 1.0
    }

    /// Samples the current blended map for the union of start and dest keys.
    #[must_use]
    pub fn sample(&self) -> ResolvedParameters {
        let shaped = self.interpolation.shape(self.progress());
        let mut out = ResolvedParameters::new();
        for (&id, &start_value) in self.start.iter() {
            match self.dest.get(id) {
                Some(dest_value) => {
                    let kind = self.kinds.get(&id).copied().unwrap_or(ParameterKind::Linear);
                    out.set(id, blend::interpolate(kind, start_value, dest_value, shaped));
                }
                None => out.set(id, start_value),
            }
        }
        for (&id, &dest_value) in self.dest.iter() {
            if self.start.get(id).is_none() {
                out.set(id, dest_value);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    fn kinds(pairs: &[(u32, ParameterKind)]) -> BTreeMap<ParameterId, ParameterKind> {
        pairs.iter().map(|&(id, k)| (ParameterId::new(id), k)).collect()
    }

    fn resolved(pairs: &[(u32, Sample)]) -> ResolvedParameters {
        let mut r = ResolvedParameters::new();
        for &(id, v) in pairs {
            r.set(ParameterId::new(id), v);
        }
        r
    }

    #[test]
    fn progress_is_zero_at_start_and_advances() {
        let t = Transition::new(
            resolved(&[(1, 0.0)]),
            resolved(&[(1, 1.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            2.0,
            Interpolation::Linear,
        );
        assert!((t.progress() - 0.0).abs() < EPS);
    }

    #[test]
    fn linear_midpoint_blends_halfway() {
        let mut t = Transition::new(
            resolved(&[(1, 0.0)]),
            resolved(&[(1, 10.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            2.0,
            Interpolation::Linear,
        );
        let done = t.advance(1.0);
        assert!(!done);
        let s = t.sample();
        assert!((s.get(ParameterId::new(1)).expect("present") - 5.0).abs() < EPS);
    }

    #[test]
    fn zero_duration_completes_immediately() {
        let t = Transition::new(
            resolved(&[(1, 0.0)]),
            resolved(&[(1, 7.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            0.0,
            Interpolation::Linear,
        );
        assert!(t.is_complete());
        assert!((t.sample().get(ParameterId::new(1)).expect("present") - 7.0).abs() < EPS);
    }

    #[test]
    fn constant_curve_holds_start_until_complete() {
        let mut t = Transition::new(
            resolved(&[(1, 2.0)]),
            resolved(&[(1, 9.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            2.0,
            Interpolation::Constant,
        );
        t.advance(1.0);
        // Constant shape stays at 0 until the very end, so value holds start.
        assert!((t.sample().get(ParameterId::new(1)).expect("present") - 2.0).abs() < EPS);
    }

    #[test]
    fn hertz_blends_geometrically_at_midpoint() {
        let mut t = Transition::new(
            resolved(&[(1, 100.0)]),
            resolved(&[(1, 400.0)]),
            kinds(&[(1, ParameterKind::Hertz)]),
            2.0,
            Interpolation::Linear,
        );
        t.advance(1.0);
        assert!((t.sample().get(ParameterId::new(1)).expect("present") - 200.0).abs() < 1e-2);
    }

    #[test]
    fn missing_key_on_one_side_keeps_that_side() {
        let mut t = Transition::new(
            resolved(&[(1, 3.0)]),
            resolved(&[(2, 9.0)]),
            kinds(&[(1, ParameterKind::Linear), (2, ParameterKind::Linear)]),
            2.0,
            Interpolation::Linear,
        );
        t.advance(1.0);
        let s = t.sample();
        assert!((s.get(ParameterId::new(1)).expect("present") - 3.0).abs() < EPS);
        assert!((s.get(ParameterId::new(2)).expect("present") - 9.0).abs() < EPS);
    }

    #[test]
    fn advance_past_duration_completes_and_lands_on_dest() {
        let mut t = Transition::new(
            resolved(&[(1, 0.0)]),
            resolved(&[(1, 1.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            1.0,
            Interpolation::Linear,
        );
        assert!(t.advance(5.0));
        assert!((t.sample().get(ParameterId::new(1)).expect("present") - 1.0).abs() < EPS);
    }

    #[test]
    fn negative_dt_is_ignored() {
        let mut t = Transition::new(
            resolved(&[(1, 0.0)]),
            resolved(&[(1, 1.0)]),
            kinds(&[(1, ParameterKind::Linear)]),
            1.0,
            Interpolation::Linear,
        );
        t.advance(-1.0);
        assert!((t.progress() - 0.0).abs() < EPS);
    }
}
