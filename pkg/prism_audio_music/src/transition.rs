//! **Transitions**: when and how one piece of music hands off to another.
//!
//! A transition answers two independent questions:
//!
//! - *When* does the switch happen? [`TransitionType`] names a quantization
//!   grid -- immediately, on the next beat, the next bar, the next `1/n`
//!   subdivision, the current segment's exit cue, or a named marker -- which
//!   the planner resolves to a sample-accurate boundary against the active
//!   [`prism_audio_core::scheduler::NamedClock`].
//! - *How* does it sound during the switch? [`Fade`] carries an out/in sample
//!   count and a [`FadeCurve`] (equal-power or linear) so the outgoing and
//!   incoming material cross without a click, and an optional **bridge**
//!   segment can be inserted between them (a transition segment / fill).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The quantized
//! transition / crossfade / bridge model is reconstructed from first
//! principles over plain data and classic equal-power gain math
//! (`cos`/`sin` of a quarter turn) evaluated through
//! [`bevy_math::ops`] for cross-platform determinism. No AI/ML.
//!
//! # Relationship
//!
//! [`TransitionType`] maps onto [`prism_audio_core::scheduler::Grid`] inside
//! [`crate::system::MusicSystem`]; [`Transition`] values drive segment
//! re-sequencing and the edges of a [`crate::clip_graph::ClipGraph`]. Fade
//! sample counts and curve kinds are forwarded verbatim on the emitted
//! [`crate::action::MusicAction`]s.

use core::f32::consts::FRAC_PI_2;

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::id::{MarkerId, SegmentId};

/// The quantization grid a transition snaps to.
///
/// Every variant except [`TransitionType::Immediate`] resolves to the next
/// boundary at or after the request sample, so a transition never lands in the
/// past and is always sample-accurate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TransitionType {
    /// Switch at the exact request sample, with no quantization.
    Immediate,
    /// Snap to the next beat boundary.
    NextBeat,
    /// Snap to the next bar boundary (beat count from the time signature).
    NextBar,
    /// Snap to the next `1/n`-of-a-beat subdivision (e.g. `NextGrid(4)` is
    /// sixteenth notes in 4/4). The subdivision is clamped to at least one.
    NextGrid(u32),
    /// Snap to the current segment's exit cue (the seamless hand-off point).
    SegmentEnd,
    /// Snap to the next occurrence of a named marker in the current segment.
    Marker(MarkerId),
}

/// The gain shape applied across a fade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FadeCurve {
    /// Linear amplitude ramp: gain equals the normalized position.
    Linear,
    /// Equal-power (constant-energy) crossfade: `sin`/`cos` of a quarter turn,
    /// so a simultaneous out/in fade keeps perceived loudness constant.
    EqualPower,
}

impl FadeCurve {
    /// Returns the rising-side linear gain in `[0, 1]` at normalized position
    /// `t` (clamped to `[0, 1]`).
    ///
    /// The falling-side gain is `gain_at(1 - t)`; for [`FadeCurve::EqualPower`]
    /// the two sides satisfy `rise^2 + fall^2 == 1`, the constant-power
    /// property.
    #[must_use]
    pub fn gain_at(self, t: Sample) -> Sample {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EqualPower => ops::sin(t * FRAC_PI_2),
        }
    }
}

/// A crossfade specification in samples plus the shaping curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Fade {
    /// Samples over which the outgoing material fades out.
    pub fade_out: u64,
    /// Samples over which the incoming material fades in.
    pub fade_in: u64,
    /// Gain shape applied to both sides.
    pub curve: FadeCurve,
}

impl Fade {
    /// Builds a fade with explicit out/in lengths and curve.
    #[must_use]
    pub fn new(fade_out: u64, fade_in: u64, curve: FadeCurve) -> Self {
        Self {
            fade_out,
            fade_in,
            curve,
        }
    }

    /// An instantaneous (hard) cut: zero-length fades.
    #[must_use]
    pub fn cut() -> Self {
        Self {
            fade_out: 0,
            fade_in: 0,
            curve: FadeCurve::Linear,
        }
    }

    /// A symmetric equal-power crossfade of `samples` on each side.
    #[must_use]
    pub fn crossfade(samples: u64) -> Self {
        Self {
            fade_out: samples,
            fade_in: samples,
            curve: FadeCurve::EqualPower,
        }
    }
}

impl Default for Fade {
    fn default() -> Self {
        Self::cut()
    }
}

/// A complete transition: when to switch, how to fade, and an optional bridge
/// segment inserted between the outgoing and incoming material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    /// The quantization grid the switch snaps to.
    pub transition_type: TransitionType,
    /// The crossfade applied at the switch.
    pub fade: Fade,
    /// Optional transition segment played between the two pieces; the incoming
    /// segment follows once the bridge completes.
    pub bridge: Option<SegmentId>,
}

impl Transition {
    /// Builds a transition with no bridge segment.
    #[must_use]
    pub fn new(transition_type: TransitionType, fade: Fade) -> Self {
        Self {
            transition_type,
            fade,
            bridge: None,
        }
    }

    /// Builds a transition that bridges through `bridge` before the incoming
    /// segment plays.
    #[must_use]
    pub fn bridged(transition_type: TransitionType, fade: Fade, bridge: SegmentId) -> Self {
        Self {
            transition_type,
            fade,
            bridge: Some(bridge),
        }
    }

    /// An immediate hard cut with no fade and no bridge.
    #[must_use]
    pub fn immediate() -> Self {
        Self::new(TransitionType::Immediate, Fade::cut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-5;

    #[test]
    fn linear_gain_is_identity() {
        assert!((FadeCurve::Linear.gain_at(0.0) - 0.0).abs() < EPS);
        assert!((FadeCurve::Linear.gain_at(0.5) - 0.5).abs() < EPS);
        assert!((FadeCurve::Linear.gain_at(1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn equal_power_is_constant_energy() {
        for i in 0..=20 {
            let t = i as Sample / 20.0;
            let rise = FadeCurve::EqualPower.gain_at(t);
            let fall = FadeCurve::EqualPower.gain_at(1.0 - t);
            assert!((rise * rise + fall * fall - 1.0).abs() < 1.0e-4, "t={t}");
        }
    }

    #[test]
    fn gain_clamps_outside_unit_range() {
        assert!((FadeCurve::Linear.gain_at(-1.0) - 0.0).abs() < EPS);
        assert!((FadeCurve::Linear.gain_at(2.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn constructors_populate_fields() {
        let cut = Fade::cut();
        assert_eq!(cut.fade_out, 0);
        assert_eq!(cut.fade_in, 0);
        let xf = Fade::crossfade(512);
        assert_eq!(xf.fade_out, 512);
        assert_eq!(xf.fade_in, 512);
        assert_eq!(xf.curve, FadeCurve::EqualPower);

        let t = Transition::bridged(TransitionType::NextBar, xf, SegmentId::new(5));
        assert_eq!(t.bridge, Some(SegmentId::new(5)));
        assert_eq!(Transition::immediate().transition_type, TransitionType::Immediate);
    }
}
