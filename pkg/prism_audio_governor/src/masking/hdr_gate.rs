//! HDR-window virtualisation gate: the loudness-window half of design
//! section 33's psychoacoustic virtualisation, linked to the section 13 HDR
//! dynamic window.
//!
//! The HDR dynamic window ([`prism_audio_core`]'s `HdrWindow`) tracks the
//! loudest recent level (the window top) and maps it onto a target, bounding
//! the usable dynamic range to `window_db` below that top. A voice whose
//! perceived loudness sits below the window's lower edge is, after the HDR
//! mapping, pushed under the audible floor: rendering it spends a slot on a
//! sound the window has already suppressed. Design section 33 therefore
//! virtualises such voices directly, sharing the same loudness estimate the
//! HDR window already measures rather than running a second detector.
//!
//! This module is the strategy-side predicate. It does not depend on the
//! stateful HDR DSP core; the caller reads the window top with
//! `HdrWindow::reference_db()` and its configured width and passes both here as
//! an [`HdrWindowEdge`]. Verdicts are pure comparisons of decibel scalars, so a
//! fixed input yields a fixed verdict for golden testing.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements the HDR-linkage half of design section 33 (the masking half lives
//! in [`crate::masking::masking_model`]). Its verdicts union with the masking
//! verdicts and the importance threshold of [`crate::governor`] to drive the
//! virtual-voice promotion/demotion of design section 25. The floor-raise knob
//! is turned by the governor's budget state: a tighter budget raises the
//! effective floor so more voices fall under it and virtualise, mirroring how
//! the masking margin tightens under pressure.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

/// The lower edge of the shared HDR dynamic window, described by its top
/// (loudest-recent reference level) and width in decibels.
///
/// Build one from a live `HdrWindow` with `HdrWindow::reference_db()` for
/// `reference_db` and the configured `HdrParams::window_db` for `window_db`.
/// The gate never mutates DSP state; it only reads these two scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HdrWindowEdge {
    /// The window top in dBFS: the loudest-recent level the HDR window tracks.
    reference_db: Sample,
    /// The usable dynamic range in decibels below the top (non-negative).
    window_db: Sample,
}

impl HdrWindowEdge {
    /// Builds a window edge from its top and width.
    ///
    /// `window_db` is clamped to be non-negative; a non-finite `reference_db`
    /// is treated as the quietest possible top ([`Sample::MIN`]) so no voice is
    /// ever spuriously held audible by a corrupt reading.
    #[must_use]
    pub fn new(reference_db: Sample, window_db: Sample) -> Self {
        let reference_db = if reference_db.is_finite() {
            reference_db
        } else {
            Sample::MIN
        };
        let window_db = if window_db.is_finite() {
            window_db.max(0.0)
        } else {
            0.0
        };
        Self {
            reference_db,
            window_db,
        }
    }

    /// Returns the window top (loudest-recent reference level) in dBFS.
    #[inline]
    #[must_use]
    pub fn reference_db(self) -> Sample {
        self.reference_db
    }

    /// Returns the window width in decibels.
    #[inline]
    #[must_use]
    pub fn window_db(self) -> Sample {
        self.window_db
    }

    /// Returns the window's lower edge in dBFS: `reference_db - window_db`.
    ///
    /// Voices quieter than this edge have been suppressed under the audible
    /// floor by the HDR mapping.
    #[inline]
    #[must_use]
    pub fn lower_edge_db(self) -> Sample {
        self.reference_db - self.window_db
    }

    /// Returns a voice's headroom above the lower edge in decibels: positive
    /// means audible margin remains, negative means it sits below the edge.
    ///
    /// This is the deterministic quantity worth logging to telemetry for the
    /// golden-replay audit design section 33 calls for.
    #[inline]
    #[must_use]
    pub fn headroom_db(self, loudness_db: Sample) -> Sample {
        loudness_db - self.lower_edge_db()
    }

    /// Returns whether a voice at `loudness_db` falls below the window's lower
    /// edge and should be virtualised.
    ///
    /// A non-finite `loudness_db` is treated as silence and always virtualises.
    #[inline]
    #[must_use]
    pub fn is_below(self, loudness_db: Sample) -> bool {
        if !loudness_db.is_finite() {
            return true;
        }
        loudness_db < self.lower_edge_db()
    }

    /// Returns whether a voice at `loudness_db` falls below the window edge
    /// after raising the floor by `floor_raise_db`.
    ///
    /// The governor passes a non-negative `floor_raise_db` that grows with
    /// budget pressure: a larger raise lifts the effective floor so more voices
    /// read as below it and virtualise. A negative raise is clamped to zero so
    /// relaxing the budget never pushes the floor below the physical window
    /// edge.
    #[inline]
    #[must_use]
    pub fn is_below_floor(self, loudness_db: Sample, floor_raise_db: Sample) -> bool {
        if !loudness_db.is_finite() {
            return true;
        }
        let raise = if floor_raise_db.is_finite() {
            floor_raise_db.max(0.0)
        } else {
            0.0
        };
        loudness_db < self.lower_edge_db() + raise
    }
}

/// Produces one virtualisation verdict per voice loudness against a shared HDR
/// window edge and a budget-driven floor raise.
///
/// Entry `i` is `true` when voice `i` should be virtualised. The output length
/// matches `loudness_db`. This is the batch form of
/// [`HdrWindowEdge::is_below_floor`] for the governor's per-block pass; it
/// unions with the masking verdicts to form the full force-virtualise set.
#[must_use]
pub fn gate_voices(
    loudness_db: &[Sample],
    edge: HdrWindowEdge,
    floor_raise_db: Sample,
) -> Vec<bool> {
    loudness_db
        .iter()
        .map(|&db| edge.is_below_floor(db, floor_raise_db))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= 1e-4
    }

    #[test]
    fn lower_edge_is_top_minus_width() {
        let edge = HdrWindowEdge::new(-6.0, 24.0);
        assert!(approx(edge.lower_edge_db(), -30.0));
        assert!(approx(edge.reference_db(), -6.0));
        assert!(approx(edge.window_db(), 24.0));
    }

    #[test]
    fn width_clamps_non_negative() {
        let edge = HdrWindowEdge::new(-6.0, -10.0);
        assert!(approx(edge.window_db(), 0.0));
        assert!(approx(edge.lower_edge_db(), -6.0));
    }

    #[test]
    fn non_finite_reference_becomes_quietest_top() {
        let edge = HdrWindowEdge::new(Sample::INFINITY, 24.0);
        assert!(approx(edge.reference_db(), Sample::MIN));
    }

    #[test]
    fn voice_below_edge_virtualises() {
        let edge = HdrWindowEdge::new(-6.0, 24.0); // lower edge -30 dB
        assert!(edge.is_below(-40.0));
        assert!(!edge.is_below(-20.0));
        assert!(!edge.is_below(-30.0)); // exactly at the edge stays audible
    }

    #[test]
    fn non_finite_loudness_always_virtualises() {
        let edge = HdrWindowEdge::new(-6.0, 24.0);
        assert!(edge.is_below(Sample::NEG_INFINITY));
        assert!(edge.is_below(Sample::NAN));
        assert!(edge.is_below_floor(Sample::NAN, 0.0));
    }

    #[test]
    fn floor_raise_virtualises_more_voices() {
        let edge = HdrWindowEdge::new(-6.0, 24.0); // lower edge -30 dB
                                                   // A -25 dB voice is audible at rest but virtualises when the floor
                                                   // rises 10 dB under budget pressure.
        assert!(!edge.is_below_floor(-25.0, 0.0));
        assert!(edge.is_below_floor(-25.0, 10.0));
    }

    #[test]
    fn negative_floor_raise_is_clamped() {
        let edge = HdrWindowEdge::new(-6.0, 24.0);
        // A negative raise must not lower the floor below the physical edge.
        assert_eq!(
            edge.is_below_floor(-28.0, -100.0),
            edge.is_below_floor(-28.0, 0.0)
        );
    }

    #[test]
    fn headroom_sign_matches_audibility() {
        let edge = HdrWindowEdge::new(-6.0, 24.0); // lower edge -30 dB
        assert!(edge.headroom_db(-20.0) > 0.0);
        assert!(edge.headroom_db(-40.0) < 0.0);
        assert!(approx(edge.headroom_db(-30.0), 0.0));
    }

    #[test]
    fn gate_voices_matches_scalar_and_length() {
        let edge = HdrWindowEdge::new(-6.0, 24.0);
        let loud = [-10.0, -31.0, -29.9, -50.0];
        let out = gate_voices(&loud, edge, 0.0);
        assert_eq!(out.len(), loud.len());
        assert_eq!(out, [false, true, false, true]);
    }

    #[test]
    fn gate_voices_is_deterministic() {
        let edge = HdrWindowEdge::new(-3.0, 18.0);
        let loud = [-10.0, -22.0, -5.0, -40.0];
        assert_eq!(gate_voices(&loud, edge, 4.0), gate_voices(&loud, edge, 4.0));
    }
}
