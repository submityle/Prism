//! Confluence of per-voice virtualisation predicates into one force set.
//!
//! Design section 33 retires inaudible voices to virtual state for two
//! independent classical reasons:
//!
//! * **Spectral masking** -- a voice is buried by louder neighbours in its
//!   critical band ([`crate::masking::masking_model::MaskingAnalyzer::analyze`]).
//! * **HDR windowing** -- a voice sits below the dynamic loudness window's lower
//!   edge, so the foreground mix would never surface it
//!   ([`crate::masking::hdr_gate::gate_voices`]).
//!
//! Either reason alone is sufficient to virtualise a voice, so the final
//! decision is the per-voice **union** of the two verdicts. This module keeps
//! the two predicates decoupled (each stays a pure function of its own inputs)
//! and joins them here, preserving *why* each voice was flagged so the profiler
//! and [`crate::governor::importance`] can report and penalise accordingly.
//!
//! The combined boolean is what callers feed into
//! [`crate::governor::importance::ImportanceInputs::masked`]: a voice that is
//! masked *or* HDR-gated is deprioritised by the masking penalty and promoted
//! to virtual state by the voice pool of design section 25.
//!
//! Verdict vectors are aligned per voice. When two inputs differ in length (for
//! example only a subset of voices carries an HDR loudness estimate), the
//! missing entries default to `false`: an absent predicate never forces
//! virtualisation on its own. The output length is the longer of the inputs.
//!
//! # Determinism
//!
//! The union is a pure elementwise boolean OR, so a fixed pair of verdict
//! vectors yields a fixed force set for golden testing.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Joins the two design-section-33 virtualisation predicates
//! ([`crate::masking::masking_model`] and [`crate::masking::hdr_gate`]) into the
//! `masked` input of [`crate::governor::importance`] and the virtual-voice
//! management of design section 25.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Why a voice was flagged for virtualisation.
///
/// Both flags can be set simultaneously; [`Self::should_virtualize`] is the
/// union that drives the force decision, while the individual flags preserve
/// the reason for telemetry and debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceVirtualization {
    /// The voice is spectrally masked by louder neighbours (section 33).
    pub masked: bool,
    /// The voice sits below the HDR loudness window's lower edge (section 13).
    pub hdr_gated: bool,
}

impl VoiceVirtualization {
    /// Returns whether the voice should be virtualised for any reason.
    #[must_use]
    #[inline]
    pub const fn should_virtualize(self) -> bool {
        self.masked || self.hdr_gated
    }
}

/// Reads verdict index `i` from `verdict`, treating out-of-range as `false`.
#[inline]
fn verdict_at(verdict: &[bool], i: usize) -> bool {
    verdict.get(i).copied().unwrap_or(false)
}

/// Combines the masking and HDR-gate verdicts into one per-voice record,
/// preserving each reason.
///
/// The result length is the longer of `masking` and `hdr_gated`; missing
/// entries in either input default to `false`.
#[must_use]
pub fn combine(masking: &[bool], hdr_gated: &[bool]) -> Vec<VoiceVirtualization> {
    let n = masking.len().max(hdr_gated.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(VoiceVirtualization {
            masked: verdict_at(masking, i),
            hdr_gated: verdict_at(hdr_gated, i),
        });
    }
    out
}

/// Returns the per-voice force-virtualise set: the elementwise union of the
/// masking and HDR-gate verdicts.
///
/// This boolean is what feeds
/// [`crate::governor::importance::ImportanceInputs::masked`]. The result length
/// is the longer of the two inputs; missing entries default to `false`.
///
/// ```
/// use prism_audio_governor::masking::virtualization::force_virtualize_set;
///
/// let masked = [true, false, false];
/// let hdr = [false, true, false];
/// assert_eq!(force_virtualize_set(&masked, &hdr), vec![true, true, false]);
/// ```
#[must_use]
pub fn force_virtualize_set(masking: &[bool], hdr_gated: &[bool]) -> Vec<bool> {
    let n = masking.len().max(hdr_gated.len());
    let mut out = vec![false; n];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = verdict_at(masking, i) || verdict_at(hdr_gated, i);
    }
    out
}

/// Returns the number of voices in `verdicts` that should be virtualised.
#[must_use]
pub fn virtualized_count(verdicts: &[VoiceVirtualization]) -> usize {
    verdicts.iter().filter(|v| v.should_virtualize()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_is_elementwise_or() {
        let masked = [true, false, true, false];
        let hdr = [false, false, true, true];
        assert_eq!(
            force_virtualize_set(&masked, &hdr),
            vec![true, false, true, true]
        );
    }

    #[test]
    fn combine_preserves_each_reason() {
        let masked = [true, false, true];
        let hdr = [false, true, true];
        let combined = combine(&masked, &hdr);
        assert_eq!(combined.len(), 3);
        assert!(combined[0].masked && !combined[0].hdr_gated);
        assert!(!combined[1].masked && combined[1].hdr_gated);
        assert!(combined[2].masked && combined[2].hdr_gated);
        for v in &combined {
            assert!(v.should_virtualize());
        }
    }

    #[test]
    fn should_virtualize_matches_union() {
        let masked = [true, false, false];
        let hdr = [false, true, false];
        let combined = combine(&masked, &hdr);
        let set = force_virtualize_set(&masked, &hdr);
        for (v, &s) in combined.iter().zip(set.iter()) {
            assert_eq!(v.should_virtualize(), s);
        }
    }

    #[test]
    fn mismatched_lengths_pad_with_false() {
        // HDR estimate missing for the last two voices.
        let masked = [false, true, false, false];
        let hdr = [false, false];
        let set = force_virtualize_set(&masked, &hdr);
        assert_eq!(set, vec![false, true, false, false]);

        // Masking shorter than HDR.
        let set2 = force_virtualize_set(&[true], &[false, true, true]);
        assert_eq!(set2, vec![true, true, true]);
    }

    #[test]
    fn empty_inputs_yield_empty_set() {
        let set = force_virtualize_set(&[], &[]);
        assert!(set.is_empty());
        assert_eq!(combine(&[], &[]).len(), 0);
    }

    #[test]
    fn default_voice_is_audible() {
        let v = VoiceVirtualization::default();
        assert!(!v.should_virtualize());
    }

    #[test]
    fn virtualized_count_counts_union() {
        let combined = combine(&[true, false, false, true], &[false, true, false, false]);
        // Voices 0 (masked), 1 (hdr), 3 (masked) => 3.
        assert_eq!(virtualized_count(&combined), 3);
    }

    #[test]
    fn no_predicate_fires_no_virtualization() {
        let masked = [false, false, false];
        let hdr = [false, false, false];
        assert_eq!(virtualized_count(&combine(&masked, &hdr)), 0);
        assert_eq!(
            force_virtualize_set(&masked, &hdr),
            vec![false, false, false]
        );
    }
}
