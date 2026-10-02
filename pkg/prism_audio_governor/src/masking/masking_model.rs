//! Masking-aware virtualisation: deciding which voices a louder voice hides.
//!
//! Design section 33 asks for a simplified frequency-domain masking test:
//! compare per-band energies across voices, decide whether a quiet voice is
//! masked by louder neighbours in its own critical band, and virtualise the
//! masked low-contribution voices first. This module implements that over the
//! band partition from [`crate::masking::critical_bands`].
//!
//! The model is deliberately classical and cheap:
//!
//! 1. Each voice contributes an energy spectrum (one linear energy per band).
//! 2. Maskers' energy is *spread* across adjacent bands by a triangular
//!    spreading function (steeper toward lower frequencies, matching the
//!    asymmetry of real auditory masking) and summed into a masking profile.
//! 3. A probe voice is masked where its own band energy is below the summed
//!    masker energy in that band times a margin. The margin tightens as the
//!    governor's budget shrinks, so under pressure more voices read as masked
//!    and are culled.
//!
//! A voice is never counted as its own masker.
//!
//! # Determinism
//!
//! Only energy additions, multiplications, and [`bevy_math::ops`] powers are
//! used, so the masking verdicts are reproducible and golden-testable.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The spreading
//! function and in-band comparison are textbook perceptual-coding constructs.
//!
//! # Relationship
//!
//! Produces the `masked` flag consumed by
//! [`crate::governor::importance::ImportanceInputs`], and the margin is driven
//! by the budget state of [`crate::governor`]. The band layout comes from
//! [`crate::masking::critical_bands::CriticalBands`].

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::masking::critical_bands::CriticalBands;

/// Default masking margin: a probe is masked when its band energy is below the
/// masker energy there (a margin of `1.0` means "any louder masker masks").
pub const DEFAULT_MARGIN: Sample = 1.0;

/// Per-band energy spectrum of a single voice (linear energy, not decibels).
///
/// Length should match the band count of the [`CriticalBands`] partition in
/// use; shorter or longer vectors are tolerated by the model (missing bands are
/// treated as zero energy).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceSpectrum {
    /// Linear energy per critical band.
    pub bands: Vec<Sample>,
}

impl VoiceSpectrum {
    /// Creates a zero spectrum with `band_count` bands.
    #[must_use]
    pub fn zeros(band_count: usize) -> Self {
        Self { bands: vec![0.0; band_count] }
    }

    /// Builds a single-band spectrum: all energy concentrated in the band that
    /// contains `center_hz`, given a band partition. A convenience for simple
    /// tonal sources.
    #[must_use]
    pub fn tonal(bands: &CriticalBands, center_hz: Sample, energy: Sample) -> Self {
        let mut spectrum = Self::zeros(bands.band_count());
        let idx = bands.band_of(center_hz);
        spectrum.bands[idx] = energy.max(0.0);
        spectrum
    }

    /// Returns the total linear energy summed over all bands, sanitising
    /// non-finite entries to zero.
    #[must_use]
    pub fn total_energy(&self) -> Sample {
        let mut sum = 0.0;
        for &b in &self.bands {
            if b.is_finite() && b > 0.0 {
                sum += b;
            }
        }
        sum
    }

    /// Returns the index of the band holding the most energy, or `0` for an
    /// empty/zero spectrum.
    #[must_use]
    pub fn peak_band(&self) -> usize {
        let mut best = 0;
        let mut best_e = Sample::NEG_INFINITY;
        for (i, &b) in self.bands.iter().enumerate() {
            let e = if b.is_finite() { b } else { 0.0 };
            if e > best_e {
                best_e = e;
                best = i;
            }
        }
        best
    }
}

/// The asymmetric triangular spreading function and in-band margin used to turn
/// raw spectra into masking verdicts.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskingModel {
    /// Energy roll-off per band toward *higher* frequencies, as a linear factor
    /// in `(0, 1]` applied once per band of separation (upward spread is wide).
    pub spread_up: Sample,
    /// Energy roll-off per band toward *lower* frequencies, as a linear factor
    /// in `(0, 1]` (downward spread is steeper, so a smaller factor).
    pub spread_down: Sample,
    /// How many neighbouring bands the spread reaches on each side.
    pub reach: usize,
}

impl Default for MaskingModel {
    /// A reasonable default: upward spread of `0.5` per band, steeper downward
    /// spread of `0.25` per band, reaching three bands each way.
    fn default() -> Self {
        Self { spread_up: 0.5, spread_down: 0.25, reach: 3 }
    }
}

impl MaskingModel {
    /// Spreads a summed masker spectrum across neighbouring bands into `out`.
    ///
    /// `out` is overwritten; it must be at least as long as `maskers`. Energy in
    /// band `j` contributes to band `k` scaled by `spread^|k - j|` with the
    /// up/down factor chosen by direction. The band's own energy is included at
    /// full weight.
    pub fn spread(&self, maskers: &[Sample], out: &mut [Sample]) {
        let n = maskers.len().min(out.len());
        for o in out.iter_mut().take(n) {
            *o = 0.0;
        }
        let up = clamp_unit_pos(self.spread_up);
        let down = clamp_unit_pos(self.spread_down);
        for (j, &src_raw) in maskers.iter().enumerate().take(n) {
            let src = if src_raw.is_finite() && src_raw > 0.0 { src_raw } else { 0.0 };
            if src == 0.0 {
                continue;
            }
            let lo = j.saturating_sub(self.reach);
            let hi = (j + self.reach).min(n - 1);
            for (k, slot) in out.iter_mut().enumerate().take(hi + 1).skip(lo) {
                let dist = k.abs_diff(j);
                let factor = if k >= j {
                    ops::powf(up, dist as Sample)
                } else {
                    ops::powf(down, dist as Sample)
                };
                *slot += src * factor;
            }
        }
    }

    /// Returns the spread masking profile of a set of masker spectra as a new
    /// vector of length `band_count`.
    ///
    /// Each spectrum's bands are summed, then the sum is spread. Spectra shorter
    /// than `band_count` contribute zeros for their missing bands.
    #[must_use]
    pub fn masking_profile(&self, maskers: &[&VoiceSpectrum], band_count: usize) -> Vec<Sample> {
        let mut summed = vec![0.0; band_count];
        for spectrum in maskers {
            for (i, &e) in spectrum.bands.iter().enumerate().take(band_count) {
                if e.is_finite() && e > 0.0 {
                    summed[i] += e;
                }
            }
        }
        let mut spread = vec![0.0; band_count];
        self.spread(&summed, &mut spread);
        spread
    }
}

/// Decides whether `probe` is masked given a precomputed spread masking
/// `profile` and a `margin`.
///
/// The probe is masked when, in its peak band, its own energy is at or below
/// `margin` times the profile energy there. A larger margin masks more
/// aggressively (used when the budget tightens). Non-finite inputs are treated
/// as zero energy.
#[must_use]
pub fn is_masked(probe: &VoiceSpectrum, profile: &[Sample], margin: Sample) -> bool {
    let band = probe.peak_band();
    let probe_e = probe.bands.get(band).copied().unwrap_or(0.0);
    let probe_e = if probe_e.is_finite() { probe_e.max(0.0) } else { 0.0 };
    let masker_e = profile.get(band).copied().unwrap_or(0.0);
    let masker_e = if masker_e.is_finite() { masker_e.max(0.0) } else { 0.0 };
    let m = if margin.is_finite() { margin.max(0.0) } else { DEFAULT_MARGIN };
    // A silent probe is trivially inaudible; a zero masker never masks.
    if probe_e <= 0.0 {
        return masker_e > 0.0;
    }
    probe_e <= m * masker_e
}

/// Analyses a whole set of voices at once, reporting which are masked by the
/// others.
///
/// For each voice, the masking profile is formed from *all other* voices (never
/// itself), then [`is_masked`] is applied. Returns a boolean per input voice in
/// input order. Allocates its working buffers; intended for the block-boundary
/// decision pass, not the sample hot path.
#[derive(Debug, Clone)]
pub struct MaskingAnalyzer {
    model: MaskingModel,
    band_count: usize,
}

impl MaskingAnalyzer {
    /// Creates an analyser for a given band count and spreading model.
    #[must_use]
    pub fn new(model: MaskingModel, band_count: usize) -> Self {
        Self { model, band_count: band_count.max(1) }
    }

    /// Returns the band count the analyser expects.
    #[must_use]
    #[inline]
    pub fn band_count(&self) -> usize {
        self.band_count
    }

    /// Computes a masked flag for each voice using the shared `margin`.
    ///
    /// The margin typically comes from the governor: tighter (larger) under
    /// budget pressure. Returns a `Vec<bool>` aligned with `voices`.
    #[must_use]
    pub fn analyze(&self, voices: &[VoiceSpectrum], margin: Sample) -> Vec<bool> {
        let n = voices.len();
        let mut out = vec![false; n];
        if n == 0 {
            return out;
        }
        // Sum all voices once, then for each probe subtract its own energy to
        // get the "all others" masker set without re-summing n times.
        let mut total = vec![0.0; self.band_count];
        for v in voices {
            for (i, &e) in v.bands.iter().enumerate().take(self.band_count) {
                if e.is_finite() && e > 0.0 {
                    total[i] += e;
                }
            }
        }
        let mut others = vec![0.0; self.band_count];
        let mut profile = vec![0.0; self.band_count];
        for (idx, probe) in voices.iter().enumerate() {
            for i in 0..self.band_count {
                let own = probe.bands.get(i).copied().unwrap_or(0.0);
                let own = if own.is_finite() && own > 0.0 { own } else { 0.0 };
                others[i] = (total[i] - own).max(0.0);
            }
            self.model.spread(&others, &mut profile);
            out[idx] = is_masked(probe, &profile, margin);
        }
        out
    }
}

/// Clamps a spreading factor into `(0, 1]`, mapping non-finite or non-positive
/// input to a tiny positive value so the power stays well-defined.
#[inline]
fn clamp_unit_pos(x: Sample) -> Sample {
    if x.is_finite() && x > 0.0 { x.min(1.0) } else { Sample::MIN_POSITIVE }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masking::critical_bands::CriticalBands;

    const EPS: Sample = 1e-6;

    #[test]
    fn total_energy_sums_bands() {
        let s = VoiceSpectrum { bands: vec![1.0, 2.0, 3.0] };
        assert!((s.total_energy() - 6.0).abs() < EPS);
    }

    #[test]
    fn total_energy_ignores_non_finite() {
        let s = VoiceSpectrum { bands: vec![1.0, Sample::NAN, -5.0, 2.0] };
        assert!((s.total_energy() - 3.0).abs() < EPS);
    }

    #[test]
    fn peak_band_finds_max() {
        let s = VoiceSpectrum { bands: vec![0.1, 0.9, 0.3] };
        assert_eq!(s.peak_band(), 1);
    }

    #[test]
    fn spread_puts_energy_in_band() {
        let model = MaskingModel::default();
        let maskers = vec![0.0, 1.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 5];
        model.spread(&maskers, &mut out);
        // Own band keeps full energy, neighbours get a fraction.
        assert!((out[1] - 1.0).abs() < EPS);
        assert!(out[2] > 0.0 && out[2] < 1.0);
        assert!(out[0] > 0.0 && out[0] < 1.0);
    }

    #[test]
    fn spread_is_asymmetric() {
        let model = MaskingModel { spread_up: 0.6, spread_down: 0.2, reach: 2 };
        let maskers = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let mut out = vec![0.0; 5];
        model.spread(&maskers, &mut out);
        // Upward (higher index) spread wider than downward.
        assert!(out[3] > out[1], "up={} down={}", out[3], out[1]);
    }

    #[test]
    fn loud_masker_masks_quiet_probe() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        let masker = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let probe = VoiceSpectrum::tonal(&bands, 1000.0, 0.01);
        let model = MaskingModel::default();
        let profile = model.masking_profile(&[&masker], n);
        assert!(is_masked(&probe, &profile, DEFAULT_MARGIN));
    }

    #[test]
    fn loud_probe_is_not_masked() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        let masker = VoiceSpectrum::tonal(&bands, 1000.0, 0.01);
        let probe = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let model = MaskingModel::default();
        let profile = model.masking_profile(&[&masker], n);
        assert!(!is_masked(&probe, &profile, DEFAULT_MARGIN));
    }

    #[test]
    fn distant_band_does_not_mask() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        let masker = VoiceSpectrum::tonal(&bands, 200.0, 1.0);
        let probe = VoiceSpectrum::tonal(&bands, 12_000.0, 0.05);
        let model = MaskingModel { spread_up: 0.5, spread_down: 0.25, reach: 2 };
        let profile = model.masking_profile(&[&masker], n);
        assert!(!is_masked(&probe, &profile, DEFAULT_MARGIN));
    }

    #[test]
    fn tighter_margin_masks_more() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        let masker = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let probe = VoiceSpectrum::tonal(&bands, 1000.0, 1.5);
        let model = MaskingModel::default();
        let profile = model.masking_profile(&[&masker], n);
        // Loud probe survives at margin 1.0 but not at a tighter margin 2.0.
        assert!(!is_masked(&probe, &profile, 1.0));
        assert!(is_masked(&probe, &profile, 2.0));
    }

    #[test]
    fn analyzer_excludes_self() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        // Two identical loud voices: neither should mask itself into silence,
        // but each is masked by the OTHER (equal energy, margin 1.0 => masked).
        let a = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let b = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let analyzer = MaskingAnalyzer::new(MaskingModel::default(), n);
        let flags = analyzer.analyze(&[a, b], DEFAULT_MARGIN);
        assert_eq!(flags.len(), 2);
        // Each is masked by the equally-loud other at margin 1.0.
        assert!(flags[0]);
        assert!(flags[1]);
    }

    #[test]
    fn analyzer_single_voice_not_masked() {
        let bands = CriticalBands::bark_default();
        let n = bands.band_count();
        let only = VoiceSpectrum::tonal(&bands, 1000.0, 1.0);
        let analyzer = MaskingAnalyzer::new(MaskingModel::default(), n);
        let flags = analyzer.analyze(&[only], DEFAULT_MARGIN);
        assert!(!flags[0]);
    }

    #[test]
    fn analyzer_empty_input() {
        let analyzer = MaskingAnalyzer::new(MaskingModel::default(), 24);
        let flags = analyzer.analyze(&[], DEFAULT_MARGIN);
        assert!(flags.is_empty());
    }

    #[test]
    fn silent_probe_masked_only_when_masker_present() {
        let silent = VoiceSpectrum::zeros(4);
        let profile_quiet = vec![0.0; 4];
        let profile_loud = vec![1.0; 4];
        assert!(!is_masked(&silent, &profile_quiet, 1.0));
        assert!(is_masked(&silent, &profile_loud, 1.0));
    }
}
