//! Spectrum band-energy snapshots built on the core FFT analyzer.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the spectrum-analysis part of design section 26 (`SpectrumNode`
//! band energy for music visualization, beat triggering, and debugging). This
//! module reduces the magnitude bins produced by
//! [`SpectrumAnalyzer`](prism_audio_core::nodes::analysis::spectrum::SpectrumAnalyzer)
//! into perceptually spaced bands; it does not re-implement the FFT.

use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::math::{linear_to_db, Sample};
use prism_audio_core::nodes::analysis::spectrum::SpectrumAnalyzer;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Reference band center for octave / third-octave grids, in Hz (`IEC` 1 kHz).
pub const REFERENCE_BAND_HZ: Sample = 1000.0;

/// How the magnitude bins are grouped into bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum BandScale {
    /// One band per raw FFT bin (center is the bin frequency).
    Linear,
    /// Octave bands centered on the `IEC` 1 kHz grid.
    Octave,
    /// Third-octave bands centered on the `IEC` 1 kHz grid.
    ThirdOctave,
}

/// A read-only reduction of one analysis frame's magnitude spectrum into
/// perceptually spaced bands.
///
/// `center_hz[i]` is the nominal center frequency of band `i`; `energy[i]` is
/// the summed squared magnitude of every bin assigned to that band (linear
/// energy, never negative). Bands with no bins report zero energy.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SpectrumSnapshot {
    /// Band grouping used to build this snapshot.
    pub scale: BandScale,
    /// Nominal center frequency of each band, in Hz, ascending.
    pub center_hz: Vec<Sample>,
    /// Linear energy (sum of squared magnitudes) assigned to each band.
    pub energy: Vec<Sample>,
}

impl SpectrumSnapshot {
    /// Build a snapshot from an analyzer's current magnitude bins.
    ///
    /// `sample_rate` is used to map bins to frequencies. Returns an empty
    /// snapshot when the analyzer has produced no bins or the sample rate is
    /// zero.
    #[must_use]
    pub fn from_analyzer(
        analyzer: &SpectrumAnalyzer,
        sample_rate: u32,
        scale: BandScale,
    ) -> Self {
        let magnitudes = analyzer.magnitudes();
        if magnitudes.is_empty() || sample_rate == 0 {
            return Self {
                scale,
                center_hz: Vec::new(),
                energy: Vec::new(),
            };
        }

        match scale {
            BandScale::Linear => {
                let mut center_hz = Vec::with_capacity(magnitudes.len());
                let mut energy = Vec::with_capacity(magnitudes.len());
                for (bin, &mag) in magnitudes.iter().enumerate() {
                    center_hz.push(analyzer.bin_frequency(bin, sample_rate));
                    energy.push(mag * mag);
                }
                Self {
                    scale,
                    center_hz,
                    energy,
                }
            }
            BandScale::Octave => {
                Self::fractional_octave(analyzer, sample_rate, scale, 1)
            }
            BandScale::ThirdOctave => {
                Self::fractional_octave(analyzer, sample_rate, scale, 3)
            }
        }
    }

    /// Energy of band `index` expressed in decibels (`10*log10`), or the dB
    /// floor when the band is empty or absent.
    #[must_use]
    pub fn band_db(&self, index: usize) -> Sample {
        match self.energy.get(index) {
            // `linear_to_db` is a 20*log10 amplitude map; energy is already a
            // squared quantity, so its amplitude equivalent is the square root.
            Some(&value) => linear_to_db(ops::sqrt(value.max(0.0))),
            None => linear_to_db(0.0),
        }
    }

    /// Total energy summed across every band.
    #[must_use]
    pub fn total_energy(&self) -> Sample {
        self.energy.iter().copied().sum()
    }

    /// Index of the band carrying the most energy, or `None` when empty.
    #[must_use]
    pub fn dominant_band(&self) -> Option<usize> {
        let mut best: Option<(usize, Sample)> = None;
        for (index, &value) in self.energy.iter().enumerate() {
            match best {
                Some((_, best_value)) if value <= best_value => {}
                _ => best = Some((index, value)),
            }
        }
        best.map(|(index, _)| index)
    }

    /// Number of bands in this snapshot.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.center_hz.len()
    }

    /// `true` when the snapshot has no bands.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.center_hz.is_empty()
    }

    /// Group bins into `1/divisions`-octave bands on the `IEC` 1 kHz grid,
    /// spanning only bands that overlap the analyzer's resolvable range.
    fn fractional_octave(
        analyzer: &SpectrumAnalyzer,
        sample_rate: u32,
        scale: BandScale,
        divisions: i32,
    ) -> Self {
        let bins = analyzer.magnitudes().len();
        let nyquist = analyzer.bin_frequency(bins - 1, sample_rate);
        let lowest = analyzer.bin_frequency(1, sample_rate).max(1.0);

        // Half-band ratio: 2^(1/(2*divisions)).
        let half_ratio = ops::powf(2.0, 1.0 / (2.0 * divisions as Sample));

        // Determine the band-index range (k) whose centers 1000*2^(k/div) stay
        // within [lowest, nyquist].
        let k_lo = fractional_band_index(lowest, divisions);
        let k_hi = fractional_band_index(nyquist, divisions);

        let mut center_hz = Vec::new();
        let mut energy = Vec::new();

        for k in k_lo..=k_hi {
            let center = REFERENCE_BAND_HZ * ops::powf(2.0, k as Sample / divisions as Sample);
            let low_edge = center / half_ratio;
            let high_edge = center * half_ratio;

            let mut band_energy = 0.0 as Sample;
            for (bin, &mag) in analyzer.magnitudes().iter().enumerate() {
                let freq = analyzer.bin_frequency(bin, sample_rate);
                if freq >= low_edge && freq < high_edge {
                    band_energy += mag * mag;
                }
            }
            center_hz.push(center);
            energy.push(band_energy);
        }

        Self {
            scale,
            center_hz,
            energy,
        }
    }
}

/// Nearest fractional-octave band index `k` such that `1000 * 2^(k/div)`
/// approximates `freq`.
fn fractional_band_index(freq: Sample, divisions: i32) -> i32 {
    // k = div * log2(freq / 1000); rounded to the nearest integer band.
    let ratio = (freq / REFERENCE_BAND_HZ).max(Sample::MIN_POSITIVE);
    let log2 = ops::ln(ratio) / core::f32::consts::LN_2;
    let k = divisions as Sample * log2;
    round_half_away(k) as i32
}

/// Round to the nearest integer, ties away from zero, without `std`.
#[inline]
fn round_half_away(value: Sample) -> Sample {
    if value >= 0.0 {
        ops::floor(value + 0.5)
    } else {
        ops::ceil(value - 0.5)
    }
}
