//! Simplified critical-band scales (Bark / ERB) for masking analysis.
//!
//! Human hearing resolves frequency into overlapping *critical bands*: energy
//! within one band largely masks other energy in the same band. Design section
//! 33 calls for a simplified critical-band model to decide masking, so this
//! module provides the two standard warped scales and a fixed band partition
//! built from them.
//!
//! - **Bark** (Zwicker): [`hz_to_bark`] / [`bark_to_hz`] via the classic
//!   analytic approximation.
//! - **ERB** (Glasberg & Moore): [`erb_bandwidth`] and [`hz_to_erb_number`]
//!   giving the equivalent-rectangular-bandwidth number.
//!
//! A [`CriticalBands`] partition slices a frequency range into contiguous bands
//! uniformly spaced on the chosen scale, with [`CriticalBands::band_of`]
//! mapping any frequency to a band index. The partition is built once off the
//! real-time thread.
//!
//! # Determinism
//!
//! Every transcendental routes through [`bevy_math::ops`], so band edges and
//! lookups are bit-reproducible across targets.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The Bark and ERB
//! formulae are standard published psychoacoustic approximations.
//!
//! # Relationship
//!
//! Provides the band partition that [`crate::masking::masking_model`] fills
//! with per-voice energy and spreads to decide masking. The band count also
//! shapes the timbre descriptor used by [`crate::clustering`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// Converts a frequency in hertz to the Bark scale (Zwicker 1961
/// approximation).
///
/// Negative inputs are clamped to `0 Hz`. The mapping is monotonic increasing.
///
/// # Examples
///
/// ```
/// # use prism_audio_governor::masking::critical_bands::hz_to_bark;
/// assert!(hz_to_bark(0.0) >= 0.0);
/// assert!(hz_to_bark(1000.0) < hz_to_bark(4000.0));
/// ```
#[must_use]
pub fn hz_to_bark(hz: Sample) -> Sample {
    let f = hz.max(0.0);
    // z = 13*atan(0.00076*f) + 3.5*atan((f/7500)^2)
    let a = 13.0 * ops::atan(0.000_76 * f);
    let r = f / 7500.0;
    let b = 3.5 * ops::atan(r * r);
    a + b
}

/// Converts a Bark value back to an approximate frequency in hertz.
///
/// This inverts [`hz_to_bark`] numerically by bisection (the forward map has no
/// closed-form inverse). Negative inputs are clamped to `0`. Accurate to well
/// under 1 Hz across the audible range.
#[must_use]
pub fn bark_to_hz(bark: Sample) -> Sample {
    let target = bark.max(0.0);
    let mut lo: Sample = 0.0;
    let mut hi: Sample = 24_000.0;
    // 40 bisection steps resolve the 0..24 kHz range to sub-millihertz.
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if hz_to_bark(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Returns the equivalent rectangular bandwidth (ERB), in hertz, of the
/// auditory filter centred at `hz` (Glasberg & Moore 1990).
///
/// Negative inputs are clamped to `0 Hz`.
///
/// # Examples
///
/// ```
/// # use prism_audio_governor::masking::critical_bands::erb_bandwidth;
/// // ERB widens with centre frequency.
/// assert!(erb_bandwidth(500.0) < erb_bandwidth(5000.0));
/// ```
#[must_use]
pub fn erb_bandwidth(hz: Sample) -> Sample {
    let f = hz.max(0.0);
    // ERB(f) = 24.7 * (4.37 * f/1000 + 1)
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

/// Returns the ERB-rate number (ERBS) for a frequency (Glasberg & Moore 1990).
///
/// This is the integral of `1/ERB(f)`, a warped axis on which auditory filters
/// are roughly uniformly spaced. Negative inputs are clamped to `0 Hz`.
#[must_use]
pub fn hz_to_erb_number(hz: Sample) -> Sample {
    let f = hz.max(0.0);
    // ERBS = 21.4 * log10(4.37 * f/1000 + 1)
    21.4 * ops::log10(4.37 * f / 1000.0 + 1.0)
}

/// A contiguous partition of a frequency range into critical bands.
///
/// Band `i` covers `[edges[i], edges[i + 1])`; there are `edges.len() - 1`
/// bands. Edges are uniformly spaced on the chosen warped scale, so bands are
/// narrow at low frequencies and wide at high ones, matching the ear.
#[derive(Debug, Clone)]
pub struct CriticalBands {
    /// Band edge frequencies in hertz, strictly increasing, length
    /// `band_count + 1`.
    edges: Vec<Sample>,
}

/// Which warped scale a [`CriticalBands`] partition is built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BandScale {
    /// Zwicker Bark scale.
    Bark,
    /// Glasberg & Moore ERB-rate scale.
    Erb,
}

impl CriticalBands {
    /// Builds a partition of `[low_hz, high_hz]` into `band_count` bands,
    /// uniformly spaced on `scale`.
    ///
    /// Returns `None` if `band_count` is zero or the range is non-finite or
    /// non-increasing, so callers always get a usable partition or nothing.
    #[must_use]
    pub fn new(scale: BandScale, low_hz: Sample, high_hz: Sample, band_count: usize) -> Option<Self> {
        if band_count == 0
            || !low_hz.is_finite()
            || !high_hz.is_finite()
            || low_hz < 0.0
            || high_hz <= low_hz
        {
            return None;
        }
        let (low_w, high_w) = match scale {
            BandScale::Bark => (hz_to_bark(low_hz), hz_to_bark(high_hz)),
            BandScale::Erb => (hz_to_erb_number(low_hz), hz_to_erb_number(high_hz)),
        };
        let mut edges = Vec::with_capacity(band_count + 1);
        for i in 0..=band_count {
            let t = i as Sample / band_count as Sample;
            let warped = low_w + (high_w - low_w) * t;
            let hz = match scale {
                BandScale::Bark => bark_to_hz(warped),
                BandScale::Erb => erb_number_to_hz(warped),
            };
            edges.push(hz);
        }
        // Guard strict monotonicity against rounding at the very bottom.
        for i in 1..edges.len() {
            if edges[i] <= edges[i - 1] {
                edges[i] = edges[i - 1] + Sample::MIN_POSITIVE;
            }
        }
        Some(Self { edges })
    }

    /// A convenient default: 24 Bark-spaced bands from 20 Hz to 20 kHz, roughly
    /// the 24 classical Bark bands over the audible range.
    #[must_use]
    pub fn bark_default() -> Self {
        // These arguments are always valid, so the unwrap cannot fail; we
        // avoid `unwrap` to keep the panic-free contract explicit.
        match Self::new(BandScale::Bark, 20.0, 20_000.0, 24) {
            Some(bands) => bands,
            None => Self { edges: alloc_fallback_edges() },
        }
    }

    /// Returns the number of bands.
    #[must_use]
    #[inline]
    pub fn band_count(&self) -> usize {
        self.edges.len() - 1
    }

    /// Returns the band index a frequency falls into, clamped to the valid
    /// range so out-of-range frequencies map to the edge bands.
    #[must_use]
    pub fn band_of(&self, hz: Sample) -> usize {
        let count = self.band_count();
        if !hz.is_finite() || hz <= self.edges[0] {
            return 0;
        }
        if hz >= self.edges[count] {
            return count - 1;
        }
        // Linear scan is fine: band counts are tiny (tens) and this runs off
        // the hot path.
        let mut idx = 0;
        while idx + 1 < self.edges.len() && hz >= self.edges[idx + 1] {
            idx += 1;
        }
        idx.min(count - 1)
    }

    /// Returns the geometric centre frequency of band `i`, clamped to a valid
    /// index.
    #[must_use]
    pub fn band_center(&self, i: usize) -> Sample {
        let count = self.band_count();
        let idx = i.min(count - 1);
        0.5 * (self.edges[idx] + self.edges[idx + 1])
    }

    /// Returns the lower and upper edge frequencies of band `i`, clamped to a
    /// valid index.
    #[must_use]
    pub fn band_edges(&self, i: usize) -> (Sample, Sample) {
        let count = self.band_count();
        let idx = i.min(count - 1);
        (self.edges[idx], self.edges[idx + 1])
    }
}

/// Inverts [`hz_to_erb_number`] in closed form.
#[inline]
fn erb_number_to_hz(erbs: Sample) -> Sample {
    // f = 1000/4.37 * (10^(erbs/21.4) - 1)
    let p = ops::powf(10.0, erbs / 21.4) - 1.0;
    (1000.0 / 4.37) * p.max(0.0)
}

/// Fallback edge table (never reached in practice) that keeps
/// [`CriticalBands::bark_default`] total and panic-free.
#[inline]
fn alloc_fallback_edges() -> Vec<Sample> {
    let mut edges = Vec::with_capacity(25);
    for i in 0..=24 {
        edges.push(20.0 + i as Sample * ((20_000.0 - 20.0) / 24.0));
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-3;

    #[test]
    fn bark_is_monotonic() {
        let mut prev = hz_to_bark(0.0);
        for k in 1..=200 {
            let f = k as Sample * 100.0;
            let z = hz_to_bark(f);
            assert!(z > prev - 1e-6, "f={f} z={z} prev={prev}");
            prev = z;
        }
    }

    #[test]
    fn bark_round_trip() {
        for &f in &[100.0, 500.0, 1000.0, 4000.0, 10_000.0] {
            let back = bark_to_hz(hz_to_bark(f));
            assert!((back - f).abs() < 1.0, "f={f} back={back}");
        }
    }

    #[test]
    fn erb_widens_with_frequency() {
        assert!(erb_bandwidth(500.0) < erb_bandwidth(5000.0));
        assert!(hz_to_erb_number(500.0) < hz_to_erb_number(5000.0));
    }

    #[test]
    fn erb_round_trip() {
        for &f in &[100.0, 1000.0, 8000.0] {
            let back = erb_number_to_hz(hz_to_erb_number(f));
            assert!((back - f).abs() < 1.0, "f={f} back={back}");
        }
    }

    #[test]
    fn partition_is_strictly_increasing() {
        let bands = CriticalBands::bark_default();
        assert_eq!(bands.band_count(), 24);
        for i in 0..bands.band_count() {
            let (lo, hi) = bands.band_edges(i);
            assert!(hi > lo, "band {i}: lo={lo} hi={hi}");
        }
    }

    #[test]
    fn band_of_clamps_out_of_range() {
        let bands = CriticalBands::bark_default();
        assert_eq!(bands.band_of(-100.0), 0);
        assert_eq!(bands.band_of(0.0), 0);
        assert_eq!(bands.band_of(1.0e9), bands.band_count() - 1);
        assert_eq!(bands.band_of(Sample::NAN), 0);
    }

    #[test]
    fn band_of_increases_with_frequency() {
        let bands = CriticalBands::bark_default();
        let low = bands.band_of(100.0);
        let mid = bands.band_of(2000.0);
        let high = bands.band_of(12_000.0);
        assert!(low <= mid);
        assert!(mid <= high);
        assert!(low < high);
    }

    #[test]
    fn band_center_within_edges() {
        let bands = CriticalBands::bark_default();
        for i in 0..bands.band_count() {
            let (lo, hi) = bands.band_edges(i);
            let c = bands.band_center(i);
            assert!(c > lo - EPS && c < hi + EPS);
        }
    }

    #[test]
    fn erb_partition_builds() {
        let bands = CriticalBands::new(BandScale::Erb, 50.0, 16_000.0, 16).unwrap();
        assert_eq!(bands.band_count(), 16);
    }

    #[test]
    fn invalid_partitions_rejected() {
        assert!(CriticalBands::new(BandScale::Bark, 20.0, 20_000.0, 0).is_none());
        assert!(CriticalBands::new(BandScale::Bark, 100.0, 50.0, 8).is_none());
        assert!(CriticalBands::new(BandScale::Bark, Sample::NAN, 1000.0, 8).is_none());
    }
}
