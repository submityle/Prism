//! Dual-band, energy-optimised Ambisonic decoding (max-rE weighting).
//!
//! Where [`crate::hoa::decode_hoa`] performs the plain "basic"/projection
//! decode, this module adds the perceptually motivated **dual-band** decode
//! used by modern scene-based renderers: the low band keeps the phase-matched
//! *basic* decode (accurate interaural time / low-frequency localisation) while
//! the high band applies **max-rE** weighting (maximising the energy-vector
//! length for stable, well-externalised imaging and even timbre off the sweet
//! spot). Both bands share [`crate::hoa`]'s `SN3D`/`ACN` convention and axis
//! definitions.
//!
//! This module deliberately only produces the two bands' *decode gains*; the
//! actual band split is a time-domain Linkwitz-Riley crossover owned by the
//! caller (built from the core `Biquad`). Keeping the crossover out of here
//! keeps the module single-purpose.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**. The max-rE weighting and the dual-band (basic + max-rE) decode are
//! publicly documented Ambisonic decoding practice: see J. Daniel,
//! "Representation de champs acoustiques" (2001) for the energy-vector
//! criterion and F. Zotter and M. Frank, "All-Round Ambisonic Panning and
//! Decoding" (`AllRAD`, JAES 2012) for the max-rE gains and dual-band strategy.
//! Everything here is implemented from that public literature.
//!
//! # Coordinate / axis convention
//!
//! Identical to [`crate::hoa`]: directions are listener-local and match Bevy
//! (`-Z` forward, `+X` right, `+Y` up); acoustic axes are `front = -z`,
//! `left = -x`, `up = +y`.
//!
//! # Real-time contract
//!
//! Every decode entry point is **allocation free, lock free, and panic free**:
//! per-degree gains are precomputed once in [`DualBandDecoder::new`] (control
//! rate) and the hot path is a bounded dot product over fixed-size stack
//! arrays. Degenerate inputs (empty layout, short buffers, zero
//! normalisation) are clamped, never panicked.
//!
//! # Determinism
//!
//! All transcendental / length math routes through [`bevy_math::ops`]
//! (libm-backed) rather than `f32` intrinsics, so decoding is bit-reproducible
//! across targets and can be golden-compared sample-for-sample. This is
//! enforced by the workspace lints.

use bevy_math::{Vec3, ops};

use prism_audio_core::math::Sample;

use crate::hoa::{MAX_HOA_CHANNELS, MAX_HOA_ORDER, acn_index, encode_hoa, hoa_channel_count};

/// The number of per-order max-rE weights, `MAX_HOA_ORDER + 1`.
pub const MAX_ORDER_WEIGHTS: usize = MAX_HOA_ORDER + 1;

/// Maximum number of loudspeaker directions a [`SpeakerLayout`] can hold. Sized
/// for large studio rigs (7.1.4 immersive beds and small domes) while staying
/// stack allocatable.
pub const MAX_DECODE_SPEAKERS: usize = 32;

/// The empirical central-angle constant (degrees) in the closed-form 3D max-rE
/// radius `r_E ~= cos(137.9 deg / (order + 1.51))` (Zotter and Frank, `AllRAD`).
const MAX_RE_ANGLE_DEG: Sample = 137.9;

/// The `1.51` offset in the closed-form max-rE radius approximation.
const MAX_RE_ORDER_OFFSET: Sample = 1.51;

/// Degrees-to-radians factor.
const DEG_TO_RAD: Sample = core::f32::consts::PI / 180.0;

/// Below this magnitude a normalisation denominator is treated as degenerate
/// and replaced by `1.0` to avoid a divide-by-zero.
const NORM_EPSILON: Sample = 1.0e-9;

/// Which decode band's gains to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DecodeBand {
    /// Phase-matched basic/projection decode (low frequencies).
    Low,
    /// Energy-optimised max-rE decode (high frequencies).
    High,
}

/// Returns the closed-form 3D max-rE energy-vector radius for `order`,
/// `cos(137.9 deg / (order + 1.51))`.
///
/// This is the direction-independent scalar that the per-degree weights
/// [`max_re_gains`] are evaluated at. `order` is clamped to
/// [`MAX_HOA_ORDER`].
#[inline]
#[must_use]
pub fn max_re_radius(order: usize) -> Sample {
    let order = order.min(MAX_HOA_ORDER);
    let denom = order as Sample + MAX_RE_ORDER_OFFSET;
    let angle_rad = (MAX_RE_ANGLE_DEG / denom) * DEG_TO_RAD;
    ops::cos(angle_rad)
}

/// Computes the per-degree **max-rE** weights `g_n = P_n(r_E)` for `order`,
/// normalised so that `g_0 = 1`.
///
/// `P_n` is the Legendre polynomial (evaluated by the standard three-term
/// recurrence) and `r_E` is [`max_re_radius`]. Weights for degrees beyond
/// `order` are `0`. `order` is clamped to [`MAX_HOA_ORDER`].
///
/// These weights, applied per Ambisonic degree, maximise the length of the
/// energy vector, concentrating reproduced energy toward the source direction.
#[must_use]
pub fn max_re_gains(order: usize) -> [Sample; MAX_ORDER_WEIGHTS] {
    let order = order.min(MAX_HOA_ORDER);
    let r_e = max_re_radius(order);

    let mut g = [0.0 as Sample; MAX_ORDER_WEIGHTS];
    g[0] = 1.0;
    if order >= 1 {
        g[1] = r_e;
    }
    // Legendre recurrence: (n + 1) P_{n+1} = (2n + 1) r P_n - n P_{n-1}.
    let mut n = 1usize;
    while n < order {
        let nn = n as Sample;
        g[n + 1] = ((2.0 * nn + 1.0) * r_e * g[n] - nn * g[n - 1]) / (nn + 1.0);
        n += 1;
    }

    // Normalise to g_0 = 1 (P_0 is already 1, but guard against drift).
    let g0 = g[0];
    if g0.abs() > NORM_EPSILON {
        for w in &mut g {
            *w /= g0;
        }
    }
    g
}

/// Expands per-degree max-rE weights into the high-band per-channel gain table
/// (`ACN` order): channel `acn_index(n, m)` receives `(2n + 1) * g[n]`.
///
/// The `(2n + 1)` degree multiplicity is required because [`crate::hoa`] uses an
/// `SN3D`-scaled convention whose per-degree self-energy is unity (its
/// projection decode sums `P_n` without the modal count). Re-injecting the
/// `(2n + 1)` term makes the projection decode reproduce the physical N3D max-rE
/// energy-vector optimisation, so the high band genuinely lengthens the energy
/// vector relative to the basic (low) band. Channels beyond `order` stay `0`.
fn per_channel_gains(order: usize, weights: &[Sample; MAX_ORDER_WEIGHTS]) -> [Sample; MAX_HOA_CHANNELS] {
    let order = order.min(MAX_HOA_ORDER);
    let mut gains = [0.0 as Sample; MAX_HOA_CHANNELS];
    for n in 0..=order {
        let n_isize = n as isize;
        let mut m = -n_isize;
        while m <= n_isize {
            let multiplicity = 2.0 * n as Sample + 1.0;
            gains[acn_index(n, m)] = multiplicity * weights[n];
            m += 1;
        }
    }
    gains
}

/// A fixed-capacity set of loudspeaker directions (unit vectors in the
/// listener-local frame), stored on the stack.
///
/// Holds up to [`MAX_DECODE_SPEAKERS`] directions; pushes beyond capacity are
/// ignored rather than panicking.
#[derive(Debug, Clone)]
pub struct SpeakerLayout {
    directions: [Vec3; MAX_DECODE_SPEAKERS],
    count: usize,
}

impl Default for SpeakerLayout {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl SpeakerLayout {
    /// Creates an empty layout.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { directions: [Vec3::ZERO; MAX_DECODE_SPEAKERS], count: 0 }
    }

    /// Builds a layout from a slice of directions, normalising each and keeping
    /// at most [`MAX_DECODE_SPEAKERS`] entries (extras are ignored). Zero-length
    /// directions are skipped.
    #[must_use]
    pub fn from_directions(directions: &[Vec3]) -> Self {
        let mut layout = Self::new();
        for &d in directions {
            layout.push(d);
        }
        layout
    }

    /// Appends a loudspeaker direction (normalised). Returns `true` if it was
    /// stored, `false` if the layout was full or the direction was degenerate.
    #[inline]
    pub fn push(&mut self, direction: Vec3) -> bool {
        if self.count >= MAX_DECODE_SPEAKERS {
            return false;
        }
        let unit = direction.normalize_or_zero();
        if unit == Vec3::ZERO {
            return false;
        }
        self.directions[self.count] = unit;
        self.count += 1;
        true
    }

    /// The number of stored loudspeakers.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether the layout has no loudspeakers.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The stored loudspeaker directions.
    #[inline]
    #[must_use]
    pub fn directions(&self) -> &[Vec3] {
        &self.directions[..self.count]
    }
}

/// A dual-band, energy-optimised Ambisonic decoder for a fixed order.
///
/// The low band is the phase-matched basic decode (equivalent to
/// [`crate::hoa::decode_hoa`]); the high band applies precomputed max-rE
/// per-degree weights. Build once (control rate), then decode any number of
/// frames on the audio thread with the allocation/lock/panic-free
/// [`decode_low`](Self::decode_low) / [`decode_high`](Self::decode_high) /
/// [`decode_to_speakers`](Self::decode_to_speakers).
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::hoa::{encode_hoa, MAX_HOA_CHANNELS};
/// use prism_audio_spatial::hoa_decode::DualBandDecoder;
///
/// let order = 3;
/// let source = Vec3::new(0.0, 0.0, -1.0); // straight ahead
/// let mut coeffs = [0.0f32; MAX_HOA_CHANNELS];
/// encode_hoa(source, order, &mut coeffs);
///
/// // The phase-matched low band returns unit gain at a coincident speaker.
/// let decoder = DualBandDecoder::new(order);
/// let here = decoder.decode_low(&coeffs, source);
/// assert!((here - 1.0).abs() < 1.0e-4);
/// ```
#[derive(Debug, Clone)]
pub struct DualBandDecoder {
    order: usize,
    /// Per-channel max-rE gains (ACN order) for the high band, including the
    /// (2n + 1) degree multiplicity that compensates the SN3D-scaled projection.
    high_channel_gains: [Sample; MAX_HOA_CHANNELS],
    /// High-band normalisation: sum of (2n + 1) * `g_n` over active degrees.
    high_norm: Sample,
    /// Low-band normalisation: the basic decode's `order + 1`.
    low_norm: Sample,
    /// The per-degree max-rE weights, exposed for inspection.
    weights: [Sample; MAX_ORDER_WEIGHTS],
}

impl DualBandDecoder {
    /// Builds a decoder for `order` (clamped to [`MAX_HOA_ORDER`]),
    /// precomputing the max-rE weights. **Control rate / non-real-time** but
    /// allocation, lock, and panic free.
    #[must_use]
    pub fn new(order: usize) -> Self {
        let order = order.min(MAX_HOA_ORDER);
        let weights = max_re_gains(order);
        let high_channel_gains = per_channel_gains(order, &weights);

        // High-band normalisation: sum of the per-channel degree weights so a
        // unit source still decodes to ~1 at a coincident speaker. Matches the
        // (2n + 1) * g_n multiplicity injected by per_channel_gains.
        let mut high_norm = 0.0 as Sample;
        for (n, &w) in weights.iter().take(order + 1).enumerate() {
            high_norm += (2.0 * n as Sample + 1.0) * w;
        }

        Self {
            order,
            high_channel_gains,
            high_norm,
            low_norm: (order + 1) as Sample,
            weights,
        }
    }

    /// The Ambisonic order this decoder targets.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// The precomputed per-degree max-rE weights (`g_0 = 1`).
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &[Sample; MAX_ORDER_WEIGHTS] {
        &self.weights
    }

    /// Decodes the field toward `speaker_direction` using the **low** (basic,
    /// phase-matched) band. Equivalent to [`crate::hoa::decode_hoa`].
    ///
    /// **Real-time**: allocation, lock, and panic free.
    #[inline]
    #[must_use]
    pub fn decode_low(&self, coeffs: &[Sample], speaker_direction: Vec3) -> Sample {
        band_dot(coeffs, speaker_direction, self.order, None, self.low_norm)
    }

    /// Decodes the field toward `speaker_direction` using the **high**
    /// (max-rE, energy-optimised) band.
    ///
    /// **Real-time**: allocation, lock, and panic free.
    #[inline]
    #[must_use]
    pub fn decode_high(&self, coeffs: &[Sample], speaker_direction: Vec3) -> Sample {
        band_dot(
            coeffs,
            speaker_direction,
            self.order,
            Some(&self.high_channel_gains),
            self.high_norm,
        )
    }

    /// Decodes toward `speaker_direction` using the selected `band`.
    #[inline]
    #[must_use]
    pub fn decode(&self, band: DecodeBand, coeffs: &[Sample], speaker_direction: Vec3) -> Sample {
        match band {
            DecodeBand::Low => self.decode_low(coeffs, speaker_direction),
            DecodeBand::High => self.decode_high(coeffs, speaker_direction),
        }
    }

    /// Decodes one Ambisonic frame `coeffs` to every loudspeaker in `layout`
    /// with the selected `band`, writing into `out`.
    ///
    /// Only `min(layout.len(), out.len())` speakers are written, so a short
    /// `out` slice never panics. **Real-time**: allocation, lock, and panic
    /// free.
    pub fn decode_to_speakers(
        &self,
        coeffs: &[Sample],
        layout: &SpeakerLayout,
        band: DecodeBand,
        out: &mut [Sample],
    ) {
        let dirs = layout.directions();
        let n = dirs.len().min(out.len());
        for i in 0..n {
            out[i] = self.decode(band, coeffs, dirs[i]);
        }
    }
}

/// The shared decode kernel: re-encode `speaker_direction`, dot with `coeffs`
/// (optionally scaled per channel by `channel_gains`), and divide by `norm`.
fn band_dot(
    coeffs: &[Sample],
    speaker_direction: Vec3,
    order: usize,
    channel_gains: Option<&[Sample; MAX_HOA_CHANNELS]>,
    norm: Sample,
) -> Sample {
    let order = order.min(MAX_HOA_ORDER);
    let count = hoa_channel_count(order);
    let mut enc = [0.0 as Sample; MAX_HOA_CHANNELS];
    encode_hoa(speaker_direction, order, &mut enc);

    let n = count.min(coeffs.len());
    let mut acc = 0.0 as Sample;
    match channel_gains {
        Some(gains) => {
            for k in 0..n {
                acc += coeffs[k] * enc[k] * gains[k];
            }
        }
        None => {
            for k in 0..n {
                acc += coeffs[k] * enc[k];
            }
        }
    }

    let denom = if norm.abs() > NORM_EPSILON { norm } else { 1.0 };
    acc / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hoa::encode_hoa;
    use core::f32::consts::PI;

    const EPS: Sample = 1.0e-4;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    fn horizontal_ring(count: usize) -> SpeakerLayout {
        let mut layout = SpeakerLayout::new();
        for i in 0..count {
            let az = 2.0 * PI * (i as Sample) / (count as Sample);
            // Bevy: front = -z, left = -x. Place on the horizontal plane.
            let (s, c) = ops::sin_cos(az);
            let dir = Vec3::new(-s, 0.0, -c);
            layout.push(dir);
        }
        layout
    }

    fn energy_vector_radius(decoder: &DualBandDecoder, coeffs: &[Sample], layout: &SpeakerLayout, band: DecodeBand) -> Sample {
        let dirs = layout.directions();
        let mut num = Vec3::ZERO;
        let mut den = 0.0 as Sample;
        for &d in dirs {
            let g = decoder.decode(band, coeffs, d);
            let e = g * g;
            num += d * e;
            den += e;
        }
        if den.abs() <= NORM_EPSILON {
            return 0.0;
        }
        ops::sqrt(num.dot(num)) / den
    }

    #[test]
    fn order_zero_both_bands_are_omnidirectional() {
        let decoder = DualBandDecoder::new(0);
        let coeffs = [1.0 as Sample; 1];
        // Any speaker direction decodes the omni component identically.
        for dir in [Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)] {
            assert!(approx(decoder.decode_low(&coeffs, dir), 1.0));
            assert!(approx(decoder.decode_high(&coeffs, dir), 1.0));
        }
    }

    #[test]
    fn basic_band_returns_unit_at_coincident_speaker() {
        for order in 1..=MAX_HOA_ORDER {
            let decoder = DualBandDecoder::new(order);
            let dir = Vec3::new(-0.3, 0.4, -0.866).normalize();
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(dir, order, &mut coeffs);
            assert!(approx(decoder.decode_low(&coeffs, dir), 1.0), "order {order}");
        }
    }

    #[test]
    fn high_band_returns_unit_at_coincident_speaker() {
        // The high-band normalisation (sum of weights) is chosen so a unit
        // source still decodes to ~1 at a coincident speaker.
        for order in 1..=MAX_HOA_ORDER {
            let decoder = DualBandDecoder::new(order);
            let dir = Vec3::new(0.5, -0.2, -0.84).normalize();
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(dir, order, &mut coeffs);
            assert!(approx(decoder.decode_high(&coeffs, dir), 1.0), "order {order}");
        }
    }

    #[test]
    fn max_re_gains_start_at_one_and_decrease() {
        for order in 1..=MAX_HOA_ORDER {
            let g = max_re_gains(order);
            assert!(approx(g[0], 1.0), "order {order}: g0 = {}", g[0]);
            for n in 1..=order {
                assert!(g[n] < g[n - 1], "order {order}: g[{n}] not below g[{}]", n - 1);
                assert!(g[n] > 0.0, "order {order}: g[{n}] should stay positive");
            }
            // Degrees beyond the order are untouched (zero).
            for n in (order + 1)..MAX_ORDER_WEIGHTS {
                assert!(approx(g[n], 0.0));
            }
        }
    }

    #[test]
    fn max_re_radius_increases_with_order() {
        let mut prev = max_re_radius(1);
        for order in 2..=MAX_HOA_ORDER {
            let r = max_re_radius(order);
            assert!(r > prev, "order {order}: r_E did not increase ({r} <= {prev})");
            prev = r;
        }
    }

    #[test]
    fn high_band_lengthens_the_energy_vector() {
        // A ring of speakers reproduces a source; the max-rE (high) band should
        // concentrate energy more tightly -> longer energy vector than basic.
        let layout = horizontal_ring(8);
        let source = Vec3::new(0.0, 0.0, -1.0);
        for order in 1..=MAX_HOA_ORDER {
            let decoder = DualBandDecoder::new(order);
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(source, order, &mut coeffs);
            let r_low = energy_vector_radius(&decoder, &coeffs, &layout, DecodeBand::Low);
            let r_high = energy_vector_radius(&decoder, &coeffs, &layout, DecodeBand::High);
            assert!(
                r_high > r_low,
                "order {order}: high-band r_E {r_high} not above low-band {r_low}"
            );
        }
    }

    #[test]
    fn decode_to_speakers_matches_scalar_decode() {
        let layout = horizontal_ring(6);
        let decoder = DualBandDecoder::new(2);
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(Vec3::new(0.2, 0.1, -0.97).normalize(), 2, &mut coeffs);

        let mut out = [0.0 as Sample; 6];
        decoder.decode_to_speakers(&coeffs, &layout, DecodeBand::High, &mut out);
        for (i, &d) in layout.directions().iter().enumerate() {
            assert!(approx(out[i], decoder.decode_high(&coeffs, d)));
        }
    }

    #[test]
    fn speaker_layout_ignores_overflow_and_degenerate() {
        let mut layout = SpeakerLayout::new();
        for _ in 0..(MAX_DECODE_SPEAKERS + 5) {
            layout.push(Vec3::new(0.0, 0.0, -1.0));
        }
        assert_eq!(layout.len(), MAX_DECODE_SPEAKERS);
        // A zero direction is rejected without panic.
        assert!(!layout.push(Vec3::ZERO));
    }

    #[test]
    fn decode_to_speakers_does_not_panic_on_short_output() {
        let layout = horizontal_ring(8);
        let decoder = DualBandDecoder::new(3);
        let coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        // Output shorter than the layout: only the leading speakers are filled.
        let mut out = [0.0 as Sample; 3];
        decoder.decode_to_speakers(&coeffs, &layout, DecodeBand::Low, &mut out);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn scalar_decode_does_not_panic_on_short_coeffs() {
        let decoder = DualBandDecoder::new(3);
        // Only two channels supplied for a nominal third-order decode.
        let coeffs = [1.0 as Sample; 2];
        let g_low = decoder.decode_low(&coeffs, Vec3::new(0.0, 0.0, -1.0));
        let g_high = decoder.decode_high(&coeffs, Vec3::new(0.0, 0.0, -1.0));
        assert!(g_low.is_finite());
        assert!(g_high.is_finite());
    }

    #[test]
    fn order_is_clamped_to_max() {
        let decoder = DualBandDecoder::new(99);
        assert_eq!(decoder.order(), MAX_HOA_ORDER);
    }
}
