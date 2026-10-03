//! Delay-and-sum beam steering for a loudspeaker aperture (design
//! section 49.3).
//!
//! A beamformer drives the array as a steered aperture: every element gets a
//! time delay that aligns its contribution along one steering direction, plus
//! an amplitude taper across the aperture that trades main-lobe width for lower
//! side lobes. Projecting a steered beam at a wall or ceiling turns a reflected
//! wavefront into a virtual surround image. This module precomputes the
//! per-element delay and weight taps and exposes the narrowband array factor so
//! the resulting beam pattern can be inspected and verified.
//!
//! The taper is a raised-cosine (Hamming) window over element index, which
//! keeps a small non-zero pedestal at the aperture edges. Delays are returned
//! relative to the earliest element (minimum delay is zero).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only textbook delay-and-sum beamforming and the raised-cosine
//! (Hamming) aperture taper.
//!
//! # Relationship
//!
//! Consumes [`crate::array::layout::ArrayLayout`] speaker positions and reuses
//! `prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS` for the propagation
//! speed. Routed from [`crate::array::decode`] when the array is suited to
//! aperture steering.

use alloc::vec::Vec;
use core::f32::consts::PI;

use bevy_math::{ops, Vec3};
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;

use crate::array::layout::ArrayLayout;

/// One element tap: a relative delay and an aperture weight.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BeamTap {
    /// Layout slot this tap drives.
    pub slot: usize,
    /// Delay relative to the earliest element, in seconds (never negative).
    pub delay_seconds: Sample,
    /// Aperture-taper weight (never negative).
    pub weight: Sample,
}

impl BeamTap {
    /// The delay in whole-and-fractional samples for `sample_rate` hertz.
    #[must_use]
    pub fn delay_samples(&self, sample_rate: Sample) -> Sample {
        self.delay_seconds * sample_rate
    }
}

/// A delay-and-sum beamformer steered toward one direction.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::array::layout::{ArrayLayout, ArraySpeaker};
/// use prism_audio_immersive::array::beamforming::Beamformer;
///
/// let mut layout = ArrayLayout::new();
/// for i in 0..9 {
///     let x = (i as f32 - 4.0) * 0.2;
///     layout.push(ArraySpeaker::at(i, Vec3::new(x, 0.0, -2.0), "S"));
/// }
/// let steer = Vec3::new(0.3, 0.0, -1.0).normalize();
/// let beam = Beamformer::new(&layout, steer);
/// // The array factor peaks (value one) along the steering direction.
/// assert!((beam.response(steer, 2_000.0) - 1.0).abs() < 1.0e-4);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Beamformer {
    steer: Vec3,
    speed: Sample,
    elements: Vec<ElementTap>,
    weight_sum: Sample,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ElementTap {
    slot: usize,
    position: Vec3,
    weight: Sample,
    delay_seconds: Sample,
}

impl Beamformer {
    /// Builds a beamformer for the directional speakers of `layout`, steering
    /// the main lobe toward `steer_direction`, using the default speed of
    /// sound.
    #[must_use]
    pub fn new(layout: &ArrayLayout, steer_direction: Vec3) -> Self {
        Self::with_speed(layout, steer_direction, SPEED_OF_SOUND_MPS)
    }

    /// Builds a beamformer with an explicit `speed` of sound in metres per
    /// second.
    #[must_use]
    pub fn with_speed(layout: &ArrayLayout, steer_direction: Vec3, speed: Sample) -> Self {
        let steer = unit_or_forward(steer_direction);
        let speed = if speed > 1.0e-6 {
            speed
        } else {
            SPEED_OF_SOUND_MPS
        };
        let slots: Vec<(usize, Vec3)> = layout
            .speakers
            .iter()
            .enumerate()
            .filter(|(_, speaker)| !speaker.is_lfe)
            .map(|(slot, speaker)| (slot, speaker.position))
            .collect();
        let count = slots.len();
        let mut elements = Vec::with_capacity(count);
        let mut weight_sum = 0.0 as Sample;
        let mut min_delay = Sample::INFINITY;
        for (index, (slot, position)) in slots.iter().enumerate() {
            let weight = hamming_weight(index, count);
            weight_sum += weight;
            let delay = position.dot(steer) / speed;
            if delay < min_delay {
                min_delay = delay;
            }
            elements.push(ElementTap {
                slot: *slot,
                position: *position,
                weight,
                delay_seconds: delay,
            });
        }
        if !min_delay.is_finite() {
            min_delay = 0.0;
        }
        for element in &mut elements {
            element.delay_seconds = (element.delay_seconds - min_delay).max(0.0);
        }
        Self {
            steer,
            speed,
            elements,
            weight_sum,
        }
    }

    /// The number of driven elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether the beamformer has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// The steering direction.
    #[must_use]
    pub fn steer_direction(&self) -> Vec3 {
        self.steer
    }

    /// The per-element driving taps.
    #[must_use]
    pub fn taps(&self) -> Vec<BeamTap> {
        self.elements
            .iter()
            .map(|element| BeamTap {
                slot: element.slot,
                delay_seconds: element.delay_seconds,
                weight: element.weight,
            })
            .collect()
    }

    /// The normalised narrowband array-factor magnitude for a plane wave from
    /// `direction` at `frequency_hz`.
    ///
    /// The response is one at the steering direction and decays away from it;
    /// it is the magnitude of the weighted sum of element phases divided by the
    /// total aperture weight.
    #[must_use]
    pub fn response(&self, direction: Vec3, frequency_hz: Sample) -> Sample {
        if self.elements.is_empty() || self.weight_sum <= 1.0e-9 {
            return 0.0;
        }
        let probe = unit_or_forward(direction);
        let offset = probe - self.steer;
        let wave_number = 2.0 * PI * frequency_hz / self.speed;
        let mut real = 0.0 as Sample;
        let mut imag = 0.0 as Sample;
        for element in &self.elements {
            let phase = wave_number * element.position.dot(offset);
            real += element.weight * ops::cos(phase);
            imag += element.weight * ops::sin(phase);
        }
        ops::sqrt(real * real + imag * imag) / self.weight_sum
    }
}

/// The raised-cosine (Hamming) aperture weight for element `index` of `count`.
fn hamming_weight(index: usize, count: usize) -> Sample {
    if count <= 1 {
        return 1.0;
    }
    let ratio = index as Sample / (count as Sample - 1.0);
    0.54 - 0.46 * ops::cos(2.0 * PI * ratio)
}

/// Returns the normalised vector, or `-Z` (forward) if it is degenerate.
fn unit_or_forward(vector: Vec3) -> Vec3 {
    let normalized = vector.normalize_or_zero();
    if normalized == Vec3::ZERO {
        Vec3::new(0.0, 0.0, -1.0)
    } else {
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn line_array() -> ArrayLayout {
        let mut layout = ArrayLayout::new();
        for i in 0..9 {
            let x = (i as Sample - 4.0) * 0.2;
            layout.push(crate::array::layout::ArraySpeaker::at(
                i,
                Vec3::new(x, 0.0, -2.0),
                "S",
            ));
        }
        layout
    }

    #[test]
    fn response_peaks_at_steer_direction() {
        let steer = Vec3::new(0.3, 0.0, -1.0).normalize();
        let beam = Beamformer::new(&line_array(), steer);
        let peak = beam.response(steer, 2_000.0);
        assert!(close(peak, 1.0));
        let off = beam.response(Vec3::new(-0.6, 0.0, -1.0).normalize(), 2_000.0);
        assert!(off < peak);
    }

    #[test]
    fn delay_minimum_is_zero() {
        let beam = Beamformer::new(&line_array(), Vec3::new(0.5, 0.0, -1.0));
        let min = beam
            .taps()
            .iter()
            .map(|tap| tap.delay_seconds)
            .fold(Sample::INFINITY, Sample::min);
        assert!(close(min, 0.0));
    }

    #[test]
    fn taper_edges_are_below_center() {
        let beam = Beamformer::new(&line_array(), Vec3::new(0.0, 0.0, -1.0));
        let taps = beam.taps();
        let center = taps[taps.len() / 2].weight;
        assert!(taps[0].weight < center);
        assert!(taps[taps.len() - 1].weight < center);
    }

    #[test]
    fn tap_count_matches_directional_speakers() {
        let layout = line_array();
        let beam = Beamformer::new(&layout, Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(beam.len(), layout.directional_count());
    }

    #[test]
    fn empty_layout_has_zero_response() {
        let beam = Beamformer::new(&ArrayLayout::new(), Vec3::new(0.0, 0.0, -1.0));
        assert!(beam.is_empty());
        assert!(close(beam.response(Vec3::new(0.0, 0.0, -1.0), 1_000.0), 0.0));
    }
}
