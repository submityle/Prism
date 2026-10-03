//! Wave-field-synthesis driving functions (design section 49.3).
//!
//! Wave-field synthesis reconstructs the sound field of a virtual source by
//! driving every physical loudspeaker (a secondary source) with a delayed and
//! weighted copy of the source signal, following Huygens' principle. This
//! module computes those per-speaker delay and gain taps for two virtual
//! source kinds:
//!
//! * a point source at a position, whose wavefront is spherical, giving a
//!   distance delay and a 1/sqrt(distance) amplitude with an obliquity factor;
//! * a plane wave travelling in a direction, giving a projection delay and a
//!   constant amplitude with an obliquity factor.
//!
//! The obliquity (secondary-source directivity) factor is the clamped cosine
//! between the wavefront arrival direction and each speaker's outward normal,
//! so only speakers facing the incoming wave contribute. Delays are returned
//! relative to the earliest speaker (minimum delay is zero) and also expressed
//! in samples for a given sample rate. This is an asset-time / offline driving
//! computation; it allocates one tap list per call.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the published wave-field-synthesis driving functions
//! (Berkhout 1988) under Huygens' principle.
//!
//! # Relationship
//!
//! Consumes [`crate::array::layout::ArrayLayout`] speaker positions and
//! reuses `prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS` for the
//! propagation speed. Routed from [`crate::array::decode`] when the array has
//! usable physical positions.

use alloc::vec::Vec;

use bevy_math::{ops, Vec3};
use prism_audio_core::math::Sample;
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;

use crate::array::layout::ArrayLayout;

/// The kind of virtual source a [`WfsDriver`] reconstructs.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum WfsSourceKind {
    /// A spherical-wave point source (position carried by [`WfsSource`]).
    PointSource,
    /// A plane wave travelling along the given unit direction of propagation.
    PlaneWave {
        /// Unit direction the wavefront travels toward.
        direction: Vec3,
    },
}

/// A virtual source to synthesise.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WfsSource {
    /// Source position in metres (used for [`WfsSourceKind::PointSource`]).
    pub position: Vec3,
    /// The source kind.
    pub kind: WfsSourceKind,
}

impl WfsSource {
    /// A point source at `position`.
    #[must_use]
    pub fn point(position: Vec3) -> Self {
        Self {
            position,
            kind: WfsSourceKind::PointSource,
        }
    }

    /// A plane wave travelling toward `direction`.
    #[must_use]
    pub fn plane_wave(direction: Vec3) -> Self {
        Self {
            position: Vec3::ZERO,
            kind: WfsSourceKind::PlaneWave {
                direction: unit_or_forward(direction),
            },
        }
    }
}

/// One loudspeaker driving tap: a relative delay and an amplitude gain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WfsTap {
    /// Layout slot this tap drives.
    pub slot: usize,
    /// Delay relative to the earliest speaker, in seconds (never negative).
    pub delay_seconds: Sample,
    /// Amplitude gain (never negative).
    pub gain: Sample,
}

impl WfsTap {
    /// The delay in whole-and-fractional samples for `sample_rate` hertz.
    #[must_use]
    pub fn delay_samples(&self, sample_rate: Sample) -> Sample {
        self.delay_seconds * sample_rate
    }
}

/// A reusable wave-field-synthesis driver bound to one array geometry.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::array::layout::{ArrayLayout, ArraySpeaker};
/// use prism_audio_immersive::array::wfs::{WfsDriver, WfsSource};
///
/// let mut layout = ArrayLayout::new();
/// for i in 0..5 {
///     let x = (i as f32 - 2.0) * 0.5;
///     layout.push(ArraySpeaker::at(i, Vec3::new(x, 0.0, -2.0), "S"));
/// }
/// let driver = WfsDriver::new(&layout);
/// let taps = driver.taps(WfsSource::point(Vec3::new(0.0, 0.0, -6.0)));
/// assert_eq!(taps.len(), 5);
/// let earliest = taps.iter().map(|t| t.delay_seconds).fold(f32::INFINITY, f32::min);
/// assert!(earliest.abs() < 1.0e-4);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WfsDriver {
    speakers: Vec<SpeakerGeometry>,
    speed: Sample,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpeakerGeometry {
    slot: usize,
    position: Vec3,
    normal: Vec3,
}

impl WfsDriver {
    /// Builds a driver for the directional speakers of `layout`, using the
    /// default speed of sound.
    #[must_use]
    pub fn new(layout: &ArrayLayout) -> Self {
        Self::with_speed(layout, SPEED_OF_SOUND_MPS)
    }

    /// Builds a driver for the directional speakers of `layout` with an
    /// explicit `speed` of sound in metres per second.
    #[must_use]
    pub fn with_speed(layout: &ArrayLayout, speed: Sample) -> Self {
        let mut speakers = Vec::new();
        for (slot, speaker) in layout.speakers.iter().enumerate() {
            if speaker.is_lfe {
                continue;
            }
            speakers.push(SpeakerGeometry {
                slot,
                position: speaker.position,
                normal: speaker.direction,
            });
        }
        let speed = if speed > 1.0e-6 {
            speed
        } else {
            SPEED_OF_SOUND_MPS
        };
        Self { speakers, speed }
    }

    /// The number of driven (directional) speakers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    /// Whether the driver has no speakers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.speakers.is_empty()
    }

    /// Computes the driving taps for `source`.
    ///
    /// Delays are shifted so the earliest contributing speaker has zero delay.
    #[must_use]
    pub fn taps(&self, source: WfsSource) -> Vec<WfsTap> {
        let mut raw = Vec::with_capacity(self.speakers.len());
        let mut min_delay = Sample::INFINITY;
        for speaker in &self.speakers {
            let (delay, gain) = self.tap_for(speaker, source);
            if delay < min_delay {
                min_delay = delay;
            }
            raw.push((speaker.slot, delay, gain));
        }
        if !min_delay.is_finite() {
            min_delay = 0.0;
        }
        let mut taps = Vec::with_capacity(raw.len());
        for (slot, delay, gain) in raw {
            taps.push(WfsTap {
                slot,
                delay_seconds: (delay - min_delay).max(0.0),
                gain,
            });
        }
        taps
    }

    fn tap_for(&self, speaker: &SpeakerGeometry, source: WfsSource) -> (Sample, Sample) {
        match source.kind {
            WfsSourceKind::PointSource => {
                let to_speaker = speaker.position - source.position;
                let distance = to_speaker.length();
                if distance <= 1.0e-6 {
                    return (0.0, 0.0);
                }
                let arrival = to_speaker / distance;
                let obliquity = arrival.dot(-speaker.normal).max(0.0);
                let delay = distance / self.speed;
                let gain = obliquity / ops::sqrt(distance);
                (delay, gain)
            }
            WfsSourceKind::PlaneWave { direction } => {
                let projection = speaker.position.dot(direction);
                let delay = projection / self.speed;
                let obliquity = (-direction).dot(speaker.normal).max(0.0);
                (delay, obliquity)
            }
        }
    }
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
    use crate::array::layout::ArrayLayout;

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn line_array() -> ArrayLayout {
        let mut layout = ArrayLayout::new();
        for i in 0..5 {
            let x = (i as Sample - 2.0) * 0.5;
            layout.push(crate::array::layout::ArraySpeaker::at(
                i,
                Vec3::new(x, 0.0, -2.0),
                "S",
            ));
        }
        layout
    }

    #[test]
    fn minimum_delay_is_zero() {
        let driver = WfsDriver::new(&line_array());
        let taps = driver.taps(WfsSource::point(Vec3::new(0.0, 0.0, -6.0)));
        let min = taps
            .iter()
            .map(|tap| tap.delay_seconds)
            .fold(Sample::INFINITY, Sample::min);
        assert!(close(min, 0.0));
    }

    #[test]
    fn point_source_gain_decreases_with_distance() {
        let driver = WfsDriver::new(&line_array());
        let near = driver.taps(WfsSource::point(Vec3::new(0.0, 0.0, -3.0)));
        let far = driver.taps(WfsSource::point(Vec3::new(0.0, 0.0, -30.0)));
        let near_center = near[2].gain;
        let far_center = far[2].gain;
        assert!(near_center > far_center);
    }

    #[test]
    fn plane_wave_delay_follows_projection() {
        let driver = WfsDriver::new(&line_array());
        let taps = driver.taps(WfsSource::plane_wave(Vec3::new(1.0, 0.0, 0.0)));
        assert!(taps[4].delay_seconds > taps[0].delay_seconds);
    }

    #[test]
    fn delay_samples_scales_with_rate() {
        let tap = WfsTap {
            slot: 0,
            delay_seconds: 0.01,
            gain: 1.0,
        };
        assert!(close(tap.delay_samples(48_000.0), 480.0));
    }

    #[test]
    fn empty_layout_yields_no_taps() {
        let driver = WfsDriver::new(&ArrayLayout::new());
        assert!(driver.is_empty());
        assert!(driver.taps(WfsSource::point(Vec3::ZERO)).is_empty());
    }
}
