//! Channel-bed layouts and their speaker geometry.
//!
//! A *bed* is the fixed channel foundation of an Atmos-style mix: a set of
//! loudspeakers at known directions (plus an optional non-directional LFE).
//! Dynamic objects are panned on top of this bed, or folded into it when a
//! playback device cannot render objects natively. This module enumerates the
//! supported layouts (2.0, 5.1.4, 7.1.4) and resolves each speaker to a unit
//! direction in the listener-local frame.
//!
//! # Coordinate convention
//!
//! Directions use the engine/Bevy convention: `-Z` forward, `+X` right, `+Y`
//! up. A speaker given as an azimuth `az` (0 straight ahead, positive toward
//! the right) and an elevation `el` (positive upward) maps to
//! `(cos(el) * sin(az), sin(el), -cos(el) * cos(az))`.
//!
//! # Determinism
//!
//! All trigonometry routes through [`bevy_math::ops`] (libm-backed) so the
//! resolved directions are bit-reproducible across targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports design section 44.2 (the channel bed of the bed-plus-objects
//! model). The speaker azimuths follow common ITU / immersive production
//! placements. Consumed by [`crate::pan`] (VBAP targets) and [`crate::fold`]
//! (downmix destinations).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

use prism_audio_core::math::Sample;

/// A single speaker of a [`BedLayout`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BedChannel {
    /// Short human-readable speaker label (for example `"FL"`, `"TRR"`).
    pub label: &'static str,
    /// Unit direction toward the speaker in the listener-local frame.
    pub direction: Vec3,
    /// Whether this is the low-frequency-effects channel (no direction, never
    /// a VBAP or Ambisonic target).
    pub is_lfe: bool,
}

/// A fixed loudspeaker arrangement that forms the bed of a mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BedLayout {
    /// Stereo 2.0: front left and front right only.
    Stereo,
    /// 5.1.4: a 5.1 base (L, R, C, LFE, side surrounds) plus four height
    /// speakers.
    Surround5_1_4,
    /// 7.1.4: a 7.1 base (L, R, C, LFE, side and rear surrounds) plus four
    /// height speakers.
    Surround7_1_4,
}

/// One entry of a static layout table: `(label, azimuth_deg, elevation_deg,
/// is_lfe)`.
type SpeakerSpec = (&'static str, Sample, Sample, bool);

/// Stereo 2.0 placement (L/R at +/-30 degrees).
const STEREO: &[SpeakerSpec] = &[("L", -30.0, 0.0, false), ("R", 30.0, 0.0, false)];

/// 5.1.4 placement: 5.1 base with side surrounds at +/-110 degrees plus four
/// 45-degree-elevated height speakers.
const SURROUND_5_1_4: &[SpeakerSpec] = &[
    ("FL", -30.0, 0.0, false),
    ("FR", 30.0, 0.0, false),
    ("C", 0.0, 0.0, false),
    ("LFE", 0.0, 0.0, true),
    ("SL", -110.0, 0.0, false),
    ("SR", 110.0, 0.0, false),
    ("TFL", -45.0, 45.0, false),
    ("TFR", 45.0, 45.0, false),
    ("TRL", -135.0, 45.0, false),
    ("TRR", 135.0, 45.0, false),
];

/// 7.1.4 placement: 7.1 base with side surrounds at +/-90 and rear surrounds
/// at +/-150 degrees plus four 45-degree-elevated height speakers.
const SURROUND_7_1_4: &[SpeakerSpec] = &[
    ("FL", -30.0, 0.0, false),
    ("FR", 30.0, 0.0, false),
    ("C", 0.0, 0.0, false),
    ("LFE", 0.0, 0.0, true),
    ("SL", -90.0, 0.0, false),
    ("SR", 90.0, 0.0, false),
    ("RL", -150.0, 0.0, false),
    ("RR", 150.0, 0.0, false),
    ("TFL", -45.0, 45.0, false),
    ("TFR", 45.0, 45.0, false),
    ("TRL", -135.0, 45.0, false),
    ("TRR", 135.0, 45.0, false),
];

/// Converts whole-degree azimuth/elevation into a unit listener-local
/// direction using only [`bevy_math::ops`] trigonometry.
#[must_use]
pub fn direction_from_angles(azimuth_deg: Sample, elevation_deg: Sample) -> Vec3 {
    let az = azimuth_deg * (PI / 180.0);
    let el = elevation_deg * (PI / 180.0);
    let (sin_az, cos_az) = ops::sin_cos(az);
    let (sin_el, cos_el) = ops::sin_cos(el);
    Vec3::new(cos_el * sin_az, sin_el, -cos_el * cos_az)
}

impl BedLayout {
    /// Returns the raw speaker specification table for this layout.
    fn spec(self) -> &'static [SpeakerSpec] {
        match self {
            BedLayout::Stereo => STEREO,
            BedLayout::Surround5_1_4 => SURROUND_5_1_4,
            BedLayout::Surround7_1_4 => SURROUND_7_1_4,
        }
    }

    /// Returns the total number of channels (including any LFE).
    #[must_use]
    pub fn channel_count(self) -> usize {
        self.spec().len()
    }

    /// Returns the number of directional (non-LFE) channels.
    #[must_use]
    pub fn directional_count(self) -> usize {
        self.spec().iter().filter(|s| !s.3).count()
    }

    /// Resolves every channel of this layout into a [`BedChannel`] with its
    /// unit direction, in canonical channel order.
    #[must_use]
    pub fn channels(self) -> Vec<BedChannel> {
        self.spec()
            .iter()
            .map(|&(label, az, el, is_lfe)| BedChannel {
                label,
                direction: if is_lfe {
                    Vec3::ZERO
                } else {
                    direction_from_angles(az, el)
                },
                is_lfe,
            })
            .collect()
    }

    /// Returns the canonical channel index of the LFE, if this layout has one.
    #[must_use]
    pub fn lfe_index(self) -> Option<usize> {
        self.spec().iter().position(|s| s.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_1_SQRT_2;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn channel_counts_match_layout_names() {
        assert_eq!(BedLayout::Stereo.channel_count(), 2);
        assert_eq!(BedLayout::Surround5_1_4.channel_count(), 10);
        assert_eq!(BedLayout::Surround7_1_4.channel_count(), 12);
    }

    #[test]
    fn directional_count_excludes_lfe() {
        assert_eq!(BedLayout::Surround5_1_4.directional_count(), 9);
        assert_eq!(BedLayout::Surround7_1_4.directional_count(), 11);
        assert_eq!(BedLayout::Stereo.directional_count(), 2);
    }

    #[test]
    fn front_center_points_forward() {
        let dir = direction_from_angles(0.0, 0.0);
        assert!(close(dir.x, 0.0));
        assert!(close(dir.y, 0.0));
        assert!(close(dir.z, -1.0));
    }

    #[test]
    fn right_speaker_has_positive_x() {
        let dir = direction_from_angles(90.0, 0.0);
        assert!(close(dir.x, 1.0));
        assert!(close(dir.y, 0.0));
        assert!(close(dir.z, 0.0));
    }

    #[test]
    fn height_speaker_elevation_is_positive() {
        let dir = direction_from_angles(-45.0, 45.0);
        assert!(dir.y > 0.0);
        // Unit length preserved.
        let len = ops::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        assert!(close(len, 1.0));
        // Elevation of 45 degrees => y = sin(45) = 1/sqrt(2).
        assert!(close(dir.y, FRAC_1_SQRT_2));
    }

    #[test]
    fn lfe_is_zero_direction_and_indexed() {
        let channels = BedLayout::Surround7_1_4.channels();
        let lfe = BedLayout::Surround7_1_4.lfe_index().unwrap();
        assert!(channels[lfe].is_lfe);
        assert_eq!(channels[lfe].direction, Vec3::ZERO);
        assert!(BedLayout::Stereo.lfe_index().is_none());
    }

    #[test]
    fn all_directional_speakers_are_unit_length() {
        for layout in [
            BedLayout::Stereo,
            BedLayout::Surround5_1_4,
            BedLayout::Surround7_1_4,
        ] {
            for ch in layout.channels() {
                if ch.is_lfe {
                    continue;
                }
                let d = ch.direction;
                let len = ops::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
                assert!(close(len, 1.0));
            }
        }
    }
}
