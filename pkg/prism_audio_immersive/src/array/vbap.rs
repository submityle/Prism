//! Vector-base amplitude panning over an arbitrary array (design section 49.3).
//!
//! This is a thin, layout-aware wrapper around the shared constant-power VBAP
//! kernel in `prism_audio_object::pan::vbap`. The kernel pans a direction over
//! a flat list of speaker directions; this wrapper restricts panning to the
//! directional (non-LFE) speakers of an [`ArrayLayout`] and scatters the
//! resulting gains back into a full per-slot gain vector so low-frequency-
//! effects feeds stay silent and the output index matches the layout order.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the published VBAP algorithm (Pulkki 1997).
//!
//! # Relationship
//!
//! Wraps `prism_audio_object::pan::vbap::vbap_gains` and consumes
//! [`crate::array::layout::ArrayLayout`]. Used by [`crate::array::decode`] as
//! the object-panning path for physical arrays and bed fallbacks.

use alloc::vec::Vec;

use bevy_math::Vec3;
use prism_audio_core::math::Sample;
use prism_audio_object::pan::vbap::vbap_gains;

use crate::array::layout::ArrayLayout;

/// A reusable amplitude panner bound to one [`ArrayLayout`].
///
/// Construction precomputes the directional speaker directions and the map
/// from the panning subset back to full layout slots. [`VbapArrayPanner::gains`]
/// then performs one VBAP solve per call and returns a gain vector whose length
/// equals the layout's slot count.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::array::layout::{ArrayLayout, ArraySpeaker};
/// use prism_audio_immersive::array::vbap::VbapArrayPanner;
///
/// let mut layout = ArrayLayout::new();
/// layout.push(ArraySpeaker::from_direction(0, Vec3::new(-1.0, 0.0, -1.0), 2.0, "FL"));
/// layout.push(ArraySpeaker::from_direction(1, Vec3::new(1.0, 0.0, -1.0), 2.0, "FR"));
/// let panner = VbapArrayPanner::new(&layout);
/// let gains = panner.gains(Vec3::new(0.0, 0.0, -1.0));
/// assert_eq!(gains.len(), 2);
/// let power: f32 = gains.iter().map(|g| g * g).sum();
/// assert!((power - 1.0).abs() < 1.0e-4);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VbapArrayPanner {
    slot_count: usize,
    directional_slots: Vec<usize>,
    directions: Vec<Vec3>,
}

impl VbapArrayPanner {
    /// Builds a panner for `layout`.
    #[must_use]
    pub fn new(layout: &ArrayLayout) -> Self {
        Self {
            slot_count: layout.len(),
            directional_slots: layout.directional_indices(),
            directions: layout.directional_directions(),
        }
    }

    /// The number of layout slots (equal to the output gain-vector length).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// The number of directional (pannable) speakers.
    #[must_use]
    pub fn directional_count(&self) -> usize {
        self.directions.len()
    }

    /// Whether this panner has any directional speaker to pan over.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    /// Computes constant-power panning gains for `direction`, one per layout
    /// slot. Low-frequency-effects slots and any unused slot are zero.
    #[must_use]
    pub fn gains(&self, direction: Vec3) -> Vec<Sample> {
        let mut out = Vec::new();
        out.resize(self.slot_count, 0.0);
        let local = vbap_gains(direction, &self.directions);
        for (local_index, &slot) in self.directional_slots.iter().enumerate() {
            if let (Some(&gain), Some(cell)) = (local.get(local_index), out.get_mut(slot)) {
                *cell = gain;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use prism_audio_object::bed::{direction_from_angles, BedLayout};

    use crate::array::layout::ArrayLayout;

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn sum_sq(gains: &[Sample]) -> Sample {
        gains.iter().map(|&g| g * g).sum()
    }

    #[test]
    fn output_length_matches_slot_count() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround5_1_4);
        let panner = VbapArrayPanner::new(&layout);
        let gains = panner.gains(direction_from_angles(0.0, 0.0));
        assert_eq!(gains.len(), layout.len());
    }

    #[test]
    fn lfe_slot_stays_silent() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround5_1_4);
        let panner = VbapArrayPanner::new(&layout);
        let gains = panner.gains(direction_from_angles(30.0, 10.0));
        for (slot, speaker) in layout.speakers.iter().enumerate() {
            if speaker.is_lfe {
                assert!(close(gains[slot], 0.0));
            }
        }
    }

    #[test]
    fn gains_are_constant_power() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
        let panner = VbapArrayPanner::new(&layout);
        let gains = panner.gains(direction_from_angles(20.0, 15.0));
        assert!(close(sum_sq(&gains), 1.0));
    }

    #[test]
    fn direction_on_speaker_concentrates_gain() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround5_1_4);
        let panner = VbapArrayPanner::new(&layout);
        let target = layout
            .speakers
            .iter()
            .find(|speaker| !speaker.is_lfe)
            .expect("surround has directional speakers");
        let gains = panner.gains(target.direction);
        let target_slot = layout
            .speakers
            .iter()
            .position(|speaker| speaker.direction == target.direction && !speaker.is_lfe)
            .expect("target slot exists");
        assert!(gains[target_slot] > 0.9);
    }

    #[test]
    fn empty_layout_yields_empty_gains() {
        let panner = VbapArrayPanner::new(&ArrayLayout::new());
        assert!(panner.is_empty());
        assert!(panner.gains(Vec3::new(0.0, 0.0, -1.0)).is_empty());
    }
}
