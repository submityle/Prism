//! Speaker-geometry adapter between a bed layout and the VBAP panner.
//!
//! [`crate::pan::vbap`] works on a flat list of directional unit vectors and
//! produces one gain per vector. A [`crate::bed::BedLayout`], however,
//! interleaves an optional non-directional LFE among its channels. This module
//! bridges the two: [`SpeakerArray`] extracts the directional speakers of a
//! layout (dropping the LFE) and remembers each one's original channel index,
//! so a short VBAP gain vector can be scattered back into a full
//! per-channel-of-the-layout gain vector with the LFE left silent.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports design section 44.2 object-to-bed panning. Built from
//! [`crate::bed::BedLayout`] and consumed by [`crate::pan`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;

use prism_audio_core::math::Sample;

use crate::bed::BedLayout;

/// The directional speakers of a bed layout, with a map back to channel order.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerArray {
    layout: BedLayout,
    dirs: Vec<Vec3>,
    indices: Vec<usize>,
}

impl SpeakerArray {
    /// Builds the directional speaker array for `layout` (the LFE, if any, is
    /// excluded).
    #[must_use]
    pub fn new(layout: BedLayout) -> Self {
        let mut dirs = Vec::new();
        let mut indices = Vec::new();
        for (idx, ch) in layout.channels().iter().enumerate() {
            if ch.is_lfe {
                continue;
            }
            dirs.push(ch.direction);
            indices.push(idx);
        }
        Self {
            layout,
            dirs,
            indices,
        }
    }

    /// Returns the bed layout this array was built from.
    #[must_use]
    pub fn layout(&self) -> BedLayout {
        self.layout
    }

    /// Returns the directional speaker unit vectors (VBAP input order).
    #[must_use]
    pub fn directions(&self) -> &[Vec3] {
        &self.dirs
    }

    /// Returns the number of directional speakers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    /// Returns whether the array has no directional speakers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }

    /// Maps a VBAP-local speaker index to its channel index in the layout.
    #[must_use]
    pub fn channel_index(&self, local: usize) -> Option<usize> {
        self.indices.get(local).copied()
    }

    /// Scatters a VBAP-local gain vector (`local_gains.len() == self.len()`)
    /// into a full per-channel gain vector sized for the whole layout, with
    /// the LFE channel (and any channel not covered) left at zero.
    #[must_use]
    pub fn scatter(&self, local_gains: &[Sample]) -> Vec<Sample> {
        let mut full = Vec::new();
        full.resize(self.layout.channel_count(), 0.0);
        for (local, &gain) in local_gains.iter().enumerate() {
            if let Some(idx) = self.indices.get(local) {
                full[*idx] = gain;
            }
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excludes_lfe_from_directional_speakers() {
        let arr = SpeakerArray::new(BedLayout::Surround7_1_4);
        assert_eq!(arr.len(), 11);
        assert_eq!(arr.directions().len(), 11);
        assert!(!arr.is_empty());
    }

    #[test]
    fn scatter_leaves_lfe_silent() {
        let arr = SpeakerArray::new(BedLayout::Surround5_1_4);
        let local = vec![1.0; arr.len()];
        let full = arr.scatter(&local);
        let lfe = BedLayout::Surround5_1_4.lfe_index().unwrap();
        assert_eq!(full.len(), BedLayout::Surround5_1_4.channel_count());
        assert_eq!(full[lfe], 0.0);
    }

    #[test]
    fn channel_index_maps_back_in_order() {
        let arr = SpeakerArray::new(BedLayout::Stereo);
        assert_eq!(arr.channel_index(0), Some(0));
        assert_eq!(arr.channel_index(1), Some(1));
        assert_eq!(arr.channel_index(2), None);
    }
}
