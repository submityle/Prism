//! **Vertical layering**: stacking tracks that fade in and out with intensity.
//!
//! Vertical re-orchestration keeps one musical bed playing while adding or
//! removing simultaneous [`Layer`]s (a string pad, a percussion track, a brass
//! stab) as a single gameplay **intensity** value rises and falls. Each layer
//! owns a pair of intensity thresholds that form a **hysteresis** band:
//!
//! - it switches *on* once intensity reaches [`Layer::enter_intensity`],
//! - and only switches *off* once intensity drops below
//!   [`Layer::exit_intensity`] (`exit <= enter`).
//!
//! The gap between the two thresholds stops a layer flickering when intensity
//! hovers around a single edge. A [`LayerSet`] groups the layers driven by one
//! intensity value. Both types are pure data; the live
//! [`crate::system::MusicSystem`] tracks which layers are currently active and
//! emits the fades.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. Threshold
//! layering with hysteresis is reconstructed from first principles as plain
//! data. No AI/ML.
//!
//! # Relationship
//!
//! A [`LayerSet`] is stored in [`crate::model::MusicModel`] and driven by
//! [`crate::system::MusicSystem::set_intensity`], which compares the live
//! intensity against each [`Layer`]'s band to emit
//! [`crate::action::MusicAction::StartLayer`] and
//! [`crate::action::MusicAction::StopLayer`].

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::id::{LayerId, LayerSetId, SoundId};

/// One track of a vertical layer set, gated by an intensity hysteresis band.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Layer {
    /// Stable id this layer is referenced by.
    pub id: LayerId,
    /// Leaf audio asset the layer plays.
    pub sound: SoundId,
    /// Intensity at or above which an inactive layer switches on.
    pub enter_intensity: Sample,
    /// Intensity below which an active layer switches off (`<= enter`).
    pub exit_intensity: Sample,
    /// Playback gain in decibels applied while the layer is active.
    pub gain_db: Sample,
}

impl Layer {
    /// Builds a layer with the given thresholds and unity gain (`0 dB`).
    ///
    /// `exit_intensity` is clamped to at most `enter_intensity` so the
    /// hysteresis band is always well formed even for corrupt authoring data.
    #[must_use]
    pub fn new(
        id: LayerId,
        sound: SoundId,
        enter_intensity: Sample,
        exit_intensity: Sample,
    ) -> Self {
        Self {
            id,
            sound,
            enter_intensity,
            exit_intensity: exit_intensity.min(enter_intensity),
            gain_db: 0.0,
        }
    }

    /// Sets the active playback gain in decibels, returning `self` for
    /// builder-style chaining.
    #[must_use]
    pub fn with_gain_db(mut self, gain_db: Sample) -> Self {
        self.gain_db = gain_db;
        self
    }

    /// Resolves whether the layer should be active at `intensity` given its
    /// previous active state, honouring the hysteresis band.
    ///
    /// An inactive layer turns on only at or above [`Layer::enter_intensity`];
    /// an active layer stays on until `intensity` falls below
    /// [`Layer::exit_intensity`].
    #[must_use]
    pub fn is_active(&self, intensity: Sample, was_active: bool) -> bool {
        if was_active {
            intensity >= self.exit_intensity
        } else {
            intensity >= self.enter_intensity
        }
    }
}

/// A group of [`Layer`]s driven by a single intensity value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerSet {
    /// Stable id this layer set is referenced by.
    pub id: LayerSetId,
    /// Member layers in authored order.
    pub layers: Vec<Layer>,
}

impl LayerSet {
    /// Builds an empty layer set.
    #[must_use]
    pub fn new(id: LayerSetId) -> Self {
        Self {
            id,
            layers: Vec::new(),
        }
    }

    /// Appends a layer, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Looks up a member layer by id.
    #[must_use]
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-5;

    fn layer() -> Layer {
        Layer::new(LayerId::new(1), SoundId::new(10), 0.6, 0.4)
    }

    #[test]
    fn exit_is_clamped_below_enter() {
        let l = Layer::new(LayerId::new(1), SoundId::new(1), 0.5, 0.9);
        assert!(l.exit_intensity <= l.enter_intensity);
        assert!((l.exit_intensity - 0.5).abs() < EPS);
    }

    #[test]
    fn hysteresis_band_holds_state() {
        let l = layer();
        // Rising edge: must reach enter to turn on.
        assert!(!l.is_active(0.5, false));
        assert!(l.is_active(0.6, false));
        // In the band while already active: stays on.
        assert!(l.is_active(0.5, true));
        // Below exit: turns off.
        assert!(!l.is_active(0.39, true));
    }

    #[test]
    fn set_lookup_by_id() {
        let set = LayerSet::new(LayerSetId::new(1))
            .with_layer(layer())
            .with_layer(Layer::new(LayerId::new(2), SoundId::new(11), 0.8, 0.7));
        assert!(set.layer(LayerId::new(2)).is_some());
        assert!(set.layer(LayerId::new(9)).is_none());
        assert_eq!(set.layers.len(), 2);
    }
}
