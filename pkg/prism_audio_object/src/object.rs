//! A single dynamic audio object and its stable identifier.
//!
//! An *audio object* is a mono sound with 3D placement metadata that is mixed
//! on top of the channel [`crate::bed::BedLayout`]. Each object carries a
//! listener-local position, a linear gain, an angular size / divergence
//! (`spread`) that widens its apparent source, a rendering `priority` used by
//! the budget/clustering fallback, and a `snap` flag requesting hard placement
//! on the nearest speaker.
//!
//! This is a plain data type: no DSP happens here. The companion
//! [`crate::metadata`] module describes how these values evolve over time.
//!
//! # Determinism
//!
//! Direction/energy helpers route length math through [`bevy_math::ops`] and
//! are bit-reproducible across targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the per-object metadata of design section 44.2 (MPEG-H-style
//! object descriptors: position, gain, size, priority). Consumed by
//! [`crate::scene`], [`crate::clustering`], [`crate::pan`], and
//! [`crate::fold`].

use bevy_math::{ops, Vec3};

use prism_audio_core::math::Sample;

/// A stable identifier for an audio object within a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectId(pub u32);

impl ObjectId {
    /// Returns the raw numeric value of this id.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A dynamic, positioned mono object layered over the bed.
///
/// `position` is a listener-local point (metres); its normalised direction is
/// what panning and encoding use, while its length can inform distance models
/// elsewhere. `gain` is a linear amplitude. `spread` in `[0, 1]` widens the
/// apparent source (0 = point, 1 = fully diffuse). `priority` ranks the object
/// for the hardware budget (higher = more important to keep discrete). `snap`
/// asks the renderer to place the object directly on the nearest speaker
/// rather than panning between speakers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioObject {
    /// Stable identifier.
    pub id: ObjectId,
    /// Listener-local position in metres.
    pub position: Vec3,
    /// Linear amplitude gain.
    pub gain: Sample,
    /// Angular size / divergence in `[0, 1]` (0 = point source).
    pub spread: Sample,
    /// Rendering priority (higher is kept discrete longer under budget).
    pub priority: Sample,
    /// Whether to snap the object onto the nearest speaker.
    pub snap: bool,
}

impl AudioObject {
    /// Creates a point object at `position` with the given linear `gain`,
    /// zero spread, unit priority, and no snapping.
    #[must_use]
    pub fn new(id: ObjectId, position: Vec3, gain: Sample) -> Self {
        Self {
            id,
            position,
            gain,
            spread: 0.0,
            priority: 1.0,
            snap: false,
        }
    }

    /// Returns the unit direction toward the object, or `None` when the
    /// position is at (or numerically at) the listener and carries no bearing.
    #[must_use]
    pub fn direction(&self) -> Option<Vec3> {
        let len_sq = self.position.dot(self.position);
        let len = ops::sqrt(len_sq);
        if len <= 1e-6 {
            None
        } else {
            Some(self.position / len)
        }
    }

    /// Returns the object's acoustic energy, `gain^2`.
    #[must_use]
    pub fn energy(&self) -> Sample {
        self.gain * self.gain
    }

    /// Returns the distance (position length, metres) from the listener.
    #[must_use]
    pub fn distance(&self) -> Sample {
        ops::sqrt(self.position.dot(self.position))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn new_sets_sensible_defaults() {
        let o = AudioObject::new(ObjectId(7), Vec3::new(0.0, 0.0, -2.0), 0.5);
        assert_eq!(o.id.get(), 7);
        assert!(close(o.gain, 0.5));
        assert!(close(o.spread, 0.0));
        assert!(close(o.priority, 1.0));
        assert!(!o.snap);
    }

    #[test]
    fn direction_is_unit_or_none() {
        let o = AudioObject::new(ObjectId(1), Vec3::new(3.0, 0.0, 0.0), 1.0);
        let d = o.direction().unwrap();
        assert!(close(d.x, 1.0));
        assert!(close(d.length(), 1.0));

        let at_listener = AudioObject::new(ObjectId(2), Vec3::ZERO, 1.0);
        assert!(at_listener.direction().is_none());
    }

    #[test]
    fn energy_is_gain_squared() {
        let o = AudioObject::new(ObjectId(3), Vec3::new(0.0, 0.0, -1.0), 0.25);
        assert!(close(o.energy(), 0.0625));
    }

    #[test]
    fn distance_is_position_length() {
        let o = AudioObject::new(ObjectId(4), Vec3::new(0.0, 3.0, 4.0), 1.0);
        assert!(close(o.distance(), 5.0));
    }
}
