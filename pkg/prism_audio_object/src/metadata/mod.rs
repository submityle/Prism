//! Time-varying object metadata: snapshots, keyframes, and streams.
//!
//! An object's placement is not static. This module models a metadata
//! *snapshot* ([`ObjectMetadata`]: position, gain, spread, priority at an
//! instant), a timestamped [`keyframe::Keyframe`], and a sampleable
//! [`stream::MetadataStream`] that linearly interpolates between keyframes.
//! This mirrors the MPEG-H object-metadata idea where each object's descriptor
//! evolves along the timeline and the renderer samples it at the control rate.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the object metadata stream of design section 44.2. Produced for
//! each [`crate::object::AudioObject`] and consumed by [`crate::scene`] when
//! advancing the scene in time.

pub mod keyframe;
pub mod stream;

use bevy_math::Vec3;

use prism_audio_core::math::Sample;

use crate::object::AudioObject;

/// An instantaneous object metadata snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectMetadata {
    /// Listener-local position in metres.
    pub position: Vec3,
    /// Linear amplitude gain.
    pub gain: Sample,
    /// Angular size / divergence in `[0, 1]`.
    pub spread: Sample,
    /// Rendering priority (higher is kept discrete longer under budget).
    pub priority: Sample,
}

impl ObjectMetadata {
    /// Builds a snapshot directly from an [`AudioObject`]'s current fields.
    #[must_use]
    pub fn from_object(object: &AudioObject) -> Self {
        Self {
            position: object.position,
            gain: object.gain,
            spread: object.spread,
            priority: object.priority,
        }
    }

    /// Applies this snapshot onto `object` in place, preserving its id and
    /// snap flag.
    pub fn apply_to(&self, object: &mut AudioObject) {
        object.position = self.position;
        object.gain = self.gain;
        object.spread = self.spread;
        object.priority = self.priority;
    }

    /// Linearly interpolates between `self` (at `t == 0`) and `other` (at
    /// `t == 1`). `t` is clamped to `[0, 1]`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: Sample) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            position: self.position.lerp(other.position, t),
            gain: self.gain + (other.gain - self.gain) * t,
            spread: self.spread + (other.spread - self.spread) * t,
            priority: self.priority + (other.priority - self.priority) * t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectId;
    use bevy_math::ops;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn meta(gain: Sample) -> ObjectMetadata {
        ObjectMetadata {
            position: Vec3::new(0.0, 0.0, -1.0),
            gain,
            spread: 0.0,
            priority: 1.0,
        }
    }

    #[test]
    fn from_and_apply_round_trip() {
        let mut o = AudioObject::new(ObjectId(1), Vec3::new(1.0, 2.0, 3.0), 0.7);
        o.spread = 0.3;
        o.priority = 2.0;
        let m = ObjectMetadata::from_object(&o);
        let mut target = AudioObject::new(ObjectId(1), Vec3::ZERO, 0.0);
        m.apply_to(&mut target);
        assert_eq!(target.position, o.position);
        assert!(close(target.gain, 0.7));
        assert!(close(target.spread, 0.3));
        assert!(close(target.priority, 2.0));
    }

    #[test]
    fn lerp_midpoint_averages() {
        let a = meta(0.0);
        let b = meta(1.0);
        let mid = a.lerp(&b, 0.5);
        assert!(close(mid.gain, 0.5));
    }

    #[test]
    fn lerp_clamps_t() {
        let a = meta(0.0);
        let b = meta(1.0);
        assert!(close(a.lerp(&b, -1.0).gain, 0.0));
        assert!(close(a.lerp(&b, 2.0).gain, 1.0));
    }
}
