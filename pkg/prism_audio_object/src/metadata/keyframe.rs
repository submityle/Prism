//! A single timestamped object-metadata keyframe.
//!
//! A keyframe binds an [`ObjectMetadata`] snapshot to a point in time (seconds
//! from the stream origin). A [`crate::metadata::stream::MetadataStream`] holds
//! an ordered list of these and interpolates between neighbours.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Part of the object metadata stream of design section 44.2. The atomic unit
//! consumed by [`crate::metadata::stream`].

use prism_audio_core::math::Sample;

use crate::metadata::ObjectMetadata;

/// A metadata snapshot anchored at a time (seconds).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Keyframe {
    /// Time of this keyframe in seconds from the stream origin.
    pub time: Sample,
    /// The metadata snapshot that holds at `time`.
    pub value: ObjectMetadata,
}

impl Keyframe {
    /// Creates a keyframe at `time` seconds holding `value`.
    #[must_use]
    pub fn new(time: Sample, value: ObjectMetadata) -> Self {
        Self { time, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn stores_time_and_value() {
        let v = ObjectMetadata {
            position: Vec3::new(0.0, 0.0, -1.0),
            gain: 0.5,
            spread: 0.1,
            priority: 1.0,
        };
        let k = Keyframe::new(1.5, v);
        assert_eq!(k.time, 1.5);
        assert_eq!(k.value.gain, 0.5);
    }
}
