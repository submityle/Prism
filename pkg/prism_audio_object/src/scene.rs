//! An object scene: a bed plus dynamic objects advanced over time.
//!
//! [`ObjectScene`] is the authoring/runtime aggregate of the bed-plus-objects
//! model: a fixed [`crate::bed::BedLayout`] and a set of
//! [`crate::object::AudioObject`]s, each optionally driven by a
//! [`crate::metadata::stream::MetadataStream`]. Advancing the scene to a time
//! samples every stream and writes the interpolated metadata back onto its
//! object, giving a consistent per-tick snapshot for the renderer.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Ties together design section 44.2's bed, objects, and metadata streams.
//! Consumed by [`crate::render`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::bed::BedLayout;
use crate::metadata::stream::MetadataStream;
use crate::object::AudioObject;

/// One scene entry: an object and the optional metadata stream that drives it.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SceneEntry {
    /// The object's current (last-sampled) state.
    pub object: AudioObject,
    /// Optional time-varying metadata driving the object.
    pub stream: Option<MetadataStream>,
}

/// A bed plus a collection of (optionally animated) objects.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectScene {
    bed: BedLayout,
    entries: Vec<SceneEntry>,
}

impl ObjectScene {
    /// Creates an empty scene with the given bed layout.
    #[must_use]
    pub fn new(bed: BedLayout) -> Self {
        Self {
            bed,
            entries: Vec::new(),
        }
    }

    /// Returns the bed layout.
    #[must_use]
    pub fn bed(&self) -> BedLayout {
        self.bed
    }

    /// Adds a static object (no metadata stream) and returns its entry index.
    pub fn add_object(&mut self, object: AudioObject) -> usize {
        self.entries.push(SceneEntry {
            object,
            stream: None,
        });
        self.entries.len() - 1
    }

    /// Adds an animated object driven by `stream` and returns its entry index.
    pub fn add_animated_object(&mut self, object: AudioObject, stream: MetadataStream) -> usize {
        self.entries.push(SceneEntry {
            object,
            stream: Some(stream),
        });
        self.entries.len() - 1
    }

    /// Returns the scene entries.
    #[must_use]
    pub fn entries(&self) -> &[SceneEntry] {
        &self.entries
    }

    /// Returns the number of objects in the scene.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the scene holds no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Advances every animated object to `time` seconds by sampling its stream
    /// and applying the interpolated metadata (id and snap flag preserved).
    /// Static objects are left untouched.
    pub fn advance_to(&mut self, time: Sample) {
        for entry in &mut self.entries {
            if let Some(stream) = &entry.stream
                && let Some(meta) = stream.sample(time)
            {
                meta.apply_to(&mut entry.object);
            }
        }
    }

    /// Returns a snapshot of every object's current state.
    #[must_use]
    pub fn current_objects(&self) -> Vec<AudioObject> {
        self.entries.iter().map(|e| e.object).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ObjectMetadata;
    use crate::object::ObjectId;
    use bevy_math::{ops, Vec3};

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn meta(gain: Sample, z: Sample) -> ObjectMetadata {
        ObjectMetadata {
            position: Vec3::new(0.0, 0.0, z),
            gain,
            spread: 0.0,
            priority: 1.0,
        }
    }

    #[test]
    fn new_scene_is_empty() {
        let s = ObjectScene::new(BedLayout::Surround7_1_4);
        assert!(s.is_empty());
        assert_eq!(s.bed(), BedLayout::Surround7_1_4);
    }

    #[test]
    fn add_objects_grows_scene() {
        let mut s = ObjectScene::new(BedLayout::Stereo);
        let i0 = s.add_object(AudioObject::new(ObjectId(0), Vec3::NEG_Z, 1.0));
        let i1 = s.add_object(AudioObject::new(ObjectId(1), Vec3::X, 1.0));
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(s.len(), 2);
        assert_eq!(s.current_objects().len(), 2);
    }

    #[test]
    fn advance_applies_stream_metadata() {
        let mut s = ObjectScene::new(BedLayout::Surround5_1_4);
        let mut stream = MetadataStream::new();
        stream.push(0.0, meta(0.0, 0.0));
        stream.push(2.0, meta(1.0, -2.0));
        let mut obj = AudioObject::new(ObjectId(9), Vec3::ZERO, 0.0);
        obj.snap = true;
        s.add_animated_object(obj, stream);

        s.advance_to(1.0);
        let o = s.current_objects()[0];
        assert!(close(o.gain, 0.5));
        assert!(close(o.position.z, -1.0));
        // Identity and snap preserved across sampling.
        assert_eq!(o.id, ObjectId(9));
        assert!(o.snap);
    }

    #[test]
    fn static_object_is_untouched_by_advance() {
        let mut s = ObjectScene::new(BedLayout::Stereo);
        s.add_object(AudioObject::new(ObjectId(3), Vec3::new(1.0, 0.0, 0.0), 0.7));
        s.advance_to(5.0);
        let o = s.current_objects()[0];
        assert!(close(o.gain, 0.7));
        assert_eq!(o.position, Vec3::new(1.0, 0.0, 0.0));
    }
}
