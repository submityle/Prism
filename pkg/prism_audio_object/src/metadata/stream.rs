//! A time-ordered, sampleable stream of object-metadata keyframes.
//!
//! A [`MetadataStream`] owns an ascending-by-time list of
//! [`Keyframe`]s and resolves the object's metadata at any requested time by
//! linear interpolation between the two bracketing keyframes. Before the first
//! keyframe the first value is held; after the last, the last value is held
//! (clamped ends, no extrapolation).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the sampleable object metadata stream of design section 44.2.
//! Keyframes come from [`crate::metadata::keyframe`]; the sampled
//! [`ObjectMetadata`] is applied to objects by [`crate::scene`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::metadata::keyframe::Keyframe;
use crate::metadata::ObjectMetadata;

/// An ascending-by-time collection of metadata keyframes.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_object::metadata::stream::MetadataStream;
/// use prism_audio_object::metadata::ObjectMetadata;
///
/// let a = ObjectMetadata { position: Vec3::new(0.0, 0.0, -1.0), gain: 0.0, spread: 0.0, priority: 1.0 };
/// let b = ObjectMetadata { position: Vec3::new(0.0, 0.0, -1.0), gain: 1.0, spread: 0.0, priority: 1.0 };
/// let mut s = MetadataStream::new();
/// s.push(0.0, a);
/// s.push(2.0, b);
/// let mid = s.sample(1.0).unwrap();
/// assert!((mid.gain - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MetadataStream {
    keyframes: Vec<Keyframe>,
}

impl MetadataStream {
    /// Creates an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Builds a stream from an unordered keyframe list, sorting it by time.
    #[must_use]
    pub fn from_keyframes(mut keyframes: Vec<Keyframe>) -> Self {
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        Self { keyframes }
    }

    /// Inserts a keyframe at `time` holding `value`, keeping the list sorted
    /// by time (stable with respect to equal times).
    pub fn push(&mut self, time: Sample, value: ObjectMetadata) {
        let kf = Keyframe::new(time, value);
        let pos = self
            .keyframes
            .iter()
            .position(|k| k.time > time)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(pos, kf);
    }

    /// Returns the keyframes in ascending time order.
    #[must_use]
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// Returns the number of keyframes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns whether the stream holds no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Samples the metadata at `time` seconds.
    ///
    /// Returns `None` only for an empty stream. Times before the first (after
    /// the last) keyframe clamp to the first (last) value; interior times are
    /// linearly interpolated between the two bracketing keyframes.
    #[must_use]
    pub fn sample(&self, time: Sample) -> Option<ObjectMetadata> {
        if self.keyframes.is_empty() {
            return None;
        }
        let first = &self.keyframes[0];
        if time <= first.time {
            return Some(first.value);
        }
        let last = &self.keyframes[self.keyframes.len() - 1];
        if time >= last.time {
            return Some(last.value);
        }
        // Find the first keyframe with time strictly greater than `time`; its
        // predecessor is the lower bracket. Both exist because of the guards
        // above.
        let upper = self
            .keyframes
            .iter()
            .position(|k| k.time > time)
            .unwrap_or(self.keyframes.len() - 1);
        let lo = &self.keyframes[upper - 1];
        let hi = &self.keyframes[upper];
        let span = hi.time - lo.time;
        if span <= 0.0 {
            return Some(hi.value);
        }
        let t = (time - lo.time) / span;
        Some(lo.value.lerp(&hi.value, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{ops, Vec3};

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn meta(gain: Sample, pos_z: Sample) -> ObjectMetadata {
        ObjectMetadata {
            position: Vec3::new(0.0, 0.0, pos_z),
            gain,
            spread: 0.0,
            priority: 1.0,
        }
    }

    #[test]
    fn empty_stream_samples_none() {
        let s = MetadataStream::new();
        assert!(s.is_empty());
        assert!(s.sample(0.0).is_none());
    }

    #[test]
    fn clamps_before_and_after() {
        let mut s = MetadataStream::new();
        s.push(1.0, meta(0.2, -1.0));
        s.push(3.0, meta(0.8, -3.0));
        assert!(close(s.sample(0.0).unwrap().gain, 0.2));
        assert!(close(s.sample(10.0).unwrap().gain, 0.8));
    }

    #[test]
    fn interpolates_interior() {
        let mut s = MetadataStream::new();
        s.push(0.0, meta(0.0, 0.0));
        s.push(4.0, meta(1.0, -4.0));
        let m = s.sample(1.0).unwrap();
        assert!(close(m.gain, 0.25));
        assert!(close(m.position.z, -1.0));
    }

    #[test]
    fn push_keeps_sorted_order() {
        let mut s = MetadataStream::new();
        s.push(2.0, meta(0.5, -2.0));
        s.push(0.0, meta(0.0, 0.0));
        s.push(1.0, meta(0.25, -1.0));
        let times: Vec<Sample> = s.keyframes().iter().map(|k| k.time).collect();
        assert_eq!(times, [0.0, 1.0, 2.0]);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn from_keyframes_sorts() {
        let s = MetadataStream::from_keyframes(alloc_vec());
        let times: Vec<Sample> = s.keyframes().iter().map(|k| k.time).collect();
        assert_eq!(times, [0.0, 1.0, 2.0]);
    }

    fn alloc_vec() -> Vec<Keyframe> {
        [
            Keyframe::new(2.0, meta(1.0, -2.0)),
            Keyframe::new(0.0, meta(0.0, 0.0)),
            Keyframe::new(1.0, meta(0.5, -1.0)),
        ]
        .into()
    }
}
