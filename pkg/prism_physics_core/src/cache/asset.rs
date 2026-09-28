//! The top-level baked physics cache asset.
//!
//! A [`PhysicsCache`] bundles everything needed to play a simulation back
//! without solving: the frame time step, the [`PositionQuantizer`] used to
//! encode positions, the keyframe interval, and one [`CacheTrack`] per baked
//! body. It is produced by the offline [`crate::cache::bake`] baker and consumed
//! by the [`crate::cache::playback`] player and the [`crate::cache::golden`]
//! digest.
//!
//! The asset is deliberately plain data (no solver handles, no borrowed state)
//! so it serializes cleanly and can be shipped as an offline resource.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::cache::quantize::PositionQuantizer;
use crate::cache::track::{raw_component_count, CacheTrack};
use crate::math::scalar::Real;

/// A complete baked simulation: metadata plus per-body trajectory tracks.
///
/// Every track shares the same frame count and the same [`PositionQuantizer`],
/// so frame `f` of every track corresponds to the same simulated time
/// `f * frame_dt`.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PhysicsCache {
    /// Seconds of simulated time between adjacent baked frames.
    frame_dt: Real,
    /// The quantizer used to encode and decode every track's positions.
    quantizer: PositionQuantizer,
    /// The keyframe interval (in frames) shared by every track.
    keyframe_interval: u32,
    /// One trajectory track per baked body.
    tracks: Vec<CacheTrack>,
}

impl PhysicsCache {
    /// Creates an empty cache with the given frame step, quantizer, and
    /// keyframe interval.
    #[must_use]
    pub fn new(
        frame_dt: Real,
        quantizer: PositionQuantizer,
        keyframe_interval: u32,
    ) -> PhysicsCache {
        PhysicsCache {
            frame_dt,
            quantizer,
            keyframe_interval,
            tracks: Vec::new(),
        }
    }

    /// Appends a fully built track to the cache.
    pub fn push_track(&mut self, track: CacheTrack) {
        self.tracks.push(track);
    }

    /// Returns the seconds of simulated time between adjacent frames.
    #[must_use]
    pub fn frame_dt(&self) -> Real {
        self.frame_dt
    }

    /// Returns the quantizer used by every track.
    #[must_use]
    pub fn quantizer(&self) -> PositionQuantizer {
        self.quantizer
    }

    /// Returns the shared keyframe interval in frames.
    #[must_use]
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }

    /// Returns the number of frames in the cache, taken from the first track.
    /// An empty cache has zero frames.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.tracks.first().map_or(0, CacheTrack::frame_count)
    }

    /// Returns the total simulated duration in seconds, `(frame_count - 1) *
    /// frame_dt`. A cache with fewer than two frames has zero duration.
    #[must_use]
    pub fn duration(&self) -> Real {
        let frames = self.frame_count();
        if frames < 2 {
            0.0
        } else {
            (frames - 1) as Real * self.frame_dt
        }
    }

    /// Returns the number of tracks (baked bodies) in the cache.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns `true` when the cache holds no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Returns a reference to the track at `index`, or `None` when out of range.
    #[must_use]
    pub fn track(&self, index: usize) -> Option<&CacheTrack> {
        self.tracks.get(index)
    }

    /// Returns an iterator over the cache's tracks in order.
    pub fn tracks(&self) -> impl Iterator<Item = &CacheTrack> {
        self.tracks.iter()
    }

    /// Reconstructs and dequantizes the positions of `track` at `frame`.
    ///
    /// Returns `None` when the track index or frame is out of range.
    #[must_use]
    pub fn reconstruct(&self, track: usize, frame: u32) -> Option<Vec<Vec3>> {
        let quantized = self.tracks.get(track)?.reconstruct(frame)?;
        Some(
            quantized
                .into_iter()
                .map(|q| self.quantizer.decode(q))
                .collect(),
        )
    }

    /// Returns the number of raw `i32` components a per-frame-full encoding of
    /// this cache would need, summed across every track. Divide the raw byte
    /// count by the serialized size to get a compression ratio.
    #[must_use]
    pub fn raw_component_count(&self) -> u64 {
        self.tracks
            .iter()
            .map(|t| raw_component_count(t.particle_count(), t.frame_count()))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_uses_frame_count_minus_one() {
        let mut cache = PhysicsCache::new(0.5, PositionQuantizer::default(), 4);
        let mut track = CacheTrack::new(1, 4);
        for i in 0..5 {
            track.push_frame(&[[i, 0, 0]]);
        }
        cache.push_track(track);
        assert_eq!(cache.frame_count(), 5);
        assert!((cache.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn empty_cache_has_zero_frames_and_duration() {
        let cache = PhysicsCache::new(1.0 / 60.0, PositionQuantizer::default(), 8);
        assert_eq!(cache.frame_count(), 0);
        assert_eq!(cache.duration(), 0.0);
        assert!(cache.is_empty());
    }

    #[test]
    fn reconstruct_dequantizes_positions() {
        let q = PositionQuantizer::new(1.0e-3);
        let mut cache = PhysicsCache::new(1.0 / 60.0, q, 2);
        let mut track = CacheTrack::new(1, 2);
        track.push_frame(&[q.encode(Vec3::new(0.5, -0.25, 1.0))]);
        cache.push_track(track);
        let out = cache.reconstruct(0, 0).unwrap();
        let err = (out[0] - Vec3::new(0.5, -0.25, 1.0)).abs();
        assert!(err.max_element() <= q.step());
        assert_eq!(cache.reconstruct(1, 0), None);
    }
}
