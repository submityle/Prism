//! Offline baking: run a soft body forward and record its trajectory.
//!
//! Baking is the offline half of the offline-mode workflow: a [`Baker`] steps a
//! [`SoftBody`] (or several in lockstep) through a fixed number of frames and
//! records each frame's quantized particle positions into a [`PhysicsCache`].
//! The resulting asset can then be played back with zero solver cost by the
//! [`crate::cache::playback`] player, which is the point of an offline cache:
//! pay the simulation cost once, at author time, and replay cheaply forever.
//!
//! One [`SoftBody`] maps to exactly one [`CacheTrack`]. When baking several
//! bodies together they are advanced in lockstep (all stepped by the same
//! `dt` each frame) so their tracks stay time-aligned.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The bake
//! loop is a plain fixed-step integration record of the M4 XPBD soft solver.

use crate::cache::asset::PhysicsCache;
use crate::cache::quantize::{PositionQuantizer, DEFAULT_STEP};
use crate::cache::track::CacheTrack;
use crate::math::scalar::Real;
use crate::soft::body::SoftBody;

/// Default number of frames a bake records when unset.
pub const DEFAULT_FRAME_COUNT: u32 = 120;
/// Default keyframe interval used by [`BakeConfig::default`].
pub const DEFAULT_KEYFRAME_INTERVAL: u32 = 12;
/// Default frame step (60 Hz) used by [`BakeConfig::default`].
pub const DEFAULT_FRAME_DT: Real = 1.0 / 60.0;

/// Configuration for an offline bake.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BakeConfig {
    /// Total number of frames to record (including the initial frame 0). A
    /// value below 1 is treated as 1 so at least the rest pose is captured.
    pub frames: u32,
    /// Simulated seconds advanced per recorded frame.
    pub dt: Real,
    /// Keyframe interval passed to each [`CacheTrack`].
    pub keyframe_interval: u32,
    /// Quantizer used to encode recorded positions.
    pub quantizer: PositionQuantizer,
}

impl Default for BakeConfig {
    fn default() -> Self {
        BakeConfig {
            frames: DEFAULT_FRAME_COUNT,
            dt: DEFAULT_FRAME_DT,
            keyframe_interval: DEFAULT_KEYFRAME_INTERVAL,
            quantizer: PositionQuantizer::new(DEFAULT_STEP),
        }
    }
}

impl BakeConfig {
    /// Returns the effective frame count, guaranteed to be at least 1.
    #[must_use]
    pub fn effective_frames(&self) -> u32 {
        self.frames.max(1)
    }
}

/// Runs a soft-body simulation offline and records it into a [`PhysicsCache`].
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Baker {
    /// The bake settings (frame count, step, quantization, keyframes).
    config: BakeConfig,
}

impl Baker {
    /// Creates a baker with the given configuration.
    #[must_use]
    pub fn new(config: BakeConfig) -> Baker {
        Baker { config }
    }

    /// Returns the baker's configuration.
    #[must_use]
    pub fn config(&self) -> BakeConfig {
        self.config
    }

    /// Records one frame of `body` into `track` by quantizing its current
    /// particle positions.
    fn record_frame(&self, body: &SoftBody, track: &mut CacheTrack) {
        let quantizer = self.config.quantizer;
        let quantized: Vec<[i32; 3]> = body
            .particles
            .positions()
            .iter()
            .map(|&p| quantizer.encode(p))
            .collect();
        track.push_frame(&quantized);
    }

    /// Creates an empty cache configured to match this baker.
    fn new_cache(&self) -> PhysicsCache {
        PhysicsCache::new(
            self.config.dt,
            self.config.quantizer,
            self.config.keyframe_interval,
        )
    }

    /// Creates a track sized for `body` matching this baker's keyframe interval.
    fn new_track(&self, body: &SoftBody) -> CacheTrack {
        CacheTrack::new(body.particles.len() as u32, self.config.keyframe_interval)
    }

    /// Bakes a single soft `body` into a one-track [`PhysicsCache`].
    ///
    /// The initial pose is recorded as frame 0, then the body is stepped
    /// `effective_frames - 1` times, recording after each step. The body is
    /// left at its final baked state.
    pub fn bake_body(&self, body: &mut SoftBody) -> PhysicsCache {
        let mut cache = self.new_cache();
        let mut track = self.new_track(body);
        self.record_frame(body, &mut track);
        for _ in 1..self.config.effective_frames() {
            body.step(self.config.dt);
            self.record_frame(body, &mut track);
        }
        cache.push_track(track);
        cache
    }

    /// Bakes several soft bodies advanced in lockstep into a multi-track cache.
    ///
    /// Every body gets its own track; on each frame all bodies are stepped by
    /// the same `dt` before recording, so track `i` frame `f` is body `i` at time
    /// `f * dt`. Bodies are left at their final baked states.
    pub fn bake_bodies(&self, bodies: &mut [SoftBody]) -> PhysicsCache {
        let mut cache = self.new_cache();
        let mut tracks: Vec<CacheTrack> = bodies.iter().map(|b| self.new_track(b)).collect();
        for (body, track) in bodies.iter().zip(tracks.iter_mut()) {
            self.record_frame(body, track);
        }
        for _ in 1..self.config.effective_frames() {
            for (body, track) in bodies.iter_mut().zip(tracks.iter_mut()) {
                body.step(self.config.dt);
                self.record_frame(body, track);
            }
        }
        for track in tracks {
            cache.push_track(track);
        }
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::build::ClothGrid;

    #[test]
    fn bake_records_requested_frame_count() {
        let mut cloth = ClothGrid {
            columns: 4,
            rows: 4,
            ..ClothGrid::default()
        }
        .build_default();
        let baker = Baker::new(BakeConfig {
            frames: 10,
            ..BakeConfig::default()
        });
        let cache = baker.bake_body(&mut cloth.body);
        assert_eq!(cache.frame_count(), 10);
        assert_eq!(cache.track_count(), 1);
    }

    #[test]
    fn zero_frames_still_records_rest_pose() {
        let mut cloth = ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default();
        let baker = Baker::new(BakeConfig {
            frames: 0,
            ..BakeConfig::default()
        });
        let cache = baker.bake_body(&mut cloth.body);
        assert_eq!(cache.frame_count(), 1);
    }

    #[test]
    fn bake_bodies_makes_one_track_each() {
        let grid = ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        };
        let baker = Baker::new(BakeConfig {
            frames: 5,
            ..BakeConfig::default()
        });
        let mut bodies = vec![grid.build_default().body, grid.build_default().body];
        let cache = baker.bake_bodies(&mut bodies);
        assert_eq!(cache.track_count(), 2);
        assert_eq!(cache.frame_count(), 5);
    }
}
