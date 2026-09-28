//! Zero-solve playback of a baked [`PhysicsCache`].
//!
//! Where the [`crate::cache::bake`] baker pays the simulation cost once, the
//! [`Player`] replays the result for free: it maps a wall-clock time to a
//! fractional frame, reconstructs the two bracketing baked frames, and linearly
//! interpolates between them. No constraint is projected and no body is stepped,
//! so playback cost is independent of scene complexity.
//!
//! Time advances via [`Player::advance`], which scales the incoming `dt` by a
//! configurable [`time_scale`](PlaybackConfig::time_scale) and optionally wraps
//! around the cache duration when [`looping`](PlaybackConfig::looping) is set.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Sampling a
//! discrete keyframed trajectory with linear interpolation is a standard
//! animation technique.

use glam::Vec3;

use crate::cache::asset::PhysicsCache;
use crate::math::scalar::Real;

/// Playback tunables applied by a [`Player`].
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaybackConfig {
    /// Multiplier applied to the `dt` fed to [`Player::advance`]. Values above
    /// 1 play faster, below 1 slower; negative values play in reverse.
    pub time_scale: Real,
    /// When `true`, playback time wraps around the cache duration instead of
    /// clamping at the ends.
    pub looping: bool,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        PlaybackConfig {
            time_scale: 1.0,
            looping: false,
        }
    }
}

/// A cursor that samples a [`PhysicsCache`] at a running playback time.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Player {
    /// Current playback time in seconds.
    time: Real,
    /// Playback configuration (speed and looping).
    config: PlaybackConfig,
}

impl Player {
    /// Creates a player at time zero with the given configuration.
    #[must_use]
    pub fn new(config: PlaybackConfig) -> Player {
        Player { time: 0.0, config }
    }

    /// Resets the playback time to zero.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }

    /// Returns the current playback time in seconds.
    #[must_use]
    pub fn time(&self) -> Real {
        self.time
    }

    /// Sets the current playback time in seconds.
    pub fn set_time(&mut self, time: Real) {
        self.time = time;
    }

    /// Returns the playback configuration.
    #[must_use]
    pub fn config(&self) -> PlaybackConfig {
        self.config
    }

    /// Advances playback time by `dt` seconds (scaled by
    /// [`time_scale`](PlaybackConfig::time_scale)), wrapping over `duration`
    /// when looping is enabled.
    ///
    /// `duration` is the total playable span, normally
    /// [`PhysicsCache::duration`]. Wrapping uses Euclidean remainder so it works
    /// for negative (reverse) time scales as well.
    pub fn advance(&mut self, dt: Real, duration: Real) {
        self.time += dt * self.config.time_scale;
        if self.config.looping && duration > 0.0 {
            self.time = self.time.rem_euclid(duration);
        }
    }

    /// Maps the current time to a fractional frame position `(frame0, alpha)`
    /// within `cache`, honouring the looping/clamping policy.
    ///
    /// Returns `None` when the cache has no frames. `alpha` is the interpolation
    /// weight in `[0, 1]` between `frame0` and `frame0 + 1`.
    #[must_use]
    fn locate(&self, cache: &PhysicsCache) -> Option<(u32, Real)> {
        let frame_count = cache.frame_count();
        if frame_count == 0 {
            return None;
        }
        let last = frame_count - 1;
        let frame_dt = cache.frame_dt();
        if frame_dt <= 0.0 {
            return Some((0, 0.0));
        }
        let duration = cache.duration();
        let mut time = self.time;
        if self.config.looping && duration > 0.0 {
            time = time.rem_euclid(duration);
        } else {
            time = time.clamp(0.0, duration);
        }
        let frame_f = time / frame_dt;
        let f0 = frame_f.floor();
        let alpha = frame_f - f0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "f0 is a clamped/wrapped, non-negative integral Real bounded by the frame count."
        )]
        let f0 = (f0 as u32).min(last);
        if f0 == last {
            Some((last, 0.0))
        } else {
            Some((f0, alpha))
        }
    }

    /// Samples `track` of `cache` at the current playback time, returning the
    /// interpolated world positions of that track's particles.
    ///
    /// Returns `None` when the cache is empty or the track index is out of
    /// range. Playback performs no simulation: the result is a linear blend of
    /// two reconstructed baked frames.
    #[must_use]
    pub fn sample(&self, cache: &PhysicsCache, track: usize) -> Option<Vec<Vec3>> {
        let (f0, alpha) = self.locate(cache)?;
        let a = cache.reconstruct(track, f0)?;
        if alpha <= 0.0 {
            return Some(a);
        }
        let b = cache.reconstruct(track, f0 + 1)?;
        let blended = a
            .iter()
            .zip(b.iter())
            .map(|(&pa, &pb)| pa + (pb - pa) * alpha)
            .collect();
        Some(blended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::bake::{BakeConfig, Baker};
    use crate::soft::build::ClothGrid;

    fn drape_cache(frames: u32) -> PhysicsCache {
        let mut cloth = ClothGrid {
            columns: 4,
            rows: 4,
            ..ClothGrid::default()
        }
        .build_default();
        cloth.pin(0, 0);
        cloth.pin(0, 3);
        let baker = Baker::new(BakeConfig {
            frames,
            ..BakeConfig::default()
        });
        baker.bake_body(&mut cloth.body)
    }

    #[test]
    fn sample_at_zero_matches_first_frame() {
        let cache = drape_cache(8);
        let player = Player::new(PlaybackConfig::default());
        let sampled = player.sample(&cache, 0).unwrap();
        let baked = cache.reconstruct(0, 0).unwrap();
        assert_eq!(sampled.len(), baked.len());
        for (s, b) in sampled.iter().zip(baked.iter()) {
            assert!((*s - *b).length() < 1e-6);
        }
    }

    #[test]
    fn advance_clamps_at_end_without_looping() {
        let cache = drape_cache(6);
        let mut player = Player::new(PlaybackConfig::default());
        player.advance(100.0, cache.duration());
        let sampled = player.sample(&cache, 0).unwrap();
        let last = cache.reconstruct(0, cache.frame_count() - 1).unwrap();
        for (s, b) in sampled.iter().zip(last.iter()) {
            assert!((*s - *b).length() < 1e-6);
        }
    }

    #[test]
    fn looping_wraps_time() {
        let cache = drape_cache(6);
        let mut player = Player::new(PlaybackConfig {
            time_scale: 1.0,
            looping: true,
        });
        let duration = cache.duration();
        player.advance(duration * 2.5, duration);
        assert!(player.time() >= 0.0 && player.time() <= duration);
    }

    #[test]
    fn interpolation_lies_between_frames() {
        let cache = drape_cache(8);
        let mut player = Player::new(PlaybackConfig::default());
        player.set_time(cache.frame_dt() * 0.5);
        let mid = player.sample(&cache, 0).unwrap();
        let f0 = cache.reconstruct(0, 0).unwrap();
        let f1 = cache.reconstruct(0, 1).unwrap();
        for i in 0..mid.len() {
            let lo = f0[i].min(f1[i]);
            let hi = f0[i].max(f1[i]);
            let eps = Vec3::splat(1e-5);
            assert!(mid[i].cmpge(lo - eps).all() && mid[i].cmple(hi + eps).all());
        }
    }

    #[test]
    fn empty_cache_samples_none() {
        let cache = PhysicsCache::default();
        let player = Player::new(PlaybackConfig::default());
        assert_eq!(player.sample(&cache, 0), None);
    }
}
