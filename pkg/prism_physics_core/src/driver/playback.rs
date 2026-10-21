//! Time-addressed full-world cache playback with an independent render projection.
//!
//! A cache records post-step snapshots. Before the first snapshot's timestamp,
//! that snapshot is held; an unrecorded initial world is never invented. Between
//! boundaries only matching rigid-body handles are interpolated. Topology,
//! constraints, pending queues and every other authoritative field remain in
//! complete snapshots; [`PlaybackSample::restore_boundary`] never writes a
//! fractional pose into solver state.

use alloc::sync::Arc;

use super::{CacheFrame, DriverCache, DRIVER_CACHE_VERSION};
use crate::snapshot::{pose::BodyPose, StateSnapshot};
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;

/// Invalid cache or playback input. Errors do not advance a playback clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackError {
    /// There is no snapshot to sample.
    EmptyCache,
    /// This build does not understand the cache schema.
    UnsupportedVersion,
    /// Invalid frame identity, duration or timestamp ordering.
    InvalidTimeline,
    /// A positive duration was lost when accumulated onto an f64 timestamp.
    TimeResolutionExceeded,
    /// Seek time or elapsed duration is negative or non-finite.
    InvalidTime,
    /// Playback speed must be finite (zero pauses; negative plays backwards).
    InvalidSpeed,
    /// Multiplication or addition would overflow the finite playback clock.
    TimeOverflow,
}

/// Endpoint policy for a time-based playback clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaybackRepeat {
    /// Clamp at either endpoint, preserving the last sample.
    #[default]
    Once,
    /// Wrap into `[0, duration)`, including for reverse playback.
    Loop,
}

/// A borrowed, allocation-free sample of a cache's timeline.
#[derive(Clone, Copy, Debug)]
pub struct PlaybackSample<'a> {
    earlier: &'a CacheFrame,
    later: &'a CacheFrame,
    time: f64,
    alpha: f32,
}

impl PlaybackSample<'_> {
    /// Clamped time in seconds since recording began.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Complete snapshot at or before the requested time, except that the
    /// first recorded snapshot is held before its timestamp.
    #[must_use]
    pub fn boundary(&self) -> &CacheFrame {
        self.earlier
    }

    /// The next render endpoint; equals `boundary` when holding an endpoint.
    #[must_use]
    pub fn next_boundary(&self) -> &CacheFrame {
        self.later
    }

    /// Fraction between the two render endpoints.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Samples one rigid-body render pose without allocating. Bodies removed
    /// or replaced at the next boundary retain their earlier pose until then;
    /// newly born bodies are not visible early.
    #[must_use]
    pub fn pose(&self, handle: BodyHandle) -> Option<BodyPose> {
        let earlier = self.earlier.snapshot().render();
        earlier
            .interpolate(self.later.snapshot().render(), handle, self.alpha)
            .or_else(|| earlier.pose(handle))
    }

    /// Publishes all rigid-body render poses and velocities, reusing capacity
    /// in `output`. No full-world clone or solver call is performed.
    pub fn render_into(&self, output: &mut StateSnapshot) {
        self.earlier.snapshot().render().interpolate_into(
            self.later.snapshot().render(),
            self.alpha,
            output,
        );
    }

    /// Restores the entire earlier world boundary, including topology,
    /// generations and pending queues. This clones full state; consumers that
    /// need an authoritative playback world should call it only at boundary
    /// changes (and on initial binding or rebinding), not every render frame.
    /// Render interpolation must remain separate from that world.
    pub fn restore_boundary(&self, world: &mut PhysicsWorld) {
        self.earlier.snapshot().restore_into(world);
    }
}

impl DriverCache {
    fn validate_timeline(&self) -> Result<(), PlaybackError> {
        if self.version() != DRIVER_CACHE_VERSION {
            return Err(PlaybackError::UnsupportedVersion);
        }
        if self.is_empty() {
            return Err(PlaybackError::EmptyCache);
        }
        if !self.timeline_valid {
            return Err(PlaybackError::TimeResolutionExceeded);
        }
        Ok(())
    }

    /// Samples at absolute seconds since recording began. Times beyond the
    /// end hold the last frame. Negative or non-finite inputs are rejected.
    /// Lookup is O(log frames), has no allocation and does not clone a world.
    pub fn sample_at(&self, time: f64) -> Result<PlaybackSample<'_>, PlaybackError> {
        if !time.is_finite() || time < 0.0 {
            return Err(PlaybackError::InvalidTime);
        }
        self.validate_timeline()?;
        let time = time.min(self.duration());
        let upper = self.frames.partition_point(|frame| frame.time() <= time);
        let lower = upper.saturating_sub(1);
        let upper = if upper == 0 {
            0
        } else {
            upper.min(self.frames.len() - 1)
        };
        let earlier = &self.frames[lower];
        let later = &self.frames[upper];
        for (index, frame) in [(lower, earlier), (upper, later)] {
            if frame.version() != DRIVER_CACHE_VERSION
                || frame.index() as usize != index
                || frame.snapshot().frame() != index as u64
                || !frame.dt().is_finite()
                || frame.dt() <= 0.0
                || !frame.time().is_finite()
            {
                return Err(PlaybackError::InvalidTimeline);
            }
        }
        let alpha = if lower == upper {
            0.0
        } else {
            ((time - earlier.time()) / (later.time() - earlier.time())).clamp(0.0, 1.0) as f32
        };
        Ok(PlaybackSample {
            earlier,
            later,
            time,
            alpha,
        })
    }
}

/// A time-based player sharing an immutable, full-world cache.
///
/// Cloning the player shares the cache through Arc, not a deep world clone.
/// Each player owns its clock, speed and repeat policy independently. Methods
/// never invoke a solver, and pure render sampling needs no destination world.
#[derive(Clone, Debug)]
pub struct TimedPlayback {
    cache: Arc<DriverCache>,
    time: f64,
    speed: f64,
    repeat: PlaybackRepeat,
}

impl TimedPlayback {
    /// Binds a nonempty, current-version cache at time zero, speed one.
    pub fn new(cache: Arc<DriverCache>) -> Result<Self, PlaybackError> {
        cache.validate_timeline()?;
        // Validate once at binding, before a binary search can rely on ordered
        // timestamps. Future disk loaders must preserve the same invariant.
        let mut previous = 0.0;
        for (index, frame) in cache.frames.iter().enumerate() {
            if frame.version() != DRIVER_CACHE_VERSION
                || frame.index() as usize != index
                || frame.snapshot().frame() != index as u64
                || !frame.dt().is_finite()
                || frame.dt() <= 0.0
                || frame.time() != previous + f64::from(frame.dt())
                || frame.time() <= previous
            {
                return Err(PlaybackError::InvalidTimeline);
            }
            previous = frame.time();
        }
        Ok(Self {
            cache,
            time: 0.0,
            speed: 1.0,
            repeat: PlaybackRepeat::Once,
        })
    }

    /// The shared source cache, without copying its worlds.
    #[must_use]
    pub fn cache(&self) -> &Arc<DriverCache> {
        &self.cache
    }

    /// Current time in seconds since recording began.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Configured playback speed.
    #[must_use]
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Sets a finite playback speed. Zero pauses; negative reverses direction.
    pub fn set_speed(&mut self, speed: f64) -> Result<(), PlaybackError> {
        if !speed.is_finite() {
            return Err(PlaybackError::InvalidSpeed);
        }
        self.speed = speed;
        Ok(())
    }

    /// Sets the endpoint policy. Switching to Loop wraps an end-held clock.
    pub fn set_repeat(&mut self, repeat: PlaybackRepeat) {
        self.repeat = repeat;
        self.time = self.normalize(self.time);
    }

    fn normalize(&self, time: f64) -> f64 {
        match self.repeat {
            PlaybackRepeat::Once => time.clamp(0.0, self.cache.duration()),
            PlaybackRepeat::Loop => {
                let duration = self.cache.duration();
                // For a negative clock arbitrarily close to zero, floating
                // subtraction can round the remainder up to the divisor.
                // Keep reverse playback at the end of the half-open period,
                // never at the excluded endpoint or back at its beginning.
                time.rem_euclid(duration).min(duration.next_down())
            }
        }
    }

    /// Samples the current clock without advancing it.
    pub fn sample(&self) -> Result<PlaybackSample<'_>, PlaybackError> {
        self.cache.sample_at(self.time)
    }

    /// Seeks without stepping a solver. Once clamps; Loop wraps to its period.
    pub fn seek(&mut self, time: f64) -> Result<PlaybackSample<'_>, PlaybackError> {
        if !time.is_finite() || time < 0.0 {
            return Err(PlaybackError::InvalidTime);
        }
        let next = self.normalize(time);
        self.cache.sample_at(next)?;
        self.time = next;
        self.sample()
    }

    /// Advances by elapsed wall time multiplied by speed. Arbitrarily large
    /// valid steps use O(1) clock arithmetic rather than a frame-stepping loop.
    /// Invalid inputs or finite-clock overflow leave the clock unchanged.
    pub fn advance(&mut self, elapsed: f64) -> Result<PlaybackSample<'_>, PlaybackError> {
        if !elapsed.is_finite() || elapsed < 0.0 {
            return Err(PlaybackError::InvalidTime);
        }
        let time = self.time + elapsed * self.speed;
        if !time.is_finite() {
            return Err(PlaybackError::TimeOverflow);
        }
        let next = self.normalize(time);
        self.cache.sample_at(next)?;
        self.time = next;
        self.sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::OfflineStepConfig;
    use crate::snapshot::FullStateSnapshot;

    fn cache() -> DriverCache {
        let mut cache = DriverCache::new(2);
        let world = PhysicsWorld::default();
        for index in 0..2 {
            cache.push(
                FullStateSnapshot::capture(index, &world),
                0.25,
                OfflineStepConfig {
                    substeps: 1,
                    iterations: 1,
                },
            );
        }
        cache
    }

    #[test]
    fn binding_rejects_corrupt_frame_metadata_and_timestamp_order() {
        for mutate in [
            |frame: &mut CacheFrame| frame.version = 99,
            |frame: &mut CacheFrame| frame.index = 100,
            |frame: &mut CacheFrame| frame.dt = f32::NAN,
            |frame: &mut CacheFrame| frame.time = 0.1,
            |frame: &mut CacheFrame| frame.time = f64::INFINITY,
            |frame: &mut CacheFrame| {
                frame.snapshot = FullStateSnapshot::capture(9, &PhysicsWorld::default());
            },
        ] {
            let mut corrupted = cache();
            mutate(&mut corrupted.frames[1]);
            assert_eq!(
                TimedPlayback::new(Arc::new(corrupted)).unwrap_err(),
                PlaybackError::InvalidTimeline,
            );
        }
    }

    #[test]
    fn precision_loss_is_detected_instead_of_silently_merging_timestamps() {
        let mut cache = DriverCache::new(2);
        let world = PhysicsWorld::default();
        let config = OfflineStepConfig {
            substeps: 1,
            iterations: 1,
        };
        cache.push(FullStateSnapshot::capture(0, &world), f32::MAX, config);
        cache.push(
            FullStateSnapshot::capture(1, &world),
            f32::MIN_POSITIVE,
            config,
        );
        assert_eq!(
            cache.sample_at(0.0).unwrap_err(),
            PlaybackError::TimeResolutionExceeded,
        );
        assert_eq!(
            TimedPlayback::new(Arc::new(cache)).unwrap_err(),
            PlaybackError::TimeResolutionExceeded,
        );
    }
}
