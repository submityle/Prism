//! Simulation drivers for realtime, bounded offline, and cache playback runs.
//!
//! [`SimulationDriver`] keeps the existing realtime entry point intact while
//! adding two stateful, bounded modes. Offline mode calls the selected backend
//! with explicit substep/iteration parameters and records [`FullStateSnapshot`]s.
//! Playback mode applies those snapshots one frame at a time without invoking
//! the solver. The driver cache is intentionally separate from
//! [`crate::cache::PhysicsCache`], which stores deformable particle tracks and
//! is consumed by [`crate::cache::Player`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::backend::{OfflineStepConfig, OfflineStepError, PhysicsBackend};
use crate::snapshot::FullStateSnapshot;
use crate::world::PhysicsWorld;

pub mod fixed_step;
pub mod playback;

/// Version of the driver-owned snapshot cache format.
pub const DRIVER_CACHE_VERSION: u32 = 3;

/// Hard upper bound for frames retained by one driver cache.
pub const MAX_DRIVER_CACHE_FRAMES: u32 = 16_384;

/// Hard upper bound for solver substeps within one offline frame.
pub const MAX_OFFLINE_SUBSTEPS: u32 = crate::solver::MAX_OFFLINE_SOLVER_SUBSTEPS;

/// Hard upper bound for the number of caches registered with one driver.
pub const MAX_REGISTERED_CACHES: usize = 64;

/// An opaque reference to a cached/baked simulation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheHandle(pub u64);

/// How a [`SimulationDriver`] advances the simulation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DriveMode {
    /// Advance one backend step per [`SimulationDriver::drive`] call.
    Realtime,
    /// Run one bounded offline frame per drive call.
    Offline {
        /// Sub-steps per offline frame, clamped to `1..=MAX_OFFLINE_SUBSTEPS`.
        substeps: u32,
        /// Position-projection iterations per substep, applied to rigid and
        /// world-owned soft constraints by the CPU XPBD solver. Zero or values
        /// above [`crate::solver::MAX_OFFLINE_SOLVER_ITERATIONS`] are rejected.
        iterations: u32,
        /// Total number of offline frames to execute and retain. Values above
        /// [`MAX_DRIVER_CACHE_FRAMES`] are clamped.
        target_frames: u32,
    },
    /// Apply one previously registered cache frame per drive call.
    Playback(CacheHandle),
}

/// A single versioned frame in a driver playback cache.
#[derive(Clone, Debug)]
pub struct CacheFrame {
    version: u32,
    index: u32,
    snapshot: FullStateSnapshot,
    dt: f32,
    time: f64,
    config: OfflineStepConfig,
}

impl CacheFrame {
    /// Returns the cache format version carried by this frame.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the zero-based frame index.
    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the immutable world snapshot stored in this frame.
    #[must_use]
    pub fn snapshot(&self) -> &FullStateSnapshot {
        &self.snapshot
    }

    /// Returns the frame duration passed to the backend when recording.
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Time in seconds from the beginning of the recording to this post-step
    /// snapshot. Durations are accumulated in f64, including variable steps.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the effective (bounded) per-frame solver overrides.
    #[must_use]
    pub fn config(&self) -> OfflineStepConfig {
        self.config
    }
}

/// A bounded, versioned sequence of full-world driver snapshots.
///
/// This is an in-memory replay cache, not a serialized render-pose transport.
/// Each frame owns a complete world, including topology and pending queues.
#[derive(Clone, Debug)]
pub struct DriverCache {
    version: u32,
    max_frames: u32,
    frames: Vec<CacheFrame>,
    timeline_valid: bool,
}

impl DriverCache {
    /// Creates an empty cache using the current driver cache format.
    #[must_use]
    pub fn new(max_frames: u32) -> DriverCache {
        Self::with_version(DRIVER_CACHE_VERSION, max_frames)
    }

    /// Creates an empty cache with an explicit format version.
    ///
    /// This constructor makes version validation testable and lets a loader
    /// preserve an unknown version for an explicit registration error instead
    /// of silently treating it as the current format.
    #[must_use]
    pub fn with_version(version: u32, max_frames: u32) -> DriverCache {
        DriverCache {
            version,
            max_frames: max_frames.min(MAX_DRIVER_CACHE_FRAMES),
            frames: Vec::new(),
            timeline_valid: true,
        }
    }

    /// Returns the cache format version.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the retained-frame capacity after the hard safety clamp.
    #[must_use]
    pub fn max_frames(&self) -> u32 {
        self.max_frames
    }

    /// Returns the number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Returns whether no frame has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns a frame by zero-based index.
    #[must_use]
    pub fn frame(&self, index: u32) -> Option<&CacheFrame> {
        self.frames.get(index as usize)
    }

    /// Total recorded time, including the step preceding the first snapshot.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.frames.last().map_or(0.0, CacheFrame::time)
    }

    fn push(
        &mut self,
        snapshot: FullStateSnapshot,
        dt: f32,
        config: OfflineStepConfig,
    ) -> Option<u32> {
        if self.frame_count() >= self.max_frames {
            return None;
        }
        let index = self.frame_count();
        let previous_time = self.duration();
        let time = previous_time + f64::from(dt);
        self.timeline_valid &= dt.is_finite() && dt > 0.0 && time > previous_time;
        self.frames.push(CacheFrame {
            version: self.version,
            index,
            snapshot,
            dt,
            time,
            config,
        });
        Some(index)
    }
}

/// Why registering a playback cache was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CacheRegistrationError {
    /// The cache format is not understood by this driver.
    UnsupportedVersion {
        /// The version accepted by this build.
        expected: u32,
        /// The version supplied by the caller.
        actual: u32,
    },
    /// The cache exceeds the driver's hard frame bound.
    FrameLimitExceeded,
    /// The bounded registry has no slot for a new handle.
    RegistryFull,
}

/// The result of a [`SimulationDriver::drive`] call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriveOutcome {
    /// The world was advanced by one realtime backend step.
    Stepped,
    /// One offline frame was simulated and recorded.
    OfflineFrame {
        /// Zero-based index of the recorded frame.
        frame: u32,
        /// Effective bounded target frame count.
        total_frames: u32,
    },
    /// The offline configuration or solver capability was rejected without
    /// advancing the world or recording a frame.
    OfflineRejected {
        /// Exact validation/capability error reported by the backend.
        error: OfflineStepError,
    },
    /// One playback frame was applied to the world.
    PlaybackFrame {
        /// Zero-based index of the applied frame.
        frame: u32,
        /// Number of frames in the registered cache.
        total_frames: u32,
    },
    /// The bounded offline or playback sequence has no more work.
    Complete {
        /// Number of frames available or recorded.
        frames: u32,
    },
    /// The requested playback cache was not registered.
    CacheUnavailable {
        /// Missing cache handle.
        handle: CacheHandle,
    },
    /// A cache frame has inconsistent version or frame metadata.
    CacheMismatch {
        /// Frame that failed validation/application.
        frame: u32,
    },
}

#[derive(Clone, Debug, Default)]
struct DriverState {
    last_mode: Option<DriveMode>,
    offline: Option<OfflineState>,
    playback_frame: u32,
    caches: BTreeMap<CacheHandle, DriverCache>,
}

#[derive(Clone, Debug)]
struct OfflineState {
    target_frames: u32,
    cache: DriverCache,
}

/// Drives a simulation according to a [`DriveMode`].
///
/// The receiver remains shared (`&self`) for compatibility with the original
/// realtime API. Offline progress, playback cursors, and registered caches are
/// kept behind interior mutability because those modes are intentionally
/// stateful across frame calls.
#[derive(Clone, Debug)]
pub struct SimulationDriver {
    /// The active drive mode.
    pub mode: DriveMode,
    state: RefCell<DriverState>,
}

impl SimulationDriver {
    /// Creates a driver for the given `mode`.
    #[must_use]
    pub fn new(mode: DriveMode) -> SimulationDriver {
        SimulationDriver {
            mode,
            state: RefCell::new(DriverState::default()),
        }
    }

    /// Returns `true` if the driver is in [`DriveMode::Realtime`].
    #[must_use]
    pub fn is_realtime(&self) -> bool {
        matches!(self.mode, DriveMode::Realtime)
    }

    /// Registers or replaces a bounded playback cache.
    pub fn register_cache(
        &self,
        handle: CacheHandle,
        cache: DriverCache,
    ) -> Result<(), CacheRegistrationError> {
        if cache.version != DRIVER_CACHE_VERSION {
            return Err(CacheRegistrationError::UnsupportedVersion {
                expected: DRIVER_CACHE_VERSION,
                actual: cache.version,
            });
        }
        if cache.frame_count() > MAX_DRIVER_CACHE_FRAMES
            || cache.max_frames > MAX_DRIVER_CACHE_FRAMES
        {
            return Err(CacheRegistrationError::FrameLimitExceeded);
        }

        let mut state = self.state.borrow_mut();
        if !state.caches.contains_key(&handle) && state.caches.len() >= MAX_REGISTERED_CACHES {
            return Err(CacheRegistrationError::RegistryFull);
        }
        state.caches.insert(handle, cache);
        if self.mode == DriveMode::Playback(handle) {
            state.playback_frame = 0;
        }
        Ok(())
    }

    /// Removes and returns a registered cache, if present.
    pub fn unregister_cache(&self, handle: CacheHandle) -> Option<DriverCache> {
        self.state.borrow_mut().caches.remove(&handle)
    }

    /// Returns a cloned registered cache for inspection or transfer.
    #[must_use]
    pub fn cache(&self, handle: CacheHandle) -> Option<DriverCache> {
        self.state.borrow().caches.get(&handle).cloned()
    }

    /// Returns a clone of the cache recorded by the current Offline run.
    #[must_use]
    pub fn recorded_cache(&self) -> Option<DriverCache> {
        let state = self.state.borrow();
        if state.last_mode != Some(self.mode) || !matches!(self.mode, DriveMode::Offline { .. }) {
            return None;
        }
        state.offline.as_ref().map(|offline| offline.cache.clone())
    }

    /// Returns the current playback cursor, or zero for a non-playback mode.
    #[must_use]
    pub fn playback_frame(&self) -> u32 {
        let state = self.state.borrow();
        if state.last_mode == Some(self.mode) && matches!(self.mode, DriveMode::Playback(_)) {
            state.playback_frame
        } else {
            0
        }
    }

    /// Advances the simulation by one bounded frame using `backend`, honouring
    /// the current mode.
    pub fn drive(
        &self,
        backend: &mut dyn PhysicsBackend,
        world: &mut PhysicsWorld,
        dt: f32,
    ) -> DriveOutcome {
        let mode = self.mode;
        let mut state = self.state.borrow_mut();
        if state.last_mode != Some(mode) {
            state.last_mode = Some(mode);
            state.offline = None;
            state.playback_frame = 0;
        }

        match mode {
            DriveMode::Realtime => {
                drop(state);
                backend.step(world, dt);
                DriveOutcome::Stepped
            }
            DriveMode::Offline {
                substeps,
                iterations,
                target_frames,
            } => {
                let bounded_substeps = substeps.clamp(1, MAX_OFFLINE_SUBSTEPS);
                let bounded_target = target_frames.min(MAX_DRIVER_CACHE_FRAMES);
                let offline = state.offline.get_or_insert_with(|| OfflineState {
                    target_frames: bounded_target,
                    cache: DriverCache::new(bounded_target),
                });

                if offline.cache.frame_count() >= offline.target_frames {
                    return DriveOutcome::Complete {
                        frames: offline.cache.frame_count(),
                    };
                }

                let config = OfflineStepConfig {
                    substeps: bounded_substeps,
                    iterations,
                };
                if let Err(error) = config
                    .validate_step(dt)
                    .and_then(|()| backend.step_offline(world, dt, config))
                {
                    return DriveOutcome::OfflineRejected { error };
                }
                let snapshot =
                    FullStateSnapshot::capture(offline.cache.frame_count() as u64, world);
                let frame = offline
                    .cache
                    .push(snapshot, dt, config)
                    .expect("offline cache capacity matches bounded target");
                DriveOutcome::OfflineFrame {
                    frame,
                    total_frames: offline.target_frames,
                }
            }
            DriveMode::Playback(handle) => {
                let frame = state.playback_frame;
                let total_frames = {
                    let Some(cache) = state.caches.get(&handle) else {
                        return DriveOutcome::CacheUnavailable { handle };
                    };
                    let Some(cached_frame) = cache.frame(frame) else {
                        return DriveOutcome::Complete {
                            frames: cache.frame_count(),
                        };
                    };
                    if cached_frame.version() != DRIVER_CACHE_VERSION
                        || cached_frame.index() != frame
                        || cached_frame.snapshot().frame() != u64::from(frame)
                    {
                        return DriveOutcome::CacheMismatch { frame };
                    }
                    cached_frame.snapshot().restore_into(world);
                    cache.frame_count()
                };
                state.playback_frame = state.playback_frame.saturating_add(1);
                DriveOutcome::PlaybackFrame {
                    frame,
                    total_frames,
                }
            }
        }
    }
}

impl Default for SimulationDriver {
    fn default() -> Self {
        SimulationDriver::new(DriveMode::Realtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::solver::XpbdSolver;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    struct CountingBackend {
        inner: CpuBackend,
        calls: u32,
    }

    impl CountingBackend {
        fn new() -> Self {
            Self {
                inner: CpuBackend::new(Box::new(XpbdSolver::new()), 4),
                calls: 0,
            }
        }
    }

    impl PhysicsBackend for CountingBackend {
        fn step(&mut self, world: &mut PhysicsWorld, dt: f32) {
            self.calls += 1;
            self.inner.step(world, dt);
        }

        fn step_offline(
            &mut self,
            world: &mut PhysicsWorld,
            dt: f32,
            config: OfflineStepConfig,
        ) -> Result<(), OfflineStepError> {
            self.inner.step_offline(world, dt, config)?;
            self.calls += 1;
            Ok(())
        }

        fn name(&self) -> &str {
            "counting-cpu"
        }
    }

    fn falling_world() -> (PhysicsWorld, crate::state::handle::BodyHandle) {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let handle = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        (world, handle)
    }

    #[test]
    fn realtime_steps_the_world() {
        let (mut world, h) = falling_world();
        let mut backend = CpuBackend::with_defaults();
        let driver = SimulationDriver::new(DriveMode::Realtime);
        assert!(driver.is_realtime());
        assert_eq!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Stepped
        );
        assert!(world.bodies.position(h).unwrap().y < 0.0);
    }

    #[test]
    fn offline_is_bounded_and_records_one_backend_frame_per_call() {
        let (mut world, h) = falling_world();
        let mut backend = CountingBackend::new();
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: 3,
            iterations: 8,
            target_frames: 2,
        });

        assert!(matches!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::OfflineFrame {
                frame: 0,
                total_frames: 2
            }
        ));
        assert!(matches!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::OfflineFrame {
                frame: 1,
                total_frames: 2
            }
        ));
        assert_eq!(backend.calls, 2);
        assert!(world.bodies.position(h).unwrap().y < 0.0);
        assert_eq!(driver.recorded_cache().unwrap().frame_count(), 2);
        assert!(matches!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Complete { frames: 2 }
        ));
        assert_eq!(backend.calls, 2);
    }

    #[test]
    fn offline_limits_are_clamped_without_unbounded_work() {
        let (mut world, _) = falling_world();
        let mut backend = CountingBackend::new();
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: u32::MAX,
            iterations: 1,
            target_frames: u32::MAX,
        });
        let outcome = driver.drive(&mut backend, &mut world, 0.01);
        assert!(matches!(
            outcome,
            DriveOutcome::OfflineFrame {
                frame: 0,
                total_frames: MAX_DRIVER_CACHE_FRAMES
            }
        ));
        assert_eq!(backend.calls, 1);
    }

    #[test]
    fn playback_registers_and_applies_frames_without_solving() {
        let (mut source, h) = falling_world();
        let mut offline_backend = CountingBackend::new();
        let offline = SimulationDriver::new(DriveMode::Offline {
            substeps: 2,
            iterations: 1,
            target_frames: 3,
        });
        for _ in 0..3 {
            assert!(matches!(
                offline.drive(&mut offline_backend, &mut source, 0.1),
                DriveOutcome::OfflineFrame { .. }
            ));
        }
        let cache = offline.recorded_cache().expect("offline cache");
        let expected = cache
            .frame(1)
            .expect("second frame")
            .snapshot()
            .render()
            .pose(h)
            .expect("source body in snapshot")
            .position;

        let (mut playback_world, playback_handle) = falling_world();
        assert_eq!(h, playback_handle);
        let playback = SimulationDriver::new(DriveMode::Playback(CacheHandle(7)));
        playback
            .register_cache(CacheHandle(7), cache)
            .expect("current cache version");
        let mut backend = CountingBackend::new();
        assert!(matches!(
            playback.drive(&mut backend, &mut playback_world, 999.0),
            DriveOutcome::PlaybackFrame { frame: 0, .. }
        ));
        assert!(matches!(
            playback.drive(&mut backend, &mut playback_world, 999.0),
            DriveOutcome::PlaybackFrame { frame: 1, .. }
        ));
        assert_eq!(
            playback_world.bodies.position(playback_handle),
            Some(expected)
        );
        assert_eq!(backend.calls, 0, "playback must not call the backend");
        assert!(matches!(
            playback.drive(&mut backend, &mut playback_world, 999.0),
            DriveOutcome::PlaybackFrame { frame: 2, .. }
        ));
        assert!(matches!(
            playback.drive(&mut backend, &mut playback_world, 999.0),
            DriveOutcome::Complete { frames: 3 }
        ));
    }

    #[test]
    fn playback_rejects_unknown_version_and_missing_cache_without_unsupported() {
        let playback = SimulationDriver::new(DriveMode::Playback(CacheHandle(9)));
        let err = playback
            .register_cache(CacheHandle(9), DriverCache::with_version(99, 1))
            .expect_err("unknown cache version");
        assert_eq!(
            err,
            CacheRegistrationError::UnsupportedVersion {
                expected: DRIVER_CACHE_VERSION,
                actual: 99
            }
        );
        let (mut world, _) = falling_world();
        let mut backend = CountingBackend::new();
        assert!(matches!(
            playback.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::CacheUnavailable {
                handle: CacheHandle(9)
            }
        ));
    }

    #[test]
    fn playback_invalid_metadata_is_atomic() {
        let (mut source, _) = falling_world();
        let mut offline_backend = CountingBackend::new();
        let offline = SimulationDriver::new(DriveMode::Offline {
            substeps: 1,
            iterations: 1,
            target_frames: 1,
        });
        assert!(matches!(
            offline.drive(&mut offline_backend, &mut source, 0.1),
            DriveOutcome::OfflineFrame { .. }
        ));
        let mut cache = offline.recorded_cache().expect("offline cache");
        cache.frames[0].index = 1;

        let (mut world, first) = falling_world();
        world.spawn(BodyDesc::dynamic_at(Vec3::new(10.0, 0.0, 0.0)));
        let before = world.bodies.position(first).expect("first body");
        let playback = SimulationDriver::new(DriveMode::Playback(CacheHandle(11)));
        playback
            .register_cache(CacheHandle(11), cache)
            .expect("current cache version");
        let mut backend = CountingBackend::new();
        assert_eq!(
            playback.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::CacheMismatch { frame: 0 }
        );
        assert_eq!(world.bodies.position(first), Some(before));
    }

    #[test]
    fn zero_target_completes_without_backend_work() {
        let (mut world, _) = falling_world();
        let mut backend = CountingBackend::new();
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: 4,
            iterations: 1,
            target_frames: 0,
        });
        assert_eq!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Complete { frames: 0 }
        );
        assert_eq!(backend.calls, 0);
    }
}
