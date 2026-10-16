//! Simulation drivers for realtime, bounded offline, and cache playback runs.
//!
//! [`SimulationDriver`] keeps the existing realtime entry point intact while
//! adding two stateful, bounded modes. Offline mode calls the selected backend
//! for a fixed number of substeps per frame and records [`StateSnapshot`]s.
//! Playback mode applies those snapshots one frame at a time without invoking
//! the solver. The driver cache is intentionally separate from
//! [`crate::cache::PhysicsCache`], which stores deformable particle tracks and
//! is consumed by [`crate::cache::Player`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::backend::PhysicsBackend;
use crate::snapshot::StateSnapshot;
use crate::world::PhysicsWorld;

pub mod fixed_step;

/// Version of the driver-owned snapshot cache format.
pub const DRIVER_CACHE_VERSION: u32 = 1;

/// Hard upper bound for frames retained by one driver cache.
pub const MAX_DRIVER_CACHE_FRAMES: u32 = 16_384;

/// Hard upper bound for backend calls made by one offline frame.
pub const MAX_OFFLINE_SUBSTEPS: u32 = 256;

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
        /// Sub-steps per offline frame. Values above
        /// [`MAX_OFFLINE_SUBSTEPS`] are clamped.
        substeps: u32,
        /// Solver iterations reserved for backends that expose this setting.
        /// The current [`PhysicsBackend`] contract receives only `dt`, so this
        /// value is retained as part of the stable mode configuration.
        iterations: u32,
        /// Total number of offline frames to execute and retain. Values above
        /// [`MAX_DRIVER_CACHE_FRAMES`] are clamped.
        target_frames: u32,
    },
    /// Apply one previously registered cache frame per drive call.
    Playback(CacheHandle),
}

/// A single versioned frame in a driver playback cache.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheFrame {
    version: u32,
    index: u32,
    snapshot: StateSnapshot,
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
    pub fn snapshot(&self) -> &StateSnapshot {
        &self.snapshot
    }
}

/// A bounded, versioned sequence of full-world driver snapshots.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DriverCache {
    version: u32,
    max_frames: u32,
    frames: Vec<CacheFrame>,
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

    fn push(&mut self, snapshot: StateSnapshot) -> Option<u32> {
        if self.frame_count() >= self.max_frames {
            return None;
        }
        let index = self.frame_count();
        self.frames.push(CacheFrame {
            version: self.version,
            index,
            snapshot,
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
    /// A cache frame cannot be applied to the current world layout.
    CacheMismatch {
        /// Frame that failed validation/application.
        frame: u32,
    },
    /// Retained for source compatibility with the original dispatch enum.
    /// Current Offline and Playback branches never return this variant.
    #[deprecated(note = "Offline and Playback are implemented; this is legacy compatibility only")]
    Unsupported {
        /// Human-readable reason retained for old callers.
        reason: &'static str,
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
    substeps: u32,
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
        self.state
            .borrow()
            .offline
            .as_ref()
            .map(|offline| offline.cache.clone())
    }

    /// Returns the current playback cursor, or zero for a non-playback mode.
    #[must_use]
    pub fn playback_frame(&self) -> u32 {
        self.state.borrow().playback_frame
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
                iterations: _,
                target_frames,
            } => {
                let bounded_substeps = substeps.clamp(1, MAX_OFFLINE_SUBSTEPS);
                let bounded_target = target_frames.min(MAX_DRIVER_CACHE_FRAMES);
                let offline = state.offline.get_or_insert_with(|| OfflineState {
                    substeps: bounded_substeps,
                    target_frames: bounded_target,
                    cache: DriverCache::new(bounded_target),
                });

                if offline.cache.frame_count() >= offline.target_frames {
                    return DriveOutcome::Complete {
                        frames: offline.cache.frame_count(),
                    };
                }

                let sub_dt = dt / bounded_substeps as f32;
                for _ in 0..bounded_substeps {
                    backend.step(world, sub_dt);
                }
                debug_assert_eq!(offline.substeps, bounded_substeps);
                let frame = offline
                    .cache
                    .push(StateSnapshot::capture(world))
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
                        || !apply_snapshot(world, cached_frame.snapshot())
                    {
                        return DriveOutcome::CacheMismatch { frame };
                    }
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

fn apply_snapshot(world: &mut PhysicsWorld, snapshot: &StateSnapshot) -> bool {
    for slot in 0..snapshot.slot_count() {
        let occupied = snapshot.slot_occupied(slot);
        let Some(handle) = world.bodies.handle_at_slot(slot) else {
            if occupied {
                return false;
            }
            continue;
        };
        if !occupied || snapshot.generation_at_slot(slot) != Some(handle.generation()) {
            return false;
        }
        if snapshot.pose_at_slot(slot).is_none() {
            return false;
        }
    }
    if !(snapshot.slot_count()..world.bodies.slot_count())
        .all(|slot| world.bodies.handle_at_slot(slot).is_none())
    {
        return false;
    }

    for slot in 0..snapshot.slot_count() {
        let Some(handle) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        let Some(pose) = snapshot.pose_at_slot(slot) else {
            return false;
        };
        world.bodies.set_position(handle, pose.position);
        world.bodies.set_orientation(handle, pose.orientation);
        world
            .bodies
            .set_linear_velocity(handle, snapshot.linear_velocity_at_slot(slot));
        world
            .bodies
            .set_angular_velocity(handle, snapshot.angular_velocity_at_slot(slot));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    struct CountingBackend {
        inner: CpuBackend,
        calls: u32,
    }

    impl CountingBackend {
        fn new() -> Self {
            Self {
                inner: CpuBackend::with_defaults(),
                calls: 0,
            }
        }
    }

    impl PhysicsBackend for CountingBackend {
        fn step(&mut self, world: &mut PhysicsWorld, dt: f32) {
            self.calls += 1;
            self.inner.step(world, dt);
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
    fn offline_is_bounded_records_frames_and_calls_substeps() {
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
        assert_eq!(backend.calls, 6);
        assert!(world.bodies.position(h).unwrap().y < 0.0);
        assert_eq!(driver.recorded_cache().unwrap().frame_count(), 2);
        assert!(matches!(
            driver.drive(&mut backend, &mut world, 0.1),
            DriveOutcome::Complete { frames: 2 }
        ));
        assert_eq!(backend.calls, 6);
    }

    #[test]
    fn offline_limits_are_clamped_without_unbounded_work() {
        let (mut world, _) = falling_world();
        let mut backend = CountingBackend::new();
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: u32::MAX,
            iterations: u32::MAX,
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
        assert_eq!(backend.calls, MAX_OFFLINE_SUBSTEPS);
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
    fn playback_mismatch_is_atomic() {
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
        let cache = offline.recorded_cache().expect("offline cache");

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
