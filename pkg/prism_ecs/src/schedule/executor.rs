//! The runnable [`Schedule`] and the single-threaded executor that drives it.
//!
//! A [`Schedule`] owns an ordered list of [`SystemConfigs`] root nodes (one per
//! `add_systems` call). Running it lazily initializes every system, then hands
//! the roots to [`SingleThreadedExecutor`], which walks them in insertion order
//! and runs each (honouring run conditions and applying deferred commands via
//! each system's `run`).
//!
//! # Honestly deferred
//!
//! The parallel, conflict-graph executor and fiber job graph (design §8.2–§8.3)
//! are deferred until `prism_tasks` lands; this module runs strictly
//! single-threaded. `SystemSet`s and run-conditions-expressed-as-systems are
//! also future work. None of that is stubbed here — the pieces are simply
//! absent, and the per-system `Access` recorded by the system layer is already
//! sufficient to drop a parallel executor in later without touching this API.

use crate::schedule::config::{IntoSystemConfigs, SystemConfigs};
use crate::world::World;
use alloc::vec::Vec;

/// An ordered collection of configured systems that can be run against a
/// [`World`].
///
/// Add work with [`add_systems`](Schedule::add_systems); run it with
/// [`run`](Schedule::run). Initialization is lazy and idempotent: the first run
/// (or an explicit [`initialize`](Schedule::initialize)) builds each system's
/// parameter state, and adding more systems re-arms initialization for the new
/// nodes.
#[derive(Default)]
pub struct Schedule {
    configs: Vec<SystemConfigs>,
    initialized: bool,
}

impl Schedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            initialized: false,
        }
    }

    /// Append systems to this schedule.
    ///
    /// Accepts a single system, a tuple of systems, or any pre-built
    /// [`SystemConfigs`] (so `.chain()` / `.run_if(..)` results work directly).
    /// Adding systems marks the schedule uninitialized so the new nodes are
    /// initialized on the next run.
    pub fn add_systems<Marker>(&mut self, systems: impl IntoSystemConfigs<Marker>) -> &mut Self {
        self.configs.push(systems.into_configs());
        self.initialized = false;
        self
    }

    /// Initialize every not-yet-initialized system against `world`.
    ///
    /// Idempotent: a no-op once initialized until more systems are added.
    pub fn initialize(&mut self, world: &mut World) {
        if self.initialized {
            return;
        }
        for config in self.configs.iter_mut() {
            config.initialize(world);
        }
        self.initialized = true;
    }

    /// Run every configured system once, in insertion order, honouring run
    /// conditions. Initializes lazily if needed.
    pub fn run(&mut self, world: &mut World) {
        self.initialize(world);
        SingleThreadedExecutor::run(&mut self.configs, world);
    }
}

/// The default executor: runs configuration roots sequentially on the calling
/// thread, in insertion order.
///
/// This is a zero-sized dispatcher rather than a stored strategy so a
/// [`Schedule`] stays trivially movable; a future parallel executor will be a
/// sibling type selected by the schedule.
pub struct SingleThreadedExecutor;

impl SingleThreadedExecutor {
    /// Run each root configuration in order against `world`.
    pub fn run(configs: &mut [SystemConfigs], world: &mut World) {
        for config in configs.iter_mut() {
            config.run(world);
        }
    }
}
