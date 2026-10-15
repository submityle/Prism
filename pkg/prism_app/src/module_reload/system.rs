//! Host-owned schedule binding whose access never changes during reload.

use std::sync::{Arc, Mutex};

use prism_ecs::change::Tick;
use prism_ecs::query::Access;
use prism_ecs::system::{IntoSystem, System, UnsafeWorldCell};
use prism_ecs::world::{World, WorldId};

use super::{lock, Runtime};

/// Stable exclusive Schedule node. Construct with `ModuleReload::system`, then
/// pass it to `Schedule::add_systems` or its ordinary configuration combinators.
pub struct ModuleSystem {
    id: u64,
    name: String,
    runtime: Arc<Mutex<Runtime>>,
    access: Access,
    world: Option<WorldId>,
    last_run: Tick,
}

impl ModuleSystem {
    pub(super) fn new(id: u64, runtime: Arc<Mutex<Runtime>>) -> Self {
        let mut access = Access::new();
        access.set_writes_everything();
        Self {
            id,
            name: format!("module system {id}"),
            runtime,
            access,
            world: None,
            last_run: Tick::ZERO,
        }
    }
}

impl System for ModuleSystem {
    type Out = ();
    fn name(&self) -> &str {
        &self.name
    }
    fn initialize(&mut self, world: &mut World) {
        self.validate_world(world);
        lock(&self.runtime)
            .expect("module runtime poisoned")
            .bind_world(world);
        self.world = Some(world.id());
    }
    fn validate_world(&self, world: &World) {
        if let Some(id) = self.world {
            assert_eq!(
                id,
                world.id(),
                "module system initialized for another World"
            );
        }
    }
    fn access(&self) -> &Access {
        &self.access
    }
    fn is_exclusive(&self) -> bool {
        true
    }
    fn get_last_run(&self) -> Tick {
        self.last_run
    }
    fn set_last_run(&mut self, tick: Tick) {
        self.last_run = tick;
    }
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) {
        // SAFETY: this system declares writes_everything and is_exclusive, so
        // no other system can alias this whole-world borrow during invocation.
        let world = unsafe { world.world_mut() };
        self.validate_world(world);
        let mut runtime = lock(&self.runtime).expect("module runtime poisoned");
        if let Err(error) = runtime.run(self.id, world) {
            runtime.record_failure(self.id, error);
        }
    }
    fn apply_deferred(&mut self, _world: &mut World) {}
}

/// Identity-conversion marker for configuring [`ModuleSystem`] in a Schedule.
pub struct IsModuleSystem;

impl IntoSystem<(), IsModuleSystem> for ModuleSystem {
    type System = Self;
    fn into_system(self) -> Self::System {
        self
    }
}
