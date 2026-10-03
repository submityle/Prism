//! Exclusive systems: plain functions that take `&mut World`.
//!
//! Some work needs whole-world access that cannot be expressed as a set of
//! disjoint [`SystemParam`](crate::system::SystemParam)s — structural changes,
//! inserting resources, running a sub-schedule, draining events, etc. An
//! *exclusive* system receives the entire `&mut World` and therefore can never
//! run in parallel with anything else; its [`access`](System::access) is marked
//! as writing the whole world (via
//! [`Access::set_writes_everything`](crate::query::Access::set_writes_everything))
//! and [`is_exclusive`](System::is_exclusive) returns `true`.

use core::any::type_name;

use crate::query::Access;
use crate::system::function::{IntoSystem, System};
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::World;

/// A system backed by a `FnMut(&mut World)`.
pub struct ExclusiveFunctionSystem<F> {
    func: F,
    access: Access,
    name: &'static str,
}

impl<F> ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut World) + Send + Sync + 'static,
{
    /// Wrap `func`, naming the system after its Rust type and marking it
    /// exclusive (writes the whole world).
    #[inline]
    pub fn new(func: F) -> Self {
        let mut access = Access::new();
        access.set_writes_everything();
        Self {
            func,
            access,
            name: type_name::<F>(),
        }
    }
}

impl<F> System for ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut World) + Send + Sync + 'static,
{
    type Out = ();

    #[inline]
    fn name(&self) -> &str {
        self.name
    }

    #[inline]
    fn initialize(&mut self, _world: &mut World) {}

    #[inline]
    fn access(&self) -> &Access {
        &self.access
    }

    #[inline]
    fn is_exclusive(&self) -> bool {
        true
    }

    #[inline]
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) {
        // SAFETY: an exclusive system is run by the scheduler with no other
        // system in flight and the cell derived from a `&mut World`, so this is
        // the unique live borrow of the world.
        let world: &mut World = unsafe { world.world_mut() };
        (self.func)(world);
    }

    #[inline]
    fn apply_deferred(&mut self, _world: &mut World) {}

    #[inline]
    fn run(&mut self, world: &mut World) {
        (self.func)(world);
    }
}

/// Marker for the exclusive-system `IntoSystem` impl.
pub struct IsExclusiveSystem;

impl<F> IntoSystem<(), IsExclusiveSystem> for F
where
    F: FnMut(&mut World) + Send + Sync + 'static,
{
    type System = ExclusiveFunctionSystem<F>;

    #[inline]
    fn into_system(self) -> Self::System {
        ExclusiveFunctionSystem::new(self)
    }
}
