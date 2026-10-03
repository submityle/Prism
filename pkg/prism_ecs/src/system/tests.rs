//! Integration tests for the system layer: param derivation, running a function
//! as a [`System`], deferred [`Commands`] application, intra-system conflict
//! rejection, [`Local`] persistence, and [`Access`] compatibility.

use alloc::vec::Vec;

use crate::component::Component;
use crate::query::Access;
use crate::resource::Resource;
use crate::command::Commands;
use crate::system::{IntoSystem, Local, Query, Res, ResMut, System};
use crate::world::World;

#[derive(Debug, PartialEq)]
struct Health(i32);
impl Component for Health {}

#[derive(Debug, PartialEq)]
struct Spawned;
impl Component for Spawned {}

#[derive(Debug, Default, PartialEq)]
struct Config(i32);
impl Resource for Config {}

#[derive(Debug, Default, PartialEq)]
struct Tally(i32);
impl Resource for Tally {}

/// A full-featured system — `Res`, `ResMut`, `Query<&mut _>`, and `Commands` —
/// resolves its params, runs, mutates component + resource state, and defers a
/// spawn that only lands after `run` flushes the command buffer.
#[test]
fn function_system_runs_with_mixed_params() {
    let mut world = World::new();
    world.insert_resource(Config(10));
    world.insert_resource(Tally(0));
    world.spawn(Health(1));
    world.spawn(Health(2));

    fn tick(cfg: Res<Config>, mut tally: ResMut<Tally>, mut q: Query<&mut Health>, mut cmd: Commands) {
        for mut h in q.iter_mut() {
            h.0 += cfg.0;
            tally.0 += 1;
        }
        cmd.spawn(Spawned);
    }

    let mut sys = IntoSystem::into_system(tick);
    sys.initialize(&mut world);
    sys.run(&mut world);

    // Query mutation applied to every matching entity.
    let mut healths: Vec<i32> = Vec::new();
    world.for_each::<&Health>(|h| healths.push(h.0));
    healths.sort_unstable();
    assert_eq!(healths, alloc::vec![11, 12]);

    // ResMut mutation visible on the world.
    assert_eq!(world.resource::<Tally>(), &Tally(2));

    // Deferred spawn landed after `run` flushed Commands.
    assert_eq!(world.query::<&Spawned>().iter(&world).count(), 1);
}

/// `Commands` effects must be deferred: nothing structural happens until the
/// buffer is applied (which `System::run` does right after the body).
#[test]
fn commands_spawn_is_deferred_until_apply() {
    use crate::system::world_cell::UnsafeWorldCell;

    let mut world = World::new();

    fn spawner(mut cmd: Commands) {
        cmd.spawn(Spawned);
        cmd.spawn(Spawned);
    }

    let mut sys = IntoSystem::into_system(spawner);
    sys.initialize(&mut world);

    // Run only the body (no apply) via the unsafe entry point: still zero
    // entities, proving the spawn was buffered rather than immediate.
    {
        let cell = UnsafeWorldCell::new_mutable(&mut world);
        // SAFETY: single system, exclusive `&mut World`, nothing else in flight.
        unsafe { sys.run_unsafe(cell) };
    }
    assert_eq!(world.entity_count(), 0);

    // Now flush the buffer: both entities appear.
    sys.apply_deferred(&mut world);
    assert_eq!(world.query::<&Spawned>().iter(&world).count(), 2);
}

/// Two params of one system that borrow the same resource mutably + immutably
/// must be rejected deterministically at `initialize` time.
#[test]
#[should_panic(expected = "resource")]
fn intra_system_resource_conflict_panics() {
    let mut world = World::new();
    world.insert_resource(Config(0));

    fn bad(_a: Res<Config>, _b: ResMut<Config>) {}

    let mut sys = IntoSystem::into_system(bad);
    sys.initialize(&mut world);
}

/// A `&mut C` query and a `&C` query over the same component in one system
/// alias and must panic at `initialize`.
#[test]
#[should_panic]
fn intra_system_query_conflict_panics() {
    let mut world = World::new();
    world.spawn(Health(0));

    fn bad(_a: Query<&mut Health>, _b: Query<&Health>) {}

    let mut sys = IntoSystem::into_system(bad);
    sys.initialize(&mut world);
}

/// `Local<T>` is per-system state that persists across runs and is independent
/// of any resource.
#[test]
fn local_state_persists_across_runs() {
    let mut world = World::new();

    fn counter(mut n: Local<u32>, mut tally: ResMut<Tally>) {
        *n += 1;
        tally.0 = *n as i32;
    }

    world.insert_resource(Tally(0));
    let mut sys = IntoSystem::into_system(counter);
    sys.initialize(&mut world);

    sys.run(&mut world);
    assert_eq!(world.resource::<Tally>(), &Tally(1));
    sys.run(&mut world);
    assert_eq!(world.resource::<Tally>(), &Tally(2));
    sys.run(&mut world);
    assert_eq!(world.resource::<Tally>(), &Tally(3));
}

/// A system that only reads a resource reports the matching access and does not
/// mutate anything.
#[test]
fn readonly_system_access_is_recorded() {
    let mut world = World::new();
    world.insert_resource(Config(5));

    fn observe(_cfg: Res<Config>) {}

    let mut sys = IntoSystem::into_system(observe);
    sys.initialize(&mut world);

    let access = sys.access();
    assert_eq!(access.resource_reads().len(), 1);
    assert_eq!(access.resource_writes().len(), 0);
    assert!(!access.writes_everything());
}

/// Compatible vs conflicting access sets, driving the eventual conflict-graph
/// executor's parallelism decisions (design §8.2).
#[test]
fn access_compatibility() {
    let mut world = World::new();
    let a = world.resources_mut().register::<Config>();
    let b = world.resources_mut().register::<Tally>();

    // read(a) vs read(a): compatible (shared).
    let mut r1 = Access::new();
    r1.add_resource_read(a);
    let mut r2 = Access::new();
    r2.add_resource_read(a);
    assert!(r1.is_compatible(&r2));

    // read(a) vs write(a): conflict.
    let mut w = Access::new();
    w.add_resource_write(a);
    assert!(!r1.is_compatible(&w));

    // write(a) vs write(b): disjoint, compatible.
    let mut wb = Access::new();
    wb.add_resource_write(b);
    assert!(w.is_compatible(&wb));

    // writes_everything conflicts with everything, including itself.
    let mut everything = Access::new();
    everything.set_writes_everything();
    assert!(!everything.is_compatible(&r1));
    assert!(!r1.is_compatible(&everything));
}

/// An exclusive `fn(&mut World)` system takes whole-world access, reports
/// `is_exclusive`, and can perform structural changes directly.
#[test]
fn exclusive_system_has_whole_world_access() {
    let mut world = World::new();

    fn setup(world: &mut World) {
        world.insert_resource(Config(99));
        world.spawn(Health(7));
    }

    let mut sys = IntoSystem::into_system(setup);
    sys.initialize(&mut world);
    assert!(sys.is_exclusive());
    assert!(sys.access().writes_everything());

    sys.run(&mut world);
    assert_eq!(world.resource::<Config>(), &Config(99));
    assert_eq!(world.query::<&Health>().iter(&world).count(), 1);
}
