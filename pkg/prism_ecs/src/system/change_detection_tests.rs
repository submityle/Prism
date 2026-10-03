//! System-level change-detection tests: proving the executor threads a
//! *per-system* change window through [`System::run`](crate::system::System::run)
//! (design §10, M2 commit C2).
//!
//! Unlike the query-level tests (which drive `World` ticks by hand), these run
//! real [`FunctionSystem`](crate::system::function::FunctionSystem)s repeatedly
//! and assert that `Added`/`Changed` filters observe exactly the writes made
//! since *that system* last ran — the window is owned by the system, not the
//! world.

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Added, Changed};
use crate::resource::Resource;
use crate::system::{IntoSystem, Local, Query, ResMut, System};
use crate::world::World;

#[derive(Debug, PartialEq, Clone, Copy)]
struct A(i32);
impl Component for A {}

/// Append-only record of how many entities each run observed.
#[derive(Debug, Default, PartialEq)]
struct Seen(Vec<usize>);
impl Resource for Seen {}

/// `Added<A>` seen through repeated `System::run` matches only on the run whose
/// per-system window still contains the spawn tick; subsequent runs advance the
/// system's `last_run` past it and see nothing.
#[test]
fn added_filter_through_system_run_matches_only_spawn_frame() {
    let mut world = World::new();
    world.insert_resource(Seen::default());
    world.spawn(A(1));
    world.spawn(A(2));

    fn observe(q: Query<Entity, Added<A>>, mut seen: ResMut<Seen>) {
        seen.0.push(q.iter().count());
    }

    let mut sys = IntoSystem::into_system(observe);
    sys.initialize(&mut world);
    sys.run(&mut world);
    sys.run(&mut world);
    sys.run(&mut world);

    assert_eq!(
        world.resource::<Seen>().0,
        alloc::vec![2, 0, 0],
        "Added matches only the first run (spawn still inside its window)"
    );
}

/// A system that mutates `A` through a `Mut` item sees its *own* write as
/// `Changed` on the next run, then — with no further writes — the window
/// advances and the change goes stale. This exercises the per-system `last_run`
/// hand-off across runs.
#[test]
fn changed_filter_through_system_run_goes_stale_after_write() {
    let mut world = World::new();
    world.insert_resource(Seen::default());
    world.spawn(A(1));

    // Writes on the first run only; later runs leave `A` untouched so the
    // observer's window should slide past the write.
    fn mutate_once(mut q: Query<&mut A>, mut done: Local<bool>) {
        if !*done {
            for mut a in q.iter_mut() {
                a.0 += 10;
            }
            *done = true;
        }
    }

    fn observe(q: Query<Entity, Changed<A>>, mut seen: ResMut<Seen>) {
        seen.0.push(q.iter().count());
    }

    let mut writer = IntoSystem::into_system(mutate_once);
    let mut observer = IntoSystem::into_system(observe);
    writer.initialize(&mut world);
    observer.initialize(&mut world);

    // Run 1: writer stamps `A` changed; observer (ran after) sees it.
    writer.run(&mut world);
    observer.run(&mut world);
    // Run 2: writer is a no-op; observer's window has moved past the write.
    writer.run(&mut world);
    observer.run(&mut world);
    // Run 3: still stale.
    writer.run(&mut world);
    observer.run(&mut world);

    assert_eq!(
        world.resource::<Seen>().0,
        alloc::vec![1, 0, 0],
        "Changed is observed once, then goes stale as last_run advances"
    );
}

/// The spawn tick itself counts as "changed": a brand-new component is both
/// `Added` and `Changed` on the first run that observes it.
#[test]
fn spawn_is_changed_on_first_system_run() {
    let mut world = World::new();
    world.insert_resource(Seen::default());
    world.spawn(A(7));

    fn observe(q: Query<Entity, Changed<A>>, mut seen: ResMut<Seen>) {
        seen.0.push(q.iter().count());
    }

    let mut sys = IntoSystem::into_system(observe);
    sys.initialize(&mut world);
    sys.run(&mut world);
    sys.run(&mut world);

    assert_eq!(world.resource::<Seen>().0, alloc::vec![1, 0]);
}
