//! Behavioural tests for the change-driven observer bus (design §24.2).
//!
//! Pins the two guarantees the design requires of the bus — coalescing (one
//! delivery per observer per flush) and deterministic order (registration
//! order, ascending entity index) — plus registration baselining, removal, and
//! the automatic flush at the end of a [`Schedule`](crate::schedule::Schedule)
//! run.

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::resource::Resource;
use crate::schedule::Schedule;
use crate::system::Query;
use crate::world::World;

/// Table-backed health.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Health(i32);
impl Component for Health {}

struct SparseHealth(i32);
impl Component for SparseHealth {
    const STORAGE: crate::component::StorageType = crate::component::StorageType::SparseSet;
}

/// A distinct component so a second observer can watch something else.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Shield(i32);
impl Component for Shield {}

/// One recorded delivery: the observer's slot label and the changed set.
#[derive(Debug, PartialEq)]
struct Delivery {
    label: u32,
    changed: Vec<Entity>,
}

/// Append-only delivery log so tests can assert order and coalescing.
#[derive(Debug, Default, PartialEq)]
struct Log(Vec<Delivery>);
impl Resource for Log {}

/// Advance the world one change-detection frame so writes land in a later tick
/// than the observer's baseline.
fn next_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

/// Record a delivery into the world's [`Log`] resource.
fn record(label: u32) -> impl FnMut(&mut crate::change_observers::ChangeContext<'_>) + Send + Sync {
    move |ctx| {
        let changed = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<Log>()
            .0
            .push(Delivery { label, changed });
    }
}

#[test]
fn observer_fires_once_with_a_coalesced_ascending_set() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let a = w.spawn(Health(1));
    let b = w.spawn(Health(2));

    w.observe_changed::<Health, _>(record(0));

    // Nothing changed since registration → no delivery.
    assert_eq!(w.flush_changed_observers(), 0);
    assert!(w.resource::<Log>().0.is_empty());

    next_frame(&mut w);

    // Three writes across two entities, several of them to `a`, must coalesce
    // into ONE delivery naming each entity once, ascending by index.
    w.get_mut::<Health>(a).expect("live").0 += 1;
    w.get_mut::<Health>(a).expect("live").0 += 1;
    w.get_mut::<Health>(a).expect("live").0 += 1;
    w.get_mut::<Health>(b).expect("live").0 += 1;

    assert_eq!(w.flush_changed_observers(), 1);
    let log = w.resource::<Log>();
    assert_eq!(
        log.0,
        alloc::vec![Delivery {
            label: 0,
            changed: alloc::vec![a, b],
        }]
    );
}

#[test]
fn changes_before_registration_do_not_fire() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let e = w.spawn(Health(1));
    next_frame(&mut w);
    w.get_mut::<Health>(e).expect("live").0 += 1;

    // Register *after* the change: the window baselines to "now", so the earlier
    // write is not delivered.
    w.observe_changed::<Health, _>(record(0));
    assert_eq!(w.flush_changed_observers(), 0);
    assert!(w.resource::<Log>().0.is_empty());
}

#[test]
fn table_registration_excludes_prior_writes_and_includes_immediate_writes() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let before = world.spawn(Health(0));
    let after = world.spawn(Health(0));
    let both = world.spawn(Health(0));
    world.get_mut::<Health>(before).unwrap().0 += 1;
    world.get_mut::<Health>(both).unwrap().0 += 1;
    world.observe_changed::<Health, _>(record(0));
    // No manual tick advance or intermediate flush after registration.
    world.get_mut::<Health>(after).unwrap().0 += 1;
    world.get_mut::<Health>(both).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(
        world.resource::<Log>().0,
        alloc::vec![Delivery {
            label: 0,
            changed: alloc::vec![after, both],
        }]
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

#[test]
fn sparse_registration_excludes_prior_writes_and_includes_immediate_writes() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let before = world.spawn(SparseHealth(0));
    let after = world.spawn(SparseHealth(0));
    let both = world.spawn(SparseHealth(0));
    world.get_mut::<SparseHealth>(before).unwrap().0 += 1;
    world.get_mut::<SparseHealth>(both).unwrap().0 += 1;
    world.observe_changed::<SparseHealth, _>(record(0));
    world.get_mut::<SparseHealth>(after).unwrap().0 += 1;
    world.get_mut::<SparseHealth>(both).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(
        world.resource::<Log>().0,
        alloc::vec![Delivery {
            label: 0,
            changed: alloc::vec![after, both],
        }]
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

fn populate_observer_replacement(world: &mut World) -> Entity {
    world.insert_resource(Log::default());
    let entity = world.spawn(Health(0));
    world.observe_changed::<Health, _>(record(10));
    world.observe_changed::<Health, _>(record(20));
    let source = world.reaction_source::<Health>();
    world.add_reaction(&[source], move |world| {
        let changed = world.reaction_changes(source).to_vec();
        world
            .resource_mut::<Log>()
            .0
            .push(Delivery { label: 30, changed });
    });
    world.get_mut::<Health>(entity).unwrap().0 += 1;
    entity
}

fn observer_replacement_case(populated: bool, panic_after: bool, flush: impl FnOnce(&mut World)) {
    let mut replacement = World::new();
    let mut entity = populated.then(|| populate_observer_replacement(&mut replacement));
    let replacement_id = replacement.id();
    let replacement_tick = replacement.change_tick();
    let mut replacement = Some(replacement);
    let mut world = World::new();
    let old_entity = world.spawn(Health(0));
    world.observe_changed::<Health, _>(move |ctx| {
        *ctx.world = replacement
            .take()
            .expect("old observer must not be restored");
        assert!(!panic_after, "observer replaced world");
    });
    world.observe_changed::<Health, _>(|_| panic!("old later observer ran"));
    let old_source = world.reaction_source::<Health>();
    world.add_reaction(&[old_source], |_| panic!("old reaction ran"));
    world.mark_reaction_dirty(old_source);
    world.get_mut::<Health>(old_entity).unwrap().0 += 1;
    flush(&mut world);
    assert_eq!(world.id(), replacement_id);
    assert_eq!(world.change_tick(), replacement_tick);
    if populated {
        assert!(world.resource::<Log>().0.is_empty());
    } else {
        // Empty replacements must remain usable, without stale slots or flags.
        assert_eq!(world.flush_changed_observers(), 0);
        assert_eq!(world.flush_reactions(), 0);
        entity = Some(populate_observer_replacement(&mut world));
    }
    assert_eq!(world.flush_changed_observers(), 3);
    let entity = entity.unwrap();
    assert_eq!(
        world.resource::<Log>().0,
        [10, 20, 30]
            .into_iter()
            .map(|label| Delivery {
                label,
                changed: alloc::vec![entity],
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

#[test]
fn replacing_world_discards_old_observers_and_preserves_replacement_registries() {
    for populated in [false, true] {
        observer_replacement_case(populated, false, |world| {
            assert_eq!(world.flush_changed_observers(), 1);
        });
    }
}

#[cfg(feature = "std")]
#[test]
fn replacing_world_then_panicking_does_not_restore_old_observer_slots() {
    for populated in [false, true] {
        observer_replacement_case(populated, true, |world| {
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                world.flush_changed_observers();
            }))
            .expect_err("replacement callback must panic");
            assert_eq!(
                error.downcast_ref::<&str>(),
                Some(&"observer replaced world")
            );
        });
    }
}

#[test]
fn observers_fire_in_registration_order() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let e = w.spawn((Health(1), Shield(2)));

    w.observe_changed::<Health, _>(record(10));
    w.observe_changed::<Shield, _>(record(20));
    next_frame(&mut w);

    w.get_mut::<Health>(e).expect("live").0 += 1;
    w.get_mut::<Shield>(e).expect("live").0 += 1;

    assert_eq!(w.flush_changed_observers(), 2);
    let labels: Vec<u32> = w.resource::<Log>().0.iter().map(|d| d.label).collect();
    assert_eq!(labels, [10, 20]);
}

#[test]
fn removed_observer_stops_firing() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let e = w.spawn(Health(1));

    let id = w.observe_changed::<Health, _>(record(0));
    assert!(w.remove_change_observer(id));
    assert!(!w.remove_change_observer(id));

    next_frame(&mut w);
    w.get_mut::<Health>(e).expect("live").0 += 1;
    assert_eq!(w.flush_changed_observers(), 0);
    assert!(w.resource::<Log>().0.is_empty());
}

#[test]
fn a_callback_change_lands_in_the_next_flush() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let e = w.spawn(Health(1));

    // The callback writes to the same component it observes.
    w.observe_changed::<Health, _>(move |ctx| {
        let changed = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<Log>()
            .0
            .push(Delivery { label: 0, changed });
        ctx.world.get_mut::<Health>(e).expect("live").0 += 1;
    });

    next_frame(&mut w);
    w.get_mut::<Health>(e).expect("live").0 += 1;

    // First flush delivers the initial change; the callback's own write is
    // scheduled for the next window, not re-delivered within this one.
    assert_eq!(w.flush_changed_observers(), 1);
    assert_eq!(w.resource::<Log>().0.len(), 1);
    assert_eq!(w.resource::<Log>().0[0].changed, alloc::vec![e]);

    next_frame(&mut w);
    assert_eq!(w.flush_changed_observers(), 1);
    assert_eq!(w.resource::<Log>().0.len(), 2);
}

fn bump_health(mut q: Query<&mut Health>) {
    for mut h in &mut q {
        h.0 += 1;
    }
}

#[test]
fn schedule_run_flushes_change_observers() {
    let mut w = World::new();
    w.insert_resource(Log::default());
    let a = w.spawn(Health(1));
    let b = w.spawn(Health(2));
    w.observe_changed::<Health, _>(record(0));

    let mut schedule = Schedule::new();
    schedule.add_systems(bump_health);
    next_frame(&mut w);
    schedule.run(&mut w);

    assert_eq!(
        w.resource::<Log>().0,
        alloc::vec![Delivery {
            label: 0,
            changed: alloc::vec![a, b],
        }]
    );
}

#[test]
fn structural_spawn_and_insert_are_indexed_observer_sources() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    world.observe_changed::<Health, _>(record(0));

    let spawned = world.spawn(Health(1));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.resource::<Log>().0[0].changed, alloc::vec![spawned]);

    let inserted = world.spawn(());
    assert!(world.insert(inserted, Health(2)));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.resource::<Log>().0[1].changed, alloc::vec![inserted]);
}

#[test]
fn an_observer_can_remove_itself_and_recursive_flush_does_not_reenter() {
    struct OwnObserver(crate::observer::ObserverId);
    impl Resource for OwnObserver {}
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entity = world.spawn(Health(1));
    let id = world.observe_changed::<Health, _>(move |ctx| {
        let id = ctx.world.resource::<OwnObserver>().0;
        assert!(ctx.world.remove_change_observer(id));
        assert!(!ctx.world.remove_change_observer(id));
        assert_eq!(ctx.world.flush_changed_observers(), 0);
        let changed = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<Log>()
            .0
            .push(Delivery { label: 0, changed });
    });
    world.insert_resource(OwnObserver(id));
    next_frame(&mut world);
    world.get_mut::<Health>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    next_frame(&mut world);
    world.get_mut::<Health>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.resource::<Log>().0.len(), 1);
}

#[test]
fn later_observers_receive_the_batch_collected_before_callback_mutation() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entity = world.spawn(Health(1));
    world.observe_changed::<Health, _>(move |ctx| {
        ctx.world.remove::<Health>(entity);
        assert_eq!(ctx.world.flush_changed_observers(), 0);
    });
    world.observe_changed::<Health, _>(record(1));
    next_frame(&mut world);
    world.get_mut::<Health>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(world.resource::<Log>().0[0].changed, alloc::vec![entity]);
}

#[cfg(feature = "std")]
#[test]
fn a_panicking_observer_is_preserved_and_retried_without_replaying_successes() {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entity = world.spawn(Health(1));
    world.observe_changed::<Health, _>(record(0));
    let panic_once = Arc::new(AtomicBool::new(true));
    world.observe_changed::<Health, _>(move |ctx| {
        assert!(
            !panic_once.swap(false, Ordering::Relaxed),
            "change callback failed"
        );
        record(1)(ctx);
    });
    world.observe_changed::<Health, _>(record(2));
    next_frame(&mut world);
    world.get_mut::<Health>(entity).unwrap().0 += 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert!(
        !world.change_observers.flushing,
        "panic unwind must release the change-observer flush barrier"
    );
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(
        world
            .resource::<Log>()
            .0
            .iter()
            .map(|batch| batch.label)
            .collect::<Vec<_>>(),
        alloc::vec![0, 1, 2]
    );
    assert_eq!(world.flush_changed_observers(), 0);
}
