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
