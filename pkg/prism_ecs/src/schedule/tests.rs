//! Integration tests for the schedule core: insertion-order execution, chained
//! configs, tuple grouping, run-condition gating (including whole-group gating),
//! the one-shot [`run_once`] latch, fixed-phase ordering, `before`/`after`
//! against a user [`SystemSet`], set-level condition gating via
//! [`configure_set`](Schedule::configure_set), and cycle detection.

use crate::resource::Resource;
use crate::schedule::{
    resource_exists, run_once, IntoSystemConfigs, Phase, Schedule, SetConfig, SystemSet,
    SystemSetId,
};
use crate::system::ResMut;
use crate::world::World;
use alloc::vec::Vec;

/// Append-only execution log so tests can assert ordering and run counts.
#[derive(Debug, Default, PartialEq)]
struct Log(Vec<u32>);
impl Resource for Log {}

/// A gate resource whose mere presence toggles a run condition.
#[derive(Debug, Default, PartialEq)]
struct Gate;
impl Resource for Gate {}

fn push_a(mut log: ResMut<Log>) {
    log.0.push(1);
}
fn push_b(mut log: ResMut<Log>) {
    log.0.push(2);
}
fn push_c(mut log: ResMut<Log>) {
    log.0.push(3);
}

fn fresh() -> World {
    let mut world = World::new();
    world.insert_resource(Log::default());
    world
}

#[test]
fn systems_run_in_insertion_order() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a).add_systems(push_b).add_systems(push_c);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn chain_preserves_order_and_sets_flag() {
    let mut world = fresh();
    let configs = (push_a, push_b, push_c).chain();
    assert!(configs.chained(), "chain() must set the chained flag");

    let mut schedule = Schedule::new();
    schedule.add_systems(configs);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn tuple_add_systems_groups_in_order() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b, push_c));
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn run_if_resource_exists_gates_a_system() {
    // Gate absent → system skipped.
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a.run_if(resource_exists::<Gate>()));
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, Vec::<u32>::new());

    // Gate present → system runs.
    world.insert_resource(Gate);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1]);
}

#[test]
fn nested_run_if_skips_whole_group() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // The group as a whole is gated: when the gate is absent, none of a/b/c run.
    schedule.add_systems((push_a, push_b, push_c).run_if(resource_exists::<Gate>()));

    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, Vec::<u32>::new());

    world.insert_resource(Gate);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn run_once_fires_exactly_once() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a.run_if(run_once()));

    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(
        world.resource::<Log>().0,
        alloc::vec![1],
        "run_once must gate every run after the first"
    );
}


/// A user-defined [`SystemSet`] label for ordering/condition tests.
struct Physics;
impl SystemSet for Physics {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::of::<Self>()
    }
}

#[test]
fn phase_orders_systems_regardless_of_add_order() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // Add the default-phase (Update) system first and the First-phase system
    // second; phase ordering must still run First before Update.
    schedule.add_systems(push_b);
    schedule.add_systems(push_a.in_phase(Phase::First));
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);
}

#[test]
fn before_after_against_a_user_set() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // push_b anchors the Physics set; push_a must precede it, push_c follow it,
    // even though they are added out of order.
    schedule.add_systems(push_c.after(Physics));
    schedule.add_systems(push_b.in_set(Physics));
    schedule.add_systems(push_a.before(Physics));
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn configure_set_condition_gates_all_members() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b).in_set(Physics));
    schedule.configure_set(Physics, SetConfig::new().run_if(resource_exists::<Gate>()));

    // Gate absent → neither member runs.
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, Vec::<u32>::new());

    // Gate present → both members run, in order.
    world.insert_resource(Gate);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);
}

#[test]
#[should_panic(expected = "cycle")]
fn contradictory_order_panics() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // push_a is a member of Physics; push_b asks to run both before AND after
    // Physics, which is an unsatisfiable cycle (push_b → push_a → push_b).
    schedule.add_systems(push_a.in_set(Physics));
    schedule.add_systems(push_b.before(Physics).after(Physics));
    schedule.run(&mut world);
}
