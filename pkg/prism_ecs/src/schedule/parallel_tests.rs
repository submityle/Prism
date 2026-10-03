//! Tests for the multi-threaded conflict-graph executor (§8.2), gated behind the
//! `multi_thread` feature.
//!
//! These cover both the *analysis* (wave partitioning: compatible systems share
//! a wave, conflicting systems are serialised, exclusive systems run alone) and
//! the *execution* (parallel result equals the single-threaded executor,
//! run-conditions gate, deferred commands flush at the wave boundary, and the
//! partition is deterministic across repeats and thread counts).

use alloc::vec::Vec;

use prism_tasks::TaskPool;

use crate::command::Commands;
use crate::component::Component;
use crate::resource::Resource;
use crate::schedule::parallel_executor::{compute_waves, effective_accesses};
use crate::schedule::{ambiguity, resource_exists, IntoSystemConfigs, Schedule};
use crate::system::ResMut;
use crate::world::World;

/// Append-only execution log for asserting order and run counts.
#[derive(Debug, Default, PartialEq)]
struct Log(Vec<u32>);
impl Resource for Log {}

/// Two disjoint counters so systems touching one never conflict with the other.
#[derive(Debug, Default)]
struct CounterA(u32);
impl Resource for CounterA {}

#[derive(Debug, Default)]
struct CounterB(u32);
impl Resource for CounterB {}

/// Presence toggles a run condition.
#[derive(Debug, Default, PartialEq)]
struct Gate;
impl Resource for Gate {}

/// Trivial component spawned by the deferred-command tests.
#[derive(Debug)]
struct Spawned;
impl Component for Spawned {}

fn push_a(mut log: ResMut<Log>) {
    log.0.push(1);
}
fn push_b(mut log: ResMut<Log>) {
    log.0.push(2);
}
fn push_c(mut log: ResMut<Log>) {
    log.0.push(3);
}

fn bump_a(mut a: ResMut<CounterA>) {
    a.0 += 1;
}
fn bump_b(mut b: ResMut<CounterB>) {
    b.0 += 1;
}

fn fresh() -> World {
    let mut world = World::new();
    world.insert_resource(Log::default());
    world
}

fn fresh_counters() -> World {
    let mut world = World::new();
    world.insert_resource(CounterA::default());
    world.insert_resource(CounterB::default());
    world
}

/// White-box reconstruction of the wave partition the executor computes, so
/// tests can assert the conflict-graph analysis directly. Mirrors the prefix of
/// [`MultiThreadedExecutor::run`](crate::schedule::parallel_executor::MultiThreadedExecutor::run).
fn waves_of(schedule: &mut Schedule, world: &mut World) -> Vec<Vec<usize>> {
    schedule.initialize(world);
    let n = schedule.nodes.len();
    let effective = effective_accesses(schedule);
    let edges: Vec<(usize, usize)> = schedule.build_edges().into_iter().collect();
    let reach = ambiguity::reachability(&edges, n);
    let order = schedule.order.clone();
    compute_waves(&order, &effective, &reach)
}

#[test]
fn parallel_matches_sequential_for_chain() {
    let pool = TaskPool::with_threads(4);
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b, push_c).chain());
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn parallel_equals_single_threaded_executor() {
    let pool = TaskPool::with_threads(4);

    let mut w1 = fresh();
    let mut s1 = Schedule::new();
    s1.add_systems((push_a, push_b, push_c).chain());
    s1.run(&mut w1);

    let mut w2 = fresh();
    let mut s2 = Schedule::new();
    s2.add_systems((push_a, push_b, push_c).chain());
    s2.run_parallel(&pool, &mut w2);

    assert_eq!(w1.resource::<Log>().0, w2.resource::<Log>().0);
}

#[test]
fn compatible_systems_share_a_wave() {
    let mut world = fresh_counters();
    let mut schedule = Schedule::new();
    schedule.add_systems((bump_a, bump_b));
    let waves = waves_of(&mut schedule, &mut world);
    assert_eq!(waves.len(), 1, "disjoint-access systems must co-schedule");
    assert_eq!(waves[0].len(), 2);
}

#[test]
fn compatible_systems_both_execute_in_one_wave() {
    let pool = TaskPool::with_threads(4);
    let mut world = fresh_counters();
    let mut schedule = Schedule::new();
    schedule.add_systems((bump_a, bump_b));
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<CounterA>().0, 1);
    assert_eq!(world.resource::<CounterB>().0, 1);
}

#[test]
fn conflicting_systems_split_into_ordered_waves() {
    // Both write `Log`, so even without an explicit ordering edge the executor
    // must place them in separate waves (serialised by topological order).
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b));
    let waves = waves_of(&mut schedule, &mut world);
    assert_eq!(waves.len(), 2, "conflicting systems cannot share a wave");
    assert_eq!(waves[0].len(), 1);
    assert_eq!(waves[1].len(), 1);
}

#[test]
fn exclusive_system_occupies_its_own_wave() {
    fn exclusive_noop(_world: &mut World) {}

    let mut world = fresh_counters();
    let mut schedule = Schedule::new();
    // Unordered: the two counter systems are mutually compatible, but the
    // whole-world exclusive system is compatible with nothing.
    schedule.add_systems((bump_a, exclusive_noop, bump_b));
    let waves = waves_of(&mut schedule, &mut world);
    assert_eq!(waves.len(), 2);
    let lens: Vec<usize> = waves.iter().map(Vec::len).collect();
    assert!(
        lens.contains(&2) && lens.contains(&1),
        "the compatible pair shares a wave; the exclusive system is alone: {lens:?}"
    );
}

#[test]
fn exclusive_system_executes_via_run_parallel() {
    fn exclusive_spawn(world: &mut World) {
        world.spawn(Spawned);
    }

    let pool = TaskPool::with_threads(4);
    let mut world = World::new();
    let mut schedule = Schedule::new();
    schedule.add_systems(exclusive_spawn);
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.entity_count(), 1);
}

#[test]
fn run_if_gates_a_system_in_parallel() {
    let pool = TaskPool::with_threads(2);
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a.run_if(resource_exists::<Gate>()));

    // Gate absent → skipped.
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Log>().0, Vec::<u32>::new());

    // Gate present → runs.
    world.insert_resource(Gate);
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1]);
}

#[test]
fn deferred_commands_flush_at_wave_boundary() {
    fn spawner(mut commands: Commands) {
        commands.spawn(Spawned);
    }
    // Exclusive, chained after `spawner`: it runs in a later wave, so the
    // spawner's deferred spawn has already flushed at the wave boundary.
    fn observe_count(world: &mut World) {
        let n = world.entity_count();
        world.resource_mut::<Log>().0.push(n);
    }

    let pool = TaskPool::with_threads(2);
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((spawner, observe_count).chain());
    schedule.run_parallel(&pool, &mut world);

    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1]);
}

#[test]
fn parallel_run_is_deterministic_across_repeats() {
    let pool = TaskPool::with_threads(4);
    for _ in 0..32 {
        let mut world = fresh();
        let mut schedule = Schedule::new();
        schedule.add_systems((push_a, push_b, push_c).chain());
        schedule.run_parallel(&pool, &mut world);
        assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
    }
}

#[test]
fn single_threaded_pool_runs_correctly() {
    let pool = TaskPool::with_threads(1);
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b, push_c).chain());
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}
