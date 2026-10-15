//! Public executor panic-isolation and deferred-reservation regressions.
#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::vec::Vec;

use prism_ecs::command::Commands;
use prism_ecs::component::Component;
use prism_ecs::resource::Resource;
use prism_ecs::schedule::{
    IntoSystemConfigs, Schedule, SetConfig, StepReport, Stepping, SystemFailureStage,
    SystemFailures, SystemPanicPolicy, SystemSet, SystemSetId,
};
use prism_ecs::system::ResMut;
use prism_ecs::world::World;

#[derive(Default)]
struct Counter(usize);
impl Resource for Counter {}
fn bump(mut counter: ResMut<Counter>) {
    counter.0 += 1;
}
fn fail_after_queue(mut commands: Commands) {
    commands.queue(|world| world.resource_mut::<Counter>().0 += 1000);
    panic!("body failure");
}
fn fail_in_deferred(mut commands: Commands) {
    commands.queue(|world| world.resource_mut::<Counter>().0 += 10);
    commands.queue(|_| panic!("deferred failure"));
    commands.queue(|world| world.resource_mut::<Counter>().0 += 1000);
}
fn world() -> World {
    let mut world = World::new();
    world.insert_resource(Counter::default());
    world.insert_resource(SystemPanicPolicy::DisableAndContinue);
    world
}

struct Spawned;
impl Component for Spawned {}

#[derive(prism_ecs::prelude::SystemParam)]
struct NestedCommands<'w, 's> {
    commands: Commands<'w, 's>,
}

#[derive(prism_ecs::prelude::SystemParam)]
struct DerivedCommands<'w, 's>(NestedCommands<'w, 's>);

#[test]
fn derived_params_discard_nested_command_reservations() {
    fn fail(mut params: DerivedCommands) {
        params.0.commands.spawn(Spawned);
        panic!("derived param failure");
    }
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail, spawn_successfully).chain());
    schedule.run(&mut world);
    assert_eq!(world.entities().len(), 1);
    assert_eq!(world.entity_count(), 1);
    schedule.run(&mut world);
    assert_eq!(world.entities().len(), 2);
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn budgeted_labeled_schedule_is_restored_after_unwind() {
    use prism_ecs::schedule::Schedules;
    let mut world = world();
    world.insert_resource(SystemPanicPolicy::FailFast);
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, bump).chain());
    let mut schedules = Schedules::new();
    schedules.insert(Main, schedule);
    world.insert_resource(schedules);
    let budget = std::time::Duration::from_secs(1);
    assert!(catch_unwind(AssertUnwindSafe(
        || world.run_schedule_budgeted(Main, budget)
    ))
    .is_err());
    assert!(world.resource::<Schedules>().contains(Main));
    world.insert_resource(SystemPanicPolicy::DisableAndContinue);
    assert_eq!(world.run_schedule_budgeted(Main, budget), 0);
    assert_eq!(world.run_schedule_budgeted(Main, budget), 0);
    assert_eq!(world.resource::<Counter>().0, 2);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_budgeted_labeled_schedule_is_restored_after_unwind() {
    use prism_ecs::schedule::Schedules;
    let mut world = world();
    world.insert_resource(SystemPanicPolicy::FailFast);
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, bump).chain());
    let mut schedules = Schedules::new();
    schedules.insert(Main, schedule);
    world.insert_resource(schedules);
    let budget = std::time::Duration::from_secs(1);
    let pool = prism_tasks::TaskPool::with_threads(2);
    assert!(catch_unwind(AssertUnwindSafe(
        || world.run_schedule_parallel_budgeted(Main, &pool, budget)
    ))
    .is_err());
    assert!(world.resource::<Schedules>().contains(Main));
    world.insert_resource(SystemPanicPolicy::DisableAndContinue);
    assert_eq!(world.run_schedule_parallel_budgeted(Main, &pool, budget), 0);
    assert_eq!(world.run_schedule_parallel_budgeted(Main, &pool, budget), 0);
    assert_eq!(world.resource::<Counter>().0, 2);
}

// Adapt prebuilt condition systems through the existing IntoSystem contract.
struct Gate<S>(S);
impl<S: prism_ecs::system::System<Out = bool>> prism_ecs::system::IntoSystem<bool, ()> for Gate<S> {
    type System = S;
    fn into_system(self) -> S {
        self.0
    }
}

fn reserve_then_fail(mut commands: Commands) {
    commands.spawn(Spawned);
    panic!("reserved spawn");
}

fn spawn_successfully(mut commands: Commands) {
    commands.spawn(Spawned);
}

fn spawn_around_deferred_failure(mut commands: Commands) {
    commands.spawn(Spawned);
    commands.queue(|_| panic!("between spawns"));
    commands.spawn(Spawned);
}

#[test]
fn isolation_releases_only_unapplied_spawn_reservations() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems(
        (
            reserve_then_fail,
            spawn_around_deferred_failure,
            spawn_successfully,
        )
            .chain(),
    );
    schedule.run(&mut world);
    world.flush_reserved();
    assert_eq!(world.entity_count(), 2);
    assert_eq!(world.entities().len(), 2);
    schedule.run(&mut world);
    assert_eq!(world.entity_count(), 3);
    assert_eq!(world.entities().len(), 3);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_failed_reservation_does_not_cancel_successful_wave_spawn() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((reserve_then_fail, spawn_successfully));
    let pool = prism_tasks::TaskPool::with_threads(2);
    schedule.run_parallel(&pool, &mut world);
    world.flush_reserved();
    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.entities().len(), 1);
}

#[test]
fn serial_retires_failed_system_and_drops_buffered_commands() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, bump).chain());
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 2);
    assert_eq!(schedule.is_system_disabled(0), Some(true));
    assert_eq!(schedule.is_system_disabled(1), Some(false));
    assert_eq!(schedule.is_system_disabled(2), None);
    let failures = world.resource::<SystemFailures>();
    assert_eq!(failures.total(), 1);
    assert_eq!(failures.iter().next().unwrap().message, "body failure");
}

#[test]
fn deferred_panic_does_not_rollback_prior_writes_or_replay_remaining_commands() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_in_deferred, bump).chain());
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 12);
    assert_eq!(world.resource::<SystemFailures>().total(), 1);
}

#[test]
fn default_and_explicit_fail_fast_propagate() {
    for explicit in [false, true] {
        let mut world = World::new();
        if explicit {
            world.insert_resource(SystemPanicPolicy::FailFast);
        }
        let mut schedule = Schedule::new();
        schedule.add_systems(|| panic!("fail fast"));
        assert!(catch_unwind(AssertUnwindSafe(|| schedule.run(&mut world))).is_err());
        assert_eq!(schedule.is_system_disabled(0), Some(false));
        assert!(!world.contains_resource::<SystemFailures>());
    }
}

#[test]
fn wrong_world_condition_remains_fail_fast_even_when_body_is_gated_off() {
    struct Group;
    impl SystemSet for Group {
        fn set_id(&self) -> SystemSetId {
            SystemSetId::of::<Self>()
        }
    }
    for shared in [false, true] {
        let mut original = world();
        let mut other = world();
        let mut schedule = Schedule::new();
        if shared {
            schedule.configure_set(Group, SetConfig::default().run_if(|| false));
            schedule.add_systems(bump.in_set(Group));
        } else {
            schedule.add_systems(bump.run_if(|| false));
        }
        schedule.initialize(&mut original);
        assert!(catch_unwind(AssertUnwindSafe(|| schedule.run(&mut other))).is_err());
        assert!(!other.contains_resource::<SystemFailures>());
        assert_eq!(other.resource::<Counter>().0, 0);
        assert_eq!(schedule.is_system_disabled(0), Some(false));
    }
}

#[test]
fn wrapped_conditions_validate_every_inner_world_before_running() {
    use prism_ecs::schedule::{and, not};
    use prism_ecs::system::System;
    let mut original = world();
    let mut other = world();
    let mut inverted = not(|| false);
    inverted.initialize(&mut original);
    assert!(catch_unwind(AssertUnwindSafe(|| inverted.validate_world(&other))).is_err());

    let mut combined = and(|| true, || false);
    combined.initialize(&mut original);
    assert!(catch_unwind(AssertUnwindSafe(|| combined.validate_world(&other))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| combined.run(&mut other))).is_err());
}

#[test]
fn budgeted_run_isolates_failure_and_defers_only_background_work() {
    use core::time::Duration;
    use prism_ecs::schedule::{SingleThreadedExecutor, SystemLane};
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems(
        (
            fail_after_queue.lane(SystemLane::Critical),
            bump.lane(SystemLane::Normal),
            bump.lane(SystemLane::Background),
        )
            .chain(),
    );
    schedule.initialize(&mut world);
    assert_eq!(
        SingleThreadedExecutor::run_with_budget(&mut schedule, &mut world, Duration::ZERO),
        1
    );
    assert_eq!(world.resource::<Counter>().0, 1);
    assert_eq!(
        SingleThreadedExecutor::run_with_budget(&mut schedule, &mut world, Duration::from_secs(60)),
        0
    );
    assert_eq!(world.resource::<Counter>().0, 3);
    assert_eq!(world.resource::<SystemFailures>().total(), 1);
}

#[test]
fn private_condition_is_retired_once_and_later_systems_continue() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let mut schedule = Schedule::new();
    schedule.add_systems(
        (
            bump.run_if(move || -> bool {
                observed.fetch_add(1, Ordering::Relaxed);
                panic!("gate");
            }),
            bump,
        )
            .chain(),
    );
    let mut world = world();
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(world.resource::<Counter>().0, 2);
    assert_eq!(
        world
            .resource::<SystemFailures>()
            .iter()
            .next()
            .unwrap()
            .stage,
        SystemFailureStage::Condition
    );
}

#[test]
fn shared_condition_stays_closed_for_all_members_after_failure() {
    struct Group;
    impl SystemSet for Group {
        fn set_id(&self) -> SystemSetId {
            SystemSetId::of::<Self>()
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let mut schedule = Schedule::new();
    schedule.configure_set(
        Group,
        SetConfig::default().run_if(move || -> bool {
            observed.fetch_add(1, Ordering::Relaxed);
            panic!("set gate");
        }),
    );
    schedule.add_systems((bump.in_set(Group), bump.in_set(Group), bump));
    let mut world = world();
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(world.resource::<Counter>().0, 2);
    let failure = world.resource::<SystemFailures>().iter().next().unwrap();
    assert_eq!(failure.stage, SystemFailureStage::SetCondition);
    assert_eq!(failure.node, None);
}

#[test]
fn stepping_reports_failure_then_skips_retired_node() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, bump).chain());
    let mut stepping = Stepping::new();
    assert_eq!(
        stepping.step_system(&mut schedule, &mut world),
        StepReport::Failed(0)
    );
    assert_eq!(
        stepping.step_system(&mut schedule, &mut world),
        StepReport::Ran(1)
    );
    assert_eq!(
        stepping.step_system(&mut schedule, &mut world),
        StepReport::FrameBoundary
    );
    assert_eq!(
        stepping.step_system(&mut schedule, &mut world),
        StepReport::Skipped(0)
    );
    assert_eq!(stepping.run_remaining_frame(&mut schedule, &mut world), [1]);
    assert_eq!(world.resource::<Counter>().0, 2);
}

#[test]
fn failure_history_is_bounded_and_utf8_safe() {
    let mut world = world();
    world.insert_resource(SystemFailures::with_capacity(2));
    let mut schedule = Schedule::new();
    for index in 0..4 {
        schedule.add_systems(move || panic!("{index}:{}", "界".repeat(2000)));
    }
    schedule.run(&mut world);
    let history = world.resource::<SystemFailures>();
    assert_eq!(history.total(), 4);
    let messages: Vec<_> = history
        .iter()
        .map(|failure| failure.message.as_str())
        .collect();
    assert_eq!(messages.len(), 2);
    assert!(messages[0].starts_with("2:"));
    assert!(messages[1].starts_with("3:"));
    assert!(messages.iter().all(|message| message.len() <= 4096));
    world.resource_mut::<SystemFailures>().clear();
    assert_eq!(world.resource::<SystemFailures>().total(), 4);
    assert_eq!(world.resource::<SystemFailures>().iter().len(), 0);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_joins_all_workers_retires_all_failures_and_preserves_successes() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, || panic!("second failure"), bump));
    let pool = prism_tasks::TaskPool::with_threads(2);
    schedule.run_parallel(&pool, &mut world);
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Counter>().0, 2);
    let history = world.resource::<SystemFailures>();
    assert_eq!(history.total(), 2);
    assert_eq!(
        history
            .iter()
            .map(|failure| failure.node)
            .collect::<Vec<_>>(),
        [Some(0), Some(1)]
    );
    assert_eq!(schedule.is_system_disabled(0), Some(true));
    assert_eq!(schedule.is_system_disabled(1), Some(true));
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_deferred_failure_isolated_at_sync_point() {
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_in_deferred, bump).chain());
    let pool = prism_tasks::TaskPool::with_threads(2);
    schedule.run_parallel(&pool, &mut world);
    schedule.run_parallel(&pool, &mut world);
    assert_eq!(world.resource::<Counter>().0, 12);
    assert_eq!(
        world
            .resource::<SystemFailures>()
            .iter()
            .next()
            .unwrap()
            .stage,
        SystemFailureStage::Deferred
    );
}

#[test]
fn discarded_queue_preserves_other_queues_and_reuses_only_released_generations() {
    use prism_ecs::command::CommandQueue;
    let mut world = world();
    let mut failed = CommandQueue::new();
    let mut successful = CommandQueue::new();
    let cancelled = failed.commands(world.entities()).spawn_empty();
    let survivor = successful.commands(world.entities()).spawn_empty();
    failed.discard(&mut world);
    failed.discard(&mut world);
    assert!(!world.entities().contains(cancelled));
    assert!(world.entities().contains(survivor));
    successful.apply(&mut world);
    assert!(world.contains(survivor));
    let replacement = world.spawn(Spawned);
    assert_eq!(replacement.index(), cancelled.index());
    assert_ne!(replacement, cancelled);
    assert_eq!(world.entities().len(), 2);
}

#[test]
fn discard_after_partial_replay_preserves_empty_spawn_and_drops_pending_bundle_once() {
    use prism_ecs::command::CommandQueue;
    struct Tracked(Arc<AtomicUsize>);
    impl Component for Tracked {}
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = world();
    let mut queue = CommandQueue::new();
    let (applied, pending) = {
        let mut commands = queue.commands(world.entities());
        let applied = commands.spawn_empty();
        commands.queue(|_| panic!("stop replay"));
        let pending = commands.spawn(Tracked(drops.clone()));
        (applied, pending)
    };
    assert!(catch_unwind(AssertUnwindSafe(|| queue.apply(&mut world))).is_err());
    queue.discard(&mut world);
    queue.apply(&mut world);
    assert!(world.contains(applied));
    assert!(!world.entities().contains(pending));
    assert_eq!(world.entities().len(), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn wrapped_condition_failure_releases_its_own_reservations() {
    use prism_ecs::schedule::{and, not};
    fn reserve_and_fail(mut commands: Commands) -> bool {
        commands.spawn(Spawned);
        panic!("condition reservation");
    }
    let mut world = world();
    let mut schedule = Schedule::new();
    schedule.add_systems(
        (
            bump.run_if(Gate(and(|| true, Gate(not(reserve_and_fail))))),
            bump,
        )
            .chain(),
    );
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(world.entities().len(), 0);
    assert_eq!(world.resource::<Counter>().0, 2);
    assert_eq!(world.resource::<SystemFailures>().total(), 1);
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Main;

#[test]
fn labeled_schedule_is_restored_after_fail_fast_unwind() {
    use prism_ecs::schedule::Schedules;
    let mut world = World::new();
    world.insert_resource(Counter::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((fail_after_queue, bump).chain());
    let mut schedules = Schedules::new();
    schedules.insert(Main, schedule);
    world.insert_resource(schedules);
    assert!(catch_unwind(AssertUnwindSafe(|| world.run_schedule(Main))).is_err());
    assert!(world.resource::<Schedules>().contains(Main));
    world.insert_resource(SystemPanicPolicy::DisableAndContinue);
    world.run_schedule(Main);
    world.run_schedule(Main);
    assert_eq!(world.resource::<Counter>().0, 2);
    assert_eq!(
        world
            .resource::<Schedules>()
            .get(Main)
            .unwrap()
            .is_system_disabled(0),
        Some(true)
    );
}

#[cfg(feature = "multi_thread")]
#[test]
fn labeled_parallel_schedule_is_restored_after_fail_fast_unwind() {
    use prism_ecs::schedule::Schedules;
    let mut world = World::new();
    world.insert_resource(Counter::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((reserve_then_fail, spawn_successfully));
    let mut schedules = Schedules::new();
    schedules.insert(Main, schedule);
    world.insert_resource(schedules);
    let pool = prism_tasks::TaskPool::with_threads(2);
    assert!(catch_unwind(AssertUnwindSafe(|| world.run_schedule_parallel(Main, &pool))).is_err());
    assert!(world.resource::<Schedules>().contains(Main));
    world.insert_resource(SystemPanicPolicy::DisableAndContinue);
    world.run_schedule_parallel(Main, &pool);
    world.flush_reserved();
    assert_eq!(world.entities().len(), 2);
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn wrapped_cross_world_condition_fails_before_tick_or_failure_history_changes() {
    use prism_ecs::schedule::{and, not};
    let mut first = world();
    let mut second = world();
    let mut schedule = Schedule::new();
    schedule.add_systems(bump.run_if(Gate(and(|| true, Gate(not(|| true))))));
    schedule.initialize(&mut first);
    let before = second.change_tick();
    assert!(catch_unwind(AssertUnwindSafe(|| schedule.run(&mut second))).is_err());
    assert_eq!(second.change_tick(), before);
    assert!(!second.contains_resource::<SystemFailures>());
    assert_eq!(schedule.is_system_disabled(0), Some(false));
}
