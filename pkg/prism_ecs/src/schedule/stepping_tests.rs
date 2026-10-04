//! Integration tests for system-granularity single-stepping (design §23.4).
//!
//! The headline test ([`step_system_drives_full_frame_like_executor`]) is the
//! "not a fake" proof: a schedule mixing a set-level condition, per-system
//! `run_if`, chaining, per-system change detection, and deferred commands is
//! run three frames two ways — once through
//! [`SingleThreadedExecutor::run`](crate::schedule::SingleThreadedExecutor) and
//! once by driving [`Stepping::step_system`] to each frame boundary — and the
//! resulting worlds are asserted equal (log, counter, entity count). The rest
//! pin down the report values, break-before breakpoints, frame-boundary cache
//! reset, and the name→index helper.

use crate::command::Commands;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::Changed;
use crate::resource::Resource;
use crate::schedule::{
    resource_exists, ContinueStop, IntoSystemConfigs, Schedule, SetConfig, StepReport, Stepping,
    SystemSet, SystemSetId,
};
use crate::system::{Local, Query, ResMut};
use crate::world::World;
use alloc::vec::Vec;

/// Append-only execution log so tests can assert ordering and run counts.
#[derive(Debug, Default, PartialEq, Clone)]
struct Log(Vec<u32>);
impl Resource for Log {}

/// A gate resource whose mere presence toggles a run condition.
#[derive(Debug, Default, PartialEq)]
struct Gate;
impl Resource for Gate {}

/// A plain counter mutated by `writer`.
#[derive(Debug, Default, PartialEq, Clone)]
struct Counter(u32);
impl Resource for Counter {}

/// A change-detected component.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Cd(i32);
impl Component for Cd {}

/// Marker for entities spawned via deferred commands.
#[derive(Debug, PartialEq, Clone, Copy)]
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

/// Drive `step` across exactly one frame (until the frame boundary), returning
/// the ordered [`StepReport`]s for the systems processed this frame (the
/// trailing [`StepReport::FrameBoundary`] is consumed, not returned).
fn step_one_frame(step: &mut Stepping, schedule: &mut Schedule, world: &mut World) -> Vec<StepReport> {
    let mut reports = Vec::new();
    loop {
        match step.step_system(schedule, world) {
            StepReport::FrameBoundary => return reports,
            other => reports.push(other),
        }
    }
}

#[test]
fn step_system_drives_full_frame_like_executor() {
    struct Guarded;
    impl SystemSet for Guarded {
        fn set_id(&self) -> SystemSetId {
            SystemSetId::of::<Self>()
        }
    }

    fn writer(mut counter: ResMut<Counter>, mut log: ResMut<Log>) {
        counter.0 += 1;
        log.0.push(10);
    }
    fn mutate_once(mut q: Query<&mut Cd>, mut done: Local<bool>) {
        if !*done {
            for mut c in q.iter_mut() {
                c.0 += 1;
            }
            *done = true;
        }
    }
    fn observe(q: Query<Entity, Changed<Cd>>, mut log: ResMut<Log>) {
        log.0.push(100 + q.iter().count() as u32);
    }
    fn spawner(mut commands: Commands) {
        commands.spawn(Spawned);
    }
    fn observe_count(world: &mut World) {
        let n = world.entity_count();
        world.resource_mut::<Log>().0.push(1000 + n);
    }

    // Build the identical schedule twice (one instance per world); system fns
    // and construction order are the same, so both resolve the same order.
    fn build() -> Schedule {
        let mut schedule = Schedule::new();
        schedule.configure_set(Guarded, SetConfig::new().run_if(resource_exists::<Gate>()));
        schedule.add_systems(
            (writer, mutate_once, observe, spawner, observe_count)
                .chain()
                .in_set(Guarded),
        );
        schedule.add_systems(push_a.run_if(resource_exists::<Gate>()));
        schedule
    }

    fn fresh_world() -> World {
        let mut world = World::new();
        world.insert_resource(Log::default());
        world.insert_resource(Counter::default());
        world.spawn(Cd(0));
        world.spawn(Cd(0));
        world
    }

    // Reference run: three frames through the single-threaded executor, with
    // the gate present for frames 1 and 3 and absent for frame 2 — exercising
    // the set condition toggling and the per-frame set cache.
    let mut world_exec = fresh_world();
    let mut sched_exec = build();
    for frame in 0..3 {
        if frame == 1 {
            world_exec.remove_resource::<Gate>();
        } else {
            world_exec.insert_resource(Gate);
        }
        sched_exec.run(&mut world_exec);
    }

    // Stepped run: the same three frames, driven one system at a time to each
    // frame boundary.
    let mut world_step = fresh_world();
    let mut sched_step = build();
    let mut step = Stepping::new();
    for frame in 0..3 {
        if frame == 1 {
            world_step.remove_resource::<Gate>();
        } else {
            world_step.insert_resource(Gate);
        }
        step_one_frame(&mut step, &mut sched_step, &mut world_step);
        assert!(
            step.is_at_frame_boundary(),
            "stepper must be parked at the frame boundary after a full frame"
        );
    }

    // Byte-for-byte identical observable state proves stepping reproduces the
    // executor: same log (order + gating + deferred-command visibility + change
    // detection), same counter, same spawned-entity count.
    assert_eq!(
        world_step.resource::<Log>().0,
        world_exec.resource::<Log>().0,
        "stepped log must equal executor log"
    );
    assert_eq!(
        world_step.resource::<Counter>().0,
        world_exec.resource::<Counter>().0,
    );
    assert_eq!(world_step.entity_count(), world_exec.entity_count());
}

#[test]
fn step_system_reports_ran_skipped_and_boundary() {
    // Node 0 is gated off (no Gate resource), node 1 always runs.
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a.run_if(resource_exists::<Gate>())); // node 0
    schedule.add_systems(push_b); // node 1
    schedule.initialize(&mut world);

    let mut step = Stepping::new();
    assert_eq!(step.step_system(&mut schedule, &mut world), StepReport::Skipped(0));
    assert_eq!(step.step_system(&mut schedule, &mut world), StepReport::Ran(1));
    assert_eq!(
        step.step_system(&mut schedule, &mut world),
        StepReport::FrameBoundary
    );
    // Only push_b (node 1) ran.
    assert_eq!(world.resource::<Log>().0, alloc::vec![2]);
    // Cursor wrapped: a new frame begins.
    assert!(step.is_at_frame_boundary());
}

#[test]
fn breakpoint_breaks_before_then_continues_past() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b, push_c).chain()); // nodes 0,1,2
    schedule.initialize(&mut world);

    let mut step = Stepping::new();
    step.add_breakpoint(1);

    // First continue halts *before* node 1 (break-before): only push_a ran.
    let report = step.continue_frame(&mut schedule, &mut world);
    assert_eq!(report.ran, alloc::vec![0]);
    assert_eq!(report.stop, ContinueStop::Breakpoint(1));
    assert_eq!(world.resource::<Log>().0, alloc::vec![1]);

    // Resuming walks past the breakpoint it is parked on and runs to the end.
    let report = step.continue_frame(&mut schedule, &mut world);
    assert_eq!(report.ran, alloc::vec![1, 2]);
    assert_eq!(report.stop, ContinueStop::FrameBoundary);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
}

#[test]
fn breakpoint_on_first_system_stops_before_running_anything() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b).chain());
    schedule.initialize(&mut world);

    let mut step = Stepping::new();
    step.add_breakpoint(0);

    let report = step.continue_frame(&mut schedule, &mut world);
    assert!(report.ran.is_empty(), "nothing runs before a leading breakpoint");
    assert_eq!(report.stop, ContinueStop::Breakpoint(0));
    assert!(world.resource::<Log>().0.is_empty());

    // Next continue walks past it and finishes the frame.
    let report = step.continue_frame(&mut schedule, &mut world);
    assert_eq!(report.ran, alloc::vec![0, 1]);
    assert_eq!(report.stop, ContinueStop::FrameBoundary);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);
}

#[test]
fn run_remaining_frame_ignores_breakpoints() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b, push_c).chain());
    schedule.initialize(&mut world);

    let mut step = Stepping::new();
    step.add_breakpoint(0);
    step.add_breakpoint(2);

    let ran = step.run_remaining_frame(&mut schedule, &mut world);
    assert_eq!(ran, alloc::vec![0, 1, 2], "breakpoints are ignored");
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2, 3]);
    assert!(step.is_at_frame_boundary());
}

#[test]
fn set_cache_resets_each_frame() {
    // Two systems share one guarded set; the set condition keys off `Gate`.
    struct Guarded;
    impl SystemSet for Guarded {
        fn set_id(&self) -> SystemSetId {
            SystemSetId::of::<Self>()
        }
    }

    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.configure_set(Guarded, SetConfig::new().run_if(resource_exists::<Gate>()));
    schedule.add_systems((push_a, push_b).chain().in_set(Guarded));

    let mut step = Stepping::new();

    // Frame 1 with the gate: both members run (the shared set condition is
    // evaluated once and cached true for the frame).
    world.insert_resource(Gate);
    step_one_frame(&mut step, &mut schedule, &mut world);
    assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);

    // Frame 2 without the gate: the cache was cleared at the boundary, so the
    // set condition is re-evaluated (now false) and neither member runs.
    world.remove_resource::<Gate>();
    step_one_frame(&mut step, &mut schedule, &mut world);
    assert_eq!(
        world.resource::<Log>().0,
        alloc::vec![1, 2],
        "frame 2 must not run; a stale cached true would wrongly append"
    );
}

#[test]
fn find_system_resolves_name_to_node_index() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b).chain());
    schedule.initialize(&mut world);

    let step = Stepping::new();
    // Round-trip each node's reported name back to its index.
    let name0 = schedule.nodes[0].system.name();
    let name1 = schedule.nodes[1].system.name();
    assert_eq!(step.find_system(&schedule, name0), Some(0));
    assert_eq!(step.find_system(&schedule, name1), Some(1));
    assert_eq!(step.find_system(&schedule, "no::such::system"), None);
}
