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

// --- Ambiguity detection (§23.4) -------------------------------------------

use crate::system::Res;

/// Read-only accessors of `Log`: two of these never conflict.
fn read_log_x(_log: Res<Log>) {}
fn read_log_y(_log: Res<Log>) {}

/// An exclusive system borrows the whole world, so it conflicts with every
/// other system that is not ordered relative to it.
fn exclusive_noop(_world: &mut World) {}

#[test]
fn conflicting_writers_without_order_are_ambiguous() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // Both write `Log` (ResMut) and share the default Update phase with no
    // explicit order between them.
    schedule.add_systems(push_a);
    schedule.add_systems(push_b);

    let ambiguities = schedule.ambiguities(&mut world);
    assert_eq!(ambiguities.len(), 1, "one unordered write/write pair expected");
    let pair = &ambiguities.pairs()[0];
    assert!(!pair.whole_world, "neither system is exclusive");
    assert!(
        !pair.resources.is_empty(),
        "the conflict must name the shared Log resource"
    );
    assert!(pair.components.is_empty(), "the systems touch no components");
}

#[test]
fn chaining_removes_the_ambiguity() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems((push_a, push_b).chain());

    let ambiguities = schedule.ambiguities(&mut world);
    assert!(
        ambiguities.is_empty(),
        "an explicit chain fixes the order: {}",
        ambiguities.report()
    );
}

#[test]
fn before_after_removes_the_ambiguity() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a.in_set(Physics));
    schedule.add_systems(push_b.after(Physics));

    let ambiguities = schedule.ambiguities(&mut world);
    assert!(
        ambiguities.is_empty(),
        "an explicit after(..) fixes the order: {}",
        ambiguities.report()
    );
}

#[test]
fn read_only_pair_is_not_ambiguous() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // Two concurrent readers of the same resource are compatible.
    schedule.add_systems(read_log_x);
    schedule.add_systems(read_log_y);

    let ambiguities = schedule.ambiguities(&mut world);
    assert!(
        ambiguities.is_empty(),
        "shared reads do not conflict: {}",
        ambiguities.report()
    );
}

#[test]
fn systems_in_different_phases_are_never_ambiguous() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // Both write `Log`, but the phases are totally chained, so there is always
    // an ordering edge between them.
    schedule.add_systems(push_a.in_phase(Phase::First));
    schedule.add_systems(push_b.in_phase(Phase::Last));

    let ambiguities = schedule.ambiguities(&mut world);
    assert!(
        ambiguities.is_empty(),
        "phase ordering already fixes the order: {}",
        ambiguities.report()
    );
}

#[test]
fn exclusive_system_conflicts_with_whole_world() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    // The exclusive system borrows the whole world; the writer shares the
    // Update phase with no explicit order.
    schedule.add_systems(push_a);
    schedule.add_systems(exclusive_noop);

    let ambiguities = schedule.ambiguities(&mut world);
    assert_eq!(ambiguities.len(), 1, "exclusive vs writer is one ambiguity");
    assert!(
        ambiguities.pairs()[0].whole_world,
        "the exclusive system is flagged as whole-world access"
    );
}

#[test]
#[should_panic(expected = "ambiguit")]
fn assert_no_ambiguities_panics_on_conflict() {
    let mut world = fresh();
    let mut schedule = Schedule::new();
    schedule.add_systems(push_a);
    schedule.add_systems(push_b);
    schedule.assert_no_ambiguities(&mut world);
}

/// A chained writer→observer schedule proves per-system change windows under
/// the single-threaded executor: the observer's `Changed<Cd>` filter sees the
/// writer's mutation on the run it happens, then — once the writer stops
/// writing — the observer's `last_run` advances and the change goes stale
/// (design §10, M2 commit C2).
#[test]
fn chained_writer_observer_change_detection_is_per_system() {
    use crate::component::Component;
    use crate::entity::Entity;
    use crate::query::Changed;
    use crate::system::{Local, Query};

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Cd(i32);
    impl Component for Cd {}

    #[derive(Debug, Default, PartialEq)]
    struct Seen(Vec<usize>);
    impl Resource for Seen {}

    fn mutate_once(mut q: Query<&mut Cd>, mut done: Local<bool>) {
        if !*done {
            for mut c in q.iter_mut() {
                c.0 += 1;
            }
            *done = true;
        }
    }
    fn observe(q: Query<Entity, Changed<Cd>>, mut seen: ResMut<Seen>) {
        seen.0.push(q.iter().count());
    }

    let mut world = World::new();
    world.insert_resource(Seen::default());
    world.spawn(Cd(0));
    world.spawn(Cd(0));

    let mut schedule = Schedule::new();
    schedule.add_systems((mutate_once, observe).chain());

    schedule.run(&mut world);
    schedule.run(&mut world);
    schedule.run(&mut world);

    assert_eq!(
        world.resource::<Seen>().0,
        alloc::vec![2, 0, 0],
        "both entities seen once (writer's run), then stale"
    );
}
