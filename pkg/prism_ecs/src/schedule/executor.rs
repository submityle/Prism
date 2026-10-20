//! The single-threaded [`Schedule`](crate::schedule::Schedule) executor
//! (design §8.2).
//!
//! [`SingleThreadedExecutor`] walks a schedule's precomputed, deterministic
//! topological [`order`](crate::schedule::Schedule) and runs each system via
//! [`System::run`](crate::system::System::run) — which fetches the system's
//! params, runs its body, and immediately applies its deferred
//! [`Commands`](crate::command::Commands), i.e. a sync point after every
//! system.
//!
//! Run-conditions gate execution at two levels:
//!
//! * **set conditions** — the conditions configured on each [`SystemSet`] a
//!   system belongs to (including a group's anonymous collective-condition set)
//!   are evaluated at most once per run and cached, so a shared condition fires
//!   once no matter how many members reference it;
//! * **system conditions** — a system's own `run_if` conditions, evaluated
//!   in order with short-circuit AND.
//!
//! A system runs only when every gate passes.
//!
//! With `std`, a world's [`SystemPanicPolicy`](crate::schedule::SystemPanicPolicy)
//! can opt into retiring failed systems and continuing the remaining schedule.

use hashbrown::HashMap;

use crate::diagnostics::profiler::SystemInstrument;
use crate::schedule::graph::Schedule;
#[cfg(feature = "std")]
use crate::schedule::lane::SystemLane;
use crate::schedule::set::SystemSetId;
use crate::world::World;
use core::time::Duration;

/// The default executor: runs a [`Schedule`]'s systems sequentially on the
/// calling thread, in the schedule's deterministic topological order, honouring
/// per-set and per-system run-conditions.
///
/// This is a zero-sized dispatcher rather than a stored strategy so a
/// [`Schedule`] stays trivially movable; the parallel conflict-graph executor
/// (§8.2) is selected separately by the schedule.
pub struct SingleThreadedExecutor;

impl SingleThreadedExecutor {
    /// Run every system in `schedule` once against `world`, in dependency
    /// order.
    ///
    /// The schedule must already be initialised (see
    /// [`Schedule::initialize`](crate::schedule::Schedule::initialize)); the
    /// public [`Schedule::run`](crate::schedule::Schedule::run) entry point
    /// does that before delegating here. This routine deliberately uses the
    /// schedule's lower-level fields rather than calling back into
    /// [`Schedule::run`], which would recurse forever.
    pub fn run(schedule: &mut Schedule, world: &mut World) {
        Self::run_inner(schedule, world, &mut (), None);
    }

    /// Run every system once, honouring a **frame-time budget** (design §24.1).
    ///
    /// [`SystemLane::Critical`] and [`SystemLane::Normal`] systems always run —
    /// the frame's required work is never dropped. Once `budget` has elapsed,
    /// every remaining [`SystemLane::Background`] system (deferrable async
    /// precompute) is **skipped this run and left for a later frame**, exactly
    /// as design §24.1 specifies. Returns the number of systems deferred.
    ///
    /// `std`-gated because it reads the wall clock ([`std::time::Instant`]).
    /// Budgeted runs are for the deferrable/background tier; a lockstep
    /// deterministic schedule should use [`run`](SingleThreadedExecutor::run)
    /// (the `determinism` tier needs a fixed plan, not a wall-clock gate).
    #[cfg(feature = "std")]
    pub fn run_with_budget(schedule: &mut Schedule, world: &mut World, budget: Duration) -> usize {
        Self::run_inner(schedule, world, &mut (), Some(budget))
    }

    /// Run every system in `schedule` once against `world`, in dependency
    /// order, timing each system through `recorder` so the caller can build a
    /// [`FlameGraph`](crate::diagnostics::profiler::FlameGraph) and hotspot
    /// report (design §16.6 "系统火焰图").
    ///
    /// This is identical to [`run`](SingleThreadedExecutor::run) except that a
    /// span is opened around each system that actually runs (gated-off systems
    /// are not timed), so a flame graph reflects the systems whose bodies
    /// executed. It is `std`-gated because [`SpanRecorder`] measures wall-clock
    /// time with [`Instant`](std::time::Instant).
    ///
    /// [`SpanRecorder`]: crate::diagnostics::profiler::SpanRecorder
    #[cfg(feature = "std")]
    pub fn run_instrumented(
        schedule: &mut Schedule,
        world: &mut World,
        recorder: &mut crate::diagnostics::profiler::SpanRecorder,
    ) {
        Self::run_inner(schedule, world, recorder, None);
    }

    /// The shared run loop, generic over the [`SystemInstrument`] so the
    /// uninstrumented path (`instrument: &mut ()`) and the profiled path
    /// (`instrument: &mut SpanRecorder`) share one implementation with no logic
    /// duplication. The no-op `()` instrument monomorphises to zero overhead.
    fn run_inner<I: SystemInstrument>(
        schedule: &mut Schedule,
        world: &mut World,
        instrument: &mut I,
        budget: Option<Duration>,
    ) -> usize {
        #[cfg(not(feature = "std"))]
        let _ = budget;
        // A budget deadline, evaluated once so every system sees the same one.
        #[cfg(feature = "std")]
        let deadline = budget.map(|b| std::time::Instant::now() + b);
        #[cfg_attr(not(feature = "std"), allow(unused_mut))]
        let mut deferred = 0usize;

        // Set-condition results are memoised for the whole run so a set's
        // shared conditions are evaluated at most once even when many members
        // reference the same set.
        let mut set_cache: HashMap<SystemSetId, bool> = HashMap::new();

        for i in 0..schedule.order.len() {
            let idx = schedule.order[i];

            #[cfg_attr(not(feature = "std"), allow(unused_mut))]
            let mut should_run = schedule.eval_gates(idx, &mut set_cache, world);

            // Frame-budget gate (design §24.1): once the budget is spent, a
            // Background-lane system is deferred to a later frame. Critical and
            // Normal systems are never dropped by the budget.
            #[cfg(feature = "std")]
            if should_run
                && let Some(deadline) = deadline
                && schedule.nodes[idx].lane == SystemLane::Background
                && std::time::Instant::now() >= deadline
            {
                should_run = false;
                deferred += 1;
            }

            if should_run {
                instrument.begin(schedule.nodes[idx].system.name());
                schedule.run_node(idx, world);
                instrument.end();
            }
        }
        deferred
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::SingleThreadedExecutor;
    use crate::resource::Resource;
    use crate::schedule::{IntoSystemConfigs, Schedule};
    use crate::system::ResMut;
    use crate::world::World;

    /// Execution log so the test can confirm the instrumented path still runs
    /// each system body exactly once, in order.
    #[derive(Debug, Default, PartialEq)]
    struct Log(alloc::vec::Vec<u32>);
    impl Resource for Log {}

    fn first(mut log: ResMut<Log>) {
        log.0.push(1);
    }
    fn second(mut log: ResMut<Log>) {
        log.0.push(2);
    }

    #[test]
    fn run_instrumented_records_one_span_per_system() {
        let mut world = World::new();
        world.insert_resource(Log::default());

        let mut schedule = Schedule::new();
        schedule.add_systems((first, second).chain());
        schedule.initialize(&mut world);

        let mut recorder = crate::diagnostics::profiler::SpanRecorder::new();
        SingleThreadedExecutor::run_instrumented(&mut schedule, &mut world, &mut recorder);

        // Both bodies ran, in order, exactly once.
        assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);

        // Every `begin` was matched by an `end`, and one root span per system.
        assert!(recorder.is_balanced());
        assert_eq!(recorder.roots().len(), 2);

        // The flame graph exports one folded line per span (both are leaves),
        // and the hotspots are ordered by descending self time.
        let graph = recorder.flame_graph();
        let folded = graph.folded();
        assert_eq!(folded.len(), 2);
        assert!(!graph.is_empty());

        let hot = graph.hotspots();
        assert_eq!(hot.len(), 2);
        for pair in hot.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "hotspots must be descending");
        }
    }

    #[test]
    fn uninstrumented_run_is_unaffected() {
        let mut world = World::new();
        world.insert_resource(Log::default());

        let mut schedule = Schedule::new();
        schedule.add_systems((first, second).chain());
        schedule.run(&mut world);

        assert_eq!(world.resource::<Log>().0, alloc::vec![1, 2]);
    }

    /// Per-lane counters for the frame-budget tests.
    #[derive(Debug, Default)]
    struct Lanes {
        critical: u32,
        normal: u32,
        background: u32,
    }
    impl Resource for Lanes {}

    fn crit(mut c: ResMut<Lanes>) {
        c.critical += 1;
    }
    fn norm(mut c: ResMut<Lanes>) {
        c.normal += 1;
    }
    fn bg(mut c: ResMut<Lanes>) {
        c.background += 1;
    }

    fn budgeted_schedule(world: &mut World) -> Schedule {
        world.insert_resource(Lanes::default());
        let mut schedule = Schedule::new();
        schedule.add_systems(crit.lane(crate::schedule::lane::SystemLane::Critical));
        schedule.add_systems(norm.lane(crate::schedule::lane::SystemLane::Normal));
        schedule.add_systems(bg.lane(crate::schedule::lane::SystemLane::Background));
        schedule
    }

    #[test]
    fn background_lane_is_deferred_when_the_budget_is_spent() {
        let mut world = World::new();
        let mut schedule = budgeted_schedule(&mut world);

        // A zero budget defers the Background system but never Critical/Normal.
        let deferred = schedule.run_budgeted(&mut world, core::time::Duration::ZERO);
        assert_eq!(deferred, 1, "exactly the background system is deferred");
        {
            let c = world.resource::<Lanes>();
            assert_eq!((c.critical, c.normal, c.background), (1, 1, 0));
        }

        // A generous budget runs everything, so the deferred work catches up.
        let deferred = schedule.run_budgeted(&mut world, core::time::Duration::from_secs(60));
        assert_eq!(deferred, 0);
        let c = world.resource::<Lanes>();
        assert_eq!((c.critical, c.normal, c.background), (2, 2, 1));
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn parallel_background_lane_is_deferred_when_the_budget_is_spent() {
        let mut world = World::new();
        let mut schedule = budgeted_schedule(&mut world);
        let pool = prism_tasks::TaskPool::with_threads(2);

        let tick_before = world.change_tick().get();
        let deferred =
            schedule.run_parallel_budgeted(&pool, &mut world, core::time::Duration::ZERO);
        assert_eq!(deferred, 1);
        assert_eq!(
            world.change_tick().get(),
            tick_before.wrapping_add(2),
            "gated systems must not receive synthetic tick windows"
        );
        {
            let c = world.resource::<Lanes>();
            assert_eq!((c.critical, c.normal, c.background), (1, 1, 0));
        }

        let tick_before = world.change_tick().get();
        let deferred =
            schedule.run_parallel_budgeted(&pool, &mut world, core::time::Duration::from_secs(60));
        assert_eq!(deferred, 0);
        assert_eq!(
            world.change_tick().get(),
            tick_before.wrapping_add(3),
            "each executed system receives exactly one change-detection tick"
        );
        let c = world.resource::<Lanes>();
        assert_eq!((c.critical, c.normal, c.background), (2, 2, 1));
    }
}
