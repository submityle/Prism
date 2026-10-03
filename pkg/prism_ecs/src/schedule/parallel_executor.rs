//! The parallel conflict-graph executor (design §8.2; dispatch base §24.1).
//!
//! [`MultiThreadedExecutor`] runs a [`Schedule`] by partitioning its systems
//! into *waves*: ordered groups whose members have pairwise-compatible
//! [`Access`] and so can run concurrently. Within a wave the system bodies are
//! dispatched onto a [`prism_tasks::TaskPool`]; between waves there is a sync
//! point where run-conditions are evaluated and deferred
//! [`Commands`](crate::command::Commands) are flushed.
//!
//! # Why waves preserve determinism
//!
//! For an **ambiguity-free** schedule (see
//! [`Schedule::ambiguities`](crate::schedule::Schedule::ambiguities), §23.4) the
//! wave partition yields the same observable result as the single-threaded
//! executor:
//!
//! * A system's *effective access* is the union of its own
//!   [`Access`](crate::query::Access) with the access of every run-condition
//!   that gates it — both its own `run_if` conditions and the conditions on
//!   each [`SystemSet`](crate::schedule::SystemSet) it belongs to. Two systems
//!   share a wave only if their effective accesses are compatible, so a system
//!   never runs concurrently with a condition (or body) that reads what it
//!   writes. Conditions are still evaluated sequentially at the sync point.
//! * Wave `w` is assigned greedily in topological order: a system lands in the
//!   earliest wave strictly after every predecessor's wave in which it is also
//!   compatible with every system already placed there. Predecessors (direct or
//!   transitive ordering edges) therefore always occupy an earlier wave and
//!   have fully applied — bodies *and* deferred commands — before a dependent
//!   wave begins.
//! * Exclusive (whole-world) systems have
//!   [`Access::writes_everything`](crate::query::Access::writes_everything), so
//!   they are incompatible with everything and always occupy a wave alone; the
//!   `&mut World` they form is then the only live borrow.
//!
//! # Soundness
//!
//! Each wave hands every one of its systems a copy of a single
//! [`UnsafeWorldCell`] plus a raw pointer to that system. The wave's node
//! indices are distinct, so the system pointers are non-aliasing; the wave's
//! pairwise-compatible access guarantees the world views the bodies form are
//! disjoint. [`TaskPool::scope`](prism_tasks::TaskPool::scope) joins every
//! spawned task before returning, so all raw pointers stay valid for the whole
//! dispatch and the `&mut World` borrow is released before the next sync point.

use alloc::vec::Vec;

use hashbrown::HashMap;
use prism_tasks::TaskPool;

use crate::query::Access;
use crate::schedule::ambiguity;
use crate::schedule::graph::Schedule;
use crate::schedule::set::SystemSetId;
use crate::system::function::{BoxedSystem, System};
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::World;

/// A `Send` wrapper around a [`UnsafeWorldCell`] so it can be captured by a
/// task dispatched onto the pool.
struct SendCell<'w>(UnsafeWorldCell<'w>);

// SAFETY: an `UnsafeWorldCell` is a raw `*mut World` and so not `Send` by
// default. It is sound to move a copy into each task of a wave because the wave
// executor only ever forms disjoint world views through it: the systems sharing
// a wave have pairwise-compatible access, and an exclusive system (which forms
// `&mut World`) runs alone in its wave. The cell's target outlives the
// `scope`-joined tasks (the `&mut World` borrow spans the dispatch).
unsafe impl Send for SendCell<'_> {}

/// A `Send` wrapper around a raw pointer to one wave member's system.
struct SendSystem(*mut (dyn System<Out = ()> + 'static));

// SAFETY: the raw pointer is derived from a `&mut` into a distinct node's boxed
// system; wave members have distinct node indices, so the pointers handed to
// concurrent tasks never alias. The pointees live in `schedule.nodes`, which is
// not mutated for the duration of the wave dispatch, and `scope` joins every
// task before the borrow could end.
unsafe impl Send for SendSystem {}

/// The parallel conflict-graph executor: a zero-sized dispatcher, the sibling
/// of [`SingleThreadedExecutor`](crate::schedule::SingleThreadedExecutor)
/// selected via [`Schedule::run_parallel`](crate::schedule::Schedule::run_parallel)
/// (design §8.2).
pub struct MultiThreadedExecutor;

impl MultiThreadedExecutor {
    /// Run every system in `schedule` once against `world`, dispatching
    /// compatible systems concurrently onto `pool`.
    ///
    /// The schedule must already be initialised (see
    /// [`Schedule::initialize`](crate::schedule::Schedule::initialize)); the
    /// public [`Schedule::run_parallel`](crate::schedule::Schedule::run_parallel)
    /// entry point does that before delegating here.
    pub fn run(schedule: &mut Schedule, pool: &TaskPool, world: &mut World) {
        let n = schedule.nodes.len();
        if n == 0 {
            return;
        }

        // Partition the systems into waves once per run. Access is stable
        // across a run, so the partition only depends on the resolved order and
        // ordering edges, both already computed by `initialize`.
        let effective = effective_accesses(schedule);
        let edges: Vec<(usize, usize)> = schedule.build_edges().into_iter().collect();
        let reach = ambiguity::reachability(&edges, n);
        let order = schedule.order.clone();
        let waves = compute_waves(&order, &effective, &reach);

        // Set-condition results are memoised for the whole run, exactly as in
        // the single-threaded executor, so a shared set condition fires once.
        let mut set_cache: HashMap<SystemSetId, bool> = HashMap::new();

        for wave in &waves {
            // --- Sync point: evaluate run-conditions sequentially, in
            // topological (within-wave) order, matching the single-threaded
            // executor's gating semantics exactly.
            let mut should_run: Vec<bool> = Vec::with_capacity(wave.len());
            for &idx in wave {
                should_run.push(eval_should_run(schedule, idx, &mut set_cache, world));
            }

            // Collect a raw pointer to each wave member's system. Distinct
            // indices make the pointers non-aliasing; we only ever hold the raw
            // pointers (not references) across the dispatch.
            let mut tasks: Vec<(SendSystem, bool)> = Vec::with_capacity(wave.len());
            for (slot, &idx) in wave.iter().enumerate() {
                let system: *mut (dyn System<Out = ()> + 'static) = {
                    let boxed: &mut BoxedSystem = &mut schedule.nodes[idx].system;
                    &mut **boxed
                };
                tasks.push((SendSystem(system), should_run[slot]));
            }

            // --- Dispatch the passing bodies in parallel. The `&mut World`
            // borrow is confined to this block so it is released before the
            // deferred-command flush below.
            {
                let cell = UnsafeWorldCell::new_mutable(world);
                pool.scope(|scope| {
                    for (system, run) in &tasks {
                        if !*run {
                            continue;
                        }
                        let cell = SendCell(cell);
                        let system = SendSystem(system.0);
                        scope.spawn(move || {
                            // Re-bind the whole wrappers so the closure captures
                            // the `Send` `SendCell`/`SendSystem` values rather
                            // than their non-`Send` `.0` fields (edition-2024
                            // closures capture precise paths otherwise).
                            let cell = cell;
                            let system = system;
                            // SAFETY: `system.0` is the unique pointer to this
                            // wave member's system, and `cell.0` grants access to
                            // a world region disjoint from every other task in
                            // the wave (pairwise-compatible access; exclusive
                            // systems run alone). No conflicting borrow is live,
                            // so running the body is sound.
                            unsafe {
                                (*system.0).run_unsafe(cell.0);
                            }
                        });
                    }
                });
            }

            // --- Sync point: flush deferred commands for the systems that ran,
            // in topological order within the wave.
            for (slot, &idx) in wave.iter().enumerate() {
                if should_run[slot] {
                    schedule.nodes[idx].system.apply_deferred(world);
                }
            }
        }
    }
}

/// Each node's *effective access*: its system's own access unioned with the
/// access of every condition that can gate it (its own `run_if` conditions and
/// the conditions on each set it belongs to). This is what two systems must
/// have compatible for them to share a wave, so a body never races a condition
/// that reads what it writes.
pub(crate) fn effective_accesses(schedule: &Schedule) -> Vec<Access> {
    let mut out: Vec<Access> = Vec::with_capacity(schedule.nodes.len());
    for node in &schedule.nodes {
        let mut access = node.system.access().clone();
        for condition in &node.conditions {
            access.extend(condition.access());
        }
        for set in &node.sets {
            if let Some(config) = schedule.set_configs.get(set) {
                for condition in &config.conditions {
                    access.extend(condition.access());
                }
            }
        }
        out.push(access);
    }
    out
}

/// Greedily assign each node (visited in topological `order`) to the earliest
/// wave that is both strictly after every predecessor's wave and compatible
/// with every system already placed in it. Returns the waves as lists of node
/// indices in topological order.
pub(crate) fn compute_waves(
    order: &[usize],
    effective: &[Access],
    reach: &[Vec<bool>],
) -> Vec<Vec<usize>> {
    let mut waves: Vec<Vec<usize>> = Vec::new();
    let mut wave_of: Vec<usize> = alloc::vec![usize::MAX; effective.len()];

    for &s in order {
        // Earliest wave allowed by ordering: strictly after every predecessor.
        let mut min_wave = 0usize;
        for (p, &placed) in wave_of.iter().enumerate() {
            if placed != usize::MAX && reach[p][s] {
                min_wave = min_wave.max(placed + 1);
            }
        }

        // Earliest wave at or after `min_wave` compatible with all its members.
        let mut target = min_wave;
        loop {
            if target >= waves.len() {
                waves.push(Vec::new());
            }
            let compatible = waves[target]
                .iter()
                .all(|&other| effective[s].is_compatible(&effective[other]));
            if compatible {
                break;
            }
            target += 1;
        }

        waves[target].push(s);
        wave_of[s] = target;
    }

    waves
}

/// Evaluate whether node `idx` should run this tick, matching the
/// single-threaded executor: all of its sets' conditions (cached, no
/// short-circuit so their state advances once) AND its own conditions
/// (short-circuit).
fn eval_should_run(
    schedule: &mut Schedule,
    idx: usize,
    set_cache: &mut HashMap<SystemSetId, bool>,
    world: &mut World,
) -> bool {
    // Clone the id list so the immutable borrow of `nodes` is released before
    // `eval_set_conditions` takes a mutable borrow of the schedule.
    let node_sets = schedule.nodes[idx].sets.clone();
    let mut should_run = true;
    for set in node_sets {
        let value = match set_cache.get(&set) {
            Some(&cached) => cached,
            None => {
                let value = schedule.eval_set_conditions(set, world);
                set_cache.insert(set, value);
                value
            }
        };
        should_run &= value;
    }

    if should_run {
        for condition in &mut schedule.nodes[idx].conditions {
            if !condition.run(world) {
                should_run = false;
                break;
            }
        }
    }

    should_run
}
