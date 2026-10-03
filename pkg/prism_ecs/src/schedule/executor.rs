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
//! # Honestly deferred
//!
//! The parallel conflict-graph executor and fiber job graph (design §8.2–§8.3,
//! which need `prism_tasks`) are future milestones. They are absent, not
//! stubbed; the per-system [`Access`](crate::query::Access) recorded by the
//! system layer already carries everything a parallel executor needs, and it
//! will become a sibling of this type selected by the schedule.

use hashbrown::HashMap;

use crate::schedule::graph::Schedule;
use crate::schedule::set::SystemSetId;
use crate::world::World;

/// The default executor: runs a [`Schedule`]'s systems sequentially on the
/// calling thread, in the schedule's deterministic topological order, honouring
/// per-set and per-system run-conditions.
///
/// This is a zero-sized dispatcher rather than a stored strategy so a
/// [`Schedule`] stays trivially movable; the future parallel conflict-graph
/// executor (§8.2) will be a sibling type selected by the schedule.
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
        // Set-condition results are memoised for the whole run so a set's
        // shared conditions are evaluated at most once even when many members
        // reference the same set.
        let mut set_cache: HashMap<SystemSetId, bool> = HashMap::new();

        for i in 0..schedule.order.len() {
            let idx = schedule.order[i];

            // Evaluate every set this system belongs to. Clone the id list so
            // the immutable borrow of `nodes` is released before
            // `eval_set_conditions` takes a mutable borrow of the schedule.
            let node_sets = schedule.nodes[idx].sets.clone();
            let mut should_run = true;
            for set in node_sets {
                let value = match set_cache.get(&set) {
                    Some(&cached) => cached,
                    None => {
                        // Evaluate all of the set's conditions (no
                        // short-circuit) so their internal state advances
                        // deterministically, then cache the AND.
                        let value = schedule.eval_set_conditions(set, world);
                        set_cache.insert(set, value);
                        value
                    }
                };
                // Keep evaluating remaining sets even once gated off, so every
                // set-condition's state advances exactly once this run.
                should_run &= value;
            }

            // Evaluate the system's own conditions with short-circuit AND.
            if should_run {
                for condition in &mut schedule.nodes[idx].conditions {
                    if !condition.run(world) {
                        should_run = false;
                        break;
                    }
                }
            }

            if should_run {
                schedule.nodes[idx].system.run(world);
            }
        }
    }
}
