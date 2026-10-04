//! System-granularity single-stepping of a [`Schedule`] (design §23.4
//! "系统单步 stepping").
//!
//! Where [`SingleThreadedExecutor`](crate::schedule::SingleThreadedExecutor)
//! runs an entire schedule in one call, [`Stepping`] drives the *same*
//! deterministic order one system at a time, so a debugger or the §16.6
//! inspector can observe the [`World`] mutate between systems and reproduce
//! timing-dependent defects.
//!
//! # Faithful to the executor
//!
//! Stepping is **not** a re-implementation of scheduling: it walks
//! [`Schedule::order`](crate::schedule::Schedule) exactly as the single-threaded
//! executor does and reuses the schedule's own
//! [`run`](crate::system::System::run) / condition machinery. The gate logic in
//! [`Stepping::eval_gates`] is a line-for-line mirror of
//! [`executor::run_inner`](crate::schedule::executor): per-set conditions are
//! evaluated **without** short-circuit (so each set-condition's internal state
//! advances exactly once per frame) and memoised in a frame-scoped cache, while
//! a system's own `run_if` conditions short-circuit with AND. Driving a full
//! frame through [`step_system`](Stepping::step_system) therefore produces a
//! byte-identical [`World`] to a single [`SingleThreadedExecutor::run`]; the
//! test module proves this on a schedule mixing set conditions, `run_if`,
//! chaining, change detection, and deferred commands.
//!
//! # Frame model
//!
//! `cursor` is the next position in [`Schedule::order`] to process, in
//! `0..=len`. Processing position `len` is the **frame boundary**: the next
//! step wraps the cursor back to `0` and clears the set-condition cache,
//! matching the executor's "evaluate each set once per run" contract. A fresh
//! [`Stepping`] starts parked at the frame boundary (`cursor == 0`).
//!
//! # Breakpoints
//!
//! Breakpoints are keyed by node index (position in
//! [`Schedule::nodes`](crate::schedule::Schedule)) with **break-before**
//! semantics: [`continue_frame`](Stepping::continue_frame) halts *before*
//! processing a system at a breakpoint — before its conditions are even
//! evaluated. Resuming with another [`continue_frame`](Stepping::continue_frame)
//! walks past the breakpoint it is parked on (standard debugger "continue")
//! and then runs to the next breakpoint or the frame boundary. Use
//! [`find_system`](Stepping::find_system) to resolve a system name to its node
//! index for the inspector.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::schedule::graph::Schedule;
use crate::schedule::set::SystemSetId;
use crate::world::World;

/// The outcome of a single [`Stepping::step_system`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepReport {
    /// The system at this node index passed its gates and its body ran.
    Ran(usize),
    /// The system at this node index was gated off (a set or `run_if`
    /// condition returned `false`) and was skipped, exactly as the executor
    /// would have skipped it.
    Skipped(usize),
    /// The cursor reached the end of the frame; it has been wrapped back to the
    /// start and the per-frame set-condition cache cleared. No system ran.
    FrameBoundary,
}

/// Why [`Stepping::continue_frame`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinueStop {
    /// Ran to the end of the frame; the cursor wrapped and the set cache
    /// cleared.
    FrameBoundary,
    /// Halted *before* the system at this node index because it carries a
    /// break-before breakpoint.
    Breakpoint(usize),
}

/// The outcome of a [`Stepping::continue_frame`] call: the node indices whose
/// bodies ran (in execution order) and why the run stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinueReport {
    /// Node indices of the systems that actually ran this call, in order.
    pub ran: Vec<usize>,
    /// Why the run stopped.
    pub stop: ContinueStop,
}

/// A debugger-style cursor over a [`Schedule`]'s deterministic order (design
/// §23.4).
///
/// Hold one `Stepping` per schedule being debugged. It borrows the schedule and
/// world on each call rather than owning them, so it composes with the
/// inspector and the ordinary executor (you can step for a few systems, then
/// hand off to [`SingleThreadedExecutor::run`](crate::schedule::SingleThreadedExecutor)
/// for the rest of the frame). Reset it with [`reset`](Stepping::reset) whenever
/// the schedule's system set changes (which re-arms the schedule's lazy
/// initialization and recomputes its order).
#[derive(Debug, Default)]
pub struct Stepping {
    /// Next position in [`Schedule::order`] to process, in `0..=len`.
    cursor: usize,
    /// Frame-scoped memo of each set's AND-ed conditions, mirroring the
    /// executor's per-run cache; cleared on every frame boundary.
    set_cache: HashMap<SystemSetId, bool>,
    /// Node indices with break-before breakpoints.
    breakpoints: BTreeSet<usize>,
    /// When parked at a breakpoint, the node index the next
    /// [`continue_frame`](Stepping::continue_frame) must walk past rather than
    /// re-trigger. Cleared whenever the cursor advances by other means.
    pending_breakpoint: Option<usize>,
}

impl Stepping {
    /// A fresh stepper parked at the frame boundary with no breakpoints.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The next position in the schedule's order that will be processed
    /// (`0..=len`). `0` is the frame boundary.
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.cursor
    }

    /// Whether the cursor is parked at a frame boundary (nothing has been
    /// stepped yet this frame).
    #[inline]
    #[must_use]
    pub fn is_at_frame_boundary(&self) -> bool {
        self.cursor == 0
    }

    /// Rewind to the start of a frame and clear the per-frame set-condition
    /// cache, keeping the configured breakpoints. Call this whenever the
    /// schedule's systems or sets change (which recomputes its order), so the
    /// cursor cannot point past the end of a shortened order.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.set_cache.clear();
        self.pending_breakpoint = None;
    }

    /// Add a break-before breakpoint on the system at `node_index`. Returns
    /// `true` if it was newly added.
    pub fn add_breakpoint(&mut self, node_index: usize) -> bool {
        self.breakpoints.insert(node_index)
    }

    /// Remove the breakpoint on `node_index`. Returns `true` if one was present.
    pub fn remove_breakpoint(&mut self, node_index: usize) -> bool {
        self.breakpoints.remove(&node_index)
    }

    /// Remove every breakpoint.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Whether `node_index` carries a breakpoint.
    #[inline]
    #[must_use]
    pub fn is_breakpoint(&self, node_index: usize) -> bool {
        self.breakpoints.contains(&node_index)
    }

    /// The node indices that currently carry breakpoints, in ascending order.
    pub fn breakpoints(&self) -> impl Iterator<Item = usize> + '_ {
        self.breakpoints.iter().copied()
    }

    /// Resolve a system name (as reported by
    /// [`System::name`](crate::system::System::name)) to its node index, so the
    /// inspector can translate a human-picked system into a breakpoint or watch
    /// target. Returns the first match, or `None` if no system has that name.
    #[must_use]
    pub fn find_system(&self, schedule: &Schedule, name: &str) -> Option<usize> {
        schedule.nodes.iter().position(|n| n.system.name() == name)
    }

    /// Process exactly one position of the schedule's order.
    ///
    /// Initializes the schedule if needed (idempotent), then:
    ///
    /// * at the frame boundary (`cursor == len`), wraps the cursor to `0`,
    ///   clears the set cache, and returns [`StepReport::FrameBoundary`] without
    ///   running anything;
    /// * otherwise evaluates the system's gates exactly as the executor does
    ///   and returns [`StepReport::Ran`] (body executed, deferred commands
    ///   applied) or [`StepReport::Skipped`] (gated off).
    ///
    /// Breakpoints are ignored: this is the finest-grained manual step and
    /// always advances past wherever the cursor is parked.
    pub fn step_system(&mut self, schedule: &mut Schedule, world: &mut World) -> StepReport {
        schedule.initialize(world);
        self.pending_breakpoint = None;
        if self.cursor >= schedule.order.len() {
            self.cursor = 0;
            self.set_cache.clear();
            return StepReport::FrameBoundary;
        }
        let idx = schedule.order[self.cursor];
        self.cursor += 1;
        if Self::eval_gates(schedule, idx, &mut self.set_cache, world) {
            schedule.nodes[idx].system.run(world);
            StepReport::Ran(idx)
        } else {
            StepReport::Skipped(idx)
        }
    }

    /// Run from the cursor until the next break-before breakpoint or the frame
    /// boundary, whichever comes first (standard debugger "continue").
    ///
    /// If the cursor is parked on a breakpoint (a previous `continue_frame`
    /// stopped there), that breakpoint is walked past rather than re-triggered;
    /// the run then halts before the *next* breakpoint. Gated-off systems are
    /// skipped silently and do not appear in [`ContinueReport::ran`].
    pub fn continue_frame(&mut self, schedule: &mut Schedule, world: &mut World) -> ContinueReport {
        schedule.initialize(world);
        // The breakpoint we are resuming from, if any, is suppressed for this
        // call only. Node indices are unique within one frame's order, so this
        // can never suppress a *different* breakpoint later in the loop.
        let resume = self.pending_breakpoint.take();
        let mut ran = Vec::new();
        loop {
            if self.cursor >= schedule.order.len() {
                self.cursor = 0;
                self.set_cache.clear();
                return ContinueReport {
                    ran,
                    stop: ContinueStop::FrameBoundary,
                };
            }
            let idx = schedule.order[self.cursor];
            if self.breakpoints.contains(&idx) && Some(idx) != resume {
                self.pending_breakpoint = Some(idx);
                return ContinueReport {
                    ran,
                    stop: ContinueStop::Breakpoint(idx),
                };
            }
            self.cursor += 1;
            if Self::eval_gates(schedule, idx, &mut self.set_cache, world) {
                schedule.nodes[idx].system.run(world);
                ran.push(idx);
            }
        }
    }

    /// Run the rest of the current frame to the frame boundary, ignoring every
    /// breakpoint, and return the node indices that ran (in order). Afterwards
    /// the cursor is parked at the boundary and the set cache cleared, ready for
    /// the next frame. Useful to "finish this frame" from a breakpoint.
    pub fn run_remaining_frame(&mut self, schedule: &mut Schedule, world: &mut World) -> Vec<usize> {
        schedule.initialize(world);
        self.pending_breakpoint = None;
        let mut ran = Vec::new();
        while self.cursor < schedule.order.len() {
            let idx = schedule.order[self.cursor];
            self.cursor += 1;
            if Self::eval_gates(schedule, idx, &mut self.set_cache, world) {
                schedule.nodes[idx].system.run(world);
                ran.push(idx);
            }
        }
        self.cursor = 0;
        self.set_cache.clear();
        ran
    }

    /// Evaluate whether the system at `idx` should run, mirroring
    /// [`executor::run_inner`](crate::schedule::executor) exactly: every set the
    /// system belongs to is evaluated without short-circuit (memoised in
    /// `set_cache` so a shared set condition fires once per frame), ANDed
    /// together, and only then are the system's own conditions evaluated with
    /// short-circuit AND.
    fn eval_gates(
        schedule: &mut Schedule,
        idx: usize,
        set_cache: &mut HashMap<SystemSetId, bool>,
        world: &mut World,
    ) -> bool {
        // Clone the id list so the immutable borrow of `nodes` is released
        // before `eval_set_conditions` takes a mutable borrow of the schedule.
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
            // Keep evaluating remaining sets even once gated off, so every
            // set-condition's state advances exactly once this frame.
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
}
