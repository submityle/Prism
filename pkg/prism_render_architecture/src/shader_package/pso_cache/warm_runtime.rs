//! Runtime-side deterministic lifecycle for warming pipeline-state objects.
//!
//! [`WarmSetPlanner`](super::WarmSetPlanner) decides *what* to precompile;
//! [`WarmRuntime`] models *the execution lifecycle* of those decisions plus the
//! per-draw gate that keeps the main thread hitch-free. It is the deterministic
//! bookkeeping half of the asynchronous warm pipeline: the engine-side consumer
//! owns real `wgpu` compilation, threads, and driver I/O, while this model
//! tracks each pipeline's [`PipelineReadiness`] and answers "may I draw this
//! pipeline right now?" via [`DrawDecision`].
//!
//! # Lifecycle
//!
//! A pipeline moves `Absent → Queued → Compiling → {Ready | Failed}`:
//!
//! - [`WarmRuntime::enqueue`] / [`WarmRuntime::enqueue_plan`] move
//!   `Absent`/`Failed` keys to `Queued` (a retry re-arms a previous failure).
//! - [`WarmRuntime::begin_compile`] moves `Queued → Compiling`.
//! - [`WarmRuntime::finish_compile`] moves `Compiling → Ready`.
//! - [`WarmRuntime::fail_compile`] moves `Compiling → Failed`.
//!
//! # Draw gating
//!
//! [`WarmRuntime::request_draw`] never blocks the main thread on compilation:
//!
//! - `Ready` → [`DrawDecision::Draw`].
//! - still compiling / queued → [`DrawDecision::Placeholder`] when the caller
//!   permits a placeholder PSO, else [`DrawDecision::Skip`].
//! - `Absent` / `Failed` → the key is recorded as a runtime miss and re-queued,
//!   then the same placeholder/skip choice applies.
//!
//! Observed misses are drained via [`WarmRuntime::drain_misses`] and fed back
//! into [`WarmSetPlanner::observe_miss`](super::WarmSetPlanner::observe_miss),
//! closing the loop so the next warm pass covers whatever hitched.
//!
//! All state is pure, deterministic bookkeeping ordered by [`PsoCacheKey`]; the
//! instrumentation counters let a consumer assert the AAA acceptance target
//! ("after warming, runtime PSO compiles == 0") directly in a golden test.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::warm_set::{WarmSetPlan, WarmSetPlanner};
use super::PsoCacheKey;

/// Compilation state of a single pipeline within the warm runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineReadiness {
    /// Never requested; the runtime has no record of it yet.
    Absent,
    /// Scheduled for compilation but not yet started.
    Queued,
    /// Currently being compiled by the (external) async warm worker.
    Compiling,
    /// Compiled and resident; safe to draw.
    Ready,
    /// Compilation attempted and failed; eligible for re-queue/retry.
    Failed,
}

impl PipelineReadiness {
    /// Whether a pipeline in this state can be drawn without compilation.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether this state is eligible to be (re-)queued for compilation.
    ///
    /// `Absent` has no record yet and `Failed` warrants a retry; `Queued`,
    /// `Compiling`, and `Ready` are already in flight or done.
    #[must_use]
    pub const fn is_enqueueable(self) -> bool {
        matches!(self, Self::Absent | Self::Failed)
    }
}

/// The per-draw gate decision for a requested pipeline.
///
/// The main thread never stalls on compilation; it either draws the ready
/// pipeline, substitutes a placeholder while compilation proceeds, or skips the
/// draw entirely when no placeholder is acceptable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DrawDecision {
    /// Pipeline is `Ready`; issue the draw with the real PSO.
    Draw,
    /// Pipeline is not ready; draw with a stand-in PSO this frame.
    Placeholder,
    /// Pipeline is not ready and no placeholder is permitted; skip the draw.
    Skip,
}

/// Instrumentation counters for the warm runtime.
///
/// These make the AAA acceptance criteria assertable: once warming has
/// completed, [`WarmCounters::runtime_misses`] and
/// [`WarmCounters::compiles_started`] should stop advancing during steady-state
/// rendering (ideally zero new compiles), and
/// [`WarmCounters::distinct_missed`] bounds how many unique pipelines ever
/// hitched.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WarmCounters {
    /// Total `Queued → Compiling` transitions.
    pub compiles_started: u64,
    /// Total `Compiling → Ready` transitions.
    pub compiles_finished: u64,
    /// Total `Compiling → Failed` transitions.
    pub compiles_failed: u64,
    /// Total draw requests that found a non-ready pipeline (observed hitches).
    pub runtime_misses: u64,
    /// Count of distinct keys that have ever been a runtime miss.
    pub distinct_missed: u64,
}

/// Deterministic runtime lifecycle + draw gate for warmed pipelines.
///
/// Backend-agnostic and side-effect free: it records readiness transitions and
/// miss feedback, leaving real compilation, threading, and driver I/O to the
/// engine-side consumer.
#[derive(Clone, Debug, Default)]
pub struct WarmRuntime {
    readiness: BTreeMap<PsoCacheKey, PipelineReadiness>,
    /// Keys observed as runtime misses, pending feedback to the planner.
    pending_misses: BTreeMap<PsoCacheKey, ()>,
    /// Keys that have been a runtime miss at least once (for `distinct_missed`).
    ever_missed: BTreeMap<PsoCacheKey, ()>,
    counters: WarmCounters,
}

impl WarmRuntime {
    /// Creates an empty runtime with no tracked pipelines.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current readiness of `key` (`Absent` if never recorded).
    #[must_use]
    pub fn readiness(&self, key: &PsoCacheKey) -> PipelineReadiness {
        self.readiness
            .get(key)
            .copied()
            .unwrap_or(PipelineReadiness::Absent)
    }

    /// Instrumentation counters snapshot.
    #[must_use]
    pub fn counters(&self) -> WarmCounters {
        self.counters
    }

    /// Number of tracked pipelines in any non-`Absent` state.
    #[must_use]
    pub fn tracked_len(&self) -> usize {
        self.readiness.len()
    }

    /// Queues `key` for compilation if it is `Absent` or `Failed`.
    ///
    /// Returns `true` if the key transitioned to `Queued`, `false` if it was
    /// already `Queued`, `Compiling`, or `Ready` (idempotent no-op).
    pub fn enqueue(&mut self, key: PsoCacheKey) -> bool {
        if self.readiness(&key).is_enqueueable() {
            self.readiness.insert(key, PipelineReadiness::Queued);
            true
        } else {
            false
        }
    }

    /// Queues every pipeline in `plan` that is `Absent`/`Failed`.
    ///
    /// Returns the number of keys newly transitioned to `Queued`.
    pub fn enqueue_plan(&mut self, plan: &WarmSetPlan) -> usize {
        let mut queued = 0;
        for request in plan.entries() {
            if self.enqueue(request.key.clone()) {
                queued += 1;
            }
        }
        queued
    }

    /// Transitions `key` `Queued → Compiling`.
    ///
    /// Returns `true` on a valid transition; `false` if the key was not
    /// `Queued` (already compiling/ready/absent — the caller raced or
    /// double-started).
    pub fn begin_compile(&mut self, key: &PsoCacheKey) -> bool {
        if self.readiness(key) == PipelineReadiness::Queued {
            self.readiness
                .insert(key.clone(), PipelineReadiness::Compiling);
            self.counters.compiles_started += 1;
            true
        } else {
            false
        }
    }

    /// Transitions `key` `Compiling → Ready`.
    ///
    /// Returns `true` on a valid transition; `false` if the key was not
    /// `Compiling`.
    pub fn finish_compile(&mut self, key: &PsoCacheKey) -> bool {
        if self.readiness(key) == PipelineReadiness::Compiling {
            self.readiness.insert(key.clone(), PipelineReadiness::Ready);
            self.counters.compiles_finished += 1;
            true
        } else {
            false
        }
    }

    /// Transitions `key` `Compiling → Failed`.
    ///
    /// Returns `true` on a valid transition; `false` if the key was not
    /// `Compiling`. A failed pipeline is eligible for re-queue via
    /// [`WarmRuntime::enqueue`].
    pub fn fail_compile(&mut self, key: &PsoCacheKey) -> bool {
        if self.readiness(key) == PipelineReadiness::Compiling {
            self.readiness
                .insert(key.clone(), PipelineReadiness::Failed);
            self.counters.compiles_failed += 1;
            true
        } else {
            false
        }
    }

    /// Gates a draw of `key` against its readiness without ever blocking.
    ///
    /// - `Ready` → [`DrawDecision::Draw`].
    /// - otherwise the draw is a runtime miss: it is counted, and `Absent`/
    ///   `Failed` keys are re-queued and recorded for planner feedback. The
    ///   decision is [`DrawDecision::Placeholder`] when `allow_placeholder`,
    ///   else [`DrawDecision::Skip`].
    pub fn request_draw(&mut self, key: &PsoCacheKey, allow_placeholder: bool) -> DrawDecision {
        let state = self.readiness(key);
        if state.is_ready() {
            return DrawDecision::Draw;
        }

        // Non-ready draw is an observed hitch.
        self.counters.runtime_misses += 1;
        if self.ever_missed.insert(key.clone(), ()).is_none() {
            self.counters.distinct_missed += 1;
        }
        // Only `Absent`/`Failed` need (re-)queueing and planner feedback;
        // `Queued`/`Compiling` are already in flight.
        if state.is_enqueueable() {
            self.enqueue(key.clone());
            self.pending_misses.insert(key.clone(), ());
        }

        if allow_placeholder {
            DrawDecision::Placeholder
        } else {
            DrawDecision::Skip
        }
    }

    /// Drains and returns observed runtime misses in deterministic key order.
    ///
    /// Feed each into
    /// [`WarmSetPlanner::observe_miss`](super::WarmSetPlanner::observe_miss) to
    /// promote the hitched pipelines in the next warm pass.
    pub fn drain_misses(&mut self) -> Vec<PsoCacheKey> {
        let drained: Vec<PsoCacheKey> = self.pending_misses.keys().cloned().collect();
        self.pending_misses.clear();
        drained
    }

    /// Convenience: drain misses straight into `planner` as observed misses.
    ///
    /// Returns the number of misses fed back.
    pub fn feed_misses_into(&mut self, planner: &mut WarmSetPlanner) -> usize {
        let misses = self.drain_misses();
        let count = misses.len();
        for key in misses {
            planner.observe_miss(key);
        }
        count
    }

    /// Whether every tracked pipeline is `Ready` (nothing queued/compiling/
    /// failed). An empty runtime is trivially complete.
    #[must_use]
    pub fn is_warm_complete(&self) -> bool {
        self.readiness.values().all(|state| state.is_ready())
    }

    /// Number of tracked pipelines not yet `Ready`.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.readiness
            .values()
            .filter(|state| !state.is_ready())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::super::warm_set::WarmPriority;
    use super::super::PipelineStateHash;
    use super::*;
    use crate::shader_package::ShaderPackageId;

    fn key(pkg: &str, perm: u64) -> PsoCacheKey {
        PsoCacheKey::new(ShaderPackageId::new(pkg), perm, PipelineStateHash(0))
    }

    #[test]
    fn absent_by_default() {
        let rt = WarmRuntime::new();
        assert_eq!(rt.readiness(&key("a", 0)), PipelineReadiness::Absent);
        assert_eq!(rt.tracked_len(), 0);
        assert!(rt.is_warm_complete());
    }

    #[test]
    fn full_lifecycle_absent_to_ready() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        assert!(rt.enqueue(k.clone()));
        assert_eq!(rt.readiness(&k), PipelineReadiness::Queued);
        assert!(rt.begin_compile(&k));
        assert_eq!(rt.readiness(&k), PipelineReadiness::Compiling);
        assert!(rt.finish_compile(&k));
        assert_eq!(rt.readiness(&k), PipelineReadiness::Ready);

        let c = rt.counters();
        assert_eq!(c.compiles_started, 1);
        assert_eq!(c.compiles_finished, 1);
        assert_eq!(c.compiles_failed, 0);
        assert!(rt.is_warm_complete());
    }

    #[test]
    fn enqueue_is_idempotent_for_in_flight_and_ready() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        assert!(rt.enqueue(k.clone()));
        // Already queued → no transition.
        assert!(!rt.enqueue(k.clone()));
        rt.begin_compile(&k);
        assert!(!rt.enqueue(k.clone()));
        rt.finish_compile(&k);
        assert!(!rt.enqueue(k.clone()));
    }

    #[test]
    fn failed_pipeline_can_be_requeued() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        rt.enqueue(k.clone());
        rt.begin_compile(&k);
        assert!(rt.fail_compile(&k));
        assert_eq!(rt.readiness(&k), PipelineReadiness::Failed);
        assert_eq!(rt.counters().compiles_failed, 1);
        // Retry re-arms it.
        assert!(rt.enqueue(k.clone()));
        assert_eq!(rt.readiness(&k), PipelineReadiness::Queued);
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        // Can't begin_compile without queue.
        assert!(!rt.begin_compile(&k));
        // Can't finish/fail without compiling.
        assert!(!rt.finish_compile(&k));
        assert!(!rt.fail_compile(&k));
        rt.enqueue(k.clone());
        // Still can't finish while merely queued.
        assert!(!rt.finish_compile(&k));
        assert_eq!(rt.counters().compiles_started, 0);
    }

    #[test]
    fn enqueue_plan_counts_only_new_queues() {
        let mut planner = WarmSetPlanner::new();
        planner.require(key("a", 0), WarmPriority::Normal);
        planner.require(key("b", 0), WarmPriority::Critical);
        let plan = planner.plan();

        let mut rt = WarmRuntime::new();
        assert_eq!(rt.enqueue_plan(&plan), 2);
        // Re-applying the same plan queues nothing new.
        assert_eq!(rt.enqueue_plan(&plan), 0);
        assert_eq!(rt.tracked_len(), 2);
    }

    #[test]
    fn request_draw_ready_draws_without_miss() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        rt.enqueue(k.clone());
        rt.begin_compile(&k);
        rt.finish_compile(&k);
        assert_eq!(rt.request_draw(&k, true), DrawDecision::Draw);
        assert_eq!(rt.counters().runtime_misses, 0);
        assert!(rt.drain_misses().is_empty());
    }

    #[test]
    fn request_draw_absent_misses_and_requeues() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        assert_eq!(rt.request_draw(&k, true), DrawDecision::Placeholder);
        assert_eq!(rt.readiness(&k), PipelineReadiness::Queued);
        let c = rt.counters();
        assert_eq!(c.runtime_misses, 1);
        assert_eq!(c.distinct_missed, 1);
        assert_eq!(rt.drain_misses(), alloc::vec![k]);
    }

    #[test]
    fn request_draw_without_placeholder_skips() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        assert_eq!(rt.request_draw(&k, false), DrawDecision::Skip);
        assert_eq!(rt.counters().runtime_misses, 1);
    }

    #[test]
    fn repeated_misses_count_once_distinct() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        rt.request_draw(&k, true); // Absent → Queued, miss #1
        rt.request_draw(&k, true); // Queued → still miss #2, no requeue
        rt.request_draw(&k, true); // miss #3
        let c = rt.counters();
        assert_eq!(c.runtime_misses, 3);
        assert_eq!(c.distinct_missed, 1);
        // Only one pending-miss entry despite three draws.
        assert_eq!(rt.drain_misses(), alloc::vec![k]);
    }

    #[test]
    fn compiling_draw_does_not_requeue_but_counts_miss() {
        let mut rt = WarmRuntime::new();
        let k = key("a", 0);
        rt.enqueue(k.clone());
        rt.begin_compile(&k);
        assert_eq!(rt.request_draw(&k, true), DrawDecision::Placeholder);
        // Still Compiling (not reset to Queued), but miss recorded.
        assert_eq!(rt.readiness(&k), PipelineReadiness::Compiling);
        assert_eq!(rt.counters().runtime_misses, 1);
        // No planner feedback needed — it is already in flight.
        assert!(rt.drain_misses().is_empty());
    }

    #[test]
    fn drain_misses_is_deterministic_and_clears() {
        let mut rt = WarmRuntime::new();
        rt.request_draw(&key("b", 0), true);
        rt.request_draw(&key("a", 0), true);
        rt.request_draw(&key("a", 1), true);
        let drained = rt.drain_misses();
        assert_eq!(drained, alloc::vec![key("a", 0), key("a", 1), key("b", 0)]);
        // Draining clears the pending set.
        assert!(rt.drain_misses().is_empty());
    }

    #[test]
    fn feed_misses_into_promotes_in_planner() {
        let mut rt = WarmRuntime::new();
        rt.request_draw(&key("a", 0), true);
        let mut planner = WarmSetPlanner::new();
        assert_eq!(rt.feed_misses_into(&mut planner), 1);
        let plan = planner.plan();
        assert_eq!(plan.entries().len(), 1);
        assert_eq!(plan.entries()[0].priority, WarmPriority::High);
        // Misses were drained.
        assert!(rt.drain_misses().is_empty());
    }

    #[test]
    fn warm_complete_only_when_all_ready() {
        let mut rt = WarmRuntime::new();
        let a = key("a", 0);
        let b = key("b", 0);
        rt.enqueue(a.clone());
        rt.enqueue(b.clone());
        assert!(!rt.is_warm_complete());
        assert_eq!(rt.pending_count(), 2);

        rt.begin_compile(&a);
        rt.finish_compile(&a);
        assert!(!rt.is_warm_complete());
        assert_eq!(rt.pending_count(), 1);

        rt.begin_compile(&b);
        rt.finish_compile(&b);
        assert!(rt.is_warm_complete());
        assert_eq!(rt.pending_count(), 0);
    }

    #[test]
    fn acceptance_zero_compiles_after_warm() {
        // Model the AAA target: warm everything up front, then a steady-state
        // frame draws only ready pipelines → no new compiles, no misses.
        let mut planner = WarmSetPlanner::new();
        for i in 0..8 {
            planner.require(key("mat", i), WarmPriority::Normal);
        }
        let plan = planner.plan();

        let mut rt = WarmRuntime::new();
        rt.enqueue_plan(&plan);
        for req in plan.entries() {
            rt.begin_compile(&req.key);
            rt.finish_compile(&req.key);
        }
        assert!(rt.is_warm_complete());

        let compiles_after_warm = rt.counters().compiles_started;
        // Steady-state frame: draw every pipeline.
        for i in 0..8 {
            assert_eq!(rt.request_draw(&key("mat", i), true), DrawDecision::Draw);
        }
        let c = rt.counters();
        assert_eq!(c.compiles_started, compiles_after_warm, "no new compiles");
        assert_eq!(c.runtime_misses, 0, "no hitches after warm");
    }
}
