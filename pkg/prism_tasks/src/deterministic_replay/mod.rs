//! Deterministic parallel record/replay (design §24.7 「确定性并行（可回放）」).
//!
//! Rollback netcode and record/replay debugging need a run to be reproducible:
//! *same input → same output*, no matter how many workers were free or in what
//! order they stole and finished work. For an **order-sensitive** reduction
//! (a non-commutative fold — sequence concatenation, hashing, `wrapping_mul`
//! chains) the result depends on the order results are committed, and that
//! order is a genuine race in a work-stealing pool.
//!
//! This module makes that race reproducible without changing the result:
//!
//! * [`TaskPool::record_ordered`] runs a workload on the real pool, computing
//!   each task's value in parallel and committing it under a lock. The order in
//!   which tasks reach that commit point is the natural (non-deterministic)
//!   finish order; it is captured verbatim into an [`ExecutionOrder`] alongside
//!   the resulting value.
//! * [`TaskPool::replay_ordered`] takes a recorded [`ExecutionOrder`], recomputes
//!   every task's value in parallel, then folds the values into the accumulator
//!   **strictly in the recorded order**. Because the fold order is pinned, the
//!   result is bit-identical to the recorded run — on any worker count, on any
//!   machine.
//! * [`TaskPool::deterministic_ordered`] skips recording entirely and folds in
//!   the canonical `0, 1, …, n-1` order. This is the worker-count-independent
//!   "same seed → same result" primitive: nothing but the seed-derived inputs
//!   and the fixed order feed the fold.
//!
//! # Determinism model
//! The *only* non-deterministic input is **which commit lands when** during a
//! recording. [`ExecutionOrder`] captures that as a permutation of task
//! indices; [`fold_in_order`] is the pure serial oracle a replay reproduces.
//! The per-task compute closure must be a pure function of the task index (use
//! [`SeedStream::for_task`] to derive reproducible per-task inputs from a
//! seed); given that, record → replay and `deterministic_ordered` are fully
//! deterministic. See [`order`] for the clock-free, thread-free core.
//!
//! # Layering
//! The façade reuses [`TaskPool::scope`] (its help-on-wait join barrier and
//! single-threaded inline fallback) for the parallel compute phase and a plain
//! [`std::sync::Mutex`] for the serialized commit point. It introduces no new
//! concurrency primitive and no `unsafe`.
#![forbid(unsafe_code)]

pub mod order;

use alloc::vec::Vec;
use std::sync::Mutex;

pub use order::{fold_in_order, ExecutionOrder, ReplayOrderError, SeedStream};

use crate::TaskPool;

/// The result of a [`TaskPool::record_ordered`] call: the folded value plus the
/// [`ExecutionOrder`] that produced it.
///
/// Feed [`ReplayOutcome::order`] back into [`TaskPool::replay_ordered`] to
/// reproduce [`ReplayOutcome::value`] exactly.
#[derive(Clone, Debug)]
pub struct ReplayOutcome<A> {
    /// The accumulator after folding every task in the recorded commit order.
    pub value: A,
    /// The captured commit order, serializable and replayable.
    pub order: ExecutionOrder,
}

impl TaskPool {
    /// Run an order-sensitive parallel fold while **recording** the commit
    /// order (design §24.7).
    ///
    /// `compute(i)` produces task `i`'s value in parallel and must be a pure
    /// function of `i`. Each task then commits under a lock, folding its value
    /// into the accumulator with `combine(acc, i, value)`; the order tasks
    /// reach that lock is the genuine (non-deterministic) finish order and is
    /// captured into the returned [`ReplayOutcome::order`].
    ///
    /// The returned value equals `fold_in_order(order, init, compute, combine)`
    /// — i.e. replaying the captured order reproduces it exactly.
    ///
    /// # Panics
    /// Panics if `len` exceeds `u32::MAX`.
    ///
    /// ```
    /// # use prism_tasks::{SeedStream, TaskPool, fold_in_order};
    /// let pool = TaskPool::with_threads(4);
    /// let seed = 0xC0FF_EE00_u64;
    /// // Order-sensitive combine: hashing fold.
    /// let compute = |i: usize| SeedStream::for_task(seed, i as u64);
    /// let combine = |acc: u64, _i: usize, v: u64| acc.wrapping_mul(0x100_0000_01b3) ^ v;
    /// let out = pool.record_ordered(seed, 1000, compute, 0xcbf2_9ce4_8422_2325, combine);
    /// // The recorded run equals the serial oracle fold over its own order.
    /// assert_eq!(
    ///     out.value,
    ///     fold_in_order(out.order.order(), 0xcbf2_9ce4_8422_2325, compute, combine)
    /// );
    /// ```
    pub fn record_ordered<V, A, F, C>(
        &self,
        seed: u64,
        len: usize,
        compute: F,
        init: A,
        combine: C,
    ) -> ReplayOutcome<A>
    where
        V: Send,
        A: Send,
        F: Fn(usize) -> V + Sync,
        C: Fn(A, usize, V) -> A + Sync,
    {
        let state: Mutex<(Option<A>, Vec<u32>)> = Mutex::new((Some(init), Vec::with_capacity(len)));
        {
            let state_ref = &state;
            let compute_ref = &compute;
            let combine_ref = &combine;
            self.scope(|s| {
                for i in 0..len {
                    s.spawn(move || {
                        let value = compute_ref(i);
                        let id = u32::try_from(i).expect("task index exceeds u32::MAX");
                        let mut guard = state_ref.lock().expect("record state mutex poisoned");
                        let (acc_slot, order) = &mut *guard;
                        order.push(id);
                        let acc = acc_slot.take().expect("accumulator missing during record");
                        *acc_slot = Some(combine_ref(acc, i, value));
                    });
                }
            });
        }
        let (acc, order) = state.into_inner().expect("record state mutex poisoned");
        ReplayOutcome {
            value: acc.expect("accumulator missing after record"),
            order: ExecutionOrder::from_parts(seed, order),
        }
    }

    /// Replay a recorded [`ExecutionOrder`], folding in the recorded order
    /// (design §24.7).
    ///
    /// Each task's value is recomputed in parallel with `compute` (which must
    /// be the same pure function used when recording), then folded into `init`
    /// with `combine(acc, i, value)` strictly in the recorded commit order. The
    /// result is bit-identical to the recorded run regardless of this pool's
    /// worker count.
    ///
    /// # Errors
    /// Returns [`ReplayOrderError`] if `order` is not a valid permutation of
    /// its own length.
    pub fn replay_ordered<V, A, F, C>(
        &self,
        order: &ExecutionOrder,
        compute: F,
        init: A,
        combine: C,
    ) -> Result<A, ReplayOrderError>
    where
        V: Send,
        F: Fn(usize) -> V + Sync,
        C: Fn(A, usize, V) -> A,
    {
        order.validate()?;
        Ok(self.fold_values_in_order(order.order(), compute, init, combine))
    }

    /// Run an order-sensitive parallel fold in the **canonical** `0..len` order
    /// (design §24.7).
    ///
    /// This is the worker-count-independent determinism primitive: results are
    /// computed in parallel but committed in a fixed index order, so the result
    /// depends only on the seed-derived inputs and never on steal timing or the
    /// number of workers. `seed` is recorded for provenance; thread it into
    /// `compute` (e.g. via [`SeedStream::for_task`]) to make *same seed → same
    /// result* meaningful.
    ///
    /// # Panics
    /// Panics if `len` exceeds `u32::MAX`.
    pub fn deterministic_ordered<V, A, F, C>(
        &self,
        seed: u64,
        len: usize,
        compute: F,
        init: A,
        combine: C,
    ) -> A
    where
        V: Send,
        F: Fn(usize) -> V + Sync,
        C: Fn(A, usize, V) -> A,
    {
        let order = ExecutionOrder::canonical(seed, len);
        self.fold_values_in_order(order.order(), compute, init, combine)
    }

    /// Shared engine for [`TaskPool::replay_ordered`] and
    /// [`TaskPool::deterministic_ordered`]: compute every task's value in
    /// parallel into its own slot, then fold the slots into `init` following
    /// `order` on the calling thread. `order` must be a permutation of
    /// `0..order.len()` (the callers guarantee it).
    fn fold_values_in_order<V, A, F, C>(
        &self,
        order: &[u32],
        compute: F,
        init: A,
        combine: C,
    ) -> A
    where
        V: Send,
        F: Fn(usize) -> V + Sync,
        C: Fn(A, usize, V) -> A,
    {
        let n = order.len();
        let slots: Vec<Mutex<Option<V>>> = (0..n).map(|_| Mutex::new(None)).collect();
        {
            let slots_ref = &slots;
            let compute_ref = &compute;
            self.scope(|s| {
                for (i, slot) in slots_ref.iter().enumerate() {
                    s.spawn(move || {
                        let value = compute_ref(i);
                        *slot.lock().expect("replay slot mutex poisoned") = Some(value);
                    });
                }
            });
        }
        let mut acc = init;
        for &id in order {
            let idx = id as usize;
            let value = slots[idx]
                .lock()
                .expect("replay slot mutex poisoned")
                .take()
                .expect("replay slot empty; order is not a permutation");
            acc = combine(acc, idx, value);
        }
        acc
    }
}
