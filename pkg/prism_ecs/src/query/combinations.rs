//! [`QueryCombinationsIter`]: the unordered `K`-subsets of a query's matched
//! rows (design §7, the read-only half of Bevy's `iter_combinations`).
//!
//! Pairwise algorithms — broad-phase candidate generation, constraint solving,
//! flocking neighbours — want "every unordered pair of matching entities", not
//! the cartesian product and not `i == j`. This iterator yields each
//! combination exactly once, in a deterministic lexicographic order over the
//! matched rows, so a physics or AI pass sees a reproducible sequence.
//!
//! # Why it is read-only
//!
//! A yield hands out `K` items at once. For `&mut T` terms the same entity could
//! then appear in two live combinations (row 0 with 1, and 0 with 2), which
//! would alias one row — so combination iteration is restricted to
//! [`ReadOnlyQueryData`], where handing out the same shared row twice is
//! harmless. A mutating pairwise pass writes through a separate query (or a
//! command buffer) instead.
//!
//! # Cost
//!
//! Matching the rows costs one full pass (the same traversal
//! [`QueryIter`](crate::query::QueryIter) uses, so the per-row
//! `Added`/`Changed`/sparse gate lives in exactly one place). Each yielded
//! combination then re-fetches its `K` entities at `O(1)` each. Total work is
//! `O(matched + K * C(matched, K))`.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::change::Tick;
use crate::entity::Entity;
use crate::query::fetch::{QueryData, ReadOnlyQueryData};
use crate::query::filter::QueryFilter;
use crate::query::state::QueryState;
use crate::world::World;

/// Iterator over every `K`-subset of the rows a query matches (design §7).
///
/// Created by [`QueryState::iter_combinations`](crate::query::QueryState::iter_combinations).
/// With `K = 2` this is the unordered-pair iterator a broad phase wants; with
/// `K = 1` it is equivalent to the ordinary read-only iteration.
pub struct QueryCombinationsIter<'w, 's, D: QueryData, F: QueryFilter, const K: usize> {
    /// Resolved query state, borrowed so it stays reusable across iterations.
    state: &'s QueryState<D, F>,
    /// The world under iteration; only ever read (the read-only bound).
    world: *mut World,
    /// The change-detection window captured when the row set was collected.
    last_run: Tick,
    this_run: Tick,
    /// The matched rows, in iteration order. Combinations index into this.
    entities: Vec<Entity>,
    /// The combination currently pointed at, always strictly ascending.
    indices: [usize; K],
    /// Set once the last combination has been yielded.
    exhausted: bool,
    _marker: PhantomData<(&'w World, fn() -> (D, F))>,
}

impl<'w, 's, D: QueryData, F: QueryFilter, const K: usize> QueryCombinationsIter<'w, 's, D, F, K> {
    /// Build a combination iterator over an already-collected row set.
    ///
    /// # Safety
    /// - `world` must be valid for reads for `'w`, and the caller must guarantee
    ///   that nothing mutates the columns this query's `D`/`F` terms touch for
    ///   the duration of the iteration (upheld by the read-only entry point plus
    ///   the scheduler's access discipline).
    /// - `state` must have been built for the same `world`, and every entity in
    ///   `entities` must match it for `(last_run, this_run]`.
    pub(crate) unsafe fn new(
        state: &'s QueryState<D, F>,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
        entities: Vec<Entity>,
    ) -> Self {
        let exhausted = K == 0 || K > entities.len();
        Self {
            state,
            world,
            last_run,
            this_run,
            entities,
            indices: core::array::from_fn(|i| i),
            exhausted,
            _marker: PhantomData,
        }
    }

    /// Advance `indices` to the next `K`-subset in lexicographic order.
    ///
    /// Returns `false` when the last combination has been reached. The rightmost
    /// index that still has room is bumped, and every index after it is packed
    /// immediately behind it — the standard combination odometer.
    fn advance(&mut self) -> bool {
        let n = self.entities.len();
        let mut i = K;
        while i > 0 {
            i -= 1;
            // The largest value index `i` may take while leaving room for the
            // `K - 1 - i` indices after it.
            let headroom = n - (K - i);
            if self.indices[i] < headroom {
                self.indices[i] += 1;
                for j in (i + 1)..K {
                    self.indices[j] = self.indices[j - 1] + 1;
                }
                return true;
            }
        }
        false
    }
}

impl<'w, 's, D: ReadOnlyQueryData, F: QueryFilter, const K: usize> Iterator
    for QueryCombinationsIter<'w, 's, D, F, K>
{
    type Item = [D::Item<'w>; K];

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let indices = self.indices;
        let items = core::array::from_fn(|slot| {
            let entity = self.entities[indices[slot]];
            // SAFETY: `D: ReadOnlyQueryData`, so `get_from_ptr` only forms shared
            // fetches; every collected entity matched this state for the same
            // `(last_run, this_run]` window, so it must resolve here.
            unsafe {
                self.state
                    .get_from_ptr(self.world, entity, self.last_run, self.this_run)
            }
            .expect("a row that matched at collection must match for the same window")
        });
        if !self.advance() {
            self.exhausted = true;
        }
        Some(items)
    }
}
