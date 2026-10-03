//! Deterministic tree reduction (design §13, §24.7 确定性树形归约).
//!
//! The M1 [`TaskPool::reduce`](crate::TaskPool::reduce) is already tree-shaped,
//! but its chunk boundaries come from an *adaptive* grain that scales with the
//! worker count. For an associative combine that is irrelevant; for a
//! non-associative one (floating-point addition is the classic example) a
//! different split regroups the additions and perturbs the low bits of the
//! result. Rollback-netcode and record/replay cannot tolerate that.
//!
//! [`TaskPool::deterministic_reduce`] fixes both axes of nondeterminism:
//!
//! 1. **Fixed partitioning.** The input is split through a
//!    [`FixedPartition`](crate::FixedPartition) whose boundaries depend only on
//!    the input length — never on the worker count or steal order.
//! 2. **Fixed-order combine.** Each chunk folds its elements left-to-right into
//!    a partial, the partials are written into an index-keyed buffer (not in
//!    completion order), and the partials are then combined with a fixed,
//!    index-ordered binary tree ([`tree_combine`]).
//!
//! The upshot: for a *pure* `map`/`combine`, the result is **bit-identical**
//! across runs and across worker counts. Only the inner per-chunk folds and the
//! tree combine run — the scheduler decides *where* each chunk runs, never
//! *how the result is grouped*.

use alloc::vec::Vec;

use crate::partition::FixedPartition;
use crate::replay::{DeterministicSession, ReplayError};
use crate::TaskPool;

/// Combine `items` with a fixed, tree-shaped (pairwise, left-to-right) order.
///
/// The grouping depends only on `items.len()`, so for an associative `combine`
/// the result equals a serial fold and, crucially, is reproducible regardless
/// of the order in which the partials were produced. Returns `None` for an
/// empty input.
///
/// # Examples
/// ```
/// use prism_tasks::tree_combine;
/// // ((1+2)+(3+4)) grouping, fixed regardless of timing.
/// let sum = tree_combine(vec![1, 2, 3, 4], &|a, b| a + b);
/// assert_eq!(sum, Some(10));
/// ```
pub fn tree_combine<R>(mut items: Vec<R>, combine: &(impl Fn(R, R) -> R + Sync)) -> Option<R> {
    if items.is_empty() {
        return None;
    }
    while items.len() > 1 {
        let mut next = Vec::with_capacity(items.len().div_ceil(2));
        let mut iter = items.into_iter();
        while let Some(a) = iter.next() {
            match iter.next() {
                Some(b) => next.push(combine(a, b)),
                None => next.push(a),
            }
        }
        items = next;
    }
    items.into_iter().next()
}

impl TaskPool {
    /// Deterministic parallel reduction with a fixed split and fixed combine
    /// order.
    ///
    /// `identity` seeds each chunk and answers an empty input, `map` lifts each
    /// element into the accumulator `R`, and `combine` folds two accumulators.
    /// Unlike [`TaskPool::reduce`], the result is **bit-identical** across runs
    /// and worker counts for a pure `map`/`combine`, because the partition and
    /// the combine tree depend only on the input length.
    ///
    /// `combine` need not be mathematically associative for the result to be
    /// *reproducible* (it always is); associativity only determines whether the
    /// reproducible result also equals a left-to-right serial fold.
    ///
    /// ```
    /// # use prism_tasks::TaskPool;
    /// let pool = TaskPool::with_threads(4);
    /// let data: Vec<u64> = (1..=1000).collect();
    /// let sum = pool.deterministic_reduce(&data, || 0, |&x| x, |a, b| a + b);
    /// assert_eq!(sum, (1..=1000).sum());
    /// ```
    pub fn deterministic_reduce<T, R, ID, M, C>(
        &self,
        data: &[T],
        identity: ID,
        map: M,
        combine: C,
    ) -> R
    where
        T: Sync,
        R: Send,
        ID: Fn() -> R + Sync,
        M: Fn(&T) -> R + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let partition = FixedPartition::balanced(data.len());
        self.reduce_over(partition, data, &identity, &map, &combine)
    }

    /// Deterministic reduction whose split is recorded to (or replayed from) a
    /// [`DeterministicSession`].
    ///
    /// In record mode the session derives the split from its seed and logs it;
    /// in replay mode it reproduces the logged split. The reduction result is
    /// bit-identical to the recorded run.
    ///
    /// # Errors
    /// Propagates [`ReplayError`] from the session (e.g. a replay that diverges
    /// from the recorded length, or a record that is exhausted).
    pub fn deterministic_reduce_with<T, R, ID, M, C>(
        &self,
        session: &DeterministicSession,
        data: &[T],
        identity: ID,
        map: M,
        combine: C,
    ) -> Result<R, ReplayError>
    where
        T: Sync,
        R: Send,
        ID: Fn() -> R + Sync,
        M: Fn(&T) -> R + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let partition = session.plan(data.len())?;
        Ok(self.reduce_over(partition, data, &identity, &map, &combine))
    }

    /// Shared engine: fold each fixed chunk into an index-keyed slot, then
    /// combine the slots in fixed tree order.
    fn reduce_over<T, R, ID, M, C>(
        &self,
        partition: FixedPartition,
        data: &[T],
        identity: &ID,
        map: &M,
        combine: &C,
    ) -> R
    where
        T: Sync,
        R: Send,
        ID: Fn() -> R + Sync,
        M: Fn(&T) -> R + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        if partition.chunk_count() == 0 {
            return identity();
        }
        let mut partials: Vec<Option<R>> = Vec::with_capacity(partition.chunk_count());
        for _ in 0..partition.chunk_count() {
            partials.push(None);
        }

        // Each chunk writes its own index-keyed slot, so completion order never
        // affects which partial lands where.
        self.scope(|s| {
            for (slot, range) in partials.iter_mut().zip(partition.chunks()) {
                let chunk = &data[range];
                s.spawn(move || {
                    let mut acc = identity();
                    for item in chunk {
                        acc = combine(acc, map(item));
                    }
                    *slot = Some(acc);
                });
            }
        });

        let resolved: Vec<R> = partials.into_iter().flatten().collect();
        tree_combine(resolved, combine).unwrap_or_else(identity)
    }
}

#[cfg(test)]
mod tests {
    use super::tree_combine;

    #[test]
    fn tree_combine_empty_is_none() {
        let out = tree_combine(Vec::<i32>::new(), &|a, b| a + b);
        assert_eq!(out, None);
    }

    #[test]
    fn tree_combine_single() {
        let out = tree_combine(alloc::vec![7], &|a, b| a + b);
        assert_eq!(out, Some(7));
    }

    #[test]
    fn tree_combine_groups_pairwise() {
        // String concat is associative but not commutative: order is preserved.
        let out = tree_combine(
            alloc::vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ],
            &|a, b| a + &b,
        );
        assert_eq!(out.as_deref(), Some("abcd"));
    }
}
