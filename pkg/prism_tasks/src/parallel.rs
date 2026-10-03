//! Data-parallel primitives built on [`TaskPool::scope`].
//!
//! Everything here splits work into chunks with an *adaptive* grain size
//! (derived from the input length and worker count, with a floor so scheduling
//! overhead is amortized) and drives the chunks through a structured scope, so
//! the calling thread helps and nested use cannot deadlock.
//!
//! - [`TaskPool::parallel_for`] / [`TaskPool::par_chunks_mut`]: in-place chunked
//!   mutation.
//! - [`TaskPool::par_chunks`]: read-only chunked traversal.
//! - [`TaskPool::par_for_each`] / [`TaskPool::par_for_each_mut`]: per-element.
//! - [`TaskPool::reduce`]: parallel reduction with a deterministic tree-shaped
//!   combine order.
//! - [`TaskPool::prefix_sum`]: parallel inclusive scan.

use core::ops::Add;

use crate::TaskPool;

/// Lower bound on chunk size, so per-chunk scheduling cost stays amortized.
const MIN_GRAIN: usize = 1024;

/// Pick an adaptive chunk size for `len` items across `workers` workers.
///
/// Aims for several chunks per worker (good load balancing / stealing) but
/// never smaller than [`MIN_GRAIN`], and never larger than `len`.
fn adaptive_grain(len: usize, workers: usize) -> usize {
    if len == 0 {
        return 1;
    }
    if workers <= 1 {
        return len;
    }
    // ~8 chunks per worker gives the stealer room to balance uneven work.
    let target_chunks = workers.saturating_mul(8).max(1);
    let by_target = len.div_ceil(target_chunks);
    by_target.max(MIN_GRAIN).min(len)
}

/// Combine a vector of partials with a deterministic, tree-shaped order.
///
/// The reduction order depends only on the number of partials, not on timing,
/// so results are reproducible for an associative `combine`.
fn tree_reduce<R>(mut items: Vec<R>, combine: &(impl Fn(R, R) -> R + Sync)) -> Option<R> {
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
    /// Split `data` into adaptively sized chunks and run `body` on each chunk
    /// in parallel, in place. Returns once every chunk has been processed.
    pub fn par_chunks_mut<T, F>(&self, data: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut [T]) + Sync,
    {
        let len = data.len();
        if len == 0 {
            return;
        }
        let grain = adaptive_grain(len, self.worker_count().max(1));
        let body = &body;
        self.scope(|s| {
            for chunk in data.chunks_mut(grain) {
                s.spawn(move || body(chunk));
            }
        });
    }

    /// Chunked in-place parallel map. Alias of [`TaskPool::par_chunks_mut`] that
    /// mirrors the design doc's `parallel_for(&mut data, |chunk| ...)` shape.
    pub fn parallel_for<T, F>(&self, data: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut [T]) + Sync,
    {
        self.par_chunks_mut(data, body);
    }

    /// Split `data` into adaptively sized chunks and run `body` on each chunk
    /// in parallel, read-only.
    pub fn par_chunks<T, F>(&self, data: &[T], body: F)
    where
        T: Sync,
        F: Fn(&[T]) + Sync,
    {
        let len = data.len();
        if len == 0 {
            return;
        }
        let grain = adaptive_grain(len, self.worker_count().max(1));
        let body = &body;
        self.scope(|s| {
            for chunk in data.chunks(grain) {
                s.spawn(move || body(chunk));
            }
        });
    }

    /// Apply `body` to every element of `data` in place, in parallel.
    pub fn par_for_each_mut<T, F>(&self, data: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Sync,
    {
        let body = &body;
        self.par_chunks_mut(data, move |chunk| {
            for item in chunk {
                body(item);
            }
        });
    }

    /// Apply `body` to every element of `data`, in parallel (read-only).
    pub fn par_for_each<T, F>(&self, data: &[T], body: F)
    where
        T: Sync,
        F: Fn(&T) + Sync,
    {
        let body = &body;
        self.par_chunks(data, move |chunk| {
            for item in chunk {
                body(item);
            }
        });
    }

    /// Parallel reduction over `data`.
    ///
    /// `map` lifts each element into the accumulator type `R`, `combine` folds
    /// two accumulators (must be associative), and `identity` produces the
    /// neutral element used to seed each chunk and to answer an empty input.
    /// Per-chunk folds run in parallel; the chunk partials are then combined in
    /// a deterministic tree order, so the result is reproducible.
    pub fn reduce<T, R, ID, M, C>(&self, data: &[T], identity: ID, map: M, combine: C) -> R
    where
        T: Sync,
        R: Send,
        ID: Fn() -> R + Sync,
        M: Fn(&T) -> R + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let len = data.len();
        if len == 0 {
            return identity();
        }
        let grain = adaptive_grain(len, self.worker_count().max(1));
        let num_chunks = len.div_ceil(grain);
        let mut partials: Vec<Option<R>> = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            partials.push(None);
        }

        let identity = &identity;
        let map = &map;
        let combine = &combine;
        self.scope(|s| {
            for (slot, chunk) in partials.iter_mut().zip(data.chunks(grain)) {
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
        tree_reduce(resolved, combine).unwrap_or_else(identity)
    }

    /// In-place parallel *inclusive* prefix sum (scan) of `data`.
    ///
    /// After the call, `data[i]` holds `data[0] + data[1] + ... + data[i]` for
    /// the original values. Uses a three-phase block scan: (1) each block is
    /// scanned locally in parallel and reports its total, (2) block offsets are
    /// accumulated serially, (3) each block adds its offset in parallel. The
    /// combine (`+`) must be associative; results are deterministic.
    pub fn prefix_sum<T>(&self, data: &mut [T])
    where
        T: Copy + Send + Sync + Add<Output = T>,
    {
        let len = data.len();
        if len <= 1 {
            return;
        }
        let grain = adaptive_grain(len, self.worker_count().max(1));

        // Phase 1: local inclusive scan of each block; record block totals.
        let num_blocks = len.div_ceil(grain);
        let mut totals: Vec<Option<T>> = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            totals.push(None);
        }
        self.scope(|s| {
            for (slot, block) in totals.iter_mut().zip(data.chunks_mut(grain)) {
                s.spawn(move || {
                    let mut acc = block[0];
                    for item in block.iter_mut().skip(1) {
                        acc = acc + *item;
                        *item = acc;
                    }
                    *slot = Some(acc);
                });
            }
        });

        // Phase 2: serial exclusive prefix of the block totals -> per-block
        // offsets. Block 0 needs no offset.
        let totals: Vec<T> = totals.into_iter().flatten().collect();
        let mut offsets: Vec<Option<T>> = Vec::with_capacity(num_blocks);
        offsets.push(None);
        let mut running = totals[0];
        for total in totals.iter().skip(1) {
            offsets.push(Some(running));
            running = running + *total;
        }

        // Phase 3: add each block's offset in parallel.
        let offsets = &offsets;
        self.scope(|s| {
            for (i, block) in data.chunks_mut(grain).enumerate() {
                if let Some(offset) = offsets[i] {
                    s.spawn(move || {
                        for item in block.iter_mut() {
                            *item = *item + offset;
                        }
                    });
                }
            }
        });
    }
}
