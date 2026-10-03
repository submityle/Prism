//! ECS-style data-parallel iteration (`par_iter`) layered on this pool.
//!
//! This is the data-parallel API an `ECS` `par_iter` calls into: a parallel,
//! adaptively chunked for-each / map over slices and index ranges that
//! schedules its chunks onto the existing worker pool through
//! [`TaskPool::scope`](crate::TaskPool::scope). Because it is built on `scope`,
//! the calling thread participates as a worker and nested use cannot deadlock,
//! and every chunk is joined before the call returns.
//!
//! Three entry points, all off [`TaskPool`]:
//! - [`TaskPool::par_iter_range`] — parallelize an index range `start..end`.
//! - [`TaskPool::par_iter`] — parallelize a read-only `&[T]`.
//! - [`TaskPool::par_iter_mut`] — parallelize a mutable `&mut [T]`.
//!
//! Each returns a lightweight builder whose `for_each`, `enumerate_for_each`,
//! and `map_collect` terminals run the work. `map_collect` always returns its
//! results in input order, so a parallel `map_collect` is bit-for-bit equal to
//! the equivalent serial `map`/`collect` for a pure mapping function.
//!
//! ## Adaptive granularity
//! Chunk size is chosen from the input length and the worker count (aiming for
//! several chunks per worker so the stealer can balance uneven work), with a
//! floor so per-chunk scheduling cost stays amortized. Call
//! [`ParRange::with_min_len`] (and the slice equivalents) to override the floor
//! — useful for very small inputs where you still want cross-worker spread.

use alloc::vec::Vec;
use core::ops::Range;

use crate::TaskPool;

/// Default lower bound on chunk length, so per-chunk scheduling cost stays
/// amortized relative to the work done per element.
pub const DEFAULT_MIN_LEN: usize = 1024;

/// Target number of chunks produced per worker. More chunks than workers gives
/// the work-stealer slack to balance uneven per-element costs.
const CHUNKS_PER_WORKER: usize = 8;

/// Pick an adaptive chunk length for `len` items across `workers` workers,
/// never smaller than `min_len` (unless `len` itself is smaller) and never
/// larger than `len`.
fn adaptive_grain(len: usize, workers: usize, min_len: usize) -> usize {
    debug_assert!(len > 0, "adaptive_grain requires a non-empty range");
    let min_len = min_len.max(1);
    let workers = workers.max(1);
    if workers == 1 {
        return len;
    }
    let target_chunks = workers.saturating_mul(CHUNKS_PER_WORKER).max(1);
    let by_target = len.div_ceil(target_chunks);
    by_target.max(min_len).min(len)
}

/// Parallel iterator over an index range `start..end` (see
/// [`TaskPool::par_iter_range`]).
pub struct ParRange<'p> {
    pool: &'p TaskPool,
    start: usize,
    end: usize,
    min_len: usize,
}

impl<'p> ParRange<'p> {
    /// Override the minimum chunk length (the adaptive-granularity floor).
    /// Values below `1` are clamped to `1`.
    #[must_use]
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        self.min_len = min_len.max(1);
        self
    }

    /// Number of indices covered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Resolve the chunk length for this range given the pool's worker count.
    fn grain(&self) -> usize {
        adaptive_grain(self.len(), self.pool.worker_count(), self.min_len)
    }

    /// Run `body` on every index in parallel. Returns once every index has been
    /// visited.
    pub fn for_each<F>(self, body: F)
    where
        F: Fn(usize) + Sync,
    {
        if self.is_empty() {
            return;
        }
        let grain = self.grain();
        let body = &body;
        let (start, end) = (self.start, self.end);
        self.pool.scope(|s| {
            let mut chunk_start = start;
            while chunk_start < end {
                let chunk_end = chunk_start.saturating_add(grain).min(end);
                s.spawn(move || {
                    for i in chunk_start..chunk_end {
                        body(i);
                    }
                });
                chunk_start = chunk_end;
            }
        });
    }

    /// Map every index to a value in parallel, collecting the results **in
    /// index order**. Equal to `(start..end).map(body).collect()` run serially
    /// for a pure `body`.
    #[must_use]
    pub fn map_collect<R, F>(self, body: F) -> Vec<R>
    where
        R: Send,
        F: Fn(usize) -> R + Sync,
    {
        if self.is_empty() {
            return Vec::new();
        }
        let len = self.len();
        let grain = self.grain();
        let (start, end) = (self.start, self.end);
        let num_chunks = len.div_ceil(grain);
        let mut partials: Vec<Option<Vec<R>>> = (0..num_chunks).map(|_| None).collect();
        let body = &body;
        self.pool.scope(|s| {
            for (chunk_index, slot) in partials.iter_mut().enumerate() {
                let chunk_start = start + chunk_index * grain;
                let chunk_end = chunk_start.saturating_add(grain).min(end);
                s.spawn(move || {
                    let mut out = Vec::with_capacity(chunk_end - chunk_start);
                    for i in chunk_start..chunk_end {
                        out.push(body(i));
                    }
                    *slot = Some(out);
                });
            }
        });
        let mut result = Vec::with_capacity(len);
        for partial in partials {
            result.extend(partial.expect("every chunk ran under the scope"));
        }
        result
    }
}

/// Parallel iterator over a read-only slice (see [`TaskPool::par_iter`]).
pub struct ParSlice<'p, 'd, T> {
    pool: &'p TaskPool,
    data: &'d [T],
    min_len: usize,
}

impl<'p, 'd, T: Sync> ParSlice<'p, 'd, T> {
    /// Override the minimum chunk length (the adaptive-granularity floor).
    #[must_use]
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        self.min_len = min_len.max(1);
        self
    }

    fn grain(&self) -> usize {
        adaptive_grain(self.data.len(), self.pool.worker_count(), self.min_len)
    }

    /// Apply `body` to every element in parallel (read-only).
    pub fn for_each<F>(self, body: F)
    where
        F: Fn(&T) + Sync,
    {
        if self.data.is_empty() {
            return;
        }
        let grain = self.grain();
        let body = &body;
        self.pool.scope(|s| {
            for chunk in self.data.chunks(grain) {
                s.spawn(move || {
                    for item in chunk {
                        body(item);
                    }
                });
            }
        });
    }

    /// Apply `body` to every `(index, &element)` pair in parallel, where
    /// `index` is the element's position in the original slice.
    pub fn enumerate_for_each<F>(self, body: F)
    where
        F: Fn(usize, &T) + Sync,
    {
        if self.data.is_empty() {
            return;
        }
        let grain = self.grain();
        let body = &body;
        self.pool.scope(|s| {
            for (chunk_index, chunk) in self.data.chunks(grain).enumerate() {
                let base = chunk_index * grain;
                s.spawn(move || {
                    for (offset, item) in chunk.iter().enumerate() {
                        body(base + offset, item);
                    }
                });
            }
        });
    }

    /// Map every element to a value in parallel, collecting the results **in
    /// slice order**. Equal to `data.iter().map(body).collect()` run serially.
    #[must_use]
    pub fn map_collect<R, F>(self, body: F) -> Vec<R>
    where
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        if self.data.is_empty() {
            return Vec::new();
        }
        let grain = self.grain();
        let num_chunks = self.data.len().div_ceil(grain);
        let mut partials: Vec<Option<Vec<R>>> = (0..num_chunks).map(|_| None).collect();
        let body = &body;
        self.pool.scope(|s| {
            for (slot, chunk) in partials.iter_mut().zip(self.data.chunks(grain)) {
                s.spawn(move || {
                    let mut out = Vec::with_capacity(chunk.len());
                    for item in chunk {
                        out.push(body(item));
                    }
                    *slot = Some(out);
                });
            }
        });
        let mut result = Vec::with_capacity(self.data.len());
        for partial in partials {
            result.extend(partial.expect("every chunk ran under the scope"));
        }
        result
    }
}

/// Parallel iterator over a mutable slice (see [`TaskPool::par_iter_mut`]).
pub struct ParSliceMut<'p, 'd, T> {
    pool: &'p TaskPool,
    data: &'d mut [T],
    min_len: usize,
}

impl<'p, 'd, T: Send> ParSliceMut<'p, 'd, T> {
    /// Override the minimum chunk length (the adaptive-granularity floor).
    #[must_use]
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        self.min_len = min_len.max(1);
        self
    }

    fn grain(&self) -> usize {
        adaptive_grain(self.data.len(), self.pool.worker_count(), self.min_len)
    }

    /// Apply `body` to every element in place, in parallel.
    pub fn for_each<F>(self, body: F)
    where
        F: Fn(&mut T) + Sync,
    {
        if self.data.is_empty() {
            return;
        }
        let grain = self.grain();
        let body = &body;
        let data = self.data;
        self.pool.scope(|s| {
            for chunk in data.chunks_mut(grain) {
                s.spawn(move || {
                    for item in chunk {
                        body(item);
                    }
                });
            }
        });
    }

    /// Apply `body` to every `(index, &mut element)` pair in place, in
    /// parallel, where `index` is the element's position in the slice.
    pub fn enumerate_for_each<F>(self, body: F)
    where
        F: Fn(usize, &mut T) + Sync,
    {
        if self.data.is_empty() {
            return;
        }
        let grain = self.grain();
        let body = &body;
        let data = self.data;
        self.pool.scope(|s| {
            for (chunk_index, chunk) in data.chunks_mut(grain).enumerate() {
                let base = chunk_index * grain;
                s.spawn(move || {
                    for (offset, item) in chunk.iter_mut().enumerate() {
                        body(base + offset, item);
                    }
                });
            }
        });
    }
}

impl TaskPool {
    /// Open a parallel iterator over the index range `range` (`start..end`).
    ///
    /// The returned [`ParRange`] schedules adaptively sized chunks of the range
    /// onto this pool. This is the primitive an `ECS` `par_iter` uses to fan a
    /// query's entity indices across workers.
    ///
    /// ```
    /// # use prism_tasks::TaskPool;
    /// use core::sync::atomic::{AtomicUsize, Ordering};
    /// let pool = TaskPool::with_threads(4);
    /// let sum = AtomicUsize::new(0);
    /// pool.par_iter_range(0..1000)
    ///     .with_min_len(16)
    ///     .for_each(|i| {
    ///         sum.fetch_add(i, Ordering::Relaxed);
    ///     });
    /// assert_eq!(sum.load(Ordering::Relaxed), (0..1000).sum());
    /// ```
    #[must_use]
    pub fn par_iter_range(&self, range: Range<usize>) -> ParRange<'_> {
        ParRange {
            pool: self,
            start: range.start,
            end: range.end,
            min_len: DEFAULT_MIN_LEN,
        }
    }

    /// Open a parallel iterator over a read-only slice.
    #[must_use]
    pub fn par_iter<'d, T: Sync>(&self, data: &'d [T]) -> ParSlice<'_, 'd, T> {
        ParSlice {
            pool: self,
            data,
            min_len: DEFAULT_MIN_LEN,
        }
    }

    /// Open a parallel iterator over a mutable slice.
    #[must_use]
    pub fn par_iter_mut<'d, T: Send>(&self, data: &'d mut [T]) -> ParSliceMut<'_, 'd, T> {
        ParSliceMut {
            pool: self,
            data,
            min_len: DEFAULT_MIN_LEN,
        }
    }
}
