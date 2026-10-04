//! Deterministic, order-independent merge/reduce of keyed contributions (§24.7).
//!
//! A parallel job system produces partial results in a nondeterministic order:
//! thread A might contribute `(key, value)` before thread B on one run and
//! after it on the next. [`DeterministicMerge`] collects those contributions
//! and canonicalises them — by sorting — so that the *same multiset* of
//! contributions always yields the *same* ordered result, independent of
//! arrival order. That canonical form is what downstream determinism (hashing,
//! serialization, networked lockstep) depends on.
//!
//! Two consumption modes:
//! - [`into_sorted`](DeterministicMerge::into_sorted) keeps every contribution,
//!   sorted by `(key, value)` — a stable canonical ordering of the full bag.
//! - [`reduce`](DeterministicMerge::reduce) folds all values sharing a key with
//!   a user combine function, the deterministic "combine by key" of a
//!   map-reduce. It is order-independent **iff** the combine function is
//!   commutative and associative (the honest boundary in the module docs).

extern crate alloc;

use alloc::vec::Vec;

/// An order-independent accumulator of `(key, value)` contributions.
///
/// Contributions can be added in any order (and, with the `concurrent`
/// [`ConcurrentMerge`] wrapper, from any thread); the canonicalising consumers
/// then impose a deterministic order. The type itself is a thin, allocation-
/// backed collector — the determinism guarantee comes from the sort in
/// [`into_sorted`](DeterministicMerge::into_sorted) /
/// [`reduce`](DeterministicMerge::reduce).
#[derive(Clone, Debug)]
pub struct DeterministicMerge<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> DeterministicMerge<K, V> {
    /// Create an empty accumulator.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an empty accumulator with room for `capacity` contributions.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Record one `(key, value)` contribution. Arrival order does not affect
    /// any canonicalised result.
    #[inline]
    pub fn contribute(&mut self, key: K, value: V) {
        self.entries.push((key, value));
    }

    /// The number of contributions recorded so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no contribution has been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<K: Ord, V: Ord> DeterministicMerge<K, V> {
    /// Consume the accumulator and return every contribution in a canonical
    /// order, sorted by `(key, value)`.
    ///
    /// Because the sort key is the whole pair, the output is a pure function of
    /// the *multiset* of contributions: any arrival order produces the exact
    /// same `Vec`. The sort is stable, though with a total `(key, value)` order
    /// stability is not observable.
    #[must_use]
    pub fn into_sorted(self) -> Vec<(K, V)> {
        let mut entries = self.entries;
        entries.sort();
        entries
    }
}

impl<K: Ord, V> DeterministicMerge<K, V> {
    /// Consume the accumulator and fold all values sharing a key with
    /// `combine`, returning one `(key, value)` per distinct key in ascending
    /// key order.
    ///
    /// Within a key group the values are folded left-to-right **in a
    /// deterministic order** (the contributions are first stably sorted by
    /// key), so the result is a pure function of the contribution multiset.
    /// The fold is order-independent across the whole run **iff** `combine` is
    /// commutative and associative; a non-associative `combine` still produces
    /// a deterministic result here, but one that depends on this grouping.
    #[must_use]
    pub fn reduce<F>(self, mut combine: F) -> Vec<(K, V)>
    where
        F: FnMut(V, V) -> V,
    {
        let mut entries = self.entries;
        // Stable sort by key keeps same-key values in their post-sort order so
        // the left-to-right fold is deterministic for any `combine`.
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut out: Vec<(K, V)> = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            match out.last_mut() {
                Some(last) if last.0 == k => {
                    // Fold the new value into the running accumulator for this
                    // key. `take`-free: move the old accumulator out via a swap
                    // using a temporary is not possible without `V: Default`,
                    // so rebuild the entry.
                    let (lk, lv) = out.pop().expect("last_mut implies non-empty");
                    out.push((lk, combine(lv, v)));
                }
                _ => out.push((k, v)),
            }
        }
        out
    }
}

impl<K, V> Default for DeterministicMerge<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// A thread-safe wrapper around [`DeterministicMerge`] for concurrent
/// contribution from many workers (`concurrent` feature).
///
/// Workers call [`contribute`](ConcurrentMerge::contribute) through a shared
/// `&self` from any thread, in any interleaving; the canonicalising consumers
/// ([`into_sorted`](ConcurrentMerge::into_sorted) /
/// [`into_reduced`](ConcurrentMerge::into_reduced)) then impose the same
/// deterministic order the single-threaded accumulator would. The lock is held
/// only for the `O(1)` push, so contention is minimal.
#[cfg(feature = "concurrent")]
#[derive(Debug)]
pub struct ConcurrentMerge<K, V> {
    inner: std::sync::Mutex<DeterministicMerge<K, V>>,
}

#[cfg(feature = "concurrent")]
impl<K, V> ConcurrentMerge<K, V> {
    /// Create an empty concurrent accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(DeterministicMerge::new()),
        }
    }

    /// Record one contribution from the calling thread. Safe to call
    /// concurrently from many threads; arrival order does not affect any
    /// canonicalised result.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned by a thread that panicked while
    /// holding it.
    pub fn contribute(&self, key: K, value: V) {
        self.inner
            .lock()
            .expect("ConcurrentMerge lock poisoned")
            .contribute(key, value);
    }

    /// The number of contributions recorded so far.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("ConcurrentMerge lock poisoned").len()
    }

    /// Whether no contribution has been recorded.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .expect("ConcurrentMerge lock poisoned")
            .is_empty()
    }

    /// Consume the wrapper and return the inner single-threaded accumulator.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned.
    #[must_use]
    pub fn into_inner(self) -> DeterministicMerge<K, V> {
        self.inner.into_inner().expect("ConcurrentMerge lock poisoned")
    }
}

#[cfg(feature = "concurrent")]
impl<K: Ord, V: Ord> ConcurrentMerge<K, V> {
    /// Consume the wrapper and return every contribution sorted by
    /// `(key, value)` — the same canonical ordering regardless of which thread
    /// contributed what, in which order.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned.
    #[must_use]
    pub fn into_sorted(self) -> Vec<(K, V)> {
        self.into_inner().into_sorted()
    }
}

#[cfg(feature = "concurrent")]
impl<K: Ord, V> ConcurrentMerge<K, V> {
    /// Consume the wrapper and fold values per key with `combine`, in ascending
    /// key order. Order-independent across threads iff `combine` is commutative
    /// and associative.
    ///
    /// # Panics
    /// Panics if the internal lock was poisoned.
    #[must_use]
    pub fn into_reduced<F>(self, combine: F) -> Vec<(K, V)>
    where
        F: FnMut(V, V) -> V,
    {
        self.into_inner().reduce(combine)
    }
}

#[cfg(feature = "concurrent")]
impl<K, V> Default for ConcurrentMerge<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
