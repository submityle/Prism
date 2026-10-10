//! Content-addressed caching for the immutable, dedup-friendly GPU objects a
//! backend creates: bind-group layouts, pipeline layouts, samplers, and
//! pipeline state objects (PSOs).
//!
//! These objects are expensive to build (PSO compilation can cost
//! milliseconds) yet are frequently requested with identical parameters by
//! unrelated call sites. Keying them by their *content* — a normalized,
//! hashable description — lets the backend build each distinct object once and
//! hand back the same id on every subsequent request, which is how mature
//! engines (and wgpu's own internal layout dedup) keep pipeline creation off
//! the hot path.
//!
//! The cache is reference-counted so a renderer can `release` an object when a
//! frame stops using it and let the cache evict truly-dead entries, while a
//! long-lived "pinned" object (`get_or_create` without a later `release`)
//! stays resident. Pure, `no_std`, no `unsafe`; keys are `Ord` so the map is a
//! deterministic `BTreeMap` rather than a hasher with platform-dependent
//! iteration order.

use alloc::collections::BTreeMap;

/// An entry in the cache: the cached value plus how many live references hold
/// it.
struct Entry<V> {
    value: V,
    refcount: u32,
}

/// A content-addressed cache mapping a normalized key `K` to a backend value
/// `V` (typically a [`crate::resource::ResourceId`] or native handle), with
/// reference counting for lifetime management.
///
/// `K` must be `Ord + Clone` (it is the content key); `V` must be `Copy`
/// (an id/handle is cheap to copy and the cache never mutates it in place).
pub struct ContentCache<K: Ord + Clone, V: Copy> {
    entries: BTreeMap<K, Entry<V>>,
    hits: u64,
    misses: u64,
}

impl<K: Ord + Clone, V: Copy> ContentCache<K, V> {
    /// Creates an empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// The number of distinct cached objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no objects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cumulative cache hits and misses, for diagnostics / tuning.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            live_entries: self.entries.len(),
        }
    }

    /// Looks up `key` without affecting its reference count. Counts as a hit or
    /// miss for statistics.
    pub fn peek(&mut self, key: &K) -> Option<V> {
        match self.entries.get(key) {
            Some(e) => {
                self.hits += 1;
                Some(e.value)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Returns the cached value for `key`, incrementing its reference count. If
    /// absent, calls `create` to build the value, inserts it with a reference
    /// count of 1, and returns it. The closure runs at most once per distinct
    /// key, which is the whole point: build-once, reuse-many.
    pub fn get_or_create<F>(&mut self, key: K, create: F) -> V
    where
        F: FnOnce() -> V,
    {
        if let Some(e) = self.entries.get_mut(&key) {
            e.refcount = e.refcount.saturating_add(1);
            self.hits += 1;
            return e.value;
        }
        self.misses += 1;
        let value = create();
        self.entries.insert(key, Entry { value, refcount: 1 });
        value
    }

    /// Acquires an additional reference to an already-cached object, returning
    /// its value. Returns `None` if the key is not present.
    pub fn acquire(&mut self, key: &K) -> Option<V> {
        self.entries.get_mut(key).map(|e| {
            e.refcount = e.refcount.saturating_add(1);
            e.value
        })
    }

    /// Releases one reference to `key`. When the last reference is dropped the
    /// entry is removed and its value returned so the caller can destroy the
    /// native object; while references remain, returns `None`. Returns `None`
    /// for an unknown key as well (there is nothing to free).
    pub fn release(&mut self, key: &K) -> Option<V> {
        let evict = match self.entries.get_mut(key) {
            Some(e) => {
                e.refcount = e.refcount.saturating_sub(1);
                e.refcount == 0
            }
            None => return None,
        };
        if evict {
            self.entries.remove(key).map(|e| e.value)
        } else {
            None
        }
    }

    /// The current reference count for `key`, or 0 if absent.
    #[must_use]
    pub fn refcount(&self, key: &K) -> u32 {
        self.entries.get(key).map_or(0, |e| e.refcount)
    }

    /// Removes every entry regardless of reference count, invoking `drop_value`
    /// on each so the backend can destroy native objects. Used on device loss.
    pub fn drain_all<F>(&mut self, mut drop_value: F)
    where
        F: FnMut(V),
    {
        for (_, e) in core::mem::take(&mut self.entries) {
            drop_value(e.value);
        }
    }
}

impl<K: Ord + Clone, V: Copy> Default for ContentCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot of a [`ContentCache`]'s hit/miss counters and live entry count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CacheStats {
    /// Lookups that found an existing entry.
    pub hits: u64,
    /// Lookups that had to create (or missed) an entry.
    pub misses: u64,
    /// Entries currently resident.
    pub live_entries: usize,
}

impl CacheStats {
    /// The hit rate in `[0, 1]`, or 0 when there have been no lookups.
    #[must_use]
    pub fn hit_rate(self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_once_reuse_many() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        let mut builds = 0u32;
        let a = cache.get_or_create(7, || {
            builds += 1;
            700
        });
        let b = cache.get_or_create(7, || {
            builds += 1;
            999
        });
        assert_eq!(a, 700);
        assert_eq!(b, 700, "second request returns the cached value");
        assert_eq!(builds, 1, "closure runs exactly once per key");
        assert_eq!(cache.refcount(&7), 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn refcount_eviction() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        cache.get_or_create(1, || 10); // rc=1
        cache.acquire(&1); // rc=2
        assert_eq!(cache.release(&1), None, "still referenced");
        assert_eq!(cache.refcount(&1), 1);
        assert_eq!(cache.release(&1), Some(10), "last ref frees value");
        assert_eq!(cache.refcount(&1), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn release_unknown_is_none() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        assert_eq!(cache.release(&42), None);
    }

    #[test]
    fn acquire_absent_is_none() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        assert_eq!(cache.acquire(&42), None);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        cache.get_or_create(1, || 1); // miss
        cache.get_or_create(1, || 1); // hit
        cache.get_or_create(2, || 2); // miss
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 2);
        assert_eq!(s.live_entries, 2);
        assert!((s.hit_rate() - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn drain_all_destroys_everything() {
        let mut cache: ContentCache<u32, u64> = ContentCache::new();
        cache.get_or_create(1, || 10);
        cache.get_or_create(2, || 20);
        let mut dropped: Vec<u64> = Vec::new();
        cache.drain_all(|v| dropped.push(v));
        dropped.sort_unstable();
        assert_eq!(dropped, alloc::vec![10, 20]);
        assert!(cache.is_empty());
    }
}
