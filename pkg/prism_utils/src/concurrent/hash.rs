//! Sharded concurrent hash map.
//!
//! [`ConcurrentHashMap`] splits its key space across a fixed number of shards,
//! each a standard hash map behind its own reader-writer lock. Operations on
//! keys that hash to different shards proceed fully in parallel, and reads take
//! a shared lock so many readers on one shard also proceed in parallel. This is
//! the read-mostly asset-cache / type-registry form (Java `ConcurrentHashMap` /
//! folly `F14`) from design doc §11 / §24.2.
//!
//! Sharding is the conservative default the design doc asks for: it is
//! obviously correct (no bespoke lock-free protocol), yet it still scales with
//! core count for the read-mostly workloads the engine actually has. The lock
//! is poison-tolerant: a panic while a guard is held does not wedge the map.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use std::collections::HashMap as StdHashMap;
use std::sync::RwLock;

use crate::hash::FxBuildHasher;

/// A sharded concurrent hash map.
///
/// `S` is the shared [`BuildHasher`]; it defaults to the crate's fast
/// [`FxBuildHasher`]. The hasher must be `Clone` so each shard can own an
/// instance, and the same hasher also selects the shard, so the hasher must be
/// deterministic across shards (any `BuildHasher` whose `build_hasher` yields
/// equal hashers works; `FxBuildHasher` does).
pub struct ConcurrentHashMap<K, V, S = FxBuildHasher> {
    /// One locked map per shard; length is a power of two.
    shards: Box<[RwLock<StdHashMap<K, V, S>>]>,
    /// `shards.len() - 1`, used to pick a shard from a key hash.
    shard_mask: usize,
    /// The hasher used both for shard selection and inside each shard.
    hash_builder: S,
}

impl<K, V> ConcurrentHashMap<K, V, FxBuildHasher> {
    /// Creates a map with a sensible default shard count.
    #[must_use]
    pub fn new() -> Self {
        Self::with_shards(DEFAULT_SHARDS)
    }

    /// Creates a map with `shards` rounded up to the next power of two (and at
    /// least 1) shards.
    #[must_use]
    pub fn with_shards(shards: usize) -> Self {
        Self::with_shards_and_hasher(shards, FxBuildHasher::default())
    }
}

/// Default shard count: enough to spread contention across typical core counts
/// without wasting memory on tiny maps.
const DEFAULT_SHARDS: usize = 16;

/// Construction that only needs a cloneable hasher (no key bounds): each shard
/// owns its own clone of the hasher.
impl<K, V, S> ConcurrentHashMap<K, V, S>
where
    S: Clone,
{
    /// Creates a map with `shards` (rounded up to a power of two, minimum 1)
    /// shards, each using a clone of `hash_builder`.
    #[must_use]
    pub fn with_shards_and_hasher(shards: usize, hash_builder: S) -> Self {
        let shard_count = shards.max(1).next_power_of_two();
        let mut vec = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            vec.push(RwLock::new(StdHashMap::with_hasher(hash_builder.clone())));
        }
        Self {
            shards: vec.into_boxed_slice(),
            shard_mask: shard_count - 1,
            hash_builder,
        }
    }
}

/// Shard-count accessor, available regardless of the key/hasher bounds.
impl<K, V, S> ConcurrentHashMap<K, V, S> {
    /// The number of shards (a power of two).
    #[inline]
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_mask + 1
    }
}

impl<K, V, S> ConcurrentHashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{
    /// Computes the full hash of `key` with the shared hasher.
    #[inline]
    fn hash_of<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    /// Picks the shard index for a precomputed hash. The top bits are used so a
    /// shard choice does not correlate with the low bits a shard's own table
    /// uses for bucketing.
    #[inline]
    fn shard_index(&self, hash: u64) -> usize {
        ((hash >> 32) as usize) & self.shard_mask
    }

    #[inline]
    fn shard<Q>(&self, key: &Q) -> &RwLock<StdHashMap<K, V, S>>
    where
        Q: Hash + ?Sized,
    {
        let index = self.shard_index(self.hash_of(key));
        &self.shards[index]
    }

    /// Inserts a key/value pair, returning the previous value for that key if
    /// any.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let shard = self.shard(&key);
        let mut guard = write(shard);
        guard.insert(key, value)
    }

    /// Removes a key, returning its value if it was present.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let shard = self.shard(key);
        let mut guard = write(shard);
        guard.remove(key)
    }

    /// Returns a clone of the value for `key`, if present.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        V: Clone,
    {
        let shard = self.shard(key);
        let guard = read(shard);
        guard.get(key).cloned()
    }

    /// Reads the value for `key` under a shared lock and runs `f` on it,
    /// returning `f`'s result. This avoids cloning when the caller only needs
    /// to inspect the value.
    pub fn with<Q, R>(&self, key: &Q, f: impl FnOnce(Option<&V>) -> R) -> R
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let shard = self.shard(key);
        let guard = read(shard);
        f(guard.get(key))
    }

    /// Mutates the value for `key` under an exclusive lock and runs `f` on it,
    /// returning `f`'s result. `f` receives `None` when the key is absent.
    pub fn with_mut<Q, R>(&self, key: &Q, f: impl FnOnce(Option<&mut V>) -> R) -> R
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let shard = self.shard(key);
        let mut guard = write(shard);
        f(guard.get_mut(key))
    }

    /// Returns the value for `key`, inserting the result of `default` first if
    /// the key is absent, then handing a clone back to the caller.
    pub fn get_or_insert_with<F>(&self, key: K, default: F) -> V
    where
        V: Clone,
        F: FnOnce() -> V,
    {
        let shard = self.shard(&key);
        let mut guard = write(shard);
        guard.entry(key).or_insert_with(default).clone()
    }

    /// Returns `true` if `key` is present.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let shard = self.shard(key);
        let guard = read(shard);
        guard.contains_key(key)
    }

    /// The total number of entries across all shards. Momentary in a concurrent
    /// setting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| read(s).len()).sum()
    }

    /// Returns `true` if every shard is currently empty. Momentary in a
    /// concurrent setting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| read(s).is_empty())
    }

    /// Removes every entry from every shard.
    pub fn clear(&self) {
        for shard in &self.shards {
            write(shard).clear();
        }
    }

    /// Calls `f` on a clone of every key/value pair. Each shard is visited under
    /// its own shared lock, so this is a point-in-time, shard-consistent (not
    /// globally atomic) snapshot.
    pub fn for_each(&self, mut f: impl FnMut(&K, &V)) {
        for shard in &self.shards {
            let guard = read(shard);
            for (k, v) in guard.iter() {
                f(k, v);
            }
        }
    }
}

/// Acquires a shard's read lock, recovering transparently from poisoning (a
/// panic in another thread must not permanently wedge the map).
#[inline]
fn read<K, V, S>(
    lock: &RwLock<StdHashMap<K, V, S>>,
) -> std::sync::RwLockReadGuard<'_, StdHashMap<K, V, S>> {
    lock.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Acquires a shard's write lock, recovering transparently from poisoning.
#[inline]
fn write<K, V, S>(
    lock: &RwLock<StdHashMap<K, V, S>>,
) -> std::sync::RwLockWriteGuard<'_, StdHashMap<K, V, S>> {
    lock.write().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl<K, V> Default for ConcurrentHashMap<K, V, FxBuildHasher> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> fmt::Debug for ConcurrentHashMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcurrentHashMap")
            .field("shards", &self.shard_count())
            .finish()
    }
}
