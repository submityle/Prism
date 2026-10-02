//! Byte-budgeted LRU residency model for compiled pipeline-state objects.
//!
//! A persisted PSO cache cannot grow without bound: driver pipeline binaries
//! accumulate across sessions and permutations. [`LruPsoCache`] models the
//! residency bookkeeping — which keys are kept, their byte cost, and which are
//! evicted when a byte budget is exceeded — using a least-recently-used policy.
//!
//! The model is deterministic and backend-agnostic: it tracks sizes and access
//! recency only, leaving the actual binary bytes and disk/driver I/O to the
//! engine-side consumer. A logical clock orders accesses so eviction order is
//! reproducible across runs and in golden tests.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::PsoCacheKey;

/// Internal residency record for one cached pipeline.
#[derive(Clone, Copy, Debug)]
struct Record {
    size_bytes: u64,
    last_used: u64,
}

/// Outcome of attempting to admit a pipeline into the cache.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AdmitOutcome {
    /// Whether the pipeline now resides in the cache.
    ///
    /// `false` only when a single pipeline is larger than the whole budget; in
    /// that case the cache is left untouched rather than thrashed empty.
    pub admitted: bool,
    /// Keys evicted (least-recently-used first) to make room, in eviction order.
    pub evicted: Vec<PsoCacheKey>,
}

/// Least-recently-used residency model with a hard byte budget.
#[derive(Clone, Debug)]
pub struct LruPsoCache {
    budget_bytes: u64,
    used_bytes: u64,
    clock: u64,
    entries: BTreeMap<PsoCacheKey, Record>,
}

impl LruPsoCache {
    /// Creates an empty cache limited to `budget_bytes` of resident pipelines.
    #[must_use]
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            clock: 0,
            entries: BTreeMap::new(),
        }
    }

    /// Configured byte budget.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Total bytes currently resident.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Number of resident pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no pipelines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `key` is currently resident.
    #[must_use]
    pub fn contains(&self, key: &PsoCacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Resident keys in deterministic key order.
    #[must_use]
    pub fn resident_keys(&self) -> BTreeSet<PsoCacheKey> {
        self.entries.keys().cloned().collect()
    }

    /// Advances the logical clock and returns the new timestamp.
    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    /// Marks `key` as most-recently-used. Returns `false` if not resident.
    pub fn touch(&mut self, key: &PsoCacheKey) -> bool {
        let now = self.tick();
        match self.entries.get_mut(key) {
            Some(record) => {
                record.last_used = now;
                true
            }
            None => false,
        }
    }

    /// Admits (or re-sizes) `key`, evicting least-recently-used pipelines until
    /// the resident footprint fits the budget.
    ///
    /// Re-inserting an existing key updates its size and marks it
    /// most-recently-used. A pipeline larger than the entire budget is refused
    /// without disturbing existing residents (`admitted == false`).
    pub fn admit(&mut self, key: PsoCacheKey, size_bytes: u64) -> AdmitOutcome {
        if size_bytes > self.budget_bytes {
            return AdmitOutcome {
                admitted: false,
                evicted: Vec::new(),
            };
        }
        let now = self.tick();
        if let Some(record) = self.entries.get_mut(&key) {
            self.used_bytes = self.used_bytes - record.size_bytes + size_bytes;
            record.size_bytes = size_bytes;
            record.last_used = now;
        } else {
            self.entries.insert(
                key,
                Record {
                    size_bytes,
                    last_used: now,
                },
            );
            self.used_bytes += size_bytes;
        }
        let evicted = self.evict_to_fit();
        AdmitOutcome {
            admitted: true,
            evicted,
        }
    }

    /// Evicts least-recently-used entries until the footprint fits the budget.
    fn evict_to_fit(&mut self) -> Vec<PsoCacheKey> {
        let mut evicted = Vec::new();
        while self.used_bytes > self.budget_bytes {
            let Some(victim) = self.least_recently_used() else {
                break;
            };
            if let Some(record) = self.entries.remove(&victim) {
                self.used_bytes -= record.size_bytes;
            }
            evicted.push(victim);
        }
        evicted
    }

    /// Finds the resident key with the smallest `last_used`, breaking ties by
    /// key order so eviction is fully deterministic.
    fn least_recently_used(&self) -> Option<PsoCacheKey> {
        self.entries
            .iter()
            .min_by(|a, b| {
                a.1.last_used
                    .cmp(&b.1.last_used)
                    .then_with(|| a.0.cmp(b.0))
            })
            .map(|(key, _)| key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::super::PipelineStateHash;
    use super::*;
    use crate::shader_package::ShaderPackageId;

    fn key(pkg: &str, perm: u64) -> PsoCacheKey {
        PsoCacheKey::new(ShaderPackageId::new(pkg), perm, PipelineStateHash(0))
    }

    #[test]
    fn admits_within_budget_without_eviction() {
        let mut cache = LruPsoCache::with_budget(1000);
        let out = cache.admit(key("a", 0), 400);
        assert!(out.admitted);
        assert!(out.evicted.is_empty());
        assert_eq!(cache.used_bytes(), 400);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let mut cache = LruPsoCache::with_budget(1000);
        cache.admit(key("a", 0), 400);
        cache.admit(key("b", 0), 400);
        // Touch `a` so `b` becomes least-recently-used.
        assert!(cache.touch(&key("a", 0)));
        let out = cache.admit(key("c", 0), 400);
        assert!(out.admitted);
        assert_eq!(out.evicted, alloc::vec![key("b", 0)]);
        assert!(cache.contains(&key("a", 0)));
        assert!(cache.contains(&key("c", 0)));
        assert!(!cache.contains(&key("b", 0)));
        assert_eq!(cache.used_bytes(), 800);
    }

    #[test]
    fn evicts_multiple_entries_when_needed() {
        let mut cache = LruPsoCache::with_budget(1000);
        cache.admit(key("a", 0), 300);
        cache.admit(key("b", 0), 300);
        cache.admit(key("c", 0), 300);
        // Inserting 900 against a 1000 budget forces eviction of all three
        // existing entries (a, then b, then c) in LRU order.
        let out = cache.admit(key("d", 0), 900);
        assert!(out.admitted);
        assert_eq!(
            out.evicted,
            alloc::vec![key("a", 0), key("b", 0), key("c", 0)]
        );
        assert!(!cache.contains(&key("a", 0)));
        assert!(!cache.contains(&key("b", 0)));
        assert!(!cache.contains(&key("c", 0)));
        assert!(cache.contains(&key("d", 0)));
        assert_eq!(cache.used_bytes(), 900);
    }

    #[test]
    fn reinsert_updates_size_and_recency() {
        let mut cache = LruPsoCache::with_budget(1000);
        cache.admit(key("a", 0), 200);
        let out = cache.admit(key("a", 0), 500);
        assert!(out.admitted);
        assert!(out.evicted.is_empty());
        assert_eq!(cache.used_bytes(), 500);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn oversized_pipeline_is_refused_without_disturbing_residents() {
        let mut cache = LruPsoCache::with_budget(1000);
        cache.admit(key("a", 0), 600);
        let out = cache.admit(key("big", 0), 1001);
        assert!(!out.admitted);
        assert!(out.evicted.is_empty());
        assert!(cache.contains(&key("a", 0)));
        assert!(!cache.contains(&key("big", 0)));
        assert_eq!(cache.used_bytes(), 600);
    }

    #[test]
    fn touch_missing_key_reports_false() {
        let mut cache = LruPsoCache::with_budget(1000);
        assert!(!cache.touch(&key("absent", 0)));
    }
}
