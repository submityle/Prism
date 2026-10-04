//! Portable cache-hierarchy model (design §24.6).
//!
//! Two cores that share an inner cache (an L2 shared by a core's SMT siblings,
//! say) are *closer* for scheduling than two cores that share only the L3, and
//! closer still than two cores that share no cache at all. [`CacheTopology`]
//! records which cores share each cache instance so the placement and
//! steal-ordering policies can prefer cache-local co-location.
//!
//! Like the rest of the `topology` module this is a pure data model with no
//! probing of its own; `prism_platform` fills in the real sizes and sharer
//! sets. Every query is a deterministic function of the stored layout.

use alloc::vec::Vec;

use crate::numa::CoreInfo;

/// A level in the CPU cache hierarchy.
///
/// Ordered from innermost/closest ([`CacheLevel::L1`]) to outermost
/// ([`CacheLevel::L3`]); a smaller level shared by two cores means they are
/// physically closer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum CacheLevel {
    /// First-level cache (per physical core, often per SMT sibling pair).
    L1,
    /// Second-level cache.
    L2,
    /// Last-level / shared cache (typically shared across a package or
    /// core complex).
    L3,
}

/// One cache instance: its level, geometry, and the set of cores that share it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CacheInfo {
    /// Which level of the hierarchy this cache sits at.
    pub level: CacheLevel,
    /// Total capacity of this cache instance, in bytes.
    pub size_bytes: u64,
    /// Cache-line size, in bytes.
    pub line_bytes: u32,
    /// OS logical-core ids that share this cache instance, in ascending order.
    pub cores: Vec<usize>,
}

impl CacheInfo {
    /// Whether `core` is one of the sharers of this cache instance.
    pub fn contains(&self, core: usize) -> bool {
        self.cores.binary_search(&core).is_ok()
    }
}

/// The full set of cache instances describing a machine (or test topology).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct CacheTopology {
    caches: Vec<CacheInfo>,
}

impl CacheTopology {
    /// An empty hierarchy: no cache sharing is known. The honest fallback when
    /// the platform exposes no cache map; every query degrades to "nothing
    /// shared".
    pub fn empty() -> Self {
        Self { caches: Vec::new() }
    }

    /// Build a hierarchy from explicit cache instances. Each instance's
    /// `cores` list is sorted so that [`CacheInfo::contains`] can binary
    /// search; the input need not be pre-sorted.
    pub fn new(mut caches: Vec<CacheInfo>) -> Self {
        for cache in &mut caches {
            cache.cores.sort_unstable();
            cache.cores.dedup();
        }
        Self { caches }
    }

    /// All cache instances, in the order supplied.
    pub fn caches(&self) -> &[CacheInfo] {
        &self.caches
    }

    /// Whether cores `a` and `b` share the same cache instance at `level`.
    pub fn shares_cache(&self, level: CacheLevel, a: usize, b: usize) -> bool {
        self.caches
            .iter()
            .any(|c| c.level == level && c.contains(a) && c.contains(b))
    }

    /// The OS core ids that share `core`'s cache instance at `level`
    /// (including `core` itself), ascending. Empty if `core` has no such cache.
    pub fn sharers(&self, level: CacheLevel, core: usize) -> Vec<usize> {
        self.caches
            .iter()
            .find(|c| c.level == level && c.contains(core))
            .map(|c| c.cores.clone())
            .unwrap_or_default()
    }

    /// The innermost (smallest) cache level at which cores `a` and `b` share an
    /// instance, or `None` if they share no cache. A smaller returned level
    /// means the two cores are physically closer.
    pub fn innermost_shared_level(&self, a: usize, b: usize) -> Option<CacheLevel> {
        if a == b {
            // A core trivially shares every cache it participates in with
            // itself; report its innermost known level.
            return self
                .caches
                .iter()
                .filter(|c| c.contains(a))
                .map(|c| c.level)
                .min();
        }
        self.caches
            .iter()
            .filter(|c| c.contains(a) && c.contains(b))
            .map(|c| c.level)
            .min()
    }

    /// The cache-line size reported at `level` (the first matching instance),
    /// or `None` if no cache at that level is described.
    pub fn line_bytes(&self, level: CacheLevel) -> Option<u32> {
        self.caches
            .iter()
            .find(|c| c.level == level)
            .map(|c| c.line_bytes)
    }
}

/// Build a conventional single-level (L3) cache instance shared by every core
/// in `cores`. A convenience for tests and for platforms that only expose a
/// shared last-level cache.
pub fn shared_l3(cores: &[CoreInfo], size_bytes: u64, line_bytes: u32) -> CacheInfo {
    let mut ids: Vec<usize> = cores.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    ids.dedup();
    CacheInfo {
        level: CacheLevel::L3,
        size_bytes,
        line_bytes,
        cores: ids,
    }
}
