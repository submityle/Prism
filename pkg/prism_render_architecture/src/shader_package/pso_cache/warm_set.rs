//! Deterministic planning of which pipelines to precompile ("warm").
//!
//! Shader-compilation hitching is eliminated by precompiling the pipeline-state
//! objects a frame will need *before* they are first drawn — at load or level
//! transition. [`WarmSetPlanner`] collects the required pipelines from two
//! sources and produces a stable, prioritized [`WarmSetPlan`]:
//!
//! 1. **Static requirements** — the permutation set a package is known to need,
//!    registered via [`WarmSetPlanner::require`].
//! 2. **Runtime miss feedback** — pipelines that were *not* warmed yet were hit
//!    at runtime (an observed hitch), fed back via
//!    [`WarmSetPlanner::observe_miss`] so the next warm pass covers them.
//!
//! Duplicate keys collapse to their strongest priority, and the resulting plan
//! is ordered by priority then key so warming is reproducible and the most
//! important pipelines compile first. Planning is pure bookkeeping with no I/O.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::PsoCacheKey;

/// Relative urgency of warming a pipeline.
///
/// Lower [`WarmPriority::rank`] warms earlier. Runtime misses are promoted to
/// [`WarmPriority::High`] because an observed hitch is more urgent than a
/// speculative background variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WarmPriority {
    /// Must be resident before the first frame renders (e.g. default material).
    Critical,
    /// Promote ahead of normal work (e.g. a pipeline that already hitched).
    High,
    /// Standard load-time requirement.
    Normal,
    /// Speculative; warm only when spare budget remains.
    Background,
}

impl WarmPriority {
    /// Ordering rank; smaller warms first.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 1,
            Self::Normal => 2,
            Self::Background => 3,
        }
    }

    /// Returns the stronger (earlier-warming) of two priorities.
    #[must_use]
    pub fn strongest(self, other: Self) -> Self {
        if self.rank() <= other.rank() {
            self
        } else {
            other
        }
    }
}

/// One pipeline to warm, paired with its priority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WarmRequest {
    /// Pipeline identity to precompile.
    pub key: PsoCacheKey,
    /// Warming urgency.
    pub priority: WarmPriority,
}

/// A finalized, deterministically ordered list of pipelines to warm.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WarmSetPlan {
    entries: Vec<WarmRequest>,
}

impl WarmSetPlan {
    /// Ordered warm requests: by priority (strongest first) then by key.
    #[must_use]
    pub fn entries(&self) -> &[WarmRequest] {
        &self.entries
    }

    /// Number of pipelines to warm.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing needs warming.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keys in warm order.
    #[must_use]
    pub fn keys(&self) -> Vec<PsoCacheKey> {
        self.entries.iter().map(|e| e.key.clone()).collect()
    }
}

/// Accumulates warm requirements and emits deterministic [`WarmSetPlan`]s.
#[derive(Clone, Debug, Default)]
pub struct WarmSetPlanner {
    required: BTreeMap<PsoCacheKey, WarmPriority>,
}

impl WarmSetPlanner {
    /// Creates an empty planner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            required: BTreeMap::new(),
        }
    }

    /// Number of distinct pipelines currently required.
    #[must_use]
    pub fn len(&self) -> usize {
        self.required.len()
    }

    /// Whether no pipelines are required yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.required.is_empty()
    }

    /// Requires `key` at `priority`, keeping the strongest priority if the key
    /// was already requested.
    pub fn require(&mut self, key: PsoCacheKey, priority: WarmPriority) {
        self.required
            .entry(key)
            .and_modify(|existing| *existing = existing.strongest(priority))
            .or_insert(priority);
    }

    /// Requires every `(key, priority)` pair from an iterator.
    pub fn require_all(&mut self, items: impl IntoIterator<Item = (PsoCacheKey, WarmPriority)>) {
        for (key, priority) in items {
            self.require(key, priority);
        }
    }

    /// Records a runtime pipeline miss (an observed hitch), promoting `key` to
    /// at least [`WarmPriority::High`] for the next warm pass.
    pub fn observe_miss(&mut self, key: PsoCacheKey) {
        self.require(key, WarmPriority::High);
    }

    /// Builds the full ordered plan for every required pipeline.
    #[must_use]
    pub fn plan(&self) -> WarmSetPlan {
        self.plan_excluding(&BTreeSet::new())
    }

    /// Builds the ordered plan for required pipelines that are not already
    /// resident in `resident`, so already-warmed pipelines are skipped.
    #[must_use]
    pub fn plan_excluding(&self, resident: &BTreeSet<PsoCacheKey>) -> WarmSetPlan {
        let mut entries: Vec<WarmRequest> = self
            .required
            .iter()
            .filter(|(key, _)| !resident.contains(*key))
            .map(|(key, &priority)| WarmRequest {
                key: key.clone(),
                priority,
            })
            .collect();
        // Strongest priority first; ties broken by key for determinism.
        entries.sort_by(|a, b| {
            a.priority
                .rank()
                .cmp(&b.priority.rank())
                .then_with(|| a.key.cmp(&b.key))
        });
        WarmSetPlan { entries }
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
    fn priority_rank_orders_critical_first() {
        assert!(WarmPriority::Critical.rank() < WarmPriority::High.rank());
        assert!(WarmPriority::High.rank() < WarmPriority::Normal.rank());
        assert!(WarmPriority::Normal.rank() < WarmPriority::Background.rank());
    }

    #[test]
    fn strongest_keeps_lower_rank() {
        assert_eq!(
            WarmPriority::Normal.strongest(WarmPriority::High),
            WarmPriority::High
        );
        assert_eq!(
            WarmPriority::Critical.strongest(WarmPriority::Background),
            WarmPriority::Critical
        );
    }

    #[test]
    fn require_dedups_to_strongest_priority() {
        let mut planner = WarmSetPlanner::new();
        planner.require(key("a", 0), WarmPriority::Normal);
        planner.require(key("a", 0), WarmPriority::Critical);
        planner.require(key("a", 0), WarmPriority::Background);
        assert_eq!(planner.len(), 1);
        let plan = planner.plan();
        assert_eq!(plan.entries()[0].priority, WarmPriority::Critical);
    }

    #[test]
    fn plan_orders_by_priority_then_key() {
        let mut planner = WarmSetPlanner::new();
        planner.require(key("b", 0), WarmPriority::Normal);
        planner.require(key("a", 0), WarmPriority::Normal);
        planner.require(key("z", 0), WarmPriority::Critical);
        planner.require(key("m", 0), WarmPriority::Background);
        let plan = planner.plan();
        let keys = plan.keys();
        assert_eq!(
            keys,
            alloc::vec![key("z", 0), key("a", 0), key("b", 0), key("m", 0)]
        );
    }

    #[test]
    fn observe_miss_promotes_to_high() {
        let mut planner = WarmSetPlanner::new();
        planner.require(key("a", 0), WarmPriority::Background);
        planner.observe_miss(key("a", 0));
        let plan = planner.plan();
        assert_eq!(plan.entries()[0].priority, WarmPriority::High);
    }

    #[test]
    fn plan_excluding_skips_resident_pipelines() {
        let mut planner = WarmSetPlanner::new();
        planner.require(key("a", 0), WarmPriority::Normal);
        planner.require(key("b", 0), WarmPriority::Normal);
        let mut resident = BTreeSet::new();
        resident.insert(key("a", 0));
        let plan = planner.plan_excluding(&resident);
        assert_eq!(plan.keys(), alloc::vec![key("b", 0)]);
    }
}
