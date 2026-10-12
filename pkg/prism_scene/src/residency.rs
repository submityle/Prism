//! Residency tiers and plans (design §11.1).
//!
//! A scene may mix three residency tiers; the bake or the author annotates each
//! entity's tier into a [`ResidencyPlan`]. The world-system streaming base
//! consumes the plan to decide what is materialized as live ECS entities, what
//! is streamed in and out by range, and what stays a flat GPU-resident proxy.

use alloc::collections::BTreeMap;

use crate::identity::LocalEntityId;

/// A scene content residency tier (design §11.1).
///
/// A **live** [`SceneInstance`](crate::instance::SceneInstance) is
/// [`Interactive`](Self::Interactive) or [`Streamed`](Self::Streamed);
/// [`Proxy`](Self::Proxy) describes content that is *not* instantiated in the
/// ECS (the flat `SoA` proxy stream).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResidencyTier {
    /// Tier A — fully instantiated as live entities (interactive objects).
    Interactive,
    /// Tier B — a sub-scene block instantiated only while in range.
    Streamed,
    /// Tier C — flat proxy stream contributing GPU-resident instances; not ECS.
    Proxy,
}

impl Default for ResidencyTier {
    /// The default tier is interactive — a template entity is a live entity
    /// unless the bake or author marks it otherwise (design §11.1).
    fn default() -> Self {
        Self::Interactive
    }
}

/// Per-entity residency annotation for one template (design §5.1 `ResidencyPlan`).
///
/// Any local entity absent from the map is [`ResidencyTier::Interactive`], so an
/// all-interactive scene needs no plan at all (empty == everything live), which
/// keeps the common case zero-cost.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidencyPlan {
    tiers: BTreeMap<LocalEntityId, ResidencyTier>,
}

impl ResidencyPlan {
    /// An empty plan (every entity defaults to interactive).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Annotates `local` with `tier`, returning any previous annotation.
    pub fn set(&mut self, local: LocalEntityId, tier: ResidencyTier) -> Option<ResidencyTier> {
        self.tiers.insert(local, tier)
    }

    /// The explicit tier of `local`, if annotated.
    #[must_use]
    pub fn explicit(&self, local: LocalEntityId) -> Option<ResidencyTier> {
        self.tiers.get(&local).copied()
    }

    /// The effective tier of `local`, defaulting to
    /// [`ResidencyTier::Interactive`].
    #[must_use]
    pub fn tier_of(&self, local: LocalEntityId) -> ResidencyTier {
        self.explicit(local).unwrap_or_default()
    }

    /// Number of explicit annotations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiers.len()
    }

    /// Whether no entity is explicitly annotated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Iterates `(local, tier)` annotations in deterministic (id-sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (LocalEntityId, ResidencyTier)> + '_ {
        self.tiers.iter().map(|(id, tier)| (*id, *tier))
    }
}

#[cfg(test)]
mod tests {
    use super::{ResidencyPlan, ResidencyTier};
    use crate::identity::LocalEntityId;
    use alloc::vec::Vec;

    #[test]
    fn default_tier_is_interactive() {
        let plan = ResidencyPlan::new();
        assert_eq!(plan.tier_of(LocalEntityId::new(0)), ResidencyTier::Interactive);
        assert_eq!(ResidencyTier::default(), ResidencyTier::Interactive);
    }

    #[test]
    fn annotations_override_the_default() {
        let mut plan = ResidencyPlan::new();
        assert!(plan.is_empty());
        plan.set(LocalEntityId::new(3), ResidencyTier::Proxy);
        plan.set(LocalEntityId::new(1), ResidencyTier::Streamed);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.tier_of(LocalEntityId::new(3)), ResidencyTier::Proxy);
        assert_eq!(plan.tier_of(LocalEntityId::new(1)), ResidencyTier::Streamed);
        assert_eq!(plan.tier_of(LocalEntityId::new(2)), ResidencyTier::Interactive);
        assert_eq!(plan.explicit(LocalEntityId::new(2)), None);
    }

    #[test]
    fn iterates_sorted_by_local_id() {
        let mut plan = ResidencyPlan::new();
        plan.set(LocalEntityId::new(5), ResidencyTier::Proxy);
        plan.set(LocalEntityId::new(2), ResidencyTier::Streamed);
        let order: Vec<u32> = plan.iter().map(|(id, _)| id.index()).collect();
        assert_eq!(order, [2, 5]);
    }

    #[test]
    fn reannotating_returns_previous() {
        let mut plan = ResidencyPlan::new();
        assert_eq!(plan.set(LocalEntityId::new(0), ResidencyTier::Proxy), None);
        assert_eq!(
            plan.set(LocalEntityId::new(0), ResidencyTier::Streamed),
            Some(ResidencyTier::Proxy)
        );
    }
}
