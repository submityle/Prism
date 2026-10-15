//! Deterministic, budgeted scene residency scheduling.
//!
//! The scheduler is deliberately world- and renderer-neutral. It decides which
//! scene instances may change tier in one frame; the world system performs the
//! actual ECS materialization and the renderer consumes proxy records. Keeping
//! this decision plane separate prevents a GPU backend from becoming the scene
//! model's source of truth.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::identity::PersistentId;
use crate::residency::ResidencyTier;

/// Per-frame work limits for tier transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyBudget {
    /// Maximum estimated ECS entities touched by one frame.
    pub max_entities: usize,
    /// Maximum estimated proxy/column bytes touched by one frame.
    pub max_bytes: usize,
}

impl Default for ResidencyBudget {
    fn default() -> Self {
        Self {
            max_entities: usize::MAX,
            max_bytes: usize::MAX,
        }
    }
}

/// A requested tier transition for one stable scene instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyRequest {
    /// Stable instance identity used for deterministic ordering.
    pub instance: PersistentId,
    /// Desired residency tier.
    pub desired: ResidencyTier,
    /// Higher values are scheduled first.
    pub priority: u32,
    /// Estimated ECS entity work for the transition.
    pub entities: usize,
    /// Estimated byte work for the transition.
    pub bytes: usize,
}

/// One transition admitted to the current frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyAction {
    /// Stable instance identity.
    pub instance: PersistentId,
    /// Previous effective tier.
    pub from: ResidencyTier,
    /// New effective tier.
    pub to: ResidencyTier,
    /// Reserved entity budget.
    pub entities: usize,
    /// Reserved byte budget.
    pub bytes: usize,
}

/// Persistent residency state and deterministic frame scheduler.
#[derive(Clone, Debug, Default)]
pub struct ResidencyScheduler {
    current: BTreeMap<PersistentId, ResidencyTier>,
}

impl ResidencyScheduler {
    /// Creates an empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the currently committed tier, defaulting to `Proxy` for an
    /// instance that has not yet been materialized.
    #[must_use]
    pub fn current(&self, instance: PersistentId) -> ResidencyTier {
        self.current
            .get(&instance)
            .copied()
            .unwrap_or(ResidencyTier::Proxy)
    }

    /// Commits an admitted action after its world-side operation succeeds.
    pub fn commit(&mut self, action: ResidencyAction) {
        self.current.insert(action.instance, action.to);
    }

    /// Selects transitions that fit the frame budget.
    ///
    /// Requests are ordered by priority descending and stable id ascending.
    /// A request whose individual estimate exceeds the remaining budget is
    /// skipped, allowing smaller work behind it to make progress without
    /// introducing a non-deterministic queue rotation.
    #[must_use]
    pub fn schedule(
        &self,
        requests: &[ResidencyRequest],
        budget: ResidencyBudget,
    ) -> Vec<ResidencyAction> {
        let mut ordered = requests.to_vec();
        ordered.sort_unstable_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.instance.cmp(&b.instance))
        });
        let mut used_entities = 0usize;
        let mut used_bytes = 0usize;
        let mut actions = Vec::new();
        for request in ordered {
            let from = self.current(request.instance);
            if from == request.desired
                || used_entities.saturating_add(request.entities) > budget.max_entities
                || used_bytes.saturating_add(request.bytes) > budget.max_bytes
            {
                continue;
            }
            used_entities = used_entities.saturating_add(request.entities);
            used_bytes = used_bytes.saturating_add(request.bytes);
            actions.push(ResidencyAction {
                instance: request.instance,
                from,
                to: request.desired,
                entities: request.entities,
                bytes: request.bytes,
            });
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u128) -> PersistentId {
        PersistentId::from_u128(value)
    }

    #[test]
    fn schedule_is_priority_first_and_budgeted() {
        let scheduler = ResidencyScheduler::new();
        let requests = [
            ResidencyRequest {
                instance: id(2),
                desired: ResidencyTier::Interactive,
                priority: 10,
                entities: 3,
                bytes: 3,
            },
            ResidencyRequest {
                instance: id(1),
                desired: ResidencyTier::Streamed,
                priority: 10,
                entities: 2,
                bytes: 2,
            },
        ];
        let actions = scheduler.schedule(
            &requests,
            ResidencyBudget {
                max_entities: 3,
                max_bytes: 3,
            },
        );
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].instance, id(1));
    }

    #[test]
    fn committed_tier_is_used_by_next_frame() {
        let mut scheduler = ResidencyScheduler::new();
        let action = ResidencyAction {
            instance: id(7),
            from: ResidencyTier::Proxy,
            to: ResidencyTier::Streamed,
            entities: 2,
            bytes: 8,
        };
        scheduler.commit(action);
        assert_eq!(scheduler.current(id(7)), ResidencyTier::Streamed);
    }
}
