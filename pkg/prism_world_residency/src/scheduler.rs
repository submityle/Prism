use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::{MaterializationBudget, ResidencyPolicy, ResidencyRecord, ResidencyState};

/// Why an item stayed in its current state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyKeepReason {
    /// The hysteresis policy already selects the current state.
    Stable,
    /// The minimum resident time has not elapsed.
    MinimumResidence,
    /// The desired promotion did not fit the remaining frame budget.
    Budget,
}

/// One deterministic state decision emitted by [`ResidencyScheduler`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyAction {
    /// Promote an item and reserve its entity/byte cost.
    Promote {
        /// Stable item identity.
        id: u64,
        /// State before the promotion.
        from: ResidencyState,
        /// State after the promotion.
        to: ResidencyState,
        /// Reserved entity work.
        entities: u32,
        /// Reserved byte work.
        bytes: u64,
    },
    /// Demote an item; demotions do not consume the promotion budget.
    Demote {
        /// Stable item identity.
        id: u64,
        /// State before the demotion.
        from: ResidencyState,
        /// State after the demotion.
        to: ResidencyState,
    },
    /// Keep the current state for this frame.
    Keep {
        /// Stable item identity.
        id: u64,
        /// Current committed state.
        state: ResidencyState,
        /// State selected by distance policy before budget/hold clipping.
        desired: ResidencyState,
        /// Reason the action remains a keep.
        reason: ResidencyKeepReason,
    },
}

impl ResidencyAction {
    /// Returns the stable identity carried by this action.
    pub const fn id(self) -> u64 {
        match self {
            Self::Promote { id, .. } | Self::Demote { id, .. } | Self::Keep { id, .. } => id,
        }
    }
}

/// The complete output of one scheduler pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyPlan {
    /// Actions in execution order.
    pub actions: Vec<ResidencyAction>,
    /// Budget after all accepted promotions.
    pub budget: MaterializationBudget,
}

/// Budgeted, hysteretic, deterministic residency planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyScheduler {
    policy: ResidencyPolicy,
}

impl ResidencyScheduler {
    /// Creates a scheduler from a distance/hold policy.
    pub const fn new(policy: ResidencyPolicy) -> Self {
        Self { policy }
    }

    /// Returns the policy used by this scheduler.
    pub const fn policy(self) -> ResidencyPolicy {
        self.policy
    }

    /// Plans every record exactly once.
    ///
    /// Promotions are ordered nearest-first, then by descending priority and
    /// stable id. Demotions are ordered farthest-first, then by ascending
    /// priority and stable id. Keeps are emitted last by stable id. This makes
    /// budget clipping independent of input slice order.
    pub fn plan(
        &self,
        frame: u64,
        records: &[ResidencyRecord],
        mut budget: MaterializationBudget,
    ) -> ResidencyPlan {
        let mut candidates = Vec::with_capacity(records.len());
        for record in records {
            let desired = self.policy.target_state(record.state, record.distance);
            let held = record
                .last_transition_frame
                .map(|last| frame.saturating_sub(last) < u64::from(self.policy.min_resident_frames))
                .unwrap_or(false);
            if held && desired != record.state {
                candidates.push(Candidate::Keep {
                    id: record.id,
                    state: record.state,
                    desired,
                    reason: ResidencyKeepReason::MinimumResidence,
                });
            } else if desired == record.state {
                candidates.push(Candidate::Keep {
                    id: record.id,
                    state: record.state,
                    desired,
                    reason: ResidencyKeepReason::Stable,
                });
            } else if record.state.is_upgrade_to(desired) {
                candidates.push(Candidate::Promote {
                    record: *record,
                    to: desired,
                });
            } else {
                candidates.push(Candidate::Demote {
                    record: *record,
                    to: desired,
                });
            }
        }
        candidates.sort_by(Candidate::ordering);

        let mut actions = Vec::with_capacity(records.len());
        for candidate in candidates {
            match candidate {
                Candidate::Promote { record, to } => {
                    if budget.try_reserve(
                        record.materialization_entities,
                        record.materialization_bytes,
                    ) {
                        actions.push(ResidencyAction::Promote {
                            id: record.id,
                            from: record.state,
                            to,
                            entities: record.materialization_entities,
                            bytes: record.materialization_bytes,
                        });
                    } else {
                        actions.push(ResidencyAction::Keep {
                            id: record.id,
                            state: record.state,
                            desired: to,
                            reason: ResidencyKeepReason::Budget,
                        });
                    }
                }
                Candidate::Demote { record, to } => actions.push(ResidencyAction::Demote {
                    id: record.id,
                    from: record.state,
                    to,
                }),
                Candidate::Keep {
                    id,
                    state,
                    desired,
                    reason,
                } => actions.push(ResidencyAction::Keep {
                    id,
                    state,
                    desired,
                    reason,
                }),
            }
        }
        ResidencyPlan { actions, budget }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Candidate {
    Promote {
        record: ResidencyRecord,
        to: ResidencyState,
    },
    Demote {
        record: ResidencyRecord,
        to: ResidencyState,
    },
    Keep {
        id: u64,
        state: ResidencyState,
        desired: ResidencyState,
        reason: ResidencyKeepReason,
    },
}

impl Candidate {
    fn ordering(left: &Self, right: &Self) -> Ordering {
        match (left, right) {
            (Self::Promote { record: a, .. }, Self::Promote { record: b, .. }) => a
                .distance
                .cmp(&b.distance)
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.id.cmp(&b.id)),
            (Self::Promote { .. }, _) => Ordering::Less,
            (_, Self::Promote { .. }) => Ordering::Greater,
            (Self::Demote { record: a, .. }, Self::Demote { record: b, .. }) => b
                .distance
                .cmp(&a.distance)
                .then_with(|| a.priority.cmp(&b.priority))
                .then_with(|| a.id.cmp(&b.id)),
            (Self::Demote { .. }, Self::Keep { .. }) => Ordering::Less,
            (Self::Keep { .. }, Self::Demote { .. }) => Ordering::Greater,
            (Self::Keep { id: a, .. }, Self::Keep { id: b, .. }) => a.cmp(b),
        }
    }
}
