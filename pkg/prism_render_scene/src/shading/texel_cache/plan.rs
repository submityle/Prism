//! Deterministic visible-page shade planning.

use super::{
    key::ShadePageKey,
    residency::{ShadePageRequest, ShadePageState, ShadeResidencyTable},
};

/// A page selected for the current shade budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadePlanItem {
    /// Page to shade.
    pub key: ShadePageKey,
    /// Priority used for selection.
    pub priority: u64,
    /// Bytes reserved by this operation.
    pub byte_cost: u64,
}

/// A deterministic frame plan. `deferred` remains visible to telemetry and can
/// be fed into the next frame's request accumulator without losing demand.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadePlan {
    /// Material/light epoch represented by the plan.
    pub epoch: u64,
    /// Pages selected under the byte budget.
    pub selected: Vec<ShadePlanItem>,
    /// Pages observed but deferred by the budget.
    pub deferred: Vec<ShadePlanItem>,
    /// Sum of selected page costs.
    pub selected_bytes: u64,
}

/// Builds a plan from visible requests. Input order does not affect output.
pub fn build_shade_plan(
    table: &mut ShadeResidencyTable,
    requests: impl IntoIterator<Item = ShadePageRequest>,
    epoch: u64,
    byte_budget: u64,
) -> ShadePlan {
    for request in requests {
        table.request(request, epoch);
    }
    let mut candidates: Vec<ShadePlanItem> = table
        .iter()
        .filter_map(|(key, state, priority, byte_cost)| {
            matches!(
                state,
                ShadePageState::Requested | ShadePageState::Stale { .. }
            )
            .then_some(ShadePlanItem {
                key,
                priority,
                byte_cost,
            })
        })
        .collect();
    candidates.sort_unstable_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.byte_cost.cmp(&b.byte_cost))
            .then_with(|| a.key.cmp(&b.key))
    });

    let mut plan = ShadePlan {
        epoch,
        ..Default::default()
    };
    for item in candidates {
        if item.byte_cost <= byte_budget.saturating_sub(plan.selected_bytes) {
            plan.selected_bytes = plan.selected_bytes.saturating_add(item.byte_cost);
            plan.selected.push(item);
        } else {
            plan.deferred.push(item);
        }
    }
    plan
}
