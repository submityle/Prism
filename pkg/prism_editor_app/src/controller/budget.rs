//! Host-owned document resource contracts and sparse, reversible accounting.

use super::*;

type Contribution = dyn Fn(&SelectionItem, &dyn Reflect) -> Result<u64, String> + Send + Sync;

/// Immutable host resource policy, never executable code loaded from a document.
/// Contributions must be deterministic, side-effect free, nonnegative and based
/// only on the supplied identity/root. Cloned policies share their identity.
#[derive(Clone)]
pub struct ControllerDocumentBudget {
    id: &'static str,
    limit: u64,
    contribution: Arc<Contribution>,
}

impl ControllerDocumentBudget {
    /// Defines an aggregate bound. Zero permits only zero-cost objects.
    #[must_use]
    pub fn new<F>(id: &'static str, limit: u64, contribution: F) -> Self
    where
        F: Fn(&SelectionItem, &dyn Reflect) -> Result<u64, String> + Send + Sync + 'static,
    {
        Self {
            id,
            limit,
            contribution: Arc::new(contribution),
        }
    }

    fn same_policy(&self, other: &Self) -> bool {
        self.limit == other.limit && Arc::ptr_eq(&self.contribution, &other.contribution)
    }

    fn measure(&self, item: &SelectionItem, root: &dyn Reflect) -> Result<u64, ControllerError> {
        (self.contribution)(item, root).map_err(|reason| {
            ControllerError::DocumentBudgetContribution {
                budget: self.id,
                item: item.clone(),
                reason,
            }
        })
    }

    fn exceeded(&self) -> ControllerError {
        ControllerError::DocumentBudgetExceeded {
            budget: self.id,
            limit: self.limit,
        }
    }
}

pub(super) struct BudgetState {
    policy: ControllerDocumentBudget,
    total: u64,
    objects: BTreeMap<SelectionItem, u64>,
}

/// Only touched identities are retained in field history. Totals are derived
/// against the current baseline, so later bootstrap registrations remain counted.
#[derive(Default)]
pub(super) struct CostPatch {
    values: Vec<BTreeMap<SelectionItem, Option<u64>>>,
}

impl EditorController {
    /// Installs immutable policies atomically before authored history exists.
    /// Reinstalling the same cloned policy is a no-op, even after editing.
    /// Policies cannot be replaced while old inverse accounting is retained.
    pub fn register_document_budgets<I>(&mut self, budgets: I) -> Result<(), ControllerError>
    where
        I: IntoIterator<Item = ControllerDocumentBudget>,
    {
        self.ensure_idle()?;
        let mut candidate = self.prepare_budget_registrations(budgets)?;
        if candidate.is_empty() {
            return Ok(());
        }
        self.ensure_revision(false)?;
        self.budgets.append(&mut candidate);
        self.advance(false);
        Ok(())
    }

    /// Returns cached aggregate usage without scanning or cloning any root.
    #[must_use]
    pub fn document_budget_usage(&self, id: &str) -> Option<(u64, u64)> {
        self.budgets
            .iter()
            .find(|state| state.policy.id == id)
            .map(|state| (state.total, state.policy.limit))
    }

    pub(super) fn prepare_budget_registrations<I>(
        &self,
        budgets: I,
    ) -> Result<Vec<BudgetState>, ControllerError>
    where
        I: IntoIterator<Item = ControllerDocumentBudget>,
    {
        let mut candidate: Vec<BudgetState> = Vec::new();
        for policy in budgets {
            ensure_bound(
                policy.id.len(),
                MAX_ACTION_TEXT_BYTES,
                "document budget ID bytes",
            )?;
            if policy.id.is_empty() {
                return Err(ControllerError::DuplicateDocumentBudget(policy.id));
            }
            if let Some(existing) = self
                .budgets
                .iter()
                .chain(&candidate)
                .find(|state| state.policy.id == policy.id)
            {
                if !existing.policy.same_policy(&policy) {
                    return Err(ControllerError::DuplicateDocumentBudget(policy.id));
                }
                continue;
            }
            if !self.undo.is_empty() || !self.redo.is_empty() {
                return Err(ControllerError::EditConflict);
            }
            ensure_capacity(
                self.budgets.len() + candidate.len(),
                self.limits.max_document_budgets.min(64),
                "document budgets",
            )?;
            candidate.push(self.measure_budget(policy)?);
        }
        Ok(candidate)
    }

    fn measure_budget(
        &self,
        policy: ControllerDocumentBudget,
    ) -> Result<BudgetState, ControllerError> {
        let mut total = 0u64;
        let mut objects = BTreeMap::new();
        for (item, editor) in &self.objects {
            let value = policy.measure(item, editor.root())?;
            total = total.checked_add(value).ok_or_else(|| policy.exceeded())?;
            if total > policy.limit {
                return Err(policy.exceeded());
            }
            objects.insert(item.clone(), value);
        }
        Ok(BudgetState {
            policy,
            total,
            objects,
        })
    }

    /// Reload uses the new roots, not the old document's cached contributions.
    /// All fallible work precedes publication of either commands or policies.
    pub(crate) fn inherit_host_configuration(
        &mut self,
        source: &Self,
    ) -> Result<(), ControllerError> {
        ensure_bound(
            source.commands.len(),
            self.limits.max_commands,
            "registered commands",
        )?;
        ensure_bound(
            source.budgets.len(),
            self.limits.max_document_budgets.min(64),
            "document budgets",
        )?;
        debug_assert!(self.commands.is_empty() && self.budgets.is_empty());
        let budgets = source
            .budgets
            .iter()
            .map(|state| self.measure_budget(state.policy.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        self.commands = source.commands.clone();
        self.budgets = budgets;
        Ok(())
    }

    pub(super) fn prepare_costs<'a>(
        &self,
        roots: impl IntoIterator<Item = (&'a SelectionItem, Option<&'a dyn Reflect>)>,
    ) -> Result<CostPatch, ControllerError> {
        if self.budgets.is_empty() {
            return Ok(CostPatch::default());
        }
        let roots: Vec<_> = roots.into_iter().collect();
        let mut values = Vec::with_capacity(self.budgets.len());
        for state in &self.budgets {
            let mut objects = BTreeMap::new();
            for (item, root) in &roots {
                let cost = root
                    .map(|root| state.policy.measure(item, root))
                    .transpose()?;
                objects.insert((*item).clone(), cost);
            }
            values.push(objects);
        }
        let patch = CostPatch { values };
        self.check_costs(&patch)?;
        Ok(patch)
    }

    pub(super) fn check_costs(&self, patch: &CostPatch) -> Result<Vec<u64>, ControllerError> {
        if patch.values.len() != self.budgets.len() {
            return Err(ControllerError::EditConflict);
        }
        self.budgets
            .iter()
            .zip(&patch.values)
            .map(|(state, desired)| {
                let mut total = state.total;
                // Subtract the entire replaced set first. A balanced grow/shrink
                // batch must not depend on identity order or transient totals.
                for item in desired.keys() {
                    total = total
                        .checked_sub(state.objects.get(item).copied().unwrap_or(0))
                        .ok_or(ControllerError::EditConflict)?;
                }
                for cost in desired.values().flatten() {
                    total = total
                        .checked_add(*cost)
                        .ok_or_else(|| state.policy.exceeded())?;
                    if total > state.policy.limit {
                        return Err(state.policy.exceeded());
                    }
                }
                Ok(total)
            })
            .collect()
    }

    pub(super) fn exchange_costs(&mut self, patch: &mut CostPatch, totals: Vec<u64>) {
        for ((state, desired), total) in self.budgets.iter_mut().zip(&mut patch.values).zip(totals)
        {
            for (item, cost) in desired {
                *cost = if let Some(value) = *cost {
                    state.objects.insert(item.clone(), value)
                } else {
                    state.objects.remove(item)
                };
            }
            state.total = total;
        }
    }
}
