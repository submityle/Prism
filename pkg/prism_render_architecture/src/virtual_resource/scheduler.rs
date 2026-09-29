//! Deterministic, budget- and dependency-aware residency scheduler.
//!
//! Given a [`VirtualResourceTable`] of demands and a [`ResidencyBudget`], this
//! picks the resident set that retains the most priority within budget and
//! reports the frame's concrete work: which resources to upload, which to evict,
//! and which had to be deferred. The policy is a priority-ordered greedy fill
//! with two extra rules that a flat texture-page scheduler does not need:
//!
//! * **Parent dependencies.** A resource is admitted only once every ancestor up
//!   its parent chain is admitted first, so a child page can never be resident
//!   while its parent is missing. When a parent cannot fit, the child is deferred
//!   with it.
//! * **Two budgets.** The hard ceiling bounds the whole resident set; the
//!   per-frame upload limit bounds only *fresh* residency, so already-resident
//!   resources are kept for free while new uploads are throttled.
//!
//! Everything is checked integer arithmetic over key-ordered containers, so the
//! same table and budget always yield the same plan. The scheduler is pure and
//! never mutates the table; [`schedule_and_apply`] folds the plan back in for
//! `CPU`-side simulation. No byte is a real allocation: the backend performs the
//! uploads, pending the `GPU` backend.

use super::budget::BudgetLedger;
use super::registry::{ResourceKey, VirtualResourceTable};
use super::ResidencyBudget;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// The work one scheduling pass resolves to.
///
/// `uploads` lists resources newly admitted this frame in admission order (a
/// parent always precedes any child it unblocks). `evictions` lists resources
/// that were resident but no longer fit, least valuable first. `deferred` lists
/// demanded resources that could not be admitted this frame — blocked by the
/// upload limit, the hard ceiling, or an unmet parent — most urgent first.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResidencyPlan<K> {
    /// Resources to upload this frame, in admission order.
    pub uploads: Vec<K>,
    /// Resources to evict this frame, least valuable first.
    pub evictions: Vec<K>,
    /// Demanded resources held back this frame, most urgent first.
    pub deferred: Vec<K>,
    /// Total byte cost of the chosen resident set (within the hard ceiling).
    pub resident_bytes: u64,
    /// Bytes of fresh residency uploaded this frame (within the upload limit).
    pub uploaded_bytes: u64,
    /// Whether the resident set exceeds the soft comfort target.
    pub over_soft: bool,
}

impl<K> ResidencyPlan<K> {
    /// Whether the plan requires no uploads and no evictions.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.uploads.is_empty() && self.evictions.is_empty()
    }
}

/// A candidate resource flattened out of the table for ordering.
#[derive(Clone, Copy, Debug)]
struct Candidate<K> {
    priority: u32,
    byte_cost: u64,
    last_used_frame: u64,
    resident: bool,
    parent: Option<K>,
}

/// Mutable state threaded through one admission pass.
struct Pass<'a, K: ResourceKey> {
    lookup: &'a BTreeMap<K, Candidate<K>>,
    ledger: BudgetLedger,
    chosen: BTreeSet<K>,
    failed: BTreeSet<K>,
    uploads: Vec<K>,
}

impl<K: ResourceKey> Pass<'_, K> {
    /// Attempts to admit `key`, first admitting its parent chain.
    ///
    /// Returns `true` when the resource is resident in the chosen set after the
    /// call. A resource whose parent cannot be admitted, or that does not fit the
    /// budget, is recorded as failed so it is never retried within the pass.
    fn admit(&mut self, key: K) -> bool {
        if self.chosen.contains(&key) {
            return true;
        }
        if self.failed.contains(&key) {
            return false;
        }
        let Some(cand) = self.lookup.get(&key).copied() else {
            return false;
        };
        if let Some(parent) = cand.parent
            && !self.admit(parent)
        {
            self.failed.insert(key);
            return false;
        }
        if self.ledger.commit(cand.byte_cost, cand.resident) {
            self.chosen.insert(key);
            if !cand.resident {
                self.uploads.push(key);
            }
            true
        } else {
            self.failed.insert(key);
            false
        }
    }
}

/// Orders candidates for admission: highest priority first, then fresher use,
/// then ascending key as the final deterministic tie-break.
fn admission_order<K: Ord>(a: &(K, Candidate<K>), b: &(K, Candidate<K>)) -> core::cmp::Ordering {
    b.1.priority
        .cmp(&a.1.priority)
        .then_with(|| b.1.last_used_frame.cmp(&a.1.last_used_frame))
        .then_with(|| a.0.cmp(&b.0))
}

/// Orders resident victims for eviction: lowest priority first, then least
/// recently used, then ascending key.
fn eviction_order<K: Ord>(a: &(K, Candidate<K>), b: &(K, Candidate<K>)) -> core::cmp::Ordering {
    a.1.priority
        .cmp(&b.1.priority)
        .then_with(|| a.1.last_used_frame.cmp(&b.1.last_used_frame))
        .then_with(|| a.0.cmp(&b.0))
}

/// Plans this frame's uploads, evictions, and deferrals for `budget`.
///
/// Every registered resource is treated as a demand for residency. Candidates
/// are admitted highest priority first, subject to their parent chain and both
/// budgets; the load / evict / defer lists follow from the difference between
/// the chosen set and what is currently resident.
#[must_use]
pub fn schedule<K: ResourceKey>(
    table: &VirtualResourceTable<K>,
    budget: ResidencyBudget,
) -> ResidencyPlan<K> {
    let lookup: BTreeMap<K, Candidate<K>> = table
        .iter()
        .map(|(key, entry)| {
            (
                key,
                Candidate {
                    priority: entry.priority.0,
                    byte_cost: entry.byte_cost,
                    last_used_frame: entry.last_used_frame,
                    resident: entry.is_resident(),
                    parent: entry.parent,
                },
            )
        })
        .collect();

    let mut order: Vec<(K, Candidate<K>)> = lookup.iter().map(|(k, c)| (*k, *c)).collect();
    order.sort_by(admission_order);

    let mut pass = Pass {
        lookup: &lookup,
        ledger: BudgetLedger::new(budget),
        chosen: BTreeSet::new(),
        failed: BTreeSet::new(),
        uploads: Vec::new(),
    };
    for (key, _) in &order {
        pass.admit(*key);
    }

    let mut victims: Vec<(K, Candidate<K>)> = order
        .iter()
        .filter(|(k, c)| c.resident && !pass.chosen.contains(k))
        .map(|(k, c)| (*k, *c))
        .collect();
    victims.sort_by(eviction_order);
    let evictions: Vec<K> = victims.into_iter().map(|(k, _)| k).collect();

    // `order` is already in admission order, so deferrals come out most urgent
    // first without a second sort.
    let deferred: Vec<K> = order
        .iter()
        .filter(|(k, c)| !c.resident && !pass.chosen.contains(k))
        .map(|(k, _)| *k)
        .collect();

    ResidencyPlan {
        uploads: pass.uploads,
        evictions,
        deferred,
        resident_bytes: pass.ledger.resident_bytes(),
        uploaded_bytes: pass.ledger.uploaded_bytes(),
        over_soft: pass.ledger.over_soft(),
    }
}

/// Schedules and folds the plan back into `table` for `CPU`-side simulation.
///
/// Uploaded resources are marked [`crate::virtual_resource::ResidencyState::Resident`]
/// (modelling a completed backend upload) and evicted resources
/// [`crate::virtual_resource::ResidencyState::Missing`]. Deferred resources are
/// left as the caller staged them. The real uploads are issued by the backend,
/// pending the `GPU` backend.
pub fn schedule_and_apply<K: ResourceKey>(
    table: &mut VirtualResourceTable<K>,
    budget: ResidencyBudget,
) -> ResidencyPlan<K> {
    let plan = schedule(table, budget);
    for key in &plan.uploads {
        table.mark_resident(*key);
    }
    for key in &plan.evictions {
        table.mark_evicted(*key);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_resource::{RequestPriority, ResidencyState};

    fn table() -> VirtualResourceTable<u32> {
        VirtualResourceTable::new()
    }

    fn declare(t: &mut VirtualResourceTable<u32>, key: u32, prio: u32, bytes: u64) {
        t.declare(key, RequestPriority(prio), bytes);
    }

    #[test]
    fn all_fit_uploads_in_priority_order() {
        let mut t = table();
        declare(&mut t, 0, 100, 1_000);
        declare(&mut t, 1, 900, 1_000);
        declare(&mut t, 2, 600, 1_000);
        let plan = schedule(&t, ResidencyBudget::new(10_000, 10_000, 10_000));
        assert_eq!(plan.uploads, alloc::vec![1, 2, 0]);
        assert!(plan.evictions.is_empty());
        assert!(plan.deferred.is_empty());
        assert_eq!(plan.resident_bytes, 3_000);
        assert_eq!(plan.uploaded_bytes, 3_000);
    }

    #[test]
    fn hard_ceiling_defers_lowest_priority() {
        let mut t = table();
        declare(&mut t, 0, 100, 1_000);
        declare(&mut t, 1, 900, 1_000);
        declare(&mut t, 2, 600, 1_000);
        // Hard ceiling holds only two of the three.
        let plan = schedule(&t, ResidencyBudget::new(2_000, 2_000, 10_000));
        assert_eq!(plan.uploads, alloc::vec![1, 2]);
        assert_eq!(plan.deferred, alloc::vec![0]);
        assert_eq!(plan.resident_bytes, 2_000);
    }

    #[test]
    fn upload_limit_throttles_fresh_uploads() {
        let mut t = table();
        declare(&mut t, 0, 900, 500);
        declare(&mut t, 1, 600, 500);
        // Hard ceiling is generous, but only 500 bytes may upload per frame.
        let plan = schedule(&t, ResidencyBudget::new(10_000, 10_000, 500));
        assert_eq!(plan.uploads, alloc::vec![0]);
        assert_eq!(plan.deferred, alloc::vec![1]);
        assert_eq!(plan.uploaded_bytes, 500);
    }

    #[test]
    fn resident_bytes_do_not_consume_upload_limit() {
        let mut t = table();
        declare(&mut t, 0, 900, 500);
        t.mark_resident(0);
        declare(&mut t, 1, 600, 500);
        // Upload limit is 500: the resident page is free, the new one just fits.
        let plan = schedule(&t, ResidencyBudget::new(10_000, 10_000, 500));
        assert_eq!(plan.uploads, alloc::vec![1]);
        assert!(plan.deferred.is_empty());
        assert_eq!(plan.uploaded_bytes, 500);
        assert_eq!(plan.resident_bytes, 1_000);
    }

    #[test]
    fn over_budget_evicts_lowest_priority_resident_first() {
        let mut t = table();
        for (k, prio) in [(0u32, 300u32), (1, 900), (2, 600)] {
            declare(&mut t, k, prio, 1_000);
            t.mark_resident(k);
        }
        // Room for two of the three resident pages.
        let plan = schedule(&t, ResidencyBudget::new(2_000, 2_000, 0));
        assert!(plan.uploads.is_empty());
        assert_eq!(plan.evictions, alloc::vec![0]);
        assert_eq!(plan.resident_bytes, 2_000);
    }

    #[test]
    fn parent_is_uploaded_before_a_higher_priority_child() {
        let mut t = table();
        // Child outranks its parent, but the parent must land first.
        declare(&mut t, 1, 10, 500); // parent
        declare(&mut t, 2, 900, 500); // child
        t.set_parent(2, Some(1));
        let plan = schedule(&t, ResidencyBudget::new(10_000, 10_000, 10_000));
        assert_eq!(plan.uploads, alloc::vec![1, 2]);
    }

    #[test]
    fn child_is_deferred_when_parent_cannot_fit() {
        let mut t = table();
        declare(&mut t, 1, 10, 1_000); // parent, too big for the budget
        declare(&mut t, 2, 900, 100); // child, would fit alone
        t.set_parent(2, Some(1));
        // Budget holds the small child but not the parent it depends on.
        let plan = schedule(&t, ResidencyBudget::new(500, 500, 10_000));
        assert!(plan.uploads.is_empty());
        assert_eq!(plan.deferred, alloc::vec![2, 1]);
        assert_eq!(plan.resident_bytes, 0);
    }

    #[test]
    fn child_admitted_when_parent_already_resident() {
        let mut t = table();
        declare(&mut t, 1, 10, 1_000);
        t.mark_resident(1);
        declare(&mut t, 2, 900, 500);
        t.set_parent(2, Some(1));
        // Only 500 bytes may upload; the resident parent is free.
        let plan = schedule(&t, ResidencyBudget::new(10_000, 10_000, 500));
        assert_eq!(plan.uploads, alloc::vec![2]);
        assert_eq!(plan.resident_bytes, 1_500);
    }

    #[test]
    fn over_soft_flag_is_reported() {
        let mut t = table();
        declare(&mut t, 0, 900, 1_500);
        let plan = schedule(&t, ResidencyBudget::new(1_000, 2_000, 10_000));
        assert!(plan.over_soft);
        assert_eq!(plan.resident_bytes, 1_500);
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        let build = || {
            let mut t = table();
            for (k, prio) in [(0u32, 700u32), (1, 700), (2, 200), (3, 950)] {
                declare(&mut t, k, prio, 400);
                t.touch(k, u64::from(k));
                if k % 2 == 0 {
                    t.mark_resident(k);
                }
            }
            t
        };
        let budget = ResidencyBudget::new(1_000, 1_000, 10_000);
        let a = schedule(&build(), budget);
        let b = schedule(&build(), budget);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_settles_state_and_invalidation_forces_reupload() {
        let mut t = table();
        declare(&mut t, 0, 900, 1_000);
        declare(&mut t, 1, 300, 1_000);
        t.mark_resident(1);
        // Budget for one page: upload the 900, evict the resident 300.
        let plan = schedule_and_apply(&mut t, ResidencyBudget::new(1_000, 1_000, 10_000));
        assert_eq!(plan.uploads, alloc::vec![0]);
        assert_eq!(plan.evictions, alloc::vec![1]);
        assert_eq!(t.state(0), ResidencyState::Resident);
        assert_eq!(t.state(1), ResidencyState::Missing);

        // Invalidate the resident resource with a newer epoch: it must re-upload.
        t.invalidate_epoch(0, 7);
        assert_eq!(t.state(0), ResidencyState::Missing);
        let plan = schedule_and_apply(&mut t, ResidencyBudget::new(1_000, 1_000, 10_000));
        assert_eq!(plan.uploads, alloc::vec![0]);
        assert_eq!(t.state(0), ResidencyState::Resident);
    }
}
