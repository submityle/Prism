//! Budget-constrained, deterministic residency scheduler.
//!
//! Given a [`TextureResidencyTable`] of live page demands and a byte budget for
//! the physical pool, this picks the resident set that maximizes retained
//! priority within the budget, then reports the concrete work: which requested
//! pages to upload this frame and which resident pages to drop. The policy is a
//! priority-ordered greedy fill — pages are considered highest priority first
//! and admitted whenever they still fit, so a small lower-priority page can slip
//! into space a larger higher-priority page could not use. Everything is
//! integer arithmetic over key-ordered containers, so identical input always
//! yields identical output.
//!
//! The scheduler is pure: it never mutates the table. Callers issue the plan's
//! uploads and, as the backend confirms them, call
//! [`TextureResidencyTable::mark_resident`]; the whole plan can also be folded
//! back in one step with [`TextureResidencyTable::apply_plan`] for `CPU`-side
//! simulation, pending the `GPU` backend.

use super::residency::TextureResidencyTable;
use super::TexturePageKey;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// The work a single frame's scheduling pass resolves to.
///
/// `loads` lists requested pages selected for upload this frame, in descending
/// priority (most urgent first). `evicts` lists resident pages the budget can no
/// longer keep, in eviction order (least valuable first). `resident_bytes` is
/// the total byte cost of the chosen resident set and never exceeds the budget.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamingPlan {
    /// Requested pages to upload this frame, most urgent first.
    pub loads: Vec<TexturePageKey>,
    /// Resident pages to drop this frame, least valuable first.
    pub evicts: Vec<TexturePageKey>,
    /// Total byte cost of the resulting resident set (within budget).
    pub resident_bytes: u64,
}

impl StreamingPlan {
    /// Whether the plan requires no uploads and no evictions.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.loads.is_empty() && self.evicts.is_empty()
    }
}

/// A candidate page flattened out of the table for ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Candidate {
    key: TexturePageKey,
    priority: u64,
    byte_cost: u64,
    last_used_frame: u64,
    resident: bool,
}

/// Orders candidates for admission: highest priority first, then fresher use,
/// then ascending key as the final deterministic tie-break.
fn admission_order(a: &Candidate, b: &Candidate) -> core::cmp::Ordering {
    b.priority
        .cmp(&a.priority)
        .then_with(|| b.last_used_frame.cmp(&a.last_used_frame))
        .then_with(|| a.key.cmp(&b.key))
}

/// Orders resident victims for eviction: lowest priority first, then least
/// recently used, then ascending key.
fn eviction_order(a: &Candidate, b: &Candidate) -> core::cmp::Ordering {
    a.priority
        .cmp(&b.priority)
        .then_with(|| a.last_used_frame.cmp(&b.last_used_frame))
        .then_with(|| a.key.cmp(&b.key))
}

/// Plans this frame's uploads and evictions for `byte_budget` physical bytes.
///
/// Considers every live (requested or resident) page, admits pages in priority
/// order while they fit the budget, and derives the load/evict lists from the
/// difference between the chosen set and what is currently resident. A page too
/// large for the remaining budget is skipped, not fatal: later, smaller pages
/// are still tried. With `byte_budget == 0` the resident set is empty, so all
/// resident pages are evicted and nothing is loaded.
#[must_use]
pub fn schedule(table: &TextureResidencyTable, byte_budget: u64) -> StreamingPlan {
    let mut candidates: Vec<Candidate> = table
        .live_pages()
        .map(|(key, rec)| Candidate {
            key,
            priority: rec.priority,
            byte_cost: rec.byte_cost,
            last_used_frame: rec.last_used_frame,
            resident: rec.residency.is_resident(),
        })
        .collect();
    candidates.sort_by(admission_order);

    let mut chosen: BTreeSet<TexturePageKey> = BTreeSet::new();
    let mut loads: Vec<TexturePageKey> = Vec::new();
    let mut resident_bytes: u64 = 0;
    for cand in &candidates {
        let Some(next_total) = resident_bytes.checked_add(cand.byte_cost) else {
            continue;
        };
        if next_total > byte_budget {
            continue;
        }
        resident_bytes = next_total;
        chosen.insert(cand.key);
        if !cand.resident {
            // Requested and now admitted: it must be uploaded this frame.
            loads.push(cand.key);
        }
    }

    let mut victims: Vec<Candidate> = candidates
        .iter()
        .copied()
        .filter(|c| c.resident && !chosen.contains(&c.key))
        .collect();
    victims.sort_by(eviction_order);
    let evicts: Vec<TexturePageKey> = victims.into_iter().map(|c| c.key).collect();

    StreamingPlan {
        loads,
        evicts,
        resident_bytes,
    }
}

/// Convenience: plan against `byte_budget`, then fold the plan back into the
/// table (loads become resident, evicts become non-resident) and return it.
///
/// Collapses the asynchronous upload so a caller can advance a `CPU`-side model
/// frame by frame; the returned plan still describes the work that a real
/// runtime would hand to the backend, pending the `GPU` backend.
pub fn schedule_and_apply(table: &mut TextureResidencyTable, byte_budget: u64) -> StreamingPlan {
    let plan = schedule(table, byte_budget);
    table.apply_plan(&plan);
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_streaming::residency::{PageResidency, TextureResidencyTable};

    fn key(mip: u8, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 3,
            mip,
            layer: 0,
            x,
            y,
        }
    }

    #[test]
    fn empty_table_yields_noop_plan() {
        let table = TextureResidencyTable::new();
        let plan = schedule(&table, 1_000_000);
        assert!(plan.is_noop());
        assert_eq!(plan.resident_bytes, 0);
    }

    #[test]
    fn requested_pages_load_within_budget_priority_first() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 300, 1000, 1);
        table.request(key(0, 0, 1), 900, 1000, 1);
        table.request(key(0, 0, 2), 600, 1000, 1);
        let plan = schedule(&table, 10_000);
        // All fit; loads ordered by descending priority.
        assert_eq!(
            plan.loads,
            alloc::vec![key(0, 0, 1), key(0, 0, 2), key(0, 0, 0)]
        );
        assert!(plan.evicts.is_empty());
        assert_eq!(plan.resident_bytes, 3000);
    }

    #[test]
    fn budget_zero_loads_nothing_and_evicts_all_resident() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 300, 1000, 1);
        table.mark_resident(key(0, 0, 0));
        table.request(key(0, 0, 1), 900, 1000, 1);
        let plan = schedule(&table, 0);
        assert!(plan.loads.is_empty());
        assert_eq!(plan.evicts, alloc::vec![key(0, 0, 0)]);
        assert_eq!(plan.resident_bytes, 0);
    }

    #[test]
    fn over_budget_evicts_lowest_priority_resident_first() {
        let mut table = TextureResidencyTable::new();
        // Three resident pages of 1000 bytes each; budget only holds one.
        for (y, prio) in [(0u16, 300u64), (1, 900), (2, 600)] {
            table.request(key(0, 0, y), prio, 1000, 1);
            table.mark_resident(key(0, 0, y));
        }
        let plan = schedule(&table, 1000);
        // Highest priority (900) is kept; the other two are evicted lowest first.
        assert!(plan.loads.is_empty());
        assert_eq!(plan.evicts, alloc::vec![key(0, 0, 0), key(0, 0, 2)]);
        assert_eq!(plan.resident_bytes, 1000);
    }

    #[test]
    fn lru_breaks_priority_ties_for_admission_and_eviction() {
        let mut table = TextureResidencyTable::new();
        // Equal priority; page used at frame 9 is fresher than the one at 2.
        table.request(key(0, 0, 0), 500, 1000, 2);
        table.mark_resident(key(0, 0, 0));
        table.request(key(0, 0, 1), 500, 1000, 9);
        table.mark_resident(key(0, 0, 1));
        let plan = schedule(&table, 1000);
        // Only one fits: keep the fresher (frame 9), evict the staler (frame 2).
        assert_eq!(plan.evicts, alloc::vec![key(0, 0, 0)]);
        assert_eq!(plan.resident_bytes, 1000);
    }

    #[test]
    fn greedy_skips_oversized_page_to_pack_smaller_one() {
        let mut table = TextureResidencyTable::new();
        // Highest priority page is too big for the budget; a smaller,
        // lower-priority page should still be admitted into the leftover space.
        table.request(key(0, 0, 0), 900, 5000, 1);
        table.request(key(0, 0, 1), 400, 800, 1);
        let plan = schedule(&table, 1000);
        assert_eq!(plan.loads, alloc::vec![key(0, 0, 1)]);
        assert_eq!(plan.resident_bytes, 800);
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        let build = || {
            let mut table = TextureResidencyTable::new();
            for (i, prio) in [(0u16, 700u64), (1, 700), (2, 200), (3, 950)] {
                table.request(key(1, 0, i), prio, 400, u64::from(i));
                if i % 2 == 0 {
                    table.mark_resident(key(1, 0, i));
                }
            }
            table
        };
        let a = schedule(&build(), 1000);
        let b = schedule(&build(), 1000);
        assert_eq!(a, b);
    }

    #[test]
    fn resident_pages_never_reload() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 900, 1000, 1);
        table.mark_resident(key(0, 0, 0));
        table.request(key(0, 0, 1), 500, 1000, 1);
        let plan = schedule(&table, 10_000);
        // Only the requested (non-resident) page is a load; the resident one is kept.
        assert_eq!(plan.loads, alloc::vec![key(0, 0, 1)]);
        assert!(plan.evicts.is_empty());
        assert_eq!(plan.resident_bytes, 2000);
    }

    #[test]
    fn schedule_and_apply_advances_state() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 900, 1000, 1);
        table.request(key(0, 0, 1), 300, 1000, 1);
        table.mark_resident(key(0, 0, 1));
        // Budget for one page: load the 900, evict the resident 300.
        let plan = schedule_and_apply(&mut table, 1000);
        assert_eq!(plan.loads, alloc::vec![key(0, 0, 0)]);
        assert_eq!(plan.evicts, alloc::vec![key(0, 0, 1)]);
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::Resident);
        assert_eq!(table.residency(key(0, 0, 1)), PageResidency::NotResident);
        assert_eq!(table.resident_bytes(), 1000);
    }

    #[test]
    fn evicted_page_leaves_live_set_until_requested_again() {
        let mut table = TextureResidencyTable::new();
        table.request(key(0, 0, 0), 300, 1000, 1);
        table.mark_resident(key(0, 0, 0));
        let _ = schedule_and_apply(&mut table, 0);
        assert_eq!(table.residency(key(0, 0, 0)), PageResidency::NotResident);
        // With nothing live, scheduling is a no-op even under a huge budget.
        assert!(schedule(&table, 10_000).is_noop());
    }
}
