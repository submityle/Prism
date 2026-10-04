//! GPU draw-batch efficiency census (design §15, §16.6).
//!
//! A GPU-driven renderer compacts the visible instance population into one
//! [`GpuBatchPlan`](crate::gpu_batch::GpuBatchPlan): a flat instance-index
//! buffer plus one [`DrawBatch`](crate::gpu_batch::DrawBatch) per batch key,
//! each becoming a single indirect draw (design §15, the Horizon / Insomniac
//! shape). The *whole point* of that compaction is to collapse many instances
//! into few indirect draws, so the plan's quality is measured by one ratio:
//! **how many visible instances does each indirect draw amortise?**
//!
//! This module turns a built plan into a read-only efficiency report for the
//! inspector / performance panel (design §16.6). It does not look at the GPU or
//! issue any draws — it only accounts the CPU-side plan the renderer will
//! submit — so it is safe to run from a diagnostics system every frame.
//!
//! # What it surfaces
//!
//! * **draw-call pressure** — [`batch_count`](GpuBatchEfficiencyReport::batch_count)
//!   is exactly the indirect-draw count; fewer draws for the same visible
//!   population is strictly better.
//! * **batch fill** — [`mean_instances_per_batch_permille`](GpuBatchEfficiencyReport::mean_instances_per_batch_permille)
//!   is the average instances-per-draw (the amortisation factor). A value near
//!   `1000` (= 1.0 instance/draw) means the plan is barely batching at all.
//! * **singleton waste** — [`singleton_batch_count`](GpuBatchEfficiencyReport::singleton_batch_count)
//!   counts batches carrying a *single* instance. Each is a full indirect draw
//!   that amortises nothing; a high
//!   [`singleton_permille`](GpuBatchEfficiencyReport::singleton_permille) is the
//!   classic "too many unique mesh+material keys" smell (design §6 批次键).
//! * **spread** — [`largest_batch`](GpuBatchEfficiencyReport::largest_batch) /
//!   [`smallest_batch`](GpuBatchEfficiencyReport::smallest_batch) bound the
//!   per-draw instance distribution.
//!
//! # Determinism (design §14)
//!
//! The report preserves the plan's batch order (ascending by
//! [`SharedValueId`]) and all tie-breaks resolve to the lowest batch key, so a
//! given plan yields a byte-identical report — matching the deterministic
//! batching contract of [`crate::gpu_batch`].

use alloc::vec::Vec;

use crate::gpu_batch::GpuBatchPlan;
use crate::storage::SharedValueId;

/// One batch's contribution to the draw-call budget: its key and how many
/// visible instances the single indirect draw for that key carries.
///
/// `instance_count` is the number of visible instances compacted under
/// `batch_key`; a value of `1` makes this a *singleton* draw that amortises
/// nothing (see [`is_singleton`](BatchEntry::is_singleton)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BatchEntry {
    /// The render batch key (SharedComponent value id) this draw groups.
    pub batch_key: SharedValueId,
    /// Visible instances compacted into this one indirect draw.
    pub instance_count: u32,
}

impl BatchEntry {
    /// Whether this batch carries exactly one instance — a full indirect draw
    /// that amortises no per-draw overhead across siblings.
    #[inline]
    pub const fn is_singleton(&self) -> bool {
        self.instance_count == 1
    }

    /// Whether this batch carries no instances. A well-formed
    /// [`GpuBatchPlan`] never emits these; the predicate exists so a report
    /// built from a hand-constructed plan stays defensible.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.instance_count == 0
    }
}

/// Read-only efficiency census of one [`GpuBatchPlan`] (design §15 / §16.6).
///
/// Built with [`from_plan`](Self::from_plan); every accessor is O(1) except the
/// key lookups ([`batch`](Self::batch) / [`contains`](Self::contains) /
/// [`instance_count_of`](Self::instance_count_of)), which binary-search the
/// key-ascending [`entries`](Self::entries), and
/// [`batches_with_at_least`](Self::batches_with_at_least), which scans.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GpuBatchEfficiencyReport {
    /// Per-batch entries, ascending by `batch_key` (plan order preserved).
    entries: Vec<BatchEntry>,
    /// Total visible instances across all batches (= plan index-buffer length).
    visible_instance_count: usize,
    /// Number of batches carrying exactly one instance.
    singleton_batch_count: usize,
    /// Index into `entries` of the fattest batch (tie -> lowest key).
    largest: Option<usize>,
    /// Index into `entries` of the smallest batch (tie -> lowest key).
    smallest: Option<usize>,
}

impl GpuBatchEfficiencyReport {
    /// Account `plan` into an efficiency report.
    ///
    /// The plan's batches are already ascending by [`SharedValueId`], so the
    /// report preserves that order. Both the largest- and smallest-batch
    /// tie-breaks keep the first (lowest-key) batch at the extreme size, so the
    /// result is deterministic for a given plan.
    pub fn from_plan(plan: &GpuBatchPlan) -> Self {
        let batches = plan.batches();
        let mut entries: Vec<BatchEntry> = Vec::with_capacity(batches.len());
        let mut visible_instance_count = 0usize;
        let mut singleton_batch_count = 0usize;
        let mut largest: Option<usize> = None;
        let mut smallest: Option<usize> = None;

        for (i, batch) in batches.iter().enumerate() {
            let count = batch.instance_count;
            visible_instance_count += count as usize;
            if count == 1 {
                singleton_batch_count += 1;
            }
            // Strictly-greater wins so an equal later (higher-key) batch never
            // displaces the first maximum; the symmetric rule holds the minimum
            // at the lowest key.
            match largest {
                Some(j) if entries[j].instance_count >= count => {}
                _ => largest = Some(i),
            }
            match smallest {
                Some(j) if entries[j].instance_count <= count => {}
                _ => smallest = Some(i),
            }
            entries.push(BatchEntry {
                batch_key: batch.batch_key,
                instance_count: count,
            });
        }

        Self {
            entries,
            visible_instance_count,
            singleton_batch_count,
            largest,
            smallest,
        }
    }

    /// The per-batch entries, ascending by batch key (plan order).
    #[inline]
    pub fn entries(&self) -> &[BatchEntry] {
        &self.entries
    }

    /// Number of distinct draw batches — exactly the indirect-draw count the
    /// renderer will submit for this plan.
    #[inline]
    pub fn batch_count(&self) -> usize {
        self.entries.len()
    }

    /// Alias for [`batch_count`](Self::batch_count), named for the render side:
    /// the number of indirect draw calls this plan issues.
    #[inline]
    pub fn draw_call_count(&self) -> usize {
        self.entries.len()
    }

    /// Total visible instances across every batch (= the plan's compacted
    /// instance-index buffer length).
    #[inline]
    pub fn visible_instance_count(&self) -> usize {
        self.visible_instance_count
    }

    /// Whether the plan draws nothing (no batches, no instances).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the entire visible population collapsed into one indirect draw —
    /// the ideal batching outcome for a single-key population.
    #[inline]
    pub fn is_single_draw(&self) -> bool {
        self.entries.len() == 1
    }

    /// Number of batches carrying exactly one instance. Each is a full indirect
    /// draw that amortises nothing, so this is the primary waste signal.
    #[inline]
    pub fn singleton_batch_count(&self) -> usize {
        self.singleton_batch_count
    }

    /// Visible instances that live in non-singleton (`instance_count >= 2`)
    /// batches — the share of the population that is actually being batched.
    ///
    /// Equals [`visible_instance_count`](Self::visible_instance_count) minus
    /// [`singleton_batch_count`](Self::singleton_batch_count) because each
    /// singleton contributes exactly one instance.
    #[inline]
    pub fn batched_instance_count(&self) -> usize {
        self.visible_instance_count - self.singleton_batch_count
    }

    /// The fattest batch (most instances), or `None` when the plan is empty.
    /// Ties resolve to the lowest batch key.
    #[inline]
    pub fn largest_batch(&self) -> Option<BatchEntry> {
        self.largest.map(|i| self.entries[i])
    }

    /// The smallest batch (fewest instances), or `None` when the plan is empty.
    /// Ties resolve to the lowest batch key.
    #[inline]
    pub fn smallest_batch(&self) -> Option<BatchEntry> {
        self.smallest.map(|i| self.entries[i])
    }

    /// Instance count of the fattest batch, or `0` for an empty plan.
    #[inline]
    pub fn max_instances_in_batch(&self) -> u32 {
        self.largest.map_or(0, |i| self.entries[i].instance_count)
    }

    /// Instance count of the smallest batch, or `0` for an empty plan.
    #[inline]
    pub fn min_instances_in_batch(&self) -> u32 {
        self.smallest.map_or(0, |i| self.entries[i].instance_count)
    }

    /// Average instances per indirect draw, scaled by 1000 (so `1000` = exactly
    /// one instance per draw, `4500` = 4.5). This is the amortisation factor:
    /// the higher it is, the more each draw call pays for itself. Returns `0`
    /// for an empty plan.
    ///
    /// Float is avoided for determinism (design §14); the per-mille form keeps
    /// one fractional digit of resolution with pure integer arithmetic.
    #[inline]
    pub fn mean_instances_per_batch_permille(&self) -> u64 {
        if self.entries.is_empty() {
            return 0;
        }
        self.visible_instance_count as u64 * 1000 / self.entries.len() as u64
    }

    /// Fraction of batches that are singletons, in per-mille (`1000` = every
    /// batch is a singleton). Returns `0` for an empty plan. A high value is the
    /// "too many unique mesh+material keys" smell (design §6 批次键去重).
    #[inline]
    pub fn singleton_permille(&self) -> u64 {
        if self.entries.is_empty() {
            return 0;
        }
        self.singleton_batch_count as u64 * 1000 / self.entries.len() as u64
    }

    /// Number of batches whose instance count is at least `n`. `batches_with_at_least(2)`
    /// is the count of genuinely-batched draws; `batches_with_at_least(1)`
    /// equals [`batch_count`](Self::batch_count).
    #[inline]
    pub fn batches_with_at_least(&self, n: u32) -> usize {
        self.entries
            .iter()
            .filter(|e| e.instance_count >= n)
            .count()
    }

    /// The batch for `key`, or `None` if the plan has no draw for it.
    #[inline]
    pub fn batch(&self, key: SharedValueId) -> Option<BatchEntry> {
        self.entries
            .binary_search_by(|e| e.batch_key.cmp(&key))
            .ok()
            .map(|i| self.entries[i])
    }

    /// Whether the plan draws batch `key`.
    #[inline]
    pub fn contains(&self, key: SharedValueId) -> bool {
        self.entries
            .binary_search_by(|e| e.batch_key.cmp(&key))
            .is_ok()
    }

    /// Visible instances drawn under `key`, or `0` if the plan has no draw for
    /// it.
    #[inline]
    pub fn instance_count_of(&self, key: SharedValueId) -> u32 {
        self.batch(key).map_or(0, |e| e.instance_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_batch::{build_draw_batches, InstanceVisibility};

    fn key(i: u32) -> SharedValueId {
        SharedValueId::new(i)
    }

    #[test]
    fn empty_plan_reports_nothing() {
        let plan = build_draw_batches(&[]);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert!(report.is_empty());
        assert!(!report.is_single_draw());
        assert_eq!(report.batch_count(), 0);
        assert_eq!(report.draw_call_count(), 0);
        assert_eq!(report.visible_instance_count(), 0);
        assert_eq!(report.singleton_batch_count(), 0);
        assert_eq!(report.batched_instance_count(), 0);
        assert_eq!(report.largest_batch(), None);
        assert_eq!(report.smallest_batch(), None);
        assert_eq!(report.max_instances_in_batch(), 0);
        assert_eq!(report.min_instances_in_batch(), 0);
        assert_eq!(report.mean_instances_per_batch_permille(), 0);
        assert_eq!(report.singleton_permille(), 0);
        assert_eq!(report.batches_with_at_least(1), 0);
        assert_eq!(report.batch(key(0)), None);
        assert!(!report.contains(key(0)));
        assert_eq!(report.instance_count_of(key(0)), 0);
    }

    #[test]
    fn mixed_population_accounts_fill_and_waste() {
        // key 0: 3 instances, key 1: 1 instance (singleton), key 2: 2 instances.
        let instances = [
            InstanceVisibility::visible(0, key(0)),
            InstanceVisibility::visible(1, key(0)),
            InstanceVisibility::visible(2, key(0)),
            InstanceVisibility::visible(3, key(1)),
            InstanceVisibility::visible(4, key(2)),
            InstanceVisibility::visible(5, key(2)),
            // A culled instance must not appear anywhere in the plan.
            InstanceVisibility::culled(6, key(2)),
        ];
        let plan = build_draw_batches(&instances);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert_eq!(report.batch_count(), 3);
        assert_eq!(report.visible_instance_count(), 6);
        assert_eq!(report.singleton_batch_count(), 1);
        assert_eq!(report.batched_instance_count(), 5);
        // entries ascending by key, counts preserved.
        assert_eq!(
            report.entries(),
            &[
                BatchEntry { batch_key: key(0), instance_count: 3 },
                BatchEntry { batch_key: key(1), instance_count: 1 },
                BatchEntry { batch_key: key(2), instance_count: 2 },
            ]
        );
        assert_eq!(report.largest_batch(), Some(report.entries()[0]));
        assert_eq!(report.max_instances_in_batch(), 3);
        assert_eq!(report.smallest_batch(), Some(report.entries()[1]));
        assert_eq!(report.min_instances_in_batch(), 1);
        // 6 instances / 3 draws = 2.0 => 2000 permille.
        assert_eq!(report.mean_instances_per_batch_permille(), 2000);
        // 1 of 3 batches is a singleton => 333 permille.
        assert_eq!(report.singleton_permille(), 333);
        assert_eq!(report.batches_with_at_least(2), 2);
        assert_eq!(report.batches_with_at_least(3), 1);
        assert_eq!(report.batches_with_at_least(4), 0);
    }

    #[test]
    fn lookup_hits_and_misses() {
        let instances = [
            InstanceVisibility::visible(0, key(10)),
            InstanceVisibility::visible(1, key(10)),
            InstanceVisibility::visible(2, key(40)),
        ];
        let plan = build_draw_batches(&instances);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert!(report.contains(key(10)));
        assert!(report.contains(key(40)));
        assert!(!report.contains(key(25)));
        assert_eq!(
            report.batch(key(10)),
            Some(BatchEntry { batch_key: key(10), instance_count: 2 })
        );
        assert_eq!(report.batch(key(25)), None);
        assert_eq!(report.instance_count_of(key(40)), 1);
        assert_eq!(report.instance_count_of(key(25)), 0);
    }

    #[test]
    fn single_key_population_is_one_draw() {
        let instances = [
            InstanceVisibility::visible(0, key(7)),
            InstanceVisibility::visible(1, key(7)),
            InstanceVisibility::visible(2, key(7)),
            InstanceVisibility::visible(3, key(7)),
        ];
        let plan = build_draw_batches(&instances);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert!(report.is_single_draw());
        assert_eq!(report.batch_count(), 1);
        assert_eq!(report.visible_instance_count(), 4);
        assert_eq!(report.singleton_batch_count(), 0);
        // 4 instances in a single draw => 4000 permille.
        assert_eq!(report.mean_instances_per_batch_permille(), 4000);
        assert_eq!(report.singleton_permille(), 0);
    }

    #[test]
    fn all_singletons_is_worst_case() {
        let instances = [
            InstanceVisibility::visible(0, key(1)),
            InstanceVisibility::visible(1, key(2)),
            InstanceVisibility::visible(2, key(3)),
        ];
        let plan = build_draw_batches(&instances);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert_eq!(report.batch_count(), 3);
        assert_eq!(report.singleton_batch_count(), 3);
        assert_eq!(report.batched_instance_count(), 0);
        // every batch is a singleton: fill = 1.0, singleton share = 1.0.
        assert_eq!(report.mean_instances_per_batch_permille(), 1000);
        assert_eq!(report.singleton_permille(), 1000);
        assert_eq!(report.batches_with_at_least(2), 0);
    }

    #[test]
    fn extreme_ties_resolve_to_lowest_key() {
        // keys 2 and 5 both have 2 instances (max tie); keys 3 and 8 both have
        // 1 instance (min tie). Lowest key must win each extreme.
        let instances = [
            InstanceVisibility::visible(0, key(2)),
            InstanceVisibility::visible(1, key(2)),
            InstanceVisibility::visible(2, key(3)),
            InstanceVisibility::visible(3, key(5)),
            InstanceVisibility::visible(4, key(5)),
            InstanceVisibility::visible(5, key(8)),
        ];
        let plan = build_draw_batches(&instances);
        let report = GpuBatchEfficiencyReport::from_plan(&plan);

        assert_eq!(report.max_instances_in_batch(), 2);
        assert_eq!(report.largest_batch().unwrap().batch_key, key(2));
        assert_eq!(report.min_instances_in_batch(), 1);
        assert_eq!(report.smallest_batch().unwrap().batch_key, key(3));
    }

    #[test]
    fn report_is_order_independent() {
        let forward = [
            InstanceVisibility::visible(0, key(1)),
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::visible(2, key(4)),
        ];
        let shuffled = [
            InstanceVisibility::visible(2, key(4)),
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::visible(0, key(1)),
        ];
        let a = GpuBatchEfficiencyReport::from_plan(&build_draw_batches(&forward));
        let b = GpuBatchEfficiencyReport::from_plan(&build_draw_batches(&shuffled));
        assert_eq!(a, b);
    }
}
