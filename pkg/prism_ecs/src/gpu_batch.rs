//! GPU-driven draw-batch compaction (design §15, bridging §6 SharedComponent
//! batch keys and §13 cell visibility bits).
//!
//! [`GpuResidentColumn`](crate::gpu_resident::GpuResidentColumn) keeps the
//! instance data persistently mapped on the GPU, but a GPU-driven renderer also
//! needs to know *which* of those instances to draw this frame and *how to
//! group them into indirect draws*. That grouping is a pure CPU-side
//! bookkeeping step (the actual `multi_draw_indirect` dispatch lives in the
//! render crates), and it is exactly what this module computes.
//!
//! The inputs are, per visible instance:
//!
//! * its **slot** — the row of the instance inside its GPU-resident column
//!   (i.e. an index into the persistently-mapped instance buffer);
//! * its **batch key** — a [`SharedValueId`] interned from the entity's
//!   render batch SharedComponent (mesh + material + pipeline, design §6);
//! * a **visibility** flag — the result of culling against the active view /
//!   cell visible bits (design §13).
//!
//! Instances sharing a batch key are *not* contiguous in the column, so a
//! GPU-driven pipeline compacts the visible set into a single flat
//! **instance-index buffer** grouped by key, plus one [`DrawBatch`] per key
//! describing the `(first_instance, instance_count)` window into that buffer.
//! The GPU then issues one indirect draw per [`DrawBatch`], each reading its
//! slice of the compacted index buffer. This is the classic Horizon /
//! Insomniac GPU-driven batching shape (design §15).
//!
//! # Determinism (design §14)
//!
//! The output is sorted by `(batch_key, slot)`, so a given visible population
//! yields a byte-identical plan regardless of the order the samples were
//! supplied in — a prerequisite for the deterministic simulation / replay path.
//!
//! # Allocation reuse
//!
//! [`GpuBatchBuilder`] owns its scratch and output buffers and clears (rather
//! than frees) them on every [`build`](GpuBatchBuilder::build), so steady-state
//! frames allocate nothing once the high-water mark is reached. One-shot
//! callers can use the free [`build_draw_batches`] helper instead.

use alloc::vec::Vec;

use crate::storage::SharedValueId;

/// One GPU-resident instance's per-frame visibility and batch classification.
///
/// `slot` indexes the instance's row in its
/// [`GpuResidentColumn`](crate::gpu_resident::GpuResidentColumn); `batch_key`
/// groups instances that can share one indirect draw (design §6); `visible`
/// is the culling result (design §13). Culled instances are dropped from the
/// plan entirely.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InstanceVisibility {
    /// Row of this instance within its GPU-resident column.
    pub slot: u32,
    /// Render batch key this instance belongs to (SharedComponent value id).
    pub batch_key: SharedValueId,
    /// Whether this instance survived culling this frame.
    pub visible: bool,
}

impl InstanceVisibility {
    /// A visible instance at `slot` in batch `batch_key`.
    #[inline]
    pub const fn visible(slot: u32, batch_key: SharedValueId) -> Self {
        Self {
            slot,
            batch_key,
            visible: true,
        }
    }

    /// A culled instance (dropped from the plan) at `slot` in batch `batch_key`.
    #[inline]
    pub const fn culled(slot: u32, batch_key: SharedValueId) -> Self {
        Self {
            slot,
            batch_key,
            visible: false,
        }
    }
}

/// One contiguous run of the compacted instance-index buffer that shares a
/// batch key — the CPU-side source of exactly one GPU indirect draw
/// (design §15).
///
/// `first_instance` / `instance_count` are the window into
/// [`GpuBatchPlan::instance_indices`] for this batch; feed them straight into
/// the indirect draw's `first_instance` / `instance_count` arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DrawBatch {
    /// The render batch key shared by every instance in this batch.
    pub batch_key: SharedValueId,
    /// Offset of this batch's first entry within
    /// [`GpuBatchPlan::instance_indices`].
    pub first_instance: u32,
    /// Number of visible instances in this batch.
    pub instance_count: u32,
}

/// The compacted GPU-driven draw plan for one frame: a list of per-key
/// [`DrawBatch`]es plus the flat instance-index buffer they reference
/// (design §15).
///
/// `batches` is ascending by [`SharedValueId`]; `instance_indices` holds the
/// visible instance slots grouped in that same batch order, each run ascending
/// by slot. Together they describe a `multi_draw_indirect` submission over one
/// shared instance-index buffer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GpuBatchPlan {
    batches: Vec<DrawBatch>,
    instance_indices: Vec<u32>,
}

impl GpuBatchPlan {
    /// The per-batch draw descriptors, ascending by batch key.
    #[inline]
    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// The flat compacted instance-index buffer, grouped by batch.
    #[inline]
    pub fn instance_indices(&self) -> &[u32] {
        &self.instance_indices
    }

    /// Number of distinct draw batches (= indirect draw count).
    #[inline]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Total number of visible instances across all batches.
    #[inline]
    pub fn visible_instance_count(&self) -> usize {
        self.instance_indices.len()
    }

    /// Whether nothing is visible this frame (no batches, no instances).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// The slice of [`instance_indices`](Self::instance_indices) belonging to
    /// `batch`.
    ///
    /// # Panics
    /// Panics if `batch`'s window falls outside this plan's index buffer, which
    /// only happens if `batch` did not come from this plan.
    #[inline]
    pub fn indices_for(&self, batch: &DrawBatch) -> &[u32] {
        let start = batch.first_instance as usize;
        let end = start + batch.instance_count as usize;
        &self.instance_indices[start..end]
    }

    /// Clear the plan, keeping its buffer capacity for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.batches.clear();
        self.instance_indices.clear();
    }
}

/// Builds [`GpuBatchPlan`]s frame after frame while reusing its internal
/// buffers, so steady-state frames allocate nothing (design §15 / §17 内存回收).
///
/// Hold one builder per GPU-resident instance stream and call
/// [`build`](Self::build) each frame with the current visibility set.
#[derive(Clone, Debug, Default)]
pub struct GpuBatchBuilder {
    /// Scratch `(batch_key, slot)` pairs for the visible subset, reused.
    scratch: Vec<(SharedValueId, u32)>,
    plan: GpuBatchPlan,
}

impl GpuBatchBuilder {
    /// A builder with empty buffers.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Recompute the draw plan from `instances`, reusing internal buffers.
    ///
    /// Culled instances are dropped; the surviving set is grouped by batch key
    /// into contiguous runs of the compacted index buffer. Returns the freshly
    /// rebuilt [`GpuBatchPlan`].
    pub fn build(&mut self, instances: &[InstanceVisibility]) -> &GpuBatchPlan {
        self.scratch.clear();
        for inst in instances {
            if inst.visible {
                self.scratch.push((inst.batch_key, inst.slot));
            }
        }
        // Deterministic grouping: sorting by (key, slot) makes the plan a pure
        // function of the visible set, independent of input ordering (§14).
        self.scratch.sort_unstable();

        self.plan.clear();
        self.plan.instance_indices.reserve(self.scratch.len());

        let mut i = 0;
        while i < self.scratch.len() {
            let key = self.scratch[i].0;
            let first_instance = self.plan.instance_indices.len() as u32;
            let mut instance_count = 0u32;
            while i < self.scratch.len() && self.scratch[i].0 == key {
                self.plan.instance_indices.push(self.scratch[i].1);
                instance_count += 1;
                i += 1;
            }
            self.plan.batches.push(DrawBatch {
                batch_key: key,
                first_instance,
                instance_count,
            });
        }

        &self.plan
    }

    /// The most recently built plan (empty before the first
    /// [`build`](Self::build)).
    #[inline]
    pub fn plan(&self) -> &GpuBatchPlan {
        &self.plan
    }
}

/// One-shot [`GpuBatchPlan`] construction for callers that do not keep a
/// [`GpuBatchBuilder`] across frames.
///
/// Equivalent to `GpuBatchBuilder::new().build(instances).clone()`, but avoids
/// retaining the scratch buffer.
pub fn build_draw_batches(instances: &[InstanceVisibility]) -> GpuBatchPlan {
    let mut builder = GpuBatchBuilder::new();
    builder.build(instances);
    builder.plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u32) -> SharedValueId {
        SharedValueId::new(i)
    }

    #[test]
    fn empty_input_yields_empty_plan() {
        let plan = build_draw_batches(&[]);
        assert!(plan.is_empty());
        assert_eq!(plan.batch_count(), 0);
        assert_eq!(plan.visible_instance_count(), 0);
        assert!(plan.batches().is_empty());
        assert!(plan.instance_indices().is_empty());
    }

    #[test]
    fn all_culled_yields_empty_plan() {
        let plan = build_draw_batches(&[
            InstanceVisibility::culled(0, key(1)),
            InstanceVisibility::culled(5, key(2)),
        ]);
        assert!(plan.is_empty());
        assert_eq!(plan.visible_instance_count(), 0);
    }

    #[test]
    fn single_batch_groups_and_sorts_slots() {
        let plan = build_draw_batches(&[
            InstanceVisibility::visible(7, key(3)),
            InstanceVisibility::visible(2, key(3)),
            InstanceVisibility::visible(5, key(3)),
        ]);
        assert_eq!(plan.batch_count(), 1);
        let b = plan.batches()[0];
        assert_eq!(b.batch_key, key(3));
        assert_eq!(b.first_instance, 0);
        assert_eq!(b.instance_count, 3);
        // Slots within a batch are ascending.
        assert_eq!(plan.indices_for(&b), &[2, 5, 7]);
    }

    #[test]
    fn multiple_batches_are_ascending_and_contiguous() {
        let plan = build_draw_batches(&[
            InstanceVisibility::visible(10, key(2)),
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::visible(11, key(2)),
            InstanceVisibility::visible(4, key(1)),
            InstanceVisibility::visible(3, key(3)),
        ]);
        assert_eq!(plan.batch_count(), 3);

        let batches = plan.batches();
        // Ascending by key.
        assert_eq!(batches[0].batch_key, key(1));
        assert_eq!(batches[1].batch_key, key(2));
        assert_eq!(batches[2].batch_key, key(3));

        // Windows are contiguous and cover the whole index buffer in order.
        assert_eq!(batches[0].first_instance, 0);
        assert_eq!(batches[0].instance_count, 2);
        assert_eq!(batches[1].first_instance, 2);
        assert_eq!(batches[1].instance_count, 2);
        assert_eq!(batches[2].first_instance, 4);
        assert_eq!(batches[2].instance_count, 1);

        assert_eq!(plan.indices_for(&batches[0]), &[1, 4]);
        assert_eq!(plan.indices_for(&batches[1]), &[10, 11]);
        assert_eq!(plan.indices_for(&batches[2]), &[3]);

        assert_eq!(plan.visible_instance_count(), 5);
        assert_eq!(plan.instance_indices(), &[1, 4, 10, 11, 3]);
    }

    #[test]
    fn culled_instances_are_dropped_from_mixed_input() {
        let plan = build_draw_batches(&[
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::culled(2, key(1)),
            InstanceVisibility::visible(3, key(2)),
            InstanceVisibility::culled(4, key(2)),
        ]);
        assert_eq!(plan.batch_count(), 2);
        assert_eq!(plan.visible_instance_count(), 2);
        assert_eq!(plan.indices_for(&plan.batches()[0]), &[1]);
        assert_eq!(plan.indices_for(&plan.batches()[1]), &[3]);
    }

    #[test]
    fn plan_is_independent_of_input_order() {
        let forward = [
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::visible(2, key(1)),
            InstanceVisibility::visible(9, key(5)),
            InstanceVisibility::visible(4, key(2)),
        ];
        let mut reversed = forward;
        reversed.reverse();

        let a = build_draw_batches(&forward);
        let b = build_draw_batches(&reversed);
        assert_eq!(a, b);
    }

    #[test]
    fn builder_reuse_clears_previous_plan() {
        let mut builder = GpuBatchBuilder::new();

        builder.build(&[
            InstanceVisibility::visible(1, key(1)),
            InstanceVisibility::visible(2, key(1)),
        ]);
        assert_eq!(builder.plan().visible_instance_count(), 2);
        assert_eq!(builder.plan().batch_count(), 1);

        // A second build with a different, smaller set fully replaces the plan.
        builder.build(&[InstanceVisibility::visible(7, key(9))]);
        assert_eq!(builder.plan().visible_instance_count(), 1);
        assert_eq!(builder.plan().batch_count(), 1);
        assert_eq!(builder.plan().batches()[0].batch_key, key(9));
        assert_eq!(builder.plan().indices_for(&builder.plan().batches()[0]), &[7]);

        // Building from nothing empties it again.
        builder.build(&[]);
        assert!(builder.plan().is_empty());
    }

    #[test]
    fn duplicate_slots_in_a_batch_are_preserved() {
        // Two samples with the same (key, slot) are both kept — the builder does
        // not deduplicate, it only groups.
        let plan = build_draw_batches(&[
            InstanceVisibility::visible(5, key(1)),
            InstanceVisibility::visible(5, key(1)),
        ]);
        assert_eq!(plan.batch_count(), 1);
        assert_eq!(plan.indices_for(&plan.batches()[0]), &[5, 5]);
    }
}
