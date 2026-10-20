//! Stable-slot light tables and dirty-range upload planning.
//!
//! This module is deliberately independent of the lighting plugin and buffer
//! resources. A render-world owner can mirror extracted lights into one table
//! per ABI record type, request an [`UploadPlan`], issue its byte ranges, then
//! acknowledge the plan with its queue fence only after submission succeeds.
//!
//! Payloads are the existing GPU ABI records (for example
//! `GpuDirectionalLight` and `GpuPunctualLight`): `Pod` supplies their byte
//! representation and [`LightGpuPayload`] supplies an explicit neutral
//! tombstone. Equality is bytewise, including signed zeros and NaN payloads.
//!
//! Integration order (all mutations stay with the render-world owner): apply
//! entity-to-handle deltas; allocate/grow the GPU buffer if needed and call
//! [`IncrementalLightTable::invalidate_gpu_copy`]; obtain a plan; copy each
//! range via `payload_bytes` and a queue write; submit; acknowledge with a
//! monotonic queue serial; reclaim only from that queue's completed serial.
//! Do not interleave mutations with the writes themselves. Plans snapshot work,
//! not payload bytes; asynchronous staging must copy the bytes before mutation.
//! Failed/dropped plans consume no dirty data. A queue serial is not a frame
//! number: completion must cover all submissions that might reference a slot.
//!
//! The shader's array count is [`IncrementalLightTable::capacity`], not
//! [`IncrementalLightTable::live_len`], because neutral holes retain indices.
//! Clusters and other index lists must use these stable slots too. An empty
//! GPU binding still needs the caller's minimum-size neutral buffer. GPU
//! generations/active-index buffers are separate consumer contracts; this
//! module does not silently change the existing light ABI or add indirection.

use core::{mem::size_of, ops::Range};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Pod;

use super::abi::{GpuDirectionalLight, GpuLightEnvironment, GpuPunctualLight};

static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(1);

/// A GPU ABI record whose neutral value contributes no light.
///
/// `Pod` alone does not prove WGSL compatibility. Implementations must have
/// the shader's array stride and a nonzero size divisible by four, with no
/// hidden padding. Neutral values must also be safe for downstream culling.
pub trait LightGpuPayload: Pod {
    fn neutral() -> Self;
}

impl LightGpuPayload for GpuDirectionalLight {
    fn neutral() -> Self {
        Self::default()
    }
}

impl LightGpuPayload for GpuPunctualLight {
    fn neutral() -> Self {
        Self::default()
    }
}

impl LightGpuPayload for GpuLightEnvironment {
    fn neutral() -> Self {
        Self::default()
    }
}

/// A generational identity for one stable table slot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LightSlotHandle {
    table_id: u64,
    slot: u32,
    generation: u32,
}

impl LightSlotHandle {
    /// Zero-based stable slot index.
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Generation captured when this slot was allocated.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Upload lifecycle for a slot that has been removed from the live scene.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotRetirement {
    /// The neutral payload is dirty and must be uploaded before fencing.
    Retiring,
    /// The neutral payload was submitted; reuse waits for this queue fence.
    FencePending { fence: u64 },
}

/// A half-open range of table elements to upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirtyRange {
    pub elements: Range<u32>,
    pub byte_offset: u64,
    pub byte_len: u64,
}

/// The upload work snapshot for one table revision.
///
/// `full_rebuild` means the ranges cover the entire allocated table, including
/// neutral slots. `bytes` and `ranges` describe the exact transfer represented
/// by this plan. A plan is a snapshot: edits made after it is created remain
/// dirty when it is acknowledged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadPlan {
    table_id: u64,
    gpu_epoch: u64,
    element_count: u32,
    revision: u64,
    ranges: Vec<DirtyRange>,
    bytes: u64,
    full_rebuild: bool,
}

/// Exact work of a plan, not cumulative or estimated GPU timings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UploadStats {
    pub bytes: u64,
    pub ranges: usize,
    pub full_rebuild: bool,
}

impl UploadPlan {
    /// Immutable ranges; byte offsets and sizes match the ABI record stride.
    pub fn ranges(&self) -> &[DirtyRange] {
        &self.ranges
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub fn stats(&self) -> UploadStats {
        UploadStats {
            bytes: self.bytes,
            ranges: self.ranges.len(),
            full_rebuild: self.full_rebuild,
        }
    }

    /// Whether the plan contains no GPU writes.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

#[derive(Clone, Debug)]
enum SlotState {
    Vacant,
    Exhausted,
    Live,
    Retiring,
    FencePending { fence: u64 },
}

#[derive(Clone, Debug)]
struct SlotMeta {
    generation: u32,
    state: SlotState,
    dirty_revision: Option<u64>,
}

/// Policy controlling when dirty data is cheaper to upload as a full table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UploadPolicy {
    /// Use a full-table upload when dirty bytes reach this fraction of the
    /// allocated table. Must be finite and in `(0, 1]`.
    pub full_rebuild_dirty_fraction: f32,
    /// Use a full-table upload when the merged range count reaches this value.
    pub full_rebuild_range_count: usize,
}

impl Default for UploadPolicy {
    fn default() -> Self {
        Self {
            full_rebuild_dirty_fraction: 0.6,
            full_rebuild_range_count: 64,
        }
    }
}

/// An incrementally maintained GPU table with stable indices and safe reuse.
///
/// `T` should be one of Prism's existing `#[repr(C)]` GPU ABI records. The
/// table never packs/reorders live entries, so previously uploaded indices
/// remain valid. Vacant slots are reused only after their retirement fence has
/// completed.
#[derive(Debug)]
pub struct IncrementalLightTable<T> {
    values: Vec<T>,
    slots: Vec<SlotMeta>,
    free_slots: Vec<usize>,
    dirty_slots: BTreeSet<usize>,
    fence_pending: BTreeMap<u64, Vec<usize>>,
    live_count: usize,
    table_id: u64,
    gpu_epoch: u64,
    revision: u64,
    acknowledged_revision: u64,
    last_submitted_fence: u64,
    completed_fence: u64,
    policy: UploadPolicy,
    force_full_rebuild: bool,
}

impl<T> Default for IncrementalLightTable<T>
where
    T: LightGpuPayload,
{
    fn default() -> Self {
        Self::new(UploadPolicy::default())
    }
}

impl<T> IncrementalLightTable<T>
where
    T: LightGpuPayload,
{
    /// Creates an empty table with the supplied upload policy.
    pub fn new(policy: UploadPolicy) -> Self {
        assert!(policy.full_rebuild_dirty_fraction.is_finite());
        assert!(policy.full_rebuild_dirty_fraction > 0.0);
        assert!(policy.full_rebuild_dirty_fraction <= 1.0);
        assert!(policy.full_rebuild_range_count > 0);
        assert!(size_of::<T>() > 0 && size_of::<T>() % 4 == 0);
        Self {
            values: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            dirty_slots: BTreeSet::new(),
            fence_pending: BTreeMap::new(),
            live_count: 0,
            table_id: NEXT_TABLE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("light table identity exhausted"),
            gpu_epoch: 0,
            revision: 0,
            acknowledged_revision: 0,
            last_submitted_fence: 0,
            completed_fence: 0,
            policy,
            force_full_rebuild: false,
        }
    }

    /// Number of allocated slots, including retired and vacant slots.
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of live entries.
    pub fn live_len(&self) -> usize {
        self.live_count
    }

    /// Current payload for a valid live handle.
    pub fn get(&self, handle: LightSlotHandle) -> Option<&T> {
        if handle.table_id != self.table_id {
            return None;
        }
        let index = handle.slot as usize;
        let slot = self.slots.get(index)?;
        (slot.generation == handle.generation && matches!(slot.state, SlotState::Live))
            .then_some(&self.values[index])
    }

    /// Allocates a stable slot, reusing only fence-safe vacant slots.
    pub fn insert(&mut self, value: T) -> LightSlotHandle {
        if let Some(index) = self.free_slots.pop() {
            let generation = self.slots[index].generation;
            let revision = self.next_revision();
            self.values[index] = value;
            let slot = &mut self.slots[index];
            slot.state = SlotState::Live;
            slot.dirty_revision = Some(revision);
            self.dirty_slots.insert(index);
            self.live_count += 1;
            return LightSlotHandle {
                table_id: self.table_id,
                slot: index as u32,
                generation,
            };
        }

        let index = self.slots.len();
        assert!(
            index < u32::MAX as usize,
            "light table exceeded u32 slot space"
        );
        let revision = self.next_revision();
        self.values.push(value);
        self.slots.push(SlotMeta {
            generation: 1,
            state: SlotState::Live,
            dirty_revision: Some(revision),
        });
        self.dirty_slots.insert(index);
        self.live_count += 1;
        LightSlotHandle {
            table_id: self.table_id,
            slot: index as u32,
            generation: 1,
        }
    }

    /// Replaces a live payload and dirties its slot only if its ABI bytes change.
    pub fn update(&mut self, handle: LightSlotHandle, value: T) -> bool {
        if handle.table_id != self.table_id {
            return false;
        }
        let index = handle.slot as usize;
        let Some(slot) = self.slots.get(index) else {
            return false;
        };
        if slot.generation != handle.generation || !matches!(slot.state, SlotState::Live) {
            return false;
        }
        if bytemuck::bytes_of(&self.values[index]) == bytemuck::bytes_of(&value) {
            return true;
        }
        let revision = self.next_revision();
        self.values[index] = value;
        self.slots[index].dirty_revision = Some(revision);
        self.dirty_slots.insert(index);
        true
    }

    /// Removes a live entry and marks its slot for neutral-payload upload.
    ///
    /// The slot cannot be reused until [`Self::acknowledge_upload`] fences a
    /// successful upload of the retirement and [`Self::reclaim_completed`]
    /// observes that fence as complete. This holds even for never-uploaded
    /// entries, avoiding a second fast-path lifetime model.
    pub fn retire(&mut self, handle: LightSlotHandle) -> bool {
        if handle.table_id != self.table_id {
            return false;
        }
        let index = handle.slot as usize;
        let Some(slot) = self.slots.get(index) else {
            return false;
        };
        if slot.generation != handle.generation || !matches!(slot.state, SlotState::Live) {
            return false;
        }
        let revision = self.next_revision();
        self.values[index] = T::neutral();
        self.slots[index].state = SlotState::Retiring;
        self.slots[index].dirty_revision = Some(revision);
        self.dirty_slots.insert(index);
        self.live_count -= 1;
        true
    }

    /// Returns a removed slot's current retirement state, if the handle still
    /// identifies that generation.
    pub fn retirement(&self, handle: LightSlotHandle) -> Option<SlotRetirement> {
        if handle.table_id != self.table_id {
            return None;
        }
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        match slot.state {
            SlotState::Retiring => Some(SlotRetirement::Retiring),
            SlotState::FencePending { fence } => Some(SlotRetirement::FencePending { fence }),
            SlotState::Vacant | SlotState::Exhausted | SlotState::Live => None,
        }
    }

    /// Plans merged element-aligned writes without consuming dirty state.
    pub fn plan_upload(&self) -> UploadPlan {
        let element_size = core::mem::size_of::<T>() as u64;
        let mut ranges = Vec::<Range<u32>>::new();
        for &index in &self.dirty_slots {
            let start = index as u32;
            let end = start + 1;
            if let Some(last) = ranges.last_mut() {
                if last.end == start {
                    last.end = end;
                    continue;
                }
            }
            ranges.push(start..end);
        }

        let dirty_elements: usize = ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .sum();
        let allocated_elements = self.slots.len();
        let dirty_fraction = if allocated_elements == 0 {
            0.0
        } else {
            dirty_elements as f32 / allocated_elements as f32
        };
        let full_rebuild = self.force_full_rebuild
            || (!ranges.is_empty()
                && (dirty_fraction >= self.policy.full_rebuild_dirty_fraction
                    || ranges.len() >= self.policy.full_rebuild_range_count.max(1)));
        if full_rebuild {
            ranges.clear();
            if allocated_elements > 0 {
                ranges.push(0..allocated_elements as u32);
            }
        }

        let ranges: Vec<_> = ranges
            .into_iter()
            .map(|elements| DirtyRange {
                byte_offset: u64::from(elements.start) * element_size,
                byte_len: u64::from(elements.end - elements.start) * element_size,
                elements,
            })
            .collect();
        let bytes = ranges.iter().map(|range| range.byte_len).sum();
        UploadPlan {
            table_id: self.table_id,
            gpu_epoch: self.gpu_epoch,
            element_count: self.capacity(),
            revision: self.revision,
            ranges,
            bytes,
            full_rebuild,
        }
    }

    /// Marks a successfully submitted plan as uploaded.
    ///
    /// Dirty slots changed after `plan` was created are preserved. Returns
    /// false for a different table/buffer epoch, out-of-order plan, or invalid
    /// fence, leaving all state unchanged. `fence` is a monotonic, nonzero
    /// submission serial on one queue timeline, not the completed serial.
    /// Dirty retirements covered by this plan move atomically to FencePending.
    pub fn acknowledge_upload(&mut self, plan: &UploadPlan, fence: u64) -> bool {
        if plan.table_id != self.table_id
            || plan.gpu_epoch != self.gpu_epoch
            || plan.revision > self.revision
            || plan.revision < self.acknowledged_revision
            || fence <= self.completed_fence
            || fence < self.last_submitted_fence
        {
            return false;
        }
        // Plans are immutable to callers. Validate their internal layout too,
        // so an accidental future producer cannot acknowledge uncovered bytes.
        let stride = size_of::<T>() as u64;
        let mut previous_end = 0;
        let mut bytes = 0;
        for range in &plan.ranges {
            if range.elements.start < previous_end
                || range.elements.start >= range.elements.end
                || range.elements.end > plan.element_count
                || range.byte_offset != u64::from(range.elements.start) * stride
                || range.byte_len != u64::from(range.elements.end - range.elements.start) * stride
            {
                return false;
            }
            previous_end = range.elements.end;
            bytes += range.byte_len;
        }
        if bytes != plan.bytes
            || plan.element_count > self.capacity()
            || (plan.full_rebuild
                && ((plan.element_count == 0 && !plan.ranges.is_empty())
                    || (plan.element_count > 0
                        && (plan.ranges.len() != 1
                            || plan.ranges[0].elements != (0..plan.element_count)))))
        {
            return false;
        }
        if self.dirty_slots.iter().any(|&index| {
            self.slots[index]
                .dirty_revision
                .is_some_and(|dirty| dirty <= plan.revision)
                && !range_contains(&plan.ranges, index as u32)
        }) {
            return false;
        }

        let slots = &mut self.slots;
        let pending = &mut self.fence_pending;
        self.dirty_slots.retain(|&index| {
            let slot = &mut slots[index];
            if slot
                .dirty_revision
                .is_some_and(|dirty| dirty <= plan.revision)
            {
                slot.dirty_revision = None;
                if matches!(slot.state, SlotState::Retiring) {
                    slot.state = SlotState::FencePending { fence };
                    pending.entry(fence).or_default().push(index);
                }
                false
            } else {
                true
            }
        });
        if plan.full_rebuild {
            self.force_full_rebuild = false;
        }
        self.last_submitted_fence = fence;
        self.acknowledged_revision = plan.revision;
        true
    }

    /// Reclaims slots whose queue fence has completed and advances generation.
    pub fn reclaim_completed(&mut self, completed_fence: u64) -> usize {
        self.completed_fence = self.completed_fence.max(completed_fence);
        let mut reclaimed = 0;
        while self
            .fence_pending
            .first_key_value()
            .is_some_and(|(&fence, _)| fence <= self.completed_fence)
        {
            let (_, indices) = self
                .fence_pending
                .pop_first()
                .expect("pending fence exists");
            for index in indices {
                let slot = &mut self.slots[index];
                if let Some(generation) = slot.generation.checked_add(1) {
                    slot.generation = generation;
                    slot.state = SlotState::Vacant;
                    self.free_slots.push(index);
                } else {
                    // Never wrap a generation back onto an ancient handle.
                    slot.state = SlotState::Exhausted;
                }
                reclaimed += 1;
            }
        }
        reclaimed
    }

    /// Payload slice in stable slot order, including neutral retired slots.
    /// The GPU table must be uploaded with this same indexing scheme.
    pub fn payloads(&self) -> &[T] {
        &self.values
    }

    /// Returns a stable payload range for one planned upload.
    pub fn payload_range(&self, elements: Range<u32>) -> Option<&[T]> {
        let start = elements.start as usize;
        let end = elements.end as usize;
        (start <= end && end <= self.values.len()).then(|| &self.values[start..end])
    }

    /// Returns the exact byte view to pass to a queue write operation.
    pub fn payload_bytes(&self, elements: Range<u32>) -> Option<&[u8]> {
        self.payload_range(elements)
            .map(bytemuck::cast_slice::<T, u8>)
    }

    /// Forces the next upload to repopulate a newly created or
    /// otherwise invalidated GPU buffer from the complete CPU table.
    pub fn invalidate_gpu_copy(&mut self) {
        let epoch = self.gpu_epoch.checked_add(1).expect("GPU epoch exhausted");
        self.next_revision();
        self.gpu_epoch = epoch;
        self.force_full_rebuild = true;
    }

    fn next_revision(&mut self) -> u64 {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("light table revision exhausted");
        self.revision
    }
}

fn range_contains(ranges: &[DirtyRange], index: u32) -> bool {
    let insertion = ranges.partition_point(|range| range.elements.end <= index);
    ranges
        .get(insertion)
        .is_some_and(|range| range.elements.start <= index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::abi::{GpuDirectionalLight, GpuPunctualLight};

    fn policy() -> UploadPolicy {
        UploadPolicy {
            full_rebuild_dirty_fraction: 1.0,
            full_rebuild_range_count: usize::MAX,
        }
    }

    fn punctual(x: f32) -> GpuPunctualLight {
        GpuPunctualLight {
            position: [x, 0.0, 0.0],
            ..Default::default()
        }
    }

    #[test]
    fn inserts_keep_stable_slots_and_updates_coalesce_adjacent_ranges() {
        let mut table = IncrementalLightTable::new(policy());
        let first = table.insert(punctual(1.0));
        let second = table.insert(punctual(2.0));
        let third = table.insert(punctual(3.0));
        let plan = table.plan_upload();

        assert_eq!((first.slot(), second.slot(), third.slot()), (0, 1, 2));
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].elements, 0..3);
        assert_eq!(
            plan.bytes,
            3 * core::mem::size_of::<GpuPunctualLight>() as u64
        );
        assert!(table.acknowledge_upload(&plan, 1));

        assert!(table.update(first, punctual(10.0)));
        assert!(table.update(second, punctual(20.0)));
        let plan = table.plan_upload();
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].elements, 0..2);
    }

    #[test]
    fn sparse_dirty_slots_form_distinct_byte_ranges() {
        let mut table = IncrementalLightTable::new(policy());
        let handles: Vec<_> = (0..5).map(|i| table.insert(punctual(i as f32))).collect();
        assert!(table.acknowledge_upload(&table.plan_upload(), 1));
        assert!(table.update(handles[1], punctual(11.0)));
        assert!(table.update(handles[4], punctual(44.0)));

        let plan = table.plan_upload();
        let stride = core::mem::size_of::<GpuPunctualLight>() as u64;
        assert_eq!(plan.ranges.len(), 2);
        assert_eq!(plan.ranges[0].byte_offset, stride);
        assert_eq!(plan.ranges[0].byte_len, stride);
        assert_eq!(plan.ranges[1].byte_offset, 4 * stride);
        assert_eq!(plan.bytes, 2 * stride);
        assert!(!plan.full_rebuild);
    }

    #[test]
    fn unchanged_updates_do_not_dirty_the_table() {
        let mut table = IncrementalLightTable::new(policy());
        let handle = table.insert(punctual(1.0));
        let plan = table.plan_upload();
        assert!(table.acknowledge_upload(&plan, 1));
        assert!(table.update(handle, punctual(1.0)));
        assert!(table.plan_upload().is_empty());
    }

    #[test]
    fn retirement_upload_fence_and_generation_prevent_early_reuse() {
        let mut table = IncrementalLightTable::new(policy());
        let old = table.insert(punctual(1.0));
        assert!(table.acknowledge_upload(&table.plan_upload(), 1));
        assert!(table.retire(old));
        assert_eq!(table.retirement(old), Some(SlotRetirement::Retiring));
        assert_eq!(table.get(old), None);

        let retirement_upload = table.plan_upload();
        assert_eq!(retirement_upload.ranges[0].elements, 0..1);
        assert!(table.acknowledge_upload(&retirement_upload, 12));
        assert_eq!(
            table.retirement(old),
            Some(SlotRetirement::FencePending { fence: 12 })
        );

        let other = table.insert(punctual(2.0));
        assert_eq!(other.slot(), 1, "pending-fence slots must not be reused");
        assert_eq!(table.reclaim_completed(11), 0);
        assert_eq!(table.reclaim_completed(12), 1);
        let replacement = table.insert(punctual(3.0));
        assert_eq!(replacement.slot(), old.slot());
        assert_ne!(replacement.generation(), old.generation());
        assert_eq!(table.get(old), None, "stale generations must not alias");
    }

    #[test]
    fn edits_after_a_plan_survive_acknowledgement_of_older_snapshot() {
        let mut table = IncrementalLightTable::new(policy());
        let handle = table.insert(punctual(1.0));
        let old_plan = table.plan_upload();
        assert!(table.update(handle, punctual(2.0)));
        assert!(table.acknowledge_upload(&old_plan, 1));

        let next = table.plan_upload();
        assert_eq!(next.ranges[0].elements, 0..1);
        assert_eq!(next.bytes, core::mem::size_of::<GpuPunctualLight>() as u64);
    }

    #[test]
    fn configured_threshold_selects_full_rebuild_and_reports_full_bytes() {
        let mut table = IncrementalLightTable::new(UploadPolicy {
            full_rebuild_dirty_fraction: 0.5,
            full_rebuild_range_count: usize::MAX,
        });
        let handles: Vec<_> = (0..4).map(|i| table.insert(punctual(i as f32))).collect();
        assert!(table.acknowledge_upload(&table.plan_upload(), 1));
        assert!(table.update(handles[0], punctual(10.0)));
        assert!(table.update(handles[2], punctual(30.0)));

        let plan = table.plan_upload();
        assert!(plan.full_rebuild);
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].elements, 0..4);
        assert_eq!(
            plan.bytes,
            4 * core::mem::size_of::<GpuPunctualLight>() as u64
        );
    }

    #[test]
    fn invalid_plan_cannot_clear_uncovered_dirty_slots() {
        let mut table = IncrementalLightTable::new(policy());
        let first = table.insert(punctual(1.0));
        let second = table.insert(punctual(2.0));
        let plan = table.plan_upload();
        assert!(table.acknowledge_upload(&plan, 1));
        assert!(table.update(first, punctual(11.0)));
        assert!(table.update(second, punctual(22.0)));

        let forged = UploadPlan {
            table_id: plan.table_id,
            gpu_epoch: plan.gpu_epoch,
            element_count: 2,
            revision: table.revision,
            ranges: vec![DirtyRange {
                elements: 0..1,
                byte_offset: 0,
                byte_len: core::mem::size_of::<GpuPunctualLight>() as u64,
            }],
            bytes: core::mem::size_of::<GpuPunctualLight>() as u64,
            full_rebuild: false,
        };
        assert!(!table.acknowledge_upload(&forged, 2));
        assert_eq!(table.plan_upload().ranges.len(), 1);
        assert_eq!(table.plan_upload().ranges[0].elements, 0..2);
    }

    #[test]
    fn invalidating_a_gpu_copy_requests_a_full_table_upload() {
        let mut table = IncrementalLightTable::new(policy());
        table.insert(punctual(1.0));
        let plan = table.plan_upload();
        assert!(table.acknowledge_upload(&plan, 1));
        table.invalidate_gpu_copy();

        let plan = table.plan_upload();
        assert!(plan.full_rebuild);
        assert_eq!(plan.ranges[0].elements, 0..1);
        assert!(table.acknowledge_upload(&plan, 2));
        assert!(table.plan_upload().is_empty());
    }

    #[test]
    fn payload_ranges_are_directly_usable_as_queue_write_bytes() {
        let mut table = IncrementalLightTable::new(policy());
        table.insert(punctual(1.0));
        table.insert(punctual(2.0));
        let range = 1..2;
        let bytes = table.payload_bytes(range.clone()).unwrap();
        assert_eq!(bytes.len(), core::mem::size_of::<GpuPunctualLight>());
        assert_eq!(bytes.as_ptr(), table.payloads()[1..2].as_ptr() as *const u8);
    }

    #[test]
    fn supports_existing_directional_abi_payload_without_a_second_encoder() {
        let mut table = IncrementalLightTable::<GpuDirectionalLight>::new(policy());
        let handle = table.insert(GpuDirectionalLight {
            illuminance: [3.0, 2.0, 1.0],
            ..Default::default()
        });
        assert_eq!(table.get(handle).unwrap().illuminance, [3.0, 2.0, 1.0]);
        assert_eq!(
            table.plan_upload().bytes,
            core::mem::size_of::<GpuDirectionalLight>() as u64
        );
    }
}
