//! Stable-slot CPU contract for incrementally uploading GPU tables.
//!
//! The table owns payload bytes and slot lifetime, but deliberately knows
//! nothing about a renderer, ECS, queue, or GPU buffer. A consumer obtains an
//! [`UploadPlan`], writes each range from [`IncrementalGpuTable::payload_bytes`],
//! submits it, and only then acknowledges the plan with that queue's monotonic
//! submission proof. Reclamation is driven by a completion proof from that
//! same queue domain. Mutations may happen after planning; acknowledgement only consumes
//! dirty revisions covered by the immutable plan.
//!
//! Slots remain at stable indices. Removed values are replaced by a neutral
//! payload, uploaded, fenced, and only then made reusable with an incremented
//! generation. Consumers indexing this table must use `capacity()` rather
//! than `live_len()`; neutral holes are part of the GPU table layout.

use core::{mem::size_of, ops::Range};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Pod;

static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(1);

/// Identifies one monotonic completion-value domain.
///
/// Completion values are only ordered inside one `(device_generation,
/// queue_domain)` pair. A device recovery resets the runtime's numeric values,
/// so the device generation is part of the identity rather than metadata. The
/// queue domain distinguishes independent submission streams that may use
/// overlapping numeric values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuSubmissionDomain {
    device_generation: u64,
    queue_domain: u64,
}

impl GpuSubmissionDomain {
    /// Creates a domain for one device generation and logical queue stream.
    pub const fn new(device_generation: u64, queue_domain: u64) -> Self {
        Self {
            device_generation,
            queue_domain,
        }
    }

    /// Device generation that owns this queue domain.
    pub const fn device_generation(self) -> u64 {
        self.device_generation
    }

    /// Logical queue-domain identity.
    pub const fn queue_domain(self) -> u64 {
        self.queue_domain
    }

    /// Creates a proof for a successful nonzero submission value.
    pub const fn submission(self, value: u64) -> Option<SubmissionProof> {
        if value == 0 {
            None
        } else {
            Some(SubmissionProof {
                domain: self,
                value,
            })
        }
    }

    /// Creates a proof for a completion query. Zero is valid when no work has
    /// completed in this domain yet.
    pub const fn completion(self, value: u64) -> CompletionProof {
        CompletionProof {
            domain: self,
            value,
        }
    }
}

/// Proof that a submission was assigned a value in a particular queue domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SubmissionProof {
    domain: GpuSubmissionDomain,
    value: u64,
}

impl SubmissionProof {
    /// Queue domain that assigned this submission value.
    pub const fn domain(self) -> GpuSubmissionDomain {
        self.domain
    }

    /// Monotonic value assigned to this submission in its domain.
    pub const fn value(self) -> u64 {
        self.value
    }
}

/// Proof of the completion frontier reported for a queue domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CompletionProof {
    domain: GpuSubmissionDomain,
    value: u64,
}

impl CompletionProof {
    /// Queue domain that reported this completion value.
    pub const fn domain(self) -> GpuSubmissionDomain {
        self.domain
    }

    /// Reported completion frontier in the proof's domain.
    pub const fn value(self) -> u64 {
        self.value
    }
}

/// A plain-old-data GPU record with a safe value for unused or retired slots.
///
/// Implementations must ensure the neutral payload is valid for every GPU
/// consumer. `Pod` guarantees a byte-safe representation, not shader ABI
/// compatibility; the implementor remains responsible for matching shader
/// layout and stride.
pub trait GpuTablePayload: Pod {
    /// Returns the neutral payload used for vacant and retired slots.
    fn neutral() -> Self;
}

/// A generational identity for one stable slot in one table instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuTableHandle {
    table_id: u64,
    slot: u32,
    generation: u32,
}

impl GpuTableHandle {
    /// Zero-based stable slot index.
    pub const fn slot(self) -> u32 {
        self.slot
    }
    /// Generation captured when this slot was allocated.
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Upload lifecycle of a removed slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotRetirement {
    /// Its neutral payload still needs to be submitted.
    Retiring,
    /// Its neutral payload was submitted and reuse waits for this fence.
    FencePending { submission: SubmissionProof },
}

/// A half-open element range and its exact byte interval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirtyRange {
    /// Element indices in the stable table.
    pub elements: Range<u32>,
    /// Byte offset from the beginning of the GPU buffer.
    pub byte_offset: u64,
    /// Number of bytes to upload.
    pub byte_len: u64,
}

/// Immutable upload snapshot. It identifies work, not payload storage: async
/// staging users must copy payload bytes before allowing later mutations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadPlan {
    table_id: u64,
    gpu_epoch: u64,
    submission_domain: GpuSubmissionDomain,
    element_count: u32,
    revision: u64,
    ranges: Vec<DirtyRange>,
    bytes: u64,
    full_rebuild: bool,
}

/// Exact transfer work represented by an [`UploadPlan`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UploadStats {
    /// Total bytes represented by the plan.
    pub bytes: u64,
    /// Number of disjoint writes.
    pub ranges: usize,
    /// Whether this plan rebuilds the full allocated table.
    pub full_rebuild: bool,
}

impl UploadPlan {
    /// Element-aligned ranges represented by this plan.
    pub fn ranges(&self) -> &[DirtyRange] {
        &self.ranges
    }
    /// CPU table revision captured by this plan.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Device and queue domain in which this plan may be acknowledged.
    pub const fn submission_domain(&self) -> GpuSubmissionDomain {
        self.submission_domain
    }
    /// Exact transfer size and shape.
    pub fn stats(&self) -> UploadStats {
        UploadStats {
            bytes: self.bytes,
            ranges: self.ranges.len(),
            full_rebuild: self.full_rebuild,
        }
    }
    /// Whether the plan has no GPU writes.
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
    FencePending { submission: SubmissionProof },
}

#[derive(Clone, Debug)]
struct SlotMeta {
    generation: u32,
    state: SlotState,
    dirty_revision: Option<u64>,
}

/// Thresholds for switching from sparse writes to one full-table write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UploadPolicy {
    /// Rebuild when dirty elements reach this fraction of allocated capacity.
    /// Must be finite and in `(0, 1]`.
    pub full_rebuild_dirty_fraction: f32,
    /// Rebuild when merged range count reaches this value; must be nonzero.
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

/// Invalid caller-supplied policy or payload requirements for a GPU table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UploadPolicyError {
    /// Dirty fraction must be finite and in `(0, 1]`.
    InvalidDirtyFraction,
    /// Range count threshold must be greater than zero.
    ZeroRangeCount,
    /// Payload stride must be nonzero and four-byte aligned.
    InvalidPayloadStride,
}

/// Incrementally maintained, stable-index GPU table contract.
///
/// `T` is a GPU ABI record implementing [`GpuTablePayload`]. The table does
/// not pack live entries; retired and vacant slots retain their indices and
/// contain `T::neutral()`. Queue writes and fence ownership stay with the
/// caller, making this contract usable by native renderers without framework
/// or backend dependencies.
#[derive(Debug)]
pub struct IncrementalGpuTable<T> {
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
    submission_domain: GpuSubmissionDomain,
    last_submitted: Option<SubmissionProof>,
    completed: Option<CompletionProof>,
    policy: UploadPolicy,
    force_full_rebuild: bool,
}

impl<T: GpuTablePayload> Default for IncrementalGpuTable<T> {
    fn default() -> Self {
        Self::new(UploadPolicy::default()).expect("the default upload policy is valid")
    }
}

impl<T: GpuTablePayload> IncrementalGpuTable<T> {
    /// Creates an empty table with an upload-coalescing policy.
    pub fn new(policy: UploadPolicy) -> Result<Self, UploadPolicyError> {
        if !policy.full_rebuild_dirty_fraction.is_finite()
            || policy.full_rebuild_dirty_fraction <= 0.0
            || policy.full_rebuild_dirty_fraction > 1.0
        {
            return Err(UploadPolicyError::InvalidDirtyFraction);
        }
        if policy.full_rebuild_range_count == 0 {
            return Err(UploadPolicyError::ZeroRangeCount);
        }
        if size_of::<T>() == 0 || size_of::<T>() % 4 != 0 {
            return Err(UploadPolicyError::InvalidPayloadStride);
        }
        let table_id = NEXT_TABLE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("GPU table identity exhausted");
        Ok(Self {
            values: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
            dirty_slots: BTreeSet::new(),
            fence_pending: BTreeMap::new(),
            live_count: 0,
            table_id,
            gpu_epoch: 0,
            revision: 0,
            acknowledged_revision: 0,
            submission_domain: GpuSubmissionDomain::new(0, 0),
            last_submitted: None,
            completed: None,
            policy,
            force_full_rebuild: false,
        })
    }

    /// Allocated slots, including neutral vacant and retired entries.
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }
    /// Number of live entries.
    pub fn live_len(&self) -> usize {
        self.live_count
    }
    /// Current payload for a handle that belongs to this table and generation.
    pub fn get(&self, handle: GpuTableHandle) -> Option<&T> {
        if handle.table_id != self.table_id {
            return None;
        }
        let index = handle.slot as usize;
        let slot = self.slots.get(index)?;
        (slot.generation == handle.generation && matches!(slot.state, SlotState::Live))
            .then_some(&self.values[index])
    }

    /// Inserts a payload at a new or safely reclaimed stable slot.
    pub fn insert(&mut self, value: T) -> GpuTableHandle {
        if let Some(index) = self.free_slots.pop() {
            let generation = self.slots[index].generation;
            let revision = self.next_revision();
            self.values[index] = value;
            self.slots[index].state = SlotState::Live;
            self.slots[index].dirty_revision = Some(revision);
            self.dirty_slots.insert(index);
            self.live_count += 1;
            return GpuTableHandle {
                table_id: self.table_id,
                slot: index as u32,
                generation,
            };
        }
        let index = self.slots.len();
        assert!(
            index < u32::MAX as usize,
            "GPU table exceeded u32 slot space"
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
        GpuTableHandle {
            table_id: self.table_id,
            slot: index as u32,
            generation: 1,
        }
    }

    /// Replaces a live value, dirtying it only when its bytes differ.
    pub fn update(&mut self, handle: GpuTableHandle, value: T) -> bool {
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

    /// Removes a live entry and dirties its slot with the neutral payload.
    pub fn retire(&mut self, handle: GpuTableHandle) -> bool {
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

    /// Current retirement stage for the removed handle's generation.
    pub fn retirement(&self, handle: GpuTableHandle) -> Option<SlotRetirement> {
        if handle.table_id != self.table_id {
            return None;
        }
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        match slot.state {
            SlotState::Retiring => Some(SlotRetirement::Retiring),
            SlotState::FencePending { submission } => {
                Some(SlotRetirement::FencePending { submission })
            }
            _ => None,
        }
    }

    /// Current device and queue domain used by upload plans.
    pub const fn submission_domain(&self) -> GpuSubmissionDomain {
        self.submission_domain
    }

    /// Plans adjacent dirty slots as merged ranges without consuming dirtiness.
    pub fn plan_upload(&self) -> UploadPlan {
        let stride = size_of::<T>() as u64;
        let mut elements = Vec::<Range<u32>>::new();
        for &index in &self.dirty_slots {
            let start = index as u32;
            if let Some(last) = elements.last_mut() {
                if last.end == start {
                    last.end += 1;
                    continue;
                }
            }
            elements.push(start..start + 1);
        }
        let dirty_count: usize = elements.iter().map(|r| (r.end - r.start) as usize).sum();
        let capacity = self.slots.len();
        let fraction = if capacity == 0 {
            0.0
        } else {
            dirty_count as f32 / capacity as f32
        };
        let full_rebuild = self.force_full_rebuild
            || (!elements.is_empty()
                && (fraction >= self.policy.full_rebuild_dirty_fraction
                    || elements.len() >= self.policy.full_rebuild_range_count));
        if full_rebuild {
            elements.clear();
            if capacity > 0 {
                elements.push(0..capacity as u32);
            }
        }
        let ranges: Vec<_> = elements
            .into_iter()
            .map(|elements| DirtyRange {
                byte_offset: u64::from(elements.start) * stride,
                byte_len: u64::from(elements.end - elements.start) * stride,
                elements,
            })
            .collect();
        let bytes = ranges.iter().map(|r| r.byte_len).sum();
        UploadPlan {
            table_id: self.table_id,
            gpu_epoch: self.gpu_epoch,
            submission_domain: self.submission_domain,
            element_count: self.capacity(),
            revision: self.revision,
            ranges,
            bytes,
            full_rebuild,
        }
    }

    /// Validates whether `plan` may be acknowledged with `submission` without
    /// changing table state.
    ///
    /// Multi-table owners can validate every participating table first, then
    /// call [`Self::acknowledge_upload`] for each table as one logical commit.
    pub fn can_acknowledge_upload(&self, plan: &UploadPlan, submission: SubmissionProof) -> bool {
        if plan.table_id != self.table_id
            || plan.gpu_epoch != self.gpu_epoch
            || plan.submission_domain != self.submission_domain
            || submission.domain() != self.submission_domain
            || submission.value() == 0
            || plan.revision > self.revision
            || plan.revision < self.acknowledged_revision
            || self
                .completed
                .is_some_and(|completed| submission.value() <= completed.value())
            || self
                .last_submitted
                .is_some_and(|last| submission.value() <= last.value())
        {
            return false;
        }
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
        if self.dirty_slots.iter().any(|&i| {
            self.slots[i]
                .dirty_revision
                .is_some_and(|r| r <= plan.revision)
                && !range_contains(&plan.ranges, i as u32)
        }) {
            return false;
        }
        true
    }

    /// Acknowledges a successfully submitted plan on this table's queue.
    /// Returns false without changing state for stale, malformed, uncovered,
    /// or out-of-domain plans. The proof must belong to the same device
    /// generation and queue domain captured by the plan.
    pub fn acknowledge_upload(&mut self, plan: &UploadPlan, submission: SubmissionProof) -> bool {
        if !self.can_acknowledge_upload(plan, submission) {
            return false;
        }
        let slots = &mut self.slots;
        let pending = &mut self.fence_pending;
        self.dirty_slots.retain(|&index| {
            let slot = &mut slots[index];
            if slot.dirty_revision.is_some_and(|r| r <= plan.revision) {
                slot.dirty_revision = None;
                if matches!(slot.state, SlotState::Retiring) {
                    slot.state = SlotState::FencePending { submission };
                    pending.entry(submission.value()).or_default().push(index);
                }
                false
            } else {
                true
            }
        });
        if plan.full_rebuild {
            self.force_full_rebuild = false;
        }
        self.last_submitted = Some(submission);
        self.acknowledged_revision = plan.revision;
        true
    }

    /// Predicts allocated capacity after applying `completion` and then
    /// inserting `insertions` new values, without changing table state.
    ///
    /// This lets a fixed-capacity owner decide whether queue-draining
    /// compaction is required before beginning a multi-table mutation.
    pub fn projected_capacity_after_reclaim(
        &self,
        completion: CompletionProof,
        insertions: usize,
    ) -> usize {
        let mut reusable = self.free_slots.len();
        if completion.domain() == self.submission_domain {
            let known_submission = self.last_submitted.map_or(0, SubmissionProof::value);
            let reported = completion.value().min(known_submission);
            let current = self.completed.map_or(0, CompletionProof::value);
            let frontier = reported.max(current);
            reusable += self
                .fence_pending
                .range(..=frontier)
                .flat_map(|(_, indices)| indices)
                .filter(|&&index| self.slots[index].generation.checked_add(1).is_some())
                .count();
        }
        self.slots
            .len()
            .saturating_add(insertions.saturating_sub(reusable))
    }

    /// Reclaims slots whose queue completion is proven and advances generations.
    ///
    /// A completion from another device generation or queue domain is ignored.
    /// Even in the current domain, the frontier is clamped to this table's
    /// greatest acknowledged submission, so a runtime-wide value cannot
    /// manufacture completion for work this table never submitted.
    pub fn reclaim_completed(&mut self, completion: CompletionProof) -> usize {
        if completion.domain() != self.submission_domain {
            return 0;
        }
        let known_submission = self.last_submitted.map_or(0, SubmissionProof::value);
        let safe_value = completion.value().min(known_submission);
        let current_value = self.completed.map_or(0, |completed| completed.value());
        if safe_value > current_value {
            self.completed = Some(self.submission_domain.completion(safe_value));
        }
        let completed_value = self.completed.map_or(0, |completed| completed.value());
        let mut reclaimed = 0;
        while self
            .fence_pending
            .first_key_value()
            .is_some_and(|(&submission, _)| submission <= completed_value)
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
                    slot.state = SlotState::Exhausted;
                }
                reclaimed += 1;
            }
        }
        reclaimed
    }

    /// Payloads in stable slot order, including neutral retired slots.
    pub fn payloads(&self) -> &[T] {
        &self.values
    }
    /// Payload slice for an element range.
    pub fn payload_range(&self, elements: Range<u32>) -> Option<&[T]> {
        let start = elements.start as usize;
        let end = elements.end as usize;
        (start <= end && end <= self.values.len()).then(|| &self.values[start..end])
    }
    /// Exact byte view suitable for a queue write operation.
    pub fn payload_bytes(&self, elements: Range<u32>) -> Option<&[u8]> {
        self.payload_range(elements)
            .map(bytemuck::cast_slice::<T, u8>)
    }
    /// Invalidates the GPU copy (for example after buffer replacement), so
    /// the next plan repopulates the complete CPU table in `domain`.
    pub fn invalidate_gpu_copy(&mut self, domain: GpuSubmissionDomain) {
        let epoch = self.gpu_epoch.checked_add(1).expect("GPU epoch exhausted");
        self.next_revision();
        self.gpu_epoch = epoch;
        self.submission_domain = domain;
        self.force_full_rebuild = true;
        // The old buffer/queue epoch is no longer authoritative. Its in-flight
        // resources are retained by the runtime independently; recycle only
        // slots whose neutral retirement had already been submitted. A full
        // rebuild into the new buffer will include the resulting neutral or
        // replacement payload at every slot.
        let pending = core::mem::take(&mut self.fence_pending);
        for indices in pending.into_values() {
            for index in indices {
                let slot = &mut self.slots[index];
                if let Some(generation) = slot.generation.checked_add(1) {
                    slot.generation = generation;
                    slot.state = SlotState::Vacant;
                    self.free_slots.push(index);
                } else {
                    slot.state = SlotState::Exhausted;
                }
            }
        }
        self.last_submitted = None;
        self.completed = None;
    }

    fn next_revision(&mut self) -> u64 {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("GPU table revision exhausted");
        self.revision
    }
}

fn range_contains(ranges: &[DirtyRange], index: u32) -> bool {
    let i = ranges.partition_point(|range| range.elements.end <= index);
    ranges
        .get(i)
        .is_some_and(|range| range.elements.start <= index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct Record {
        value: u32,
        extra: u32,
    }
    impl GpuTablePayload for Record {
        fn neutral() -> Self {
            Self::default()
        }
    }

    fn policy() -> UploadPolicy {
        UploadPolicy {
            full_rebuild_dirty_fraction: 1.0,
            full_rebuild_range_count: usize::MAX,
        }
    }
    fn record(value: u32) -> Record {
        Record { value, extra: 0 }
    }

    fn domain() -> GpuSubmissionDomain {
        GpuSubmissionDomain::new(0, 0)
    }

    fn submission(value: u64) -> SubmissionProof {
        domain()
            .submission(value)
            .expect("test submission is nonzero")
    }

    fn completion(value: u64) -> CompletionProof {
        domain().completion(value)
    }

    #[test]
    fn stable_slots_dirty_ranges_and_payload_bytes() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let a = table.insert(record(1));
        let b = table.insert(record(2));
        let c = table.insert(record(3));
        assert_eq!((a.slot(), b.slot(), c.slot()), (0, 1, 2));
        let initial = table.plan_upload();
        assert_eq!(initial.ranges()[0].elements, 0..3);
        assert!(table.acknowledge_upload(&initial, submission(1)));
        assert!(table.update(a, record(10)));
        assert!(table.update(c, record(30)));
        let sparse = table.plan_upload();
        assert_eq!(
            sparse
                .ranges()
                .iter()
                .map(|r| r.elements.clone())
                .collect::<Vec<_>>(),
            vec![0..1, 2..3]
        );
        assert_eq!(
            table.payload_bytes(2..3).unwrap().len(),
            size_of::<Record>()
        );
    }

    #[test]
    fn unchanged_update_and_wrong_or_stale_handles_are_rejected_without_dirtying() {
        let mut a = IncrementalGpuTable::new(policy()).unwrap();
        let mut b = IncrementalGpuTable::new(policy()).unwrap();
        let handle = a.insert(record(7));
        let other = b.insert(record(9));
        assert!(a.acknowledge_upload(&a.plan_upload(), submission(1)));
        assert!(a.update(handle, record(7)));
        assert!(a.plan_upload().is_empty());
        assert!(!a.update(other, record(8)));
        assert!(a.retire(handle));
        assert!(!a.update(handle, record(8)));
        assert_eq!(a.get(handle), None);
    }

    #[test]
    fn snapshot_ack_preserves_later_edits_and_invalid_coverage_is_atomic() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let h = table.insert(record(1));
        let old = table.plan_upload();
        assert!(table.update(h, record(2)));
        assert!(table.acknowledge_upload(&old, submission(1)));
        assert_eq!(table.plan_upload().ranges()[0].elements, 0..1);
        let plan = table.plan_upload();
        let mut forged = plan.clone();
        forged.ranges.clear();
        forged.bytes = 0;
        assert!(!table.acknowledge_upload(&forged, submission(2)));
        assert!(!table.plan_upload().is_empty());
    }

    #[test]
    fn retirement_requires_upload_fence_completion_before_generation_reuse() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let old = table.insert(record(1));
        assert!(table.acknowledge_upload(&table.plan_upload(), submission(1)));
        assert!(table.retire(old));
        assert_eq!(table.retirement(old), Some(SlotRetirement::Retiring));
        let plan = table.plan_upload();
        assert!(table.acknowledge_upload(&plan, submission(5)));
        assert_eq!(
            table.retirement(old),
            Some(SlotRetirement::FencePending {
                submission: submission(5),
            })
        );
        assert_eq!(table.reclaim_completed(completion(4)), 0);
        assert_eq!(table.reclaim_completed(completion(5)), 1);
        let replacement = table.insert(record(2));
        assert_eq!(replacement.slot(), old.slot());
        assert_ne!(replacement.generation(), old.generation());
        assert_eq!(table.get(old), None);
    }

    #[test]
    fn gpu_invalidation_forces_complete_rebuild_and_rejects_old_epoch_plan() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        table.insert(record(1));
        let old = table.plan_upload();
        assert!(table.acknowledge_upload(&old, submission(1)));
        let recovered_domain = GpuSubmissionDomain::new(1, 0);
        table.invalidate_gpu_copy(recovered_domain);
        assert!(!table.acknowledge_upload(&old, submission(2)));
        let rebuild = table.plan_upload();
        assert!(rebuild.stats().full_rebuild);
        assert_eq!(rebuild.ranges()[0].elements, 0..1);
        assert!(table.acknowledge_upload(&rebuild, recovered_domain.submission(2).unwrap()));
        assert!(table.plan_upload().is_empty());
    }

    #[test]
    fn gpu_invalidation_starts_a_fresh_fence_domain_and_releases_old_retirements() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let old = table.insert(record(1));
        let first = table.plan_upload();
        assert!(table.acknowledge_upload(&first, submission(9)));
        assert!(table.retire(old));
        let retire = table.plan_upload();
        assert!(table.acknowledge_upload(&retire, submission(12)));
        let recovered_domain = GpuSubmissionDomain::new(1, 4);
        table.invalidate_gpu_copy(recovered_domain);

        let replacement = table.insert(record(2));
        assert_eq!(replacement.slot(), old.slot());
        assert_ne!(replacement.generation(), old.generation());
        let rebuild = table.plan_upload();
        assert!(rebuild.stats().full_rebuild);
        assert!(table.acknowledge_upload(&rebuild, recovered_domain.submission(1).unwrap()));
        assert_eq!(table.reclaim_completed(recovered_domain.completion(1)), 0);
        assert!(table.plan_upload().is_empty());
    }

    #[test]
    fn completion_frontier_cannot_advance_past_known_submissions() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let handle = table.insert(record(1));
        assert!(table.acknowledge_upload(&table.plan_upload(), submission(3)));
        assert_eq!(table.reclaim_completed(completion(u64::MAX)), 0);
        assert!(table.retire(handle));
        let retire = table.plan_upload();
        assert!(table.acknowledge_upload(&retire, submission(4)));
        assert_eq!(table.reclaim_completed(completion(3)), 0);
        assert_eq!(table.reclaim_completed(completion(4)), 1);
    }

    #[test]
    fn completion_from_another_device_or_queue_domain_cannot_reclaim() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let old = table.insert(record(1));
        assert!(table.acknowledge_upload(&table.plan_upload(), submission(1)));
        assert!(table.retire(old));
        let retire = table.plan_upload();
        assert!(table.acknowledge_upload(&retire, submission(2)));

        let other_device = GpuSubmissionDomain::new(1, 0);
        let other_queue = GpuSubmissionDomain::new(0, 1);
        assert_eq!(
            table.reclaim_completed(other_device.completion(u64::MAX)),
            0
        );
        assert_eq!(table.reclaim_completed(other_queue.completion(u64::MAX)), 0);
        assert!(matches!(
            table.retirement(old),
            Some(SlotRetirement::FencePending { .. })
        ));
        assert_eq!(table.reclaim_completed(completion(2)), 1);
    }

    #[test]
    fn stale_old_device_completion_cannot_confirm_new_device_retirement() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        let old = table.insert(record(1));
        assert!(table.acknowledge_upload(&table.plan_upload(), submission(7)));
        assert!(table.retire(old));
        let old_retirement = table.plan_upload();
        assert!(table.acknowledge_upload(&old_retirement, submission(9)));

        let recovered_domain = GpuSubmissionDomain::new(1, 0);
        table.invalidate_gpu_copy(recovered_domain);
        let replacement = table.insert(record(2));
        let rebuild = table.plan_upload();
        assert_eq!(rebuild.submission_domain(), recovered_domain);
        assert!(table.acknowledge_upload(&rebuild, recovered_domain.submission(1).unwrap()));
        assert!(table.retire(replacement));
        let new_retirement = table.plan_upload();
        assert!(table.acknowledge_upload(&new_retirement, recovered_domain.submission(2).unwrap()));

        assert_eq!(table.reclaim_completed(domain().completion(9)), 0);
        assert!(matches!(
            table.retirement(replacement),
            Some(SlotRetirement::FencePending { .. })
        ));
        assert_eq!(table.reclaim_completed(recovered_domain.completion(2)), 1);
    }

    #[test]
    fn submission_from_another_queue_domain_is_rejected_without_consuming_plan() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        table.insert(record(1));
        let plan = table.plan_upload();
        let other_queue = GpuSubmissionDomain::new(0, 1);
        assert!(!table.acknowledge_upload(&plan, other_queue.submission(1).unwrap()));
        assert_eq!(table.plan_upload(), plan);
        assert!(table.acknowledge_upload(&plan, submission(1)));
    }

    #[test]
    fn submission_proof_must_advance_strictly_within_one_domain() {
        let mut table = IncrementalGpuTable::new(policy()).unwrap();
        table.insert(record(1));
        let first = table.plan_upload();
        assert!(table.acknowledge_upload(&first, submission(4)));
        let second = table.plan_upload();
        assert!(second.is_empty());

        let handle = table.insert(record(2));
        let update = table.plan_upload();
        assert!(!table.acknowledge_upload(&update, submission(4)));
        assert!(!table.acknowledge_upload(&update, submission(3)));
        assert!(table.acknowledge_upload(&update, submission(5)));
        assert_eq!(table.get(handle).unwrap().value, 2);
    }

    #[test]
    fn policy_switches_dense_changes_to_full_rebuild() {
        let mut table = IncrementalGpuTable::new(UploadPolicy {
            full_rebuild_dirty_fraction: 0.5,
            full_rebuild_range_count: usize::MAX,
        })
        .unwrap();
        let handles: Vec<_> = (0..4).map(|v| table.insert(record(v))).collect();
        assert!(table.acknowledge_upload(&table.plan_upload(), submission(1)));
        assert!(table.update(handles[0], record(10)));
        assert!(table.update(handles[2], record(20)));
        let plan = table.plan_upload();
        assert!(plan.stats().full_rebuild);
        assert_eq!(plan.ranges()[0].elements, 0..4);
    }

    #[test]
    fn invalid_upload_policy_returns_field_specific_error() {
        assert!(matches!(
            IncrementalGpuTable::<Record>::new(UploadPolicy {
                full_rebuild_dirty_fraction: f32::NAN,
                full_rebuild_range_count: 1,
            }),
            Err(UploadPolicyError::InvalidDirtyFraction)
        ));
        assert!(matches!(
            IncrementalGpuTable::<Record>::new(UploadPolicy {
                full_rebuild_dirty_fraction: 0.5,
                full_rebuild_range_count: 0,
            }),
            Err(UploadPolicyError::ZeroRangeCount)
        ));
    }
}
