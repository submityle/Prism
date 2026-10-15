//! Engine-neutral consumer-side storage for [`prism_camera::GpuViewRecord`].
//!
//! The crate owns the CPU contract between camera publication and a renderer's
//! buffer upload code. It deliberately has no GPU, window, ECS, or engine
//! dependency. A renderer provides [`UploadBackend`] and decides how each
//! operation is copied into its native buffer.
//!
//! A [`ViewBufferRing`] has two fixed dimensions:
//!
//! * the number of frame slots that may be in flight;
//! * the maximum number of view records in each slot.
//!
//! [`ViewBufferRing::plan_uploads`] accepts a complete view snapshot. It
//! validates the ABI schema, compares the snapshot with the selected frame
//! slot's last committed bytes, and returns only changed records. Omitted
//! records are cleared, which makes removal deterministic and prevents stale
//! views from surviving a slot recycle.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use prism_camera::{GpuViewRecord, GPU_VIEW_RECORD_SIZE, VIEW_SCHEMA_VERSION};

/// The byte width consumed by one camera view record.
pub const VIEW_RECORD_SIZE: usize = GPU_VIEW_RECORD_SIZE;

/// The exact byte representation stored in a frame slot.
pub type RecordBytes = [u8; VIEW_RECORD_SIZE];

/// An acquired physical frame-buffer slot.
///
/// The token contains an acquisition generation, so a slot cannot be used by
/// an older frame after it has been recycled.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameSlot {
    slot_index: usize,
    frame_id: u64,
    generation: u64,
}

impl FrameSlot {
    /// Returns the physical slot index used by the backend.
    #[must_use]
    pub const fn slot_index(self) -> usize {
        self.slot_index
    }

    /// Returns the caller-provided frame identifier.
    #[must_use]
    pub const fn frame_id(self) -> u64 {
        self.frame_id
    }
}

/// Lifecycle state of a physical frame slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameSlotState {
    /// The slot can be acquired by a new frame.
    Free,
    /// The slot has been acquired and accepts one upload plan.
    Recording,
    /// An upload plan exists and can be submitted or cancelled.
    Prepared,
    /// The backend accepted the plan; the slot awaits completion notification.
    Submitted,
}

/// The operation a backend should apply to one view record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UploadKind {
    /// Copy [`UploadOperation::bytes`] to the record offset.
    Write,
    /// Clear the record at the offset. The bytes are zeroes for backends that
    /// use one copy path for both writes and clears.
    Clear,
}

/// One deterministic record-sized upload operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UploadOperation {
    view_index: u32,
    offset_bytes: usize,
    kind: UploadKind,
    bytes: RecordBytes,
}

impl UploadOperation {
    /// Returns the logical view index encoded by the source record.
    #[must_use]
    pub const fn view_index(self) -> u32 {
        self.view_index
    }

    /// Returns the byte offset within the selected frame slot.
    #[must_use]
    pub const fn offset_bytes(self) -> usize {
        self.offset_bytes
    }

    /// Returns whether this operation writes or clears the record.
    #[must_use]
    pub const fn kind(self) -> UploadKind {
        self.kind
    }

    /// Returns the exact little-endian bytes produced by
    /// [`GpuViewRecord::to_bytes_le`], or zero bytes for a clear.
    #[must_use]
    pub const fn bytes(self) -> RecordBytes {
        self.bytes
    }

    /// Borrows the operation bytes without copying them.
    #[must_use]
    pub const fn bytes_ref(&self) -> &RecordBytes {
        &self.bytes
    }
}

/// An immutable upload plan produced for one acquired frame slot.
///
/// The plan owns its operation bytes, so it remains valid while a backend
/// performs an asynchronous or deferred upload. The ring commits the plan
/// only after the backend reports success.
#[derive(Debug)]
pub struct UploadPlan {
    frame_slot: FrameSlot,
    plan_id: u64,
    schema_version: u32,
    operations: Vec<UploadOperation>,
    target: Vec<RecordBytes>,
}

impl UploadPlan {
    /// Returns the frame slot targeted by this plan.
    #[must_use]
    pub const fn frame_slot(&self) -> FrameSlot {
        self.frame_slot
    }

    /// Returns the frame identifier targeted by this plan.
    #[must_use]
    pub const fn frame_id(&self) -> u64 {
        self.frame_slot.frame_id
    }

    /// Returns the schema version validated for every source record.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns all changed record operations in ascending view-index order.
    #[must_use]
    pub fn operations(&self) -> &[UploadOperation] {
        &self.operations
    }

    /// Returns true when the target snapshot needs no writes for this slot.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Returns the total number of bytes represented by the operations.
    #[must_use]
    pub fn upload_bytes(&self) -> usize {
        self.operations.len() * VIEW_RECORD_SIZE
    }

    /// Returns the number of records represented by the complete target.
    #[must_use]
    pub fn view_capacity(&self) -> usize {
        self.target.len()
    }
}

/// A renderer-owned upload sink.
///
/// The implementation may map the operations to a uniform/storage buffer,
/// an API-specific staging ring, or a test recorder. It must return success
/// only after it has accepted every operation in the plan.
pub trait UploadBackend {
    /// Backend-specific failure type.
    type Error;

    /// Accepts one immutable plan for upload.
    fn upload(&mut self, plan: &UploadPlan) -> Result<(), Self::Error>;
}

/// Error returned by [`ViewBufferRing`] operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RingError {
    /// The ring cannot have zero frame slots.
    InvalidFrameSlotCount { requested: usize },
    /// The ring cannot have zero view capacity.
    InvalidViewCapacity { requested: usize },
    /// The configured byte size does not fit in `usize`.
    CapacityOverflow {
        frame_slot_count: usize,
        view_capacity: usize,
    },
    /// The frame-id-selected physical slot is still in use.
    SlotBusy {
        slot_index: usize,
        requested_frame_id: u64,
        resident_frame_id: u64,
        state: FrameSlotState,
    },
    /// A token refers to a slot index outside this ring.
    InvalidSlot { slot_index: usize },
    /// A token was used after its physical slot was recycled.
    StaleSlot {
        slot_index: usize,
        frame_id: u64,
        expected_generation: u64,
        actual_generation: u64,
    },
    /// The operation is not valid for the slot's current lifecycle state.
    InvalidState {
        slot_index: usize,
        frame_id: u64,
        expected: FrameSlotState,
        actual: FrameSlotState,
    },
    /// The complete snapshot contains more records than the configured bound.
    ViewCountExceeded { capacity: usize, actual: usize },
    /// A record points outside the fixed view array.
    ViewIndexOutOfRange { view_index: u32, capacity: usize },
    /// A complete snapshot contains the same logical view more than once.
    DuplicateViewIndex { view_index: u32 },
    /// The source record is not compatible with this consumer's schema.
    SchemaMismatch {
        view_index: u32,
        expected: u32,
        actual: u32,
    },
    /// The supplied plan is not the plan currently prepared by its slot.
    StalePlan { plan_id: u64 },
}

/// Stable category for programmatic error handling and telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RingErrorCode {
    InvalidFrameSlotCount,
    InvalidViewCapacity,
    CapacityOverflow,
    SlotBusy,
    InvalidSlot,
    StaleSlot,
    InvalidState,
    ViewCountExceeded,
    ViewIndexOutOfRange,
    DuplicateViewIndex,
    SchemaMismatch,
    StalePlan,
}

impl RingError {
    /// Returns the stable category for this error.
    #[must_use]
    pub const fn code(self) -> RingErrorCode {
        match self {
            Self::InvalidFrameSlotCount { .. } => RingErrorCode::InvalidFrameSlotCount,
            Self::InvalidViewCapacity { .. } => RingErrorCode::InvalidViewCapacity,
            Self::CapacityOverflow { .. } => RingErrorCode::CapacityOverflow,
            Self::SlotBusy { .. } => RingErrorCode::SlotBusy,
            Self::InvalidSlot { .. } => RingErrorCode::InvalidSlot,
            Self::StaleSlot { .. } => RingErrorCode::StaleSlot,
            Self::InvalidState { .. } => RingErrorCode::InvalidState,
            Self::ViewCountExceeded { .. } => RingErrorCode::ViewCountExceeded,
            Self::ViewIndexOutOfRange { .. } => RingErrorCode::ViewIndexOutOfRange,
            Self::DuplicateViewIndex { .. } => RingErrorCode::DuplicateViewIndex,
            Self::SchemaMismatch { .. } => RingErrorCode::SchemaMismatch,
            Self::StalePlan { .. } => RingErrorCode::StalePlan,
        }
    }
}

impl fmt::Display for RingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrameSlotCount { requested } => {
                write!(formatter, "frame slot count must be non-zero: {requested}")
            }
            Self::InvalidViewCapacity { requested } => {
                write!(formatter, "view capacity must be non-zero: {requested}")
            }
            Self::CapacityOverflow {
                frame_slot_count,
                view_capacity,
            } => write!(
                formatter,
                "ring byte capacity overflows usize: {frame_slot_count} x {view_capacity}"
            ),
            Self::SlotBusy {
                slot_index,
                requested_frame_id,
                resident_frame_id,
                state,
            } => write!(
                formatter,
                "slot {slot_index} is {state:?} for frame {resident_frame_id}; cannot acquire frame {requested_frame_id}"
            ),
            Self::InvalidSlot { slot_index } => write!(formatter, "invalid frame slot index: {slot_index}"),
            Self::StaleSlot {
                slot_index,
                frame_id,
                expected_generation,
                actual_generation,
            } => write!(
                formatter,
                "stale slot {slot_index} for frame {frame_id}: expected generation {expected_generation}, actual {actual_generation}"
            ),
            Self::InvalidState {
                slot_index,
                frame_id,
                expected,
                actual,
            } => write!(
                formatter,
                "slot {slot_index} for frame {frame_id} is {actual:?}; expected {expected:?}"
            ),
            Self::ViewCountExceeded { capacity, actual } => {
                write!(formatter, "view count {actual} exceeds capacity {capacity}")
            }
            Self::ViewIndexOutOfRange {
                view_index,
                capacity,
            } => write!(formatter, "view index {view_index} exceeds capacity {capacity}"),
            Self::DuplicateViewIndex { view_index } => {
                write!(formatter, "duplicate view index: {view_index}")
            }
            Self::SchemaMismatch {
                view_index,
                expected,
                actual,
            } => write!(
                formatter,
                "view {view_index} has schema {actual}; expected {expected}"
            ),
            Self::StalePlan { plan_id } => write!(formatter, "stale upload plan: {plan_id}"),
        }
    }
}

/// Error returned when backend upload or ring commit fails.
#[derive(Debug, PartialEq, Eq)]
pub enum SubmitError<E> {
    /// The plan or slot failed local validation.
    Ring(RingError),
    /// The backend rejected the plan. The slot remains prepared for retry.
    Backend(E),
}

impl<E> From<RingError> for SubmitError<E> {
    fn from(error: RingError) -> Self {
        Self::Ring(error)
    }
}

/// Result of notifying the ring about completed GPU work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReclaimReport {
    /// The highest completion frame identifier observed by this call.
    pub completed_frame_id: u64,
    /// Number of submitted slots made available by this call.
    pub reclaimed_slots: usize,
}

#[derive(Debug)]
struct SlotStorage {
    generation: u64,
    frame_id: Option<u64>,
    state: FrameSlotState,
    records: Vec<RecordBytes>,
    pending_plan_id: Option<u64>,
}

/// Fixed-capacity ring for camera view-buffer uploads.
#[derive(Debug)]
pub struct ViewBufferRing {
    expected_schema: u32,
    view_capacity: usize,
    slots: Vec<SlotStorage>,
    next_plan_id: u64,
    highest_completed_frame: Option<u64>,
}

impl ViewBufferRing {
    /// Creates a ring using [`VIEW_SCHEMA_VERSION`].
    pub fn new(frame_slot_count: usize, view_capacity: usize) -> Result<Self, RingError> {
        Self::with_schema(frame_slot_count, view_capacity, VIEW_SCHEMA_VERSION)
    }

    /// Creates a ring with an explicitly required record schema version.
    pub fn with_schema(
        frame_slot_count: usize,
        view_capacity: usize,
        expected_schema: u32,
    ) -> Result<Self, RingError> {
        if frame_slot_count == 0 {
            return Err(RingError::InvalidFrameSlotCount {
                requested: frame_slot_count,
            });
        }
        if view_capacity == 0 {
            return Err(RingError::InvalidViewCapacity {
                requested: view_capacity,
            });
        }
        if frame_slot_count
            .checked_mul(view_capacity)
            .and_then(|records| records.checked_mul(VIEW_RECORD_SIZE))
            .is_none()
        {
            return Err(RingError::CapacityOverflow {
                frame_slot_count,
                view_capacity,
            });
        }

        let mut slots = Vec::with_capacity(frame_slot_count);
        for _ in 0..frame_slot_count {
            slots.push(SlotStorage {
                generation: 0,
                frame_id: None,
                state: FrameSlotState::Free,
                records: vec![[0_u8; VIEW_RECORD_SIZE]; view_capacity],
                pending_plan_id: None,
            });
        }

        Ok(Self {
            expected_schema,
            view_capacity,
            slots,
            next_plan_id: 1,
            highest_completed_frame: None,
        })
    }

    /// Returns the number of physical frame slots.
    #[must_use]
    pub fn frame_slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Returns the maximum number of view records per frame slot.
    #[must_use]
    pub const fn view_capacity(&self) -> usize {
        self.view_capacity
    }

    /// Returns the schema required by this consumer.
    #[must_use]
    pub const fn expected_schema_version(&self) -> u32 {
        self.expected_schema
    }

    /// Returns a slot's current state without requiring an acquisition token.
    #[must_use]
    pub fn state_at(&self, slot_index: usize) -> Option<FrameSlotState> {
        self.slots.get(slot_index).map(|slot| slot.state)
    }

    /// Acquires the frame-id-selected physical slot.
    pub fn begin_frame(&mut self, frame_id: u64) -> Result<FrameSlot, RingError> {
        let slot_index = (frame_id % self.slots.len() as u64) as usize;
        let storage = &mut self.slots[slot_index];
        if storage.state != FrameSlotState::Free {
            return Err(RingError::SlotBusy {
                slot_index,
                requested_frame_id: frame_id,
                resident_frame_id: storage.frame_id.unwrap_or(frame_id),
                state: storage.state,
            });
        }

        storage.generation = storage.generation.wrapping_add(1);
        storage.frame_id = Some(frame_id);
        storage.state = FrameSlotState::Recording;
        storage.pending_plan_id = None;

        Ok(FrameSlot {
            slot_index,
            frame_id,
            generation: storage.generation,
        })
    }

    /// Returns the current state for a valid acquisition token.
    pub fn state(&self, slot: FrameSlot) -> Result<FrameSlotState, RingError> {
        let storage = self.validate_slot(slot)?;
        Ok(storage.state)
    }

    /// Builds a complete-snapshot diff for an acquired frame slot.
    pub fn plan_uploads(
        &mut self,
        slot: FrameSlot,
        views: &[GpuViewRecord],
    ) -> Result<UploadPlan, RingError> {
        self.expect_state(slot, FrameSlotState::Recording)?;
        if views.len() > self.view_capacity {
            return Err(RingError::ViewCountExceeded {
                capacity: self.view_capacity,
                actual: views.len(),
            });
        }

        let mut target = vec![[0_u8; VIEW_RECORD_SIZE]; self.view_capacity];
        let mut seen = vec![false; self.view_capacity];
        for view in views {
            if view.schema_version != self.expected_schema {
                return Err(RingError::SchemaMismatch {
                    view_index: view.view_index,
                    expected: self.expected_schema,
                    actual: view.schema_version,
                });
            }
            let view_index = view.view_index as usize;
            if view_index >= self.view_capacity {
                return Err(RingError::ViewIndexOutOfRange {
                    view_index: view.view_index,
                    capacity: self.view_capacity,
                });
            }
            if seen[view_index] {
                return Err(RingError::DuplicateViewIndex {
                    view_index: view.view_index,
                });
            }
            seen[view_index] = true;
            target[view_index] = view.to_bytes_le();
        }

        let plan_id = self.allocate_plan_id();
        let mut operations = Vec::with_capacity(self.view_capacity);
        let storage = &self.slots[slot.slot_index];
        for (view_index, (old, next)) in storage.records.iter().zip(&target).enumerate() {
            if old != next {
                let kind = if next.iter().all(|byte| *byte == 0) {
                    UploadKind::Clear
                } else {
                    UploadKind::Write
                };
                operations.push(UploadOperation {
                    view_index: view_index as u32,
                    offset_bytes: view_index * VIEW_RECORD_SIZE,
                    kind,
                    bytes: *next,
                });
            }
        }

        let storage = &mut self.slots[slot.slot_index];
        storage.state = FrameSlotState::Prepared;
        storage.pending_plan_id = Some(plan_id);

        Ok(UploadPlan {
            frame_slot: slot,
            plan_id,
            schema_version: self.expected_schema,
            operations,
            target,
        })
    }

    /// Sends a plan to a backend and commits it when the backend succeeds.
    pub fn submit<B: UploadBackend>(
        &mut self,
        plan: &UploadPlan,
        backend: &mut B,
    ) -> Result<FrameSlot, SubmitError<B::Error>> {
        self.validate_plan(plan)?;
        backend.upload(plan).map_err(SubmitError::Backend)?;
        self.commit(plan).map_err(SubmitError::Ring)
    }

    /// Commits a plan after an external upload has been accepted.
    pub fn commit(&mut self, plan: &UploadPlan) -> Result<FrameSlot, RingError> {
        self.validate_plan(plan)?;
        let storage = &mut self.slots[plan.frame_slot.slot_index];
        storage.records.clone_from(&plan.target);
        storage.pending_plan_id = None;
        storage.state = FrameSlotState::Submitted;
        Ok(plan.frame_slot)
    }

    /// Cancels a prepared plan while keeping the frame slot recordable.
    pub fn cancel_plan(&mut self, slot: FrameSlot) -> Result<(), RingError> {
        self.expect_state(slot, FrameSlotState::Prepared)?;
        let storage = &mut self.slots[slot.slot_index];
        storage.pending_plan_id = None;
        storage.state = FrameSlotState::Recording;
        Ok(())
    }

    /// Aborts a recording or prepared frame without changing its cached bytes.
    pub fn abort_frame(&mut self, slot: FrameSlot) -> Result<(), RingError> {
        let state = self.state(slot)?;
        if state != FrameSlotState::Recording && state != FrameSlotState::Prepared {
            return Err(RingError::InvalidState {
                slot_index: slot.slot_index,
                frame_id: slot.frame_id,
                expected: FrameSlotState::Recording,
                actual: state,
            });
        }
        let storage = &mut self.slots[slot.slot_index];
        storage.pending_plan_id = None;
        storage.frame_id = None;
        storage.state = FrameSlotState::Free;
        Ok(())
    }

    /// Marks submitted frames up to `completed_frame_id` as reusable.
    pub fn reclaim(&mut self, completed_frame_id: u64) -> ReclaimReport {
        let highest = self
            .highest_completed_frame
            .map_or(completed_frame_id, |previous| {
                previous.max(completed_frame_id)
            });
        self.highest_completed_frame = Some(highest);

        let mut reclaimed_slots = 0;
        for storage in &mut self.slots {
            if storage.state == FrameSlotState::Submitted
                && storage.frame_id.is_some_and(|frame_id| frame_id <= highest)
            {
                storage.frame_id = None;
                storage.state = FrameSlotState::Free;
                storage.pending_plan_id = None;
                reclaimed_slots += 1;
            }
        }

        ReclaimReport {
            completed_frame_id: highest,
            reclaimed_slots,
        }
    }

    fn allocate_plan_id(&mut self) -> u64 {
        let plan_id = self.next_plan_id;
        self.next_plan_id = self.next_plan_id.wrapping_add(1).max(1);
        plan_id
    }

    fn validate_slot(&self, slot: FrameSlot) -> Result<&SlotStorage, RingError> {
        let storage = self
            .slots
            .get(slot.slot_index)
            .ok_or(RingError::InvalidSlot {
                slot_index: slot.slot_index,
            })?;
        if storage.frame_id != Some(slot.frame_id) || storage.generation != slot.generation {
            return Err(RingError::StaleSlot {
                slot_index: slot.slot_index,
                frame_id: slot.frame_id,
                expected_generation: slot.generation,
                actual_generation: storage.generation,
            });
        }
        Ok(storage)
    }

    fn expect_state(&self, slot: FrameSlot, expected: FrameSlotState) -> Result<(), RingError> {
        let actual = self.state(slot)?;
        if actual != expected {
            return Err(RingError::InvalidState {
                slot_index: slot.slot_index,
                frame_id: slot.frame_id,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn validate_plan(&self, plan: &UploadPlan) -> Result<(), RingError> {
        let storage = self.validate_slot(plan.frame_slot)?;
        if storage.state != FrameSlotState::Prepared {
            return Err(RingError::InvalidState {
                slot_index: plan.frame_slot.slot_index,
                frame_id: plan.frame_slot.frame_id,
                expected: FrameSlotState::Prepared,
                actual: storage.state,
            });
        }
        if storage.pending_plan_id != Some(plan.plan_id) {
            return Err(RingError::StalePlan {
                plan_id: plan.plan_id,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(index: u32, exposure: f32) -> GpuViewRecord {
        GpuViewRecord {
            exposure,
            schema_version: VIEW_SCHEMA_VERSION,
            view_index: index,
            ..GpuViewRecord::default()
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        calls: usize,
        operations: usize,
        fail: bool,
    }

    impl UploadBackend for RecordingBackend {
        type Error = &'static str;

        fn upload(&mut self, plan: &UploadPlan) -> Result<(), Self::Error> {
            self.calls += 1;
            self.operations += plan.operations().len();
            if self.fail {
                Err("backend rejected plan")
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn fixed_capacity_and_schema_are_enforced() {
        assert_eq!(
            ViewBufferRing::new(0, 1).unwrap_err().code(),
            RingErrorCode::InvalidFrameSlotCount
        );
        assert_eq!(
            ViewBufferRing::new(1, 0).unwrap_err().code(),
            RingErrorCode::InvalidViewCapacity
        );

        let mut ring = ViewBufferRing::new(1, 1).unwrap();
        let slot = ring.begin_frame(0).unwrap();
        let wrong_schema = GpuViewRecord {
            schema_version: VIEW_SCHEMA_VERSION + 1,
            ..view(0, 1.0)
        };
        assert_eq!(
            ring.plan_uploads(slot, &[wrong_schema]).unwrap_err().code(),
            RingErrorCode::SchemaMismatch
        );
        assert_eq!(
            ring.plan_uploads(slot, &[view(1, 1.0)]).unwrap_err().code(),
            RingErrorCode::ViewIndexOutOfRange
        );
    }

    #[test]
    fn plan_consumes_record_bytes_and_only_reports_dirty_records() {
        let mut ring = ViewBufferRing::new(2, 4).unwrap();
        let slot = ring.begin_frame(0).unwrap();
        let first = ring.plan_uploads(slot, &[view(2, 1.0)]).unwrap();
        assert_eq!(first.operations().len(), 1);
        assert_eq!(first.operations()[0].view_index(), 2);
        assert_eq!(first.operations()[0].offset_bytes(), 2 * VIEW_RECORD_SIZE);
        assert_eq!(first.operations()[0].kind(), UploadKind::Write);
        assert_eq!(first.operations()[0].bytes(), view(2, 1.0).to_bytes_le());

        let mut backend = RecordingBackend::default();
        ring.submit(&first, &mut backend).unwrap();
        assert_eq!(backend.calls, 1);
        assert_eq!(ring.state(slot).unwrap(), FrameSlotState::Submitted);

        let report = ring.reclaim(0);
        assert_eq!(report.reclaimed_slots, 1);
        let recycled = ring.begin_frame(2).unwrap();
        let unchanged = ring.plan_uploads(recycled, &[view(2, 1.0)]).unwrap();
        assert!(unchanged.is_empty());
    }

    #[test]
    fn omitted_records_generate_clear_operations() {
        let mut ring = ViewBufferRing::new(1, 2).unwrap();
        let slot = ring.begin_frame(0).unwrap();
        let initial = ring
            .plan_uploads(slot, &[view(0, 1.0), view(1, 2.0)])
            .unwrap();
        ring.commit(&initial).unwrap();
        ring.reclaim(0);

        let slot = ring.begin_frame(1).unwrap();
        let removal = ring.plan_uploads(slot, &[view(0, 1.0)]).unwrap();
        assert_eq!(removal.operations().len(), 1);
        assert_eq!(removal.operations()[0].view_index(), 1);
        assert_eq!(removal.operations()[0].kind(), UploadKind::Clear);
        assert!(removal.operations()[0]
            .bytes_ref()
            .iter()
            .all(|byte| *byte == 0));
    }

    #[test]
    fn duplicate_indices_and_stale_tokens_are_rejected() {
        let mut ring = ViewBufferRing::new(1, 2).unwrap();
        let old = ring.begin_frame(0).unwrap();
        assert_eq!(
            ring.plan_uploads(old, &[view(0, 1.0), view(0, 2.0)])
                .unwrap_err()
                .code(),
            RingErrorCode::DuplicateViewIndex
        );
        ring.abort_frame(old).unwrap();
        let current = ring.begin_frame(1).unwrap();
        assert_eq!(
            ring.state(old).unwrap_err().code(),
            RingErrorCode::StaleSlot
        );
        assert_eq!(current.frame_id(), 1);
    }

    #[test]
    fn backend_failure_leaves_plan_retryable() {
        let mut ring = ViewBufferRing::new(1, 1).unwrap();
        let slot = ring.begin_frame(0).unwrap();
        let plan = ring.plan_uploads(slot, &[view(0, 1.0)]).unwrap();
        let mut backend = RecordingBackend {
            fail: true,
            ..RecordingBackend::default()
        };
        assert_eq!(
            ring.submit(&plan, &mut backend),
            Err(SubmitError::Backend("backend rejected plan"))
        );
        assert_eq!(ring.state(slot).unwrap(), FrameSlotState::Prepared);
        backend.fail = false;
        ring.submit(&plan, &mut backend).unwrap();
        assert_eq!(ring.state(slot).unwrap(), FrameSlotState::Submitted);
    }

    #[test]
    fn frame_id_selects_slot_and_reclaim_controls_reuse() {
        let mut ring = ViewBufferRing::new(2, 1).unwrap();
        let first = ring.begin_frame(4).unwrap();
        let second = ring.begin_frame(5).unwrap();
        assert_eq!(first.slot_index(), 0);
        assert_eq!(second.slot_index(), 1);
        assert_eq!(
            ring.begin_frame(6).unwrap_err().code(),
            RingErrorCode::SlotBusy
        );

        let first_plan = ring.plan_uploads(first, &[]).unwrap();
        ring.commit(&first_plan).unwrap();
        let second_plan = ring.plan_uploads(second, &[]).unwrap();
        ring.commit(&second_plan).unwrap();
        assert_eq!(ring.reclaim(4).reclaimed_slots, 1);
        let recycled = ring.begin_frame(6).unwrap();
        assert_eq!(recycled.slot_index(), 0);
        assert_eq!(ring.reclaim(5).reclaimed_slots, 1);
    }
}
