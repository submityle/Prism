//! Deterministic lighting/GI event replay and checkpoint contracts.
//!
//! This file is intentionally self-contained. It can be introduced into the
//! lighting module without requiring any change to the module exports, while
//! still matching the existing renderer rules: epochs advance monotonically,
//! device generations fence submissions, and every state transition is
//! represented by an ordered event.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

/// Canonical schema version for the replay wire contract.
pub const REPLAY_SCHEMA_VERSION: u16 = 1;

const DIGEST_DOMAIN: &[u8] = b"prism.lighting.replay.v1\0";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const LANE_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Stable 32-byte content digest used for replay identity and checkpoints.
///
/// This is a deterministic content fingerprint, not a cryptographic proof.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ReplayDigest(pub [u8; 32]);

impl ReplayDigest {
    /// The digest of an empty replay chain.
    pub const ZERO: Self = Self([0; 32]);

    /// Hashes one byte slice with the replay domain separator.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut builder = DigestBuilder::new();
        builder.write_framed(DIGEST_DOMAIN);
        builder.write_framed(bytes);
        builder.finish()
    }

    /// Hashes ordered framed parts. Part boundaries are significant.
    #[must_use]
    pub fn combine(parts: &[&[u8]]) -> Self {
        let mut builder = DigestBuilder::new();
        builder.write_framed(DIGEST_DOMAIN);
        for part in parts {
            builder.write_framed(part);
        }
        builder.finish()
    }

    /// Returns the canonical bytes in lane order.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
struct DigestBuilder {
    lanes: [u64; 4],
}

impl DigestBuilder {
    const fn new() -> Self {
        let mut lanes = [0; 4];
        let mut index = 0;
        while index < lanes.len() {
            lanes[index] = FNV_OFFSET_BASIS ^ LANE_SALT.wrapping_mul(index as u64 + 1);
            index += 1;
        }
        Self { lanes }
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            for lane in &mut self.lanes {
                *lane = (*lane ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
            }
        }
    }

    fn write_framed(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn finish(self) -> ReplayDigest {
        let mut output = [0; 32];
        for (index, lane) in self.lanes.into_iter().enumerate() {
            let mut value = lane;
            value ^= value >> 33;
            value = value.wrapping_mul(FNV_PRIME);
            value ^= value >> 29;
            output[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        ReplayDigest(output)
    }
}

/// Monotonic versions of the fields that affect lighting history.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayEpoch {
    pub geometry: u32,
    pub material: u32,
    pub lighting: u32,
    pub environment: u32,
    pub layout: u32,
}

impl ReplayEpoch {
    /// The initial epoch before any frame is opened.
    pub const ZERO: Self = Self {
        geometry: 0,
        material: 0,
        lighting: 0,
        environment: 0,
        layout: 0,
    };

    fn write_to(self, builder: &mut DigestBuilder) {
        builder.write_u32(self.geometry);
        builder.write_u32(self.material);
        builder.write_u32(self.lighting);
        builder.write_u32(self.environment);
        builder.write_u32(self.layout);
    }

    fn checked_advance(self, fields: ReplayChangeMask) -> Result<Self, ReplayError> {
        let mut next = self;
        if fields.contains(ReplayChangeMask::GEOMETRY) {
            next.geometry = next
                .geometry
                .checked_add(1)
                .ok_or(ReplayError::EpochOverflow)?;
        }
        if fields.contains(ReplayChangeMask::MATERIAL) {
            next.material = next
                .material
                .checked_add(1)
                .ok_or(ReplayError::EpochOverflow)?;
        }
        if fields.contains(ReplayChangeMask::LIGHTING) {
            next.lighting = next
                .lighting
                .checked_add(1)
                .ok_or(ReplayError::EpochOverflow)?;
        }
        if fields.contains(ReplayChangeMask::ENVIRONMENT) {
            next.environment = next
                .environment
                .checked_add(1)
                .ok_or(ReplayError::EpochOverflow)?;
        }
        if fields.contains(ReplayChangeMask::LAYOUT) {
            next.layout = next
                .layout
                .checked_add(1)
                .ok_or(ReplayError::EpochOverflow)?;
        }
        Ok(next)
    }
}

/// Fields invalidated by a replay change event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayChangeMask(u8);

impl ReplayChangeMask {
    pub const GEOMETRY: Self = Self(1 << 0);
    pub const MATERIAL: Self = Self(1 << 1);
    pub const LIGHTING: Self = Self(1 << 2);
    pub const ENVIRONMENT: Self = Self(1 << 3);
    pub const LAYOUT: Self = Self(1 << 4);
    const VALID_BITS: u8 = Self::GEOMETRY.0
        | Self::MATERIAL.0
        | Self::LIGHTING.0
        | Self::ENVIRONMENT.0
        | Self::LAYOUT.0;

    /// Decodes the compact mask and rejects unknown bits.
    pub const fn from_bits(bits: u8) -> Result<Self, ReplayError> {
        if bits & !Self::VALID_BITS != 0 {
            Err(ReplayError::InvalidChangeMask { bits })
        } else {
            Ok(Self(bits))
        }
    }

    /// Creates a mask from known bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// The operation carried by one ordered replay event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayOperation {
    /// Opens a new frame. Its epoch must equal the previous frame's epoch.
    BeginFrame,
    /// Advances one or more invalidation fields in the active frame.
    Change { fields: ReplayChangeMask },
    /// Records a GPU or CPU lighting submission.
    Submit {
        submission: u64,
        device_generation: u64,
        input_digest: ReplayDigest,
    },
    /// Records completion of a submission from the active device generation.
    Complete {
        submission: u64,
        output_digest: ReplayDigest,
    },
    /// Persists a durable state boundary with a pre-event state digest.
    Checkpoint {
        checkpoint: u64,
        state_digest: ReplayDigest,
    },
    /// Invalidates all in-flight work for the current device generation.
    DeviceLost { device_generation: u64 },
    /// Installs exactly the next device generation after a loss.
    DeviceRebuilt { device_generation: u64 },
    /// Closes the active frame. No work may remain in flight.
    EndFrame,
}

impl ReplayOperation {
    fn kind(self) -> ReplayOperationKind {
        match self {
            Self::BeginFrame => ReplayOperationKind::BeginFrame,
            Self::Change { .. } => ReplayOperationKind::Change,
            Self::Submit { .. } => ReplayOperationKind::Submit,
            Self::Complete { .. } => ReplayOperationKind::Complete,
            Self::Checkpoint { .. } => ReplayOperationKind::Checkpoint,
            Self::DeviceLost { .. } => ReplayOperationKind::DeviceLost,
            Self::DeviceRebuilt { .. } => ReplayOperationKind::DeviceRebuilt,
            Self::EndFrame => ReplayOperationKind::EndFrame,
        }
    }
}

/// Operation labels used in rejection diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayOperationKind {
    BeginFrame,
    Change,
    Submit,
    Complete,
    Checkpoint,
    DeviceLost,
    DeviceRebuilt,
    EndFrame,
}

/// A fully framed event. Sequence numbers begin at one and cannot be reused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayEvent {
    pub sequence: u64,
    pub frame: u64,
    pub epoch: ReplayEpoch,
    pub operation: ReplayOperation,
}

impl ReplayEvent {
    /// Creates an event without applying it to a replay state.
    pub const fn new(
        sequence: u64,
        frame: u64,
        epoch: ReplayEpoch,
        operation: ReplayOperation,
    ) -> Self {
        Self {
            sequence,
            frame,
            epoch,
            operation,
        }
    }

    fn digest(self) -> ReplayDigest {
        let mut builder = DigestBuilder::new();
        builder.write_u16(REPLAY_SCHEMA_VERSION);
        builder.write_u64(self.sequence);
        builder.write_u64(self.frame);
        self.epoch.write_to(&mut builder);
        self.write_operation(&mut builder);
        builder.finish()
    }

    fn write_operation(self, builder: &mut DigestBuilder) {
        match self.operation {
            ReplayOperation::BeginFrame => builder.write_u8(0),
            ReplayOperation::Change { fields } => {
                builder.write_u8(1);
                builder.write_u8(fields.bits());
            }
            ReplayOperation::Submit {
                submission,
                device_generation,
                input_digest,
            } => {
                builder.write_u8(2);
                builder.write_u64(submission);
                builder.write_u64(device_generation);
                builder.write_framed(&input_digest.0);
            }
            ReplayOperation::Complete {
                submission,
                output_digest,
            } => {
                builder.write_u8(3);
                builder.write_u64(submission);
                builder.write_framed(&output_digest.0);
            }
            ReplayOperation::Checkpoint {
                checkpoint,
                state_digest,
            } => {
                builder.write_u8(4);
                builder.write_u64(checkpoint);
                builder.write_framed(&state_digest.0);
            }
            ReplayOperation::DeviceLost { device_generation } => {
                builder.write_u8(5);
                builder.write_u64(device_generation);
            }
            ReplayOperation::DeviceRebuilt { device_generation } => {
                builder.write_u8(6);
                builder.write_u64(device_generation);
            }
            ReplayOperation::EndFrame => builder.write_u8(7),
        }
    }
}

impl DigestBuilder {
    fn write_u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }
}

/// A submission retained until completion or device loss.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplaySubmission {
    pub submission: u64,
    pub device_generation: u64,
    pub submitted_frame: u64,
    pub input_digest: ReplayDigest,
}

/// Serializable state at a replay checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaySnapshot {
    pub next_sequence: u64,
    pub frame: Option<u64>,
    pub last_frame: Option<u64>,
    pub epoch: ReplayEpoch,
    pub device_generation: u64,
    pub device_lost: bool,
    pub next_checkpoint: u64,
    pub in_flight: Vec<ReplaySubmission>,
    pub seen_submissions: Vec<u64>,
    pub log_digest: ReplayDigest,
}

impl ReplaySnapshot {
    /// Computes a deterministic digest over every persisted field.
    #[must_use]
    pub fn digest(&self) -> ReplayDigest {
        let mut builder = DigestBuilder::new();
        builder.write_u16(REPLAY_SCHEMA_VERSION);
        builder.write_u64(self.next_sequence);
        match self.frame {
            Some(frame) => {
                builder.write_u8(1);
                builder.write_u64(frame);
            }
            None => builder.write_u8(0),
        }
        match self.last_frame {
            Some(frame) => {
                builder.write_u8(1);
                builder.write_u64(frame);
            }
            None => builder.write_u8(0),
        }
        self.epoch.write_to(&mut builder);
        builder.write_u64(self.device_generation);
        builder.write_u8(u8::from(self.device_lost));
        builder.write_u64(self.next_checkpoint);
        builder.write_u64(self.in_flight.len() as u64);
        for submission in &self.in_flight {
            builder.write_u64(submission.submission);
            builder.write_u64(submission.device_generation);
            builder.write_u64(submission.submitted_frame);
            builder.write_framed(&submission.input_digest.0);
        }
        builder.write_u64(self.seen_submissions.len() as u64);
        for submission in &self.seen_submissions {
            builder.write_u64(*submission);
        }
        builder.write_framed(&self.log_digest.0);
        builder.finish()
    }
}

/// Errors returned when a replay stream violates its state contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayError {
    SequenceOverflow,
    SequenceMismatch {
        expected: u64,
        actual: u64,
    },
    InvalidChangeMask {
        bits: u8,
    },
    EpochMismatch {
        expected: ReplayEpoch,
        actual: ReplayEpoch,
    },
    EpochOverflow,
    FrameRequired {
        operation: ReplayOperationKind,
    },
    FrameMismatch {
        expected: u64,
        actual: u64,
    },
    InvalidOrder {
        expected: ReplayOperationKind,
        actual: ReplayOperationKind,
    },
    InvalidSubmissionId,
    DuplicateSubmission {
        submission: u64,
    },
    UnknownSubmission {
        submission: u64,
    },
    StaleSubmission {
        expected_generation: u64,
        actual_generation: u64,
    },
    DeviceGenerationMismatch {
        expected: u64,
        actual: u64,
    },
    CheckpointMismatch {
        expected: ReplayDigest,
        actual: ReplayDigest,
    },
    CheckpointIdMismatch {
        expected: u64,
        actual: u64,
    },
    InFlightWork,
    SnapshotInvalid,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceOverflow => formatter.write_str("replay sequence exhausted"),
            Self::SequenceMismatch { expected, actual } => {
                write!(formatter, "replay sequence mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidChangeMask { bits } => {
                write!(formatter, "invalid replay change mask: 0x{bits:02x}")
            }
            Self::EpochMismatch { expected, actual } => {
                write!(formatter, "replay epoch mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::EpochOverflow => formatter.write_str("replay epoch overflow"),
            Self::FrameRequired { operation } => {
                write!(formatter, "{operation:?} requires an open frame")
            }
            Self::FrameMismatch { expected, actual } => {
                write!(formatter, "replay frame mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidOrder { expected, actual } => {
                write!(formatter, "invalid replay order: expected {expected:?}, got {actual:?}")
            }
            Self::InvalidSubmissionId => formatter.write_str("submission id must be non-zero"),
            Self::DuplicateSubmission { submission } => {
                write!(formatter, "duplicate replay submission {submission}")
            }
            Self::UnknownSubmission { submission } => {
                write!(formatter, "unknown replay submission {submission}")
            }
            Self::StaleSubmission {
                expected_generation,
                actual_generation,
            } => write!(
                formatter,
                "stale replay submission generation: expected {expected_generation}, got {actual_generation}"
            ),
            Self::DeviceGenerationMismatch { expected, actual } => write!(
                formatter,
                "device generation mismatch: expected {expected}, got {actual}"
            ),
            Self::CheckpointMismatch { expected, actual } => write!(
                formatter,
                "checkpoint digest mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::CheckpointIdMismatch { expected, actual } => write!(
                formatter,
                "checkpoint id mismatch: expected {expected}, got {actual}"
            ),
            Self::InFlightWork => formatter.write_str("operation requires no in-flight work"),
            Self::SnapshotInvalid => formatter.write_str("invalid replay snapshot"),
        }
    }
}

/// A sequence container that enforces contiguous event numbering.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReplayLog {
    events: Vec<ReplayEvent>,
    digest: ReplayDigest,
}

impl ReplayLog {
    /// Creates an empty log.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            events: Vec::new(),
            digest: ReplayDigest::ZERO,
        }
    }

    /// Appends an event after checking only the log-level sequence boundary.
    pub fn append(&mut self, event: ReplayEvent) -> Result<ReplayDigest, ReplayError> {
        let expected = self.events.len() as u64 + 1;
        if event.sequence != expected {
            return Err(ReplayError::SequenceMismatch {
                expected,
                actual: event.sequence,
            });
        }
        self.events.push(event);
        self.digest = chain_digest(self.digest, event.digest());
        Ok(self.digest)
    }

    /// Returns the ordered event slice.
    #[must_use]
    pub fn events(&self) -> &[ReplayEvent] {
        &self.events
    }

    /// Returns the content digest of the ordered event chain.
    #[must_use]
    pub const fn digest(&self) -> ReplayDigest {
        self.digest
    }

    /// Replays the complete log from the initial state.
    pub fn replay(&self) -> Result<ReplayEngine, ReplayError> {
        ReplayEngine::replay(&self.events)
    }
}

/// Deterministic replay state machine for lighting/GI events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayEngine {
    next_sequence: u64,
    frame: Option<u64>,
    last_frame: Option<u64>,
    epoch: ReplayEpoch,
    device_generation: u64,
    device_lost: bool,
    next_checkpoint: u64,
    in_flight: Vec<ReplaySubmission>,
    seen_submissions: Vec<u64>,
    log_digest: ReplayDigest,
}

impl Default for ReplayEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayEngine {
    /// Creates an empty replay state at frame zero's predecessor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_sequence: 1,
            frame: None,
            last_frame: None,
            epoch: ReplayEpoch::ZERO,
            device_generation: 0,
            device_lost: false,
            next_checkpoint: 1,
            in_flight: Vec::new(),
            seen_submissions: Vec::new(),
            log_digest: ReplayDigest::ZERO,
        }
    }

    /// Replays ordered events from the initial state.
    pub fn replay(events: &[ReplayEvent]) -> Result<Self, ReplayError> {
        let mut engine = Self::new();
        for event in events {
            engine.apply(*event)?;
        }
        Ok(engine)
    }

    /// Restores a validated checkpoint state.
    pub fn restore(snapshot: ReplaySnapshot) -> Result<Self, ReplayError> {
        validate_snapshot(&snapshot)?;
        Ok(Self {
            next_sequence: snapshot.next_sequence,
            frame: snapshot.frame,
            last_frame: snapshot.last_frame,
            epoch: snapshot.epoch,
            device_generation: snapshot.device_generation,
            device_lost: snapshot.device_lost,
            next_checkpoint: snapshot.next_checkpoint,
            in_flight: snapshot.in_flight,
            seen_submissions: snapshot.seen_submissions,
            log_digest: snapshot.log_digest,
        })
    }

    /// Applies one event and returns the new chain digest.
    pub fn apply(&mut self, event: ReplayEvent) -> Result<ReplayDigest, ReplayError> {
        if self.next_sequence == u64::MAX {
            return Err(ReplayError::SequenceOverflow);
        }
        if event.sequence != self.next_sequence {
            return Err(ReplayError::SequenceMismatch {
                expected: self.next_sequence,
                actual: event.sequence,
            });
        }

        match event.operation {
            ReplayOperation::BeginFrame => self.begin_frame(event)?,
            ReplayOperation::Change { fields } => self.change(event, fields)?,
            ReplayOperation::Submit {
                submission,
                device_generation,
                input_digest,
            } => self.submit(event, submission, device_generation, input_digest)?,
            ReplayOperation::Complete {
                submission,
                output_digest: _,
            } => self.complete(event, submission)?,
            ReplayOperation::Checkpoint {
                checkpoint,
                state_digest,
            } => self.checkpoint(event, checkpoint, state_digest)?,
            ReplayOperation::DeviceLost { device_generation } => {
                self.device_lost(event, device_generation)?
            }
            ReplayOperation::DeviceRebuilt { device_generation } => {
                self.device_rebuilt(event, device_generation)?
            }
            ReplayOperation::EndFrame => self.end_frame(event)?,
        }

        self.next_sequence += 1;
        self.log_digest = chain_digest(self.log_digest, event.digest());
        Ok(self.log_digest)
    }

    /// Captures all state needed for deterministic continuation.
    #[must_use]
    pub fn snapshot(&self) -> ReplaySnapshot {
        ReplaySnapshot {
            next_sequence: self.next_sequence,
            frame: self.frame,
            last_frame: self.last_frame,
            epoch: self.epoch,
            device_generation: self.device_generation,
            device_lost: self.device_lost,
            next_checkpoint: self.next_checkpoint,
            in_flight: self.in_flight.clone(),
            seen_submissions: self.seen_submissions.clone(),
            log_digest: self.log_digest,
        }
    }

    #[must_use]
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    #[must_use]
    pub const fn epoch(&self) -> ReplayEpoch {
        self.epoch
    }

    #[must_use]
    pub const fn device_generation(&self) -> u64 {
        self.device_generation
    }

    #[must_use]
    pub const fn log_digest(&self) -> ReplayDigest {
        self.log_digest
    }

    fn begin_frame(&mut self, event: ReplayEvent) -> Result<(), ReplayError> {
        if self.frame.is_some() {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::EndFrame,
                actual: event.operation.kind(),
            });
        }
        if event.epoch != self.epoch {
            return Err(ReplayError::EpochMismatch {
                expected: self.epoch,
                actual: event.epoch,
            });
        }
        if self
            .last_frame
            .is_some_and(|last_frame| event.frame <= last_frame)
        {
            return Err(ReplayError::FrameMismatch {
                expected: self.last_frame.unwrap().saturating_add(1),
                actual: event.frame,
            });
        }
        self.frame = Some(event.frame);
        Ok(())
    }

    fn change(&mut self, event: ReplayEvent, fields: ReplayChangeMask) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        if self.device_lost {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceRebuilt,
                actual: event.operation.kind(),
            });
        }
        if fields.is_empty() {
            return Err(ReplayError::InvalidChangeMask { bits: 0 });
        }
        let expected = self.epoch.checked_advance(fields)?;
        if event.epoch != expected {
            return Err(ReplayError::EpochMismatch {
                expected,
                actual: event.epoch,
            });
        }
        self.epoch = expected;
        Ok(())
    }

    fn submit(
        &mut self,
        event: ReplayEvent,
        submission: u64,
        device_generation: u64,
        input_digest: ReplayDigest,
    ) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_live_device(event)?;
        self.require_epoch(event)?;
        if submission == 0 {
            return Err(ReplayError::InvalidSubmissionId);
        }
        if self.seen_submissions.contains(&submission) {
            return Err(ReplayError::DuplicateSubmission { submission });
        }
        self.require_generation(device_generation)?;
        self.seen_submissions.push(submission);
        self.in_flight.push(ReplaySubmission {
            submission,
            device_generation,
            submitted_frame: event.frame,
            input_digest,
        });
        Ok(())
    }

    fn complete(&mut self, event: ReplayEvent, submission: u64) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_live_device(event)?;
        self.require_epoch(event)?;
        let index = self
            .in_flight
            .iter()
            .position(|item| item.submission == submission)
            .ok_or(ReplayError::UnknownSubmission { submission })?;
        let item = self.in_flight[index];
        self.require_generation(item.device_generation)?;
        self.in_flight.swap_remove(index);
        Ok(())
    }

    fn checkpoint(
        &mut self,
        event: ReplayEvent,
        checkpoint: u64,
        state_digest: ReplayDigest,
    ) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_live_device(event)?;
        self.require_epoch(event)?;
        if !self.in_flight.is_empty() {
            return Err(ReplayError::InFlightWork);
        }
        if checkpoint != self.next_checkpoint {
            return Err(ReplayError::CheckpointIdMismatch {
                expected: self.next_checkpoint,
                actual: checkpoint,
            });
        }
        let expected = self.snapshot().digest();
        if state_digest != expected {
            return Err(ReplayError::CheckpointMismatch {
                expected,
                actual: state_digest,
            });
        }
        self.next_checkpoint =
            self.next_checkpoint
                .checked_add(1)
                .ok_or(ReplayError::CheckpointIdMismatch {
                    expected: u64::MAX,
                    actual: checkpoint,
                })?;
        Ok(())
    }

    fn device_lost(
        &mut self,
        event: ReplayEvent,
        device_generation: u64,
    ) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_epoch(event)?;
        if self.device_lost {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceRebuilt,
                actual: event.operation.kind(),
            });
        }
        self.require_generation(device_generation)?;
        self.device_lost = true;
        self.in_flight.clear();
        Ok(())
    }

    fn device_rebuilt(
        &mut self,
        event: ReplayEvent,
        device_generation: u64,
    ) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_epoch(event)?;
        if !self.device_lost {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceLost,
                actual: event.operation.kind(),
            });
        }
        let expected =
            self.device_generation
                .checked_add(1)
                .ok_or(ReplayError::DeviceGenerationMismatch {
                    expected: u64::MAX,
                    actual: device_generation,
                })?;
        if device_generation != expected {
            return Err(ReplayError::DeviceGenerationMismatch {
                expected,
                actual: device_generation,
            });
        }
        self.device_generation = device_generation;
        self.device_lost = false;
        Ok(())
    }

    fn end_frame(&mut self, event: ReplayEvent) -> Result<(), ReplayError> {
        self.require_frame(event)?;
        self.require_epoch(event)?;
        if self.device_lost {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceRebuilt,
                actual: event.operation.kind(),
            });
        }
        if !self.in_flight.is_empty() {
            return Err(ReplayError::InFlightWork);
        }
        self.last_frame = Some(event.frame);
        self.frame = None;
        Ok(())
    }

    fn require_frame(&self, event: ReplayEvent) -> Result<u64, ReplayError> {
        let frame = self.frame.ok_or(ReplayError::FrameRequired {
            operation: event.operation.kind(),
        })?;
        if event.frame != frame {
            return Err(ReplayError::FrameMismatch {
                expected: frame,
                actual: event.frame,
            });
        }
        Ok(frame)
    }

    fn require_epoch(&self, event: ReplayEvent) -> Result<(), ReplayError> {
        if event.epoch != self.epoch {
            return Err(ReplayError::EpochMismatch {
                expected: self.epoch,
                actual: event.epoch,
            });
        }
        Ok(())
    }

    fn require_live_device(&self, event: ReplayEvent) -> Result<(), ReplayError> {
        if self.device_lost {
            return Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceRebuilt,
                actual: event.operation.kind(),
            });
        }
        Ok(())
    }

    fn require_generation(&self, actual: u64) -> Result<(), ReplayError> {
        if actual != self.device_generation {
            return Err(ReplayError::DeviceGenerationMismatch {
                expected: self.device_generation,
                actual,
            });
        }
        Ok(())
    }
}

fn validate_snapshot(snapshot: &ReplaySnapshot) -> Result<(), ReplayError> {
    if snapshot.next_sequence == 0
        || snapshot.next_checkpoint == 0
        || snapshot.device_lost && !snapshot.in_flight.is_empty()
        || snapshot.device_lost && snapshot.frame.is_none()
        || snapshot.frame.is_none() && !snapshot.in_flight.is_empty()
        || snapshot.frame.is_some_and(|frame| {
            snapshot
                .last_frame
                .is_some_and(|last_frame| frame <= last_frame)
        })
    {
        return Err(ReplayError::SnapshotInvalid);
    }
    for (index, item) in snapshot.in_flight.iter().enumerate() {
        if item.submission == 0 || item.device_generation != snapshot.device_generation {
            return Err(ReplayError::SnapshotInvalid);
        }
        if !snapshot.seen_submissions.contains(&item.submission) {
            return Err(ReplayError::SnapshotInvalid);
        }
        if snapshot.frame != Some(item.submitted_frame) {
            return Err(ReplayError::SnapshotInvalid);
        }
        if snapshot.in_flight[..index]
            .iter()
            .any(|previous| previous.submission == item.submission)
        {
            return Err(ReplayError::SnapshotInvalid);
        }
    }
    for (index, submission) in snapshot.seen_submissions.iter().enumerate() {
        if *submission == 0
            || snapshot.seen_submissions[..index]
                .iter()
                .any(|previous| previous == submission)
        {
            return Err(ReplayError::SnapshotInvalid);
        }
    }
    Ok(())
}

fn chain_digest(previous: ReplayDigest, event: ReplayDigest) -> ReplayDigest {
    ReplayDigest::combine(&[&previous.0, &event.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        sequence: u64,
        frame: u64,
        epoch: ReplayEpoch,
        operation: ReplayOperation,
    ) -> ReplayEvent {
        ReplayEvent::new(sequence, frame, epoch, operation)
    }

    fn digest(byte: u8) -> ReplayDigest {
        ReplayDigest::from_bytes(&[byte])
    }

    fn begin(sequence: u64, frame: u64, epoch: ReplayEpoch) -> ReplayEvent {
        event(sequence, frame, epoch, ReplayOperation::BeginFrame)
    }

    fn end(sequence: u64, frame: u64, epoch: ReplayEpoch) -> ReplayEvent {
        event(sequence, frame, epoch, ReplayOperation::EndFrame)
    }

    #[test]
    fn deterministic_frame_change_submit_complete_checkpoint_and_replay() {
        let mut engine = ReplayEngine::new();
        let initial = ReplayEpoch::ZERO;
        let changed = ReplayEpoch {
            geometry: 1,
            ..initial
        };
        let events = [
            begin(1, 0, initial),
            event(
                2,
                0,
                changed,
                ReplayOperation::Change {
                    fields: ReplayChangeMask::GEOMETRY,
                },
            ),
            event(
                3,
                0,
                changed,
                ReplayOperation::Submit {
                    submission: 7,
                    device_generation: 0,
                    input_digest: digest(1),
                },
            ),
            event(
                4,
                0,
                changed,
                ReplayOperation::Complete {
                    submission: 7,
                    output_digest: digest(2),
                },
            ),
        ];
        for item in events {
            engine.apply(item).expect("valid event");
        }
        let checkpoint_digest = engine.snapshot().digest();
        let checkpoint = event(
            5,
            0,
            changed,
            ReplayOperation::Checkpoint {
                checkpoint: 1,
                state_digest: checkpoint_digest,
            },
        );
        engine.apply(checkpoint).expect("valid checkpoint");
        engine.apply(end(6, 0, changed)).expect("valid end");

        let mut log = ReplayLog::new();
        for item in events.into_iter().chain([checkpoint, end(6, 0, changed)]) {
            log.append(item).expect("contiguous log");
        }
        let replayed = log.replay().expect("valid replay");
        assert_eq!(engine.snapshot(), replayed.snapshot());
        assert_eq!(engine.log_digest(), replayed.log_digest());
        assert_eq!(engine.log_digest(), log.digest());
    }

    #[test]
    fn rejects_sequence_frame_epoch_and_order_violations() {
        let mut engine = ReplayEngine::new();
        assert_eq!(
            engine.apply(begin(2, 0, ReplayEpoch::ZERO)),
            Err(ReplayError::SequenceMismatch {
                expected: 1,
                actual: 2
            })
        );
        engine
            .apply(begin(1, 0, ReplayEpoch::ZERO))
            .expect("begin frame");
        assert_eq!(
            engine.apply(event(2, 1, ReplayEpoch::ZERO, ReplayOperation::EndFrame,)),
            Err(ReplayError::FrameMismatch {
                expected: 0,
                actual: 1
            })
        );
        assert_eq!(
            engine.apply(event(
                2,
                0,
                ReplayEpoch {
                    lighting: 1,
                    ..ReplayEpoch::ZERO
                },
                ReplayOperation::EndFrame,
            )),
            Err(ReplayError::EpochMismatch {
                expected: ReplayEpoch::ZERO,
                actual: ReplayEpoch {
                    lighting: 1,
                    ..ReplayEpoch::ZERO
                }
            })
        );
        assert_eq!(
            engine.apply(event(
                2,
                0,
                ReplayEpoch::ZERO,
                ReplayOperation::Complete {
                    submission: 1,
                    output_digest: digest(3),
                },
            )),
            Err(ReplayError::UnknownSubmission { submission: 1 })
        );
    }

    #[test]
    fn device_rebuild_drops_old_work_and_requires_new_generation() {
        let mut engine = ReplayEngine::new();
        let zero = ReplayEpoch::ZERO;
        engine.apply(begin(1, 4, zero)).expect("begin");
        engine
            .apply(event(
                2,
                4,
                zero,
                ReplayOperation::Submit {
                    submission: 11,
                    device_generation: 0,
                    input_digest: digest(4),
                },
            ))
            .expect("submit");
        engine
            .apply(event(
                3,
                4,
                zero,
                ReplayOperation::DeviceLost {
                    device_generation: 0,
                },
            ))
            .expect("loss");
        assert_eq!(
            engine.apply(event(
                4,
                4,
                zero,
                ReplayOperation::Complete {
                    submission: 11,
                    output_digest: digest(5),
                },
            )),
            Err(ReplayError::InvalidOrder {
                expected: ReplayOperationKind::DeviceRebuilt,
                actual: ReplayOperationKind::Complete
            })
        );
        engine
            .apply(event(
                4,
                4,
                zero,
                ReplayOperation::DeviceRebuilt {
                    device_generation: 1,
                },
            ))
            .expect("rebuild");
        engine
            .apply(event(
                5,
                4,
                zero,
                ReplayOperation::Submit {
                    submission: 12,
                    device_generation: 1,
                    input_digest: digest(6),
                },
            ))
            .expect("resubmit on rebuilt device");
        assert_eq!(engine.device_generation(), 1);
    }

    #[test]
    fn snapshot_restore_continues_bit_identically() {
        let zero = ReplayEpoch::ZERO;
        let mut original = ReplayEngine::new();
        original.apply(begin(1, 8, zero)).expect("begin");
        original
            .apply(event(
                2,
                8,
                zero,
                ReplayOperation::Submit {
                    submission: 21,
                    device_generation: 0,
                    input_digest: digest(7),
                },
            ))
            .expect("submit");
        let snapshot = original.snapshot();
        let mut restored = ReplayEngine::restore(snapshot.clone()).expect("restore");
        assert_eq!(snapshot.digest(), restored.snapshot().digest());

        let tail = [
            event(
                3,
                8,
                zero,
                ReplayOperation::Complete {
                    submission: 21,
                    output_digest: digest(8),
                },
            ),
            end(4, 8, zero),
        ];
        for item in tail {
            original.apply(item).expect("original continuation");
            restored.apply(item).expect("restored continuation");
        }
        assert_eq!(original.snapshot(), restored.snapshot());
        assert_eq!(original.log_digest(), restored.log_digest());
    }

    #[test]
    fn snapshot_validation_rejects_impossible_states() {
        let mut snapshot = ReplayEngine::new().snapshot();
        snapshot.next_sequence = 0;
        assert_eq!(
            ReplayEngine::restore(snapshot),
            Err(ReplayError::SnapshotInvalid)
        );

        let mut snapshot = ReplayEngine::new().snapshot();
        snapshot.frame = Some(0);
        snapshot.device_lost = true;
        snapshot.in_flight.push(ReplaySubmission {
            submission: 1,
            device_generation: 0,
            submitted_frame: 0,
            input_digest: digest(9),
        });
        assert_eq!(
            ReplayEngine::restore(snapshot),
            Err(ReplayError::SnapshotInvalid)
        );
    }

    #[test]
    fn digest_is_stable_framed_and_sensitive_to_event_content() {
        let zero = ReplayEpoch::ZERO;
        let first = begin(1, 0, zero);
        let second = begin(1, 1, zero);
        assert_eq!(first.digest(), first.digest());
        assert_ne!(first.digest(), second.digest());
        assert_ne!(
            ReplayDigest::combine(&[b"ab", b"c"]),
            ReplayDigest::combine(&[b"a", b"bc"])
        );

        let mut left = ReplayLog::new();
        let mut right = ReplayLog::new();
        left.append(first).expect("first");
        right.append(first).expect("first");
        assert_eq!(left.digest(), right.digest());
        right.append(end(2, 0, zero)).expect("second event");
        assert_ne!(left.digest(), right.digest());
    }

    #[test]
    fn boundary_masks_and_epoch_overflow_are_rejected() {
        assert_eq!(
            ReplayChangeMask::from_bits(0x80),
            Err(ReplayError::InvalidChangeMask { bits: 0x80 })
        );
        let max = ReplayEpoch {
            geometry: u32::MAX,
            ..ReplayEpoch::ZERO
        };
        let mut engine = ReplayEngine::new();
        engine.epoch = max;
        engine.apply(begin(1, 0, max)).expect("begin max epoch");
        assert_eq!(
            engine.apply(event(
                2,
                0,
                max,
                ReplayOperation::Change {
                    fields: ReplayChangeMask::GEOMETRY,
                },
            )),
            Err(ReplayError::EpochOverflow)
        );
        assert_eq!(
            ReplayChangeMask::from_bits(ReplayChangeMask::VALID_BITS),
            Ok(ReplayChangeMask(ReplayChangeMask::VALID_BITS)),
            "the raw mask is validated independently of field composition"
        );
    }

    #[test]
    fn sequence_exhaustion_is_atomic_and_snapshot_rejects_loss_without_frame() {
        let mut engine = ReplayEngine::new();
        engine.next_sequence = u64::MAX;
        let before = engine.snapshot();
        assert_eq!(
            engine.apply(begin(u64::MAX, 0, ReplayEpoch::ZERO)),
            Err(ReplayError::SequenceOverflow)
        );
        assert_eq!(engine.snapshot(), before);

        let mut snapshot = ReplayEngine::new().snapshot();
        snapshot.device_lost = true;
        assert_eq!(
            ReplayEngine::restore(snapshot),
            Err(ReplayError::SnapshotInvalid)
        );

        let mut snapshot = ReplayEngine::new().snapshot();
        snapshot.frame = Some(3);
        snapshot.in_flight.push(ReplaySubmission {
            submission: 1,
            device_generation: 0,
            submitted_frame: 2,
            input_digest: digest(10),
        });
        snapshot.seen_submissions.push(1);
        assert_eq!(
            ReplayEngine::restore(snapshot),
            Err(ReplayError::SnapshotInvalid)
        );
    }
}
