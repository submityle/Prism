//! Host-neutral replication, prediction, and rollback contracts for StateTree.
//!
//! This module only describes what a host may replicate and what a rollback
//! path may accept. It does not serialize values, perform transport, or apply
//! authoritative state. The host remains responsible for those operations.

use alloc::vec::Vec;

use crate::{PlanRevision, ReplaySafety};

/// Maximum number of rollback inputs retained by one contract.
pub const MAX_ROLLBACK_INPUTS: usize = 256;

/// Maximum aggregate payload bytes retained by one rollback contract.
pub const MAX_ROLLBACK_INPUT_BYTES: usize = 64 * 1024;

/// Maximum payload bytes accepted for one rollback input.
pub const MAX_ROLLBACK_INPUT_SIZE: usize = 4096;

/// The StateTree replication policy attached to a node or node-owned value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReplicationPolicy {
    /// The node is evaluated by the authority and is not sent as a prediction
    /// input or replicated state contract.
    ServerOnly,
    /// The authority replicates state snapshots or state deltas for the node.
    ReplicatedState,
    /// The authority replicates ordered transition/event information for the
    /// node.
    ReplicatedEvent,
    /// The node exists only in the local host and is outside reconciliation.
    ClientLocal,
}

/// Backward-compatible descriptive name for [`ReplicationPolicy`].
pub type StateTreeReplicationPolicy = ReplicationPolicy;

impl ReplicationPolicy {
    /// Returns whether the policy participates in network replication.
    #[must_use]
    pub const fn is_replicated(self) -> bool {
        matches!(self, Self::ReplicatedState | Self::ReplicatedEvent)
    }

    /// Returns whether the policy carries state rather than ordered events.
    #[must_use]
    pub const fn is_state_based(self) -> bool {
        matches!(self, Self::ReplicatedState)
    }

    /// Returns whether the policy carries ordered events rather than state.
    #[must_use]
    pub const fn is_event_based(self) -> bool {
        matches!(self, Self::ReplicatedEvent)
    }

    /// Judges whether this policy can enter a predicted/reconciled path.
    #[must_use]
    pub const fn prediction_safety(
        self,
        authority: AuthorityMode,
        replay_safety: ReplaySafety,
        handshake: PlanRevisionHandshakeStatus,
    ) -> PredictionSafety {
        if matches!(self, Self::ClientLocal) {
            return PredictionSafety::LocalOnly;
        }
        if matches!(self, Self::ServerOnly) {
            return PredictionSafety::Unsafe(PredictionSafetyReason::ServerOnlyPolicy);
        }
        if !matches!(handshake, PlanRevisionHandshakeStatus::Exact) {
            return PredictionSafety::Unsafe(PredictionSafetyReason::PlanRevisionMismatch);
        }
        if !replay_safety.allows_prediction() {
            return PredictionSafety::Unsafe(PredictionSafetyReason::AuthorityOnlyReplay);
        }
        match authority {
            AuthorityMode::Server | AuthorityMode::Client | AuthorityMode::Shared => {
                PredictionSafety::Predictable
            }
        }
    }

    /// Verifies prediction safety and returns a structured contract error on
    /// failure.
    pub const fn ensure_prediction_safe(
        self,
        authority: AuthorityMode,
        replay_safety: ReplaySafety,
        handshake: PlanRevisionHandshakeStatus,
    ) -> Result<(), ReplicationContractError> {
        match self.prediction_safety(authority, replay_safety, handshake) {
            PredictionSafety::Predictable | PredictionSafety::LocalOnly => Ok(()),
            PredictionSafety::Unsafe(reason) => {
                Err(ReplicationContractError::PredictionUnsafe(reason))
            }
        }
    }
}

/// The authority that owns the semantic result for a StateTree node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AuthorityMode {
    /// A server or dedicated authority owns the semantic result.
    Server,
    /// The local client owns the semantic result for this node.
    Client,
    /// Both peers run the same deterministic contract and reconcile by input
    /// or state agreement.
    Shared,
}

impl AuthorityMode {
    /// Upper-case alias for [`AuthorityMode::Server`].
    #[allow(non_upper_case_globals)]
    pub const SERVER: Self = Self::Server;
    /// Upper-case alias for [`AuthorityMode::Client`].
    #[allow(non_upper_case_globals)]
    pub const CLIENT: Self = Self::Client;
    /// Upper-case alias for [`AuthorityMode::Shared`].
    #[allow(non_upper_case_globals)]
    pub const SHARED: Self = Self::Shared;

    /// Returns whether the local host may authoritatively commit the node.
    #[must_use]
    pub const fn local_can_commit(self) -> bool {
        matches!(self, Self::Client | Self::Shared)
    }
}

/// The mismatch action used by a plan-revision handshake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanRevisionHandshakePolicy {
    /// Refuse peers whose plan revisions differ.
    Reject,
    /// Keep the connection usable but force the mismatched tree to
    /// server-only behavior.
    DowngradeToServerOnly,
}

/// Result of comparing two plan revisions during a handshake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanRevisionHandshakeStatus {
    /// Both peers use the same semantic plan revision.
    Exact,
    /// The peers differ and replication was explicitly downgraded.
    DowngradedToServerOnly,
}

/// Structured failure from a plan-revision handshake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanRevisionHandshakeError {
    /// The peers advertised different semantic plan revisions.
    RevisionMismatch {
        /// The local plan revision.
        local: PlanRevision,
        /// The remote plan revision.
        remote: PlanRevision,
    },
}

/// A completed, transport-independent plan-revision handshake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanRevisionHandshake {
    local: PlanRevision,
    remote: PlanRevision,
    status: PlanRevisionHandshakeStatus,
}

impl PlanRevisionHandshake {
    /// Compares local and remote revisions using the supplied mismatch policy.
    pub const fn negotiate(
        local: PlanRevision,
        remote: PlanRevision,
        policy: PlanRevisionHandshakePolicy,
    ) -> Result<Self, PlanRevisionHandshakeError> {
        if local.fingerprint() == remote.fingerprint() {
            return Ok(Self {
                local,
                remote,
                status: PlanRevisionHandshakeStatus::Exact,
            });
        }
        match policy {
            PlanRevisionHandshakePolicy::Reject => {
                Err(PlanRevisionHandshakeError::RevisionMismatch { local, remote })
            }
            PlanRevisionHandshakePolicy::DowngradeToServerOnly => Ok(Self {
                local,
                remote,
                status: PlanRevisionHandshakeStatus::DowngradedToServerOnly,
            }),
        }
    }

    /// Returns the local revision advertised by the handshake.
    #[must_use]
    pub const fn local(&self) -> PlanRevision {
        self.local
    }

    /// Returns the remote revision advertised by the handshake.
    #[must_use]
    pub const fn remote(&self) -> PlanRevision {
        self.remote
    }

    /// Returns the negotiated status.
    #[must_use]
    pub const fn status(&self) -> PlanRevisionHandshakeStatus {
        self.status
    }

    /// Returns whether the handshake permits the original replication policy.
    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self.status, PlanRevisionHandshakeStatus::Exact)
    }

    /// Converts a policy to the effective policy after this handshake.
    #[must_use]
    pub const fn effective_policy(&self, requested: ReplicationPolicy) -> ReplicationPolicy {
        match self.status {
            PlanRevisionHandshakeStatus::Exact => requested,
            PlanRevisionHandshakeStatus::DowngradedToServerOnly => ReplicationPolicy::ServerOnly,
        }
    }
}

/// The result of the prediction-safety judgment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictionSafety {
    /// The node may use prediction and reconciliation.
    Predictable,
    /// The node is safe to evaluate locally but is not part of network
    /// reconciliation.
    LocalOnly,
    /// The node must not enter a prediction path.
    Unsafe(PredictionSafetyReason),
}

impl PredictionSafety {
    /// Returns whether the judgment permits local evaluation.
    #[must_use]
    pub const fn is_safe(self) -> bool {
        !matches!(self, Self::Unsafe(_))
    }

    /// Returns whether the judgment permits network reconciliation.
    #[must_use]
    pub const fn is_reconciled(self) -> bool {
        matches!(self, Self::Predictable)
    }

    /// Returns the denial reason, if prediction is unsafe.
    #[must_use]
    pub const fn reason(self) -> Option<PredictionSafetyReason> {
        match self {
            Self::Unsafe(reason) => Some(reason),
            Self::Predictable | Self::LocalOnly => None,
        }
    }
}

/// Structured reason why a node cannot use network prediction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictionSafetyReason {
    /// The node is explicitly server-only.
    ServerOnlyPolicy,
    /// The node requires an authority-owned or transient replay result.
    AuthorityOnlyReplay,
    /// The peers did not complete an exact plan-revision handshake.
    PlanRevisionMismatch,
}

/// A top-level error for callers that combine the replication contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicationContractError {
    /// A plan-revision handshake failed.
    Handshake(PlanRevisionHandshakeError),
    /// A rollback input violated its bounds or ordering contract.
    RollbackInput(RollbackInputError),
    /// Prediction was rejected by the policy, authority, replay, or handshake
    /// contract.
    PredictionUnsafe(PredictionSafetyReason),
}

impl From<PlanRevisionHandshakeError> for ReplicationContractError {
    fn from(error: PlanRevisionHandshakeError) -> Self {
        Self::Handshake(error)
    }
}

impl From<RollbackInputError> for ReplicationContractError {
    fn from(error: RollbackInputError) -> Self {
        Self::RollbackInput(error)
    }
}

/// One owned, opaque input in a bounded rollback timeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollbackInput {
    frame: u64,
    sequence: u32,
    payload: Vec<u8>,
}

impl RollbackInput {
    /// Creates the first input for a frame.
    pub fn new(frame: u64, payload: &[u8]) -> Result<Self, RollbackInputError> {
        Self::with_sequence(frame, 0, payload)
    }

    /// Creates an input with an explicit deterministic order within its frame.
    pub fn with_sequence(
        frame: u64,
        sequence: u32,
        payload: &[u8],
    ) -> Result<Self, RollbackInputError> {
        if payload.len() > MAX_ROLLBACK_INPUT_SIZE {
            return Err(RollbackInputError::InputSizeLimit {
                requested: payload.len(),
                maximum: MAX_ROLLBACK_INPUT_SIZE,
            });
        }
        Ok(Self {
            frame,
            sequence,
            payload: payload.to_vec(),
        })
    }

    /// Returns the simulation frame to which this input belongs.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Returns the deterministic order within the frame.
    #[must_use]
    pub const fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Returns the opaque input payload.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

/// Structured failure from a bounded rollback input contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RollbackInputError {
    /// The configured input capacity is zero.
    ZeroCapacity,
    /// The configured input capacity exceeds the hard contract bound.
    CapacityLimit {
        /// Requested input capacity.
        requested: usize,
        /// Maximum supported input capacity.
        maximum: usize,
    },
    /// The configured aggregate payload budget exceeds the hard bound.
    ByteCapacityLimit {
        /// Requested aggregate byte capacity.
        requested: usize,
        /// Maximum supported aggregate byte capacity.
        maximum: usize,
    },
    /// One input payload exceeds the contract's per-input limit.
    InputSizeLimit {
        /// Requested payload bytes.
        requested: usize,
        /// Maximum payload bytes.
        maximum: usize,
    },
    /// An input belongs to a frame newer than the contract cursor.
    FutureFrame {
        /// Current simulation frame.
        current: u64,
        /// Input frame.
        found: u64,
    },
    /// An input is older than the retained rollback window.
    OutsideWindow {
        /// Current simulation frame.
        current: u64,
        /// Input frame.
        found: u64,
        /// Maximum rollback distance.
        maximum: u64,
    },
    /// The same frame and sequence was already retained.
    Duplicate {
        /// Duplicate input frame.
        frame: u64,
        /// Duplicate input sequence.
        sequence: u32,
    },
    /// The input count budget is full.
    InputCapacityExceeded {
        /// Configured input capacity.
        capacity: usize,
    },
    /// Adding the input would exceed the aggregate byte budget.
    ByteCapacityExceeded {
        /// Configured aggregate byte capacity.
        capacity: usize,
    },
    /// The rollback cursor may only move forward.
    FrameRewind {
        /// Current simulation frame.
        current: u64,
        /// Requested frame.
        requested: u64,
    },
}

/// Bounded, ordered rollback inputs for one StateTree instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RollbackInputContract {
    current_frame: u64,
    max_rollback_frames: u64,
    capacity: usize,
    max_input_bytes: usize,
    input_bytes: usize,
    inputs: Vec<RollbackInput>,
}

impl RollbackInputContract {
    /// Creates an empty bounded rollback input contract.
    pub fn new(
        current_frame: u64,
        max_rollback_frames: u64,
        capacity: usize,
        max_input_bytes: usize,
    ) -> Result<Self, RollbackInputError> {
        if capacity == 0 {
            return Err(RollbackInputError::ZeroCapacity);
        }
        if capacity > MAX_ROLLBACK_INPUTS {
            return Err(RollbackInputError::CapacityLimit {
                requested: capacity,
                maximum: MAX_ROLLBACK_INPUTS,
            });
        }
        if max_input_bytes > MAX_ROLLBACK_INPUT_BYTES {
            return Err(RollbackInputError::ByteCapacityLimit {
                requested: max_input_bytes,
                maximum: MAX_ROLLBACK_INPUT_BYTES,
            });
        }
        Ok(Self {
            current_frame,
            max_rollback_frames,
            capacity,
            max_input_bytes,
            input_bytes: 0,
            inputs: Vec::with_capacity(capacity),
        })
    }

    /// Returns the current simulation frame.
    #[must_use]
    pub const fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Returns the maximum accepted rollback distance in frames.
    #[must_use]
    pub const fn max_rollback_frames(&self) -> u64 {
        self.max_rollback_frames
    }

    /// Returns the maximum number of retained inputs.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the aggregate payload budget.
    #[must_use]
    pub const fn max_input_bytes(&self) -> usize {
        self.max_input_bytes
    }

    /// Returns the aggregate payload bytes currently retained.
    #[must_use]
    pub const fn input_bytes(&self) -> usize {
        self.input_bytes
    }

    /// Returns retained inputs ordered by frame and then sequence.
    #[must_use]
    pub fn inputs(&self) -> &[RollbackInput] {
        &self.inputs
    }

    /// Advances the simulation cursor and evicts inputs outside the window.
    pub fn advance_to(&mut self, frame: u64) -> Result<(), RollbackInputError> {
        if frame < self.current_frame {
            return Err(RollbackInputError::FrameRewind {
                current: self.current_frame,
                requested: frame,
            });
        }
        self.current_frame = frame;
        let oldest = frame.saturating_sub(self.max_rollback_frames);
        self.discard_before(oldest);
        Ok(())
    }

    /// Inserts one input while preserving deterministic order and all bounds.
    pub fn insert(&mut self, input: RollbackInput) -> Result<(), RollbackInputError> {
        if input.frame > self.current_frame {
            return Err(RollbackInputError::FutureFrame {
                current: self.current_frame,
                found: input.frame,
            });
        }
        if self.current_frame.saturating_sub(input.frame) > self.max_rollback_frames {
            return Err(RollbackInputError::OutsideWindow {
                current: self.current_frame,
                found: input.frame,
                maximum: self.max_rollback_frames,
            });
        }
        if input.payload.len() > self.max_input_bytes {
            return Err(RollbackInputError::InputSizeLimit {
                requested: input.payload.len(),
                maximum: self.max_input_bytes,
            });
        }
        if self
            .inputs
            .iter()
            .any(|existing| existing.frame == input.frame && existing.sequence == input.sequence)
        {
            return Err(RollbackInputError::Duplicate {
                frame: input.frame,
                sequence: input.sequence,
            });
        }
        if self.inputs.len() >= self.capacity {
            return Err(RollbackInputError::InputCapacityExceeded {
                capacity: self.capacity,
            });
        }
        let next_bytes = self.input_bytes.saturating_add(input.payload.len());
        if next_bytes > self.max_input_bytes {
            return Err(RollbackInputError::ByteCapacityExceeded {
                capacity: self.max_input_bytes,
            });
        }

        let position = self
            .inputs
            .iter()
            .position(|existing| {
                (input.frame, input.sequence) < (existing.frame, existing.sequence)
            })
            .unwrap_or(self.inputs.len());
        self.inputs.insert(position, input);
        self.input_bytes = next_bytes;
        Ok(())
    }

    /// Alias for [`RollbackInputContract::insert`].
    pub fn push(&mut self, input: RollbackInput) -> Result<(), RollbackInputError> {
        self.insert(input)
    }

    /// Returns an input by its frame and sequence, if retained.
    #[must_use]
    pub fn get(&self, frame: u64, sequence: u32) -> Option<&RollbackInput> {
        self.inputs
            .iter()
            .find(|input| input.frame == frame && input.sequence == sequence)
    }

    /// Drops all retained inputs before `frame`.
    pub fn discard_before(&mut self, frame: u64) {
        let split = self
            .inputs
            .iter()
            .position(|input| input.frame >= frame)
            .unwrap_or(self.inputs.len());
        if split == 0 {
            return;
        }
        let removed_bytes: usize = self.inputs[..split]
            .iter()
            .map(|input| input.payload.len())
            .sum();
        self.inputs.drain(..split);
        self.input_bytes -= removed_bytes;
    }

    /// Removes every retained input.
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.input_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn revision(value: u128) -> PlanRevision {
        PlanRevision::new(value)
    }

    #[test]
    fn policy_reuses_replay_safety_and_handshake_status() {
        let exact = PlanRevisionHandshake::negotiate(
            revision(7),
            revision(7),
            PlanRevisionHandshakePolicy::Reject,
        )
        .unwrap();
        assert_eq!(
            ReplicationPolicy::ReplicatedState.prediction_safety(
                AuthorityMode::Server,
                ReplaySafety::ReplaySafe,
                exact.status(),
            ),
            PredictionSafety::Predictable
        );
        assert_eq!(
            ReplicationPolicy::ReplicatedState.prediction_safety(
                AuthorityMode::Server,
                ReplaySafety::AuthorityOnly,
                exact.status(),
            ),
            PredictionSafety::Unsafe(PredictionSafetyReason::AuthorityOnlyReplay)
        );
        assert_eq!(
            ReplicationPolicy::ClientLocal.prediction_safety(
                AuthorityMode::Client,
                ReplaySafety::AuthorityOnly,
                exact.status(),
            ),
            PredictionSafety::LocalOnly
        );
    }

    #[test]
    fn revision_handshake_rejects_or_downgrades_without_partial_acceptance() {
        assert_eq!(
            PlanRevisionHandshake::negotiate(
                revision(1),
                revision(2),
                PlanRevisionHandshakePolicy::Reject,
            ),
            Err(PlanRevisionHandshakeError::RevisionMismatch {
                local: revision(1),
                remote: revision(2),
            })
        );

        let handshake = PlanRevisionHandshake::negotiate(
            revision(1),
            revision(2),
            PlanRevisionHandshakePolicy::DowngradeToServerOnly,
        )
        .unwrap();
        assert_eq!(
            handshake.status(),
            PlanRevisionHandshakeStatus::DowngradedToServerOnly
        );
        assert_eq!(
            handshake.effective_policy(ReplicationPolicy::ReplicatedEvent),
            ReplicationPolicy::ServerOnly
        );
        assert_eq!(
            ReplicationPolicy::ReplicatedEvent.prediction_safety(
                AuthorityMode::Server,
                ReplaySafety::Pure,
                handshake.status(),
            ),
            PredictionSafety::Unsafe(PredictionSafetyReason::PlanRevisionMismatch)
        );
    }

    #[test]
    fn rollback_inputs_are_bounded_sorted_and_duplicate_free() {
        let mut contract = RollbackInputContract::new(10, 4, 3, 6).unwrap();
        contract
            .insert(RollbackInput::with_sequence(9, 1, &[2, 3]).unwrap())
            .unwrap();
        contract
            .insert(RollbackInput::with_sequence(8, 0, &[1]).unwrap())
            .unwrap();
        assert_eq!(
            contract
                .inputs()
                .iter()
                .map(|input| (input.frame(), input.sequence()))
                .collect::<Vec<_>>(),
            vec![(8, 0), (9, 1)]
        );
        assert_eq!(contract.input_bytes(), 3);
        assert_eq!(
            contract.insert(RollbackInput::with_sequence(9, 1, &[4]).unwrap()),
            Err(RollbackInputError::Duplicate {
                frame: 9,
                sequence: 1,
            })
        );
        assert_eq!(
            contract.insert(RollbackInput::new(5, &[0]).unwrap()),
            Err(RollbackInputError::OutsideWindow {
                current: 10,
                found: 5,
                maximum: 4,
            })
        );
        contract.advance_to(14).unwrap();
        assert_eq!(contract.inputs().len(), 0);
    }

    #[test]
    fn rollback_contract_reports_each_hard_bound() {
        assert_eq!(
            RollbackInputContract::new(0, 1, 0, 1),
            Err(RollbackInputError::ZeroCapacity)
        );
        assert_eq!(
            RollbackInputContract::new(0, 1, MAX_ROLLBACK_INPUTS + 1, 1),
            Err(RollbackInputError::CapacityLimit {
                requested: MAX_ROLLBACK_INPUTS + 1,
                maximum: MAX_ROLLBACK_INPUTS,
            })
        );
        assert_eq!(
            RollbackInputContract::new(0, 1, 1, MAX_ROLLBACK_INPUT_BYTES + 1),
            Err(RollbackInputError::ByteCapacityLimit {
                requested: MAX_ROLLBACK_INPUT_BYTES + 1,
                maximum: MAX_ROLLBACK_INPUT_BYTES,
            })
        );

        let mut contract = RollbackInputContract::new(2, 2, 1, 1).unwrap();
        assert_eq!(
            contract.insert(RollbackInput::new(3, &[]).unwrap()),
            Err(RollbackInputError::FutureFrame {
                current: 2,
                found: 3,
            })
        );
        assert_eq!(
            contract.advance_to(1),
            Err(RollbackInputError::FrameRewind {
                current: 2,
                requested: 1,
            })
        );
    }
}
