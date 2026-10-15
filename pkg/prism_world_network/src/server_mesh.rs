//! Bounded server-mesh and authority-handoff contracts.
//!
//! This module is a deterministic control-plane state machine for partition
//! ownership. It owns bounded metadata: partition identities, authority
//! leases, epochs, admission reservations, and handoff outcomes. Transport,
//! failure detection, and persistence are connected by [`crate::ServerMeshNode`]
//! through replaceable providers; this module remains independent of sockets,
//! serializers, clocks, and filesystems.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::{ActorId, SectorId};

/// Monotonically increasing fencing epoch for one partition authority.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AuthorityEpoch(pub u64);

impl AuthorityEpoch {
    /// The initial epoch assigned to a newly registered partition.
    pub const INITIAL: Self = Self(0);

    /// Returns the next epoch, or an exhaustion error at `u64::MAX`.
    pub const fn next(self) -> Result<Self, EpochError> {
        match self.0.checked_add(1) {
            Some(next) => Ok(Self(next)),
            None => Err(EpochError::Exhausted),
        }
    }
}

/// Epoch arithmetic failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpochError {
    /// The epoch cannot be incremented without wrapping.
    Exhausted,
}

/// Opaque identity for one authority lease.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LeaseId(pub u128);

/// Opaque identity for one handoff transaction command.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HandoffTransactionId(pub u128);

/// Which side of a lease check produced an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseRole {
    /// The authority currently recorded by the mesh.
    Current,
    /// The source authority supplied by a handoff command.
    Source,
    /// The authority proposed by a handoff command.
    Target,
}

/// Authority identity plus its fencing epoch and bounded validity interval.
///
/// `expires_at` is an exclusive logical tick. The lease is valid when
/// `now < expires_at`; the host supplies the monotonic tick and remains
/// responsible for deciding how that tick maps to wall time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AuthorityLease {
    /// Authority/server identity.
    pub authority: ActorId,
    /// Fencing epoch for this authority assignment.
    pub epoch: AuthorityEpoch,
    /// Opaque lease identity, which must change across a handoff.
    pub lease: LeaseId,
    /// Exclusive logical expiry tick.
    pub expires_at: u64,
}

impl AuthorityLease {
    /// Creates an authority lease.
    pub const fn new(
        authority: ActorId,
        epoch: AuthorityEpoch,
        lease: LeaseId,
        expires_at: u64,
    ) -> Self {
        Self {
            authority,
            epoch,
            lease,
            expires_at,
        }
    }

    /// Returns whether the lease is valid at a logical tick.
    pub const fn is_valid_at(self, now: u64) -> bool {
        now < self.expires_at
    }
}

/// One partition's current authority record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionRecord {
    /// Spatial partition identity from [`prism_world_stream`].
    pub partition: SectorId,
    /// Current fenced authority lease.
    pub authority: AuthorityLease,
}

/// Limits for the in-memory mesh control plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServerMeshConfig {
    /// Maximum number of registered partitions.
    pub max_partitions: usize,
    /// Maximum number of handoff command records retained for idempotency.
    pub max_transactions: usize,
    /// Maximum number of prepared but uncommitted handoffs.
    pub max_inflight_handoffs: usize,
    /// Maximum number of current plus reserved partitions per authority.
    pub max_partitions_per_authority: usize,
}

impl ServerMeshConfig {
    /// Validates all bounded capacities.
    pub const fn validate(self) -> Result<(), MeshConfigError> {
        if self.max_partitions == 0 {
            return Err(MeshConfigError::ZeroPartitions);
        }
        if self.max_transactions == 0 {
            return Err(MeshConfigError::ZeroTransactions);
        }
        if self.max_inflight_handoffs == 0 {
            return Err(MeshConfigError::ZeroInflightHandoffs);
        }
        if self.max_partitions_per_authority == 0 {
            return Err(MeshConfigError::ZeroPartitionsPerAuthority);
        }
        Ok(())
    }
}

/// Invalid mesh capacity configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshConfigError {
    /// `max_partitions` cannot be zero.
    ZeroPartitions,
    /// `max_transactions` cannot be zero.
    ZeroTransactions,
    /// `max_inflight_handoffs` cannot be zero.
    ZeroInflightHandoffs,
    /// `max_partitions_per_authority` cannot be zero.
    ZeroPartitionsPerAuthority,
}

/// Admission failures caused by bounded mesh capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    /// The registered partition table is full.
    PartitionLimit {
        /// Configured limit.
        limit: usize,
        /// Current number of registered partitions.
        current: usize,
    },
    /// The retained idempotency table is full.
    TransactionLimit {
        /// Configured limit.
        limit: usize,
        /// Current number of retained transaction records.
        current: usize,
    },
    /// The prepared handoff window is full.
    InflightLimit {
        /// Configured limit.
        limit: usize,
        /// Current number of prepared handoffs.
        current: usize,
    },
    /// The target authority cannot own another current or reserved partition.
    AuthorityPartitionLimit {
        /// Target authority.
        authority: ActorId,
        /// Configured per-authority limit.
        limit: usize,
        /// Current plus reserved partitions for the target.
        current: usize,
    },
}

/// Lease and fencing validation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseError {
    /// A lease is no longer valid at the supplied logical tick.
    Expired {
        /// Which lease was checked.
        role: LeaseRole,
        /// Exclusive expiry tick.
        expires_at: u64,
        /// Logical tick used for validation.
        now: u64,
    },
    /// The command names a different authority than the current record.
    AuthorityMismatch {
        /// Authority recorded by the mesh.
        expected: ActorId,
        /// Authority supplied by the command.
        actual: ActorId,
    },
    /// The command carries an older epoch and is fenced out.
    StaleEpoch {
        /// Current mesh epoch.
        current: AuthorityEpoch,
        /// Epoch supplied by the command.
        received: AuthorityEpoch,
    },
    /// The command carries an epoch other than the required one.
    EpochMismatch {
        /// Required epoch.
        expected: AuthorityEpoch,
        /// Epoch supplied by the command.
        received: AuthorityEpoch,
    },
    /// The opaque lease identity does not match the current record.
    LeaseMismatch {
        /// Lease recorded by the mesh.
        expected: LeaseId,
        /// Lease supplied by the command.
        actual: LeaseId,
    },
    /// A target attempted to reuse the source lease identity.
    LeaseReuse(LeaseId),
}

/// Failures while registering or inspecting mesh partitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshError {
    /// The configuration contains a zero capacity.
    InvalidConfig(MeshConfigError),
    /// The partition is already registered.
    AlreadyRegistered(SectorId),
    /// A registration or handoff target exceeded a bounded capacity.
    Admission(AdmissionError),
    /// The initial lease is not valid at registration time.
    Lease(LeaseError),
}

/// Immutable authority handoff command.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HandoffCommand {
    /// Stable idempotency identity.
    pub transaction: HandoffTransactionId,
    /// Partition whose authority is changing.
    pub partition: SectorId,
    /// Lease that must still be current at prepare and commit.
    pub source: AuthorityLease,
    /// New authority lease. Its epoch must be current epoch plus one.
    pub target: AuthorityLease,
}

impl HandoffCommand {
    /// Creates a handoff command.
    pub const fn new(
        transaction: HandoffTransactionId,
        partition: SectorId,
        source: AuthorityLease,
        target: AuthorityLease,
    ) -> Self {
        Self {
            transaction,
            partition,
            source,
            target,
        }
    }
}

/// A prepared handoff reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandoffTicket {
    /// Stable transaction identity.
    pub transaction: HandoffTransactionId,
    /// Reserved partition.
    pub partition: SectorId,
    /// Reserved target lease.
    pub target: AuthorityLease,
}

/// The committed authority change recorded for a handoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandoffReceipt {
    /// Stable transaction identity.
    pub transaction: HandoffTransactionId,
    /// Partition whose authority changed.
    pub partition: SectorId,
    /// Authority before the commit.
    pub previous: AuthorityLease,
    /// Authority after the commit.
    pub current: AuthorityLease,
}

/// Result of preparing a handoff command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrepareOutcome {
    /// A new admission reservation was created.
    Prepared(HandoffTicket),
    /// The exact command was already prepared.
    AlreadyPrepared(HandoffTicket),
    /// The exact command was already committed.
    AlreadyCommitted(HandoffReceipt),
}

/// Result of committing a handoff transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitOutcome {
    /// The reservation changed the partition authority.
    Committed(HandoffReceipt),
    /// The exact transaction had already committed.
    AlreadyCommitted(HandoffReceipt),
}

/// Result of aborting a handoff transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbortOutcome {
    /// The prepared reservation was released.
    Aborted,
    /// The transaction had already been aborted.
    AlreadyAborted,
}

/// Result of one deterministic batch preparation item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchPrepareResult {
    /// Transaction identity from the sorted command.
    pub transaction: HandoffTransactionId,
    /// Admission result for that command.
    pub outcome: Result<PrepareOutcome, HandoffError>,
}

/// Failures while preparing, committing, or aborting a handoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffError {
    /// The partition is not registered in this mesh.
    UnknownPartition(SectorId),
    /// The transaction id was reused with different command contents.
    ConflictingTransaction(HandoffTransactionId),
    /// A previously prepared transaction was aborted.
    TransactionAborted(HandoffTransactionId),
    /// A commit or abort referred to no retained transaction.
    UnknownTransaction(HandoffTransactionId),
    /// A commit was attempted for an already aborted transaction.
    AlreadyAborted(HandoffTransactionId),
    /// An abort was attempted after the transaction committed.
    AlreadyCommitted(HandoffTransactionId),
    /// The command failed bounded admission.
    Admission(AdmissionError),
    /// The command failed lease or epoch fencing.
    Lease(LeaseError),
    /// A command would not change authority.
    SameAuthority(ActorId),
    /// No next epoch exists.
    EpochExhausted(AuthorityEpoch),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HandoffState {
    Prepared,
    Committed(HandoffReceipt),
    Aborted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HandoffRecord {
    command: HandoffCommand,
    recovery: bool,
    state: HandoffState,
}

/// Deterministic, bounded authority-handoff state machine.
///
/// The mesh tracks partition metadata only. It is suitable as the policy
/// kernel behind a host server coordinator, but it is not itself a server,
/// failure detector, transport, serializer, or durable transaction log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerMesh {
    config: ServerMeshConfig,
    partitions: BTreeMap<SectorId, PartitionRecord>,
    transactions: BTreeMap<HandoffTransactionId, HandoffRecord>,
    reserved_by_authority: BTreeMap<ActorId, usize>,
    inflight_handoffs: usize,
}

impl ServerMesh {
    /// Creates an empty mesh after validating all bounds.
    pub fn new(config: ServerMeshConfig) -> Result<Self, MeshError> {
        config.validate().map_err(MeshError::InvalidConfig)?;
        Ok(Self {
            config,
            partitions: BTreeMap::new(),
            transactions: BTreeMap::new(),
            reserved_by_authority: BTreeMap::new(),
            inflight_handoffs: 0,
        })
    }

    /// Returns the immutable mesh limits.
    pub const fn config(&self) -> ServerMeshConfig {
        self.config
    }

    /// Registers one initial partition authority.
    pub fn register_partition(
        &mut self,
        now: u64,
        partition: SectorId,
        authority: AuthorityLease,
    ) -> Result<(), MeshError> {
        if self.partitions.contains_key(&partition) {
            return Err(MeshError::AlreadyRegistered(partition));
        }
        if self.partitions.len() >= self.config.max_partitions {
            return Err(MeshError::Admission(AdmissionError::PartitionLimit {
                limit: self.config.max_partitions,
                current: self.partitions.len(),
            }));
        }
        if !authority.is_valid_at(now) {
            return Err(MeshError::Lease(LeaseError::Expired {
                role: LeaseRole::Current,
                expires_at: authority.expires_at,
                now,
            }));
        }
        let current = self.current_owned(authority.authority);
        if current >= self.config.max_partitions_per_authority {
            return Err(MeshError::Admission(
                AdmissionError::AuthorityPartitionLimit {
                    authority: authority.authority,
                    limit: self.config.max_partitions_per_authority,
                    current,
                },
            ));
        }
        self.partitions.insert(
            partition,
            PartitionRecord {
                partition,
                authority,
            },
        );
        Ok(())
    }

    /// Returns the current record for one partition.
    pub fn partition(&self, partition: SectorId) -> Option<PartitionRecord> {
        self.partitions.get(&partition).copied()
    }

    /// Returns the number of registered partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Returns the number of retained transaction records.
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    /// Returns the number of prepared handoffs.
    pub const fn inflight_handoffs(&self) -> usize {
        self.inflight_handoffs
    }

    /// Returns current plus reserved partition load for one authority.
    pub fn authority_load(&self, authority: ActorId) -> AuthorityLoad {
        AuthorityLoad {
            owned_partitions: self.current_owned(authority),
            reserved_partitions: self
                .reserved_by_authority
                .get(&authority)
                .copied()
                .unwrap_or(0),
        }
    }

    /// Captures the complete bounded control-plane state for provider storage.
    pub fn snapshot(&self) -> ServerMeshSnapshot {
        ServerMeshSnapshot {
            config: self.config,
            partitions: self.partitions.values().copied().collect(),
            transactions: self
                .transactions
                .values()
                .map(|record| HandoffSnapshot {
                    command: record.command,
                    recovery: record.recovery,
                    state: match record.state {
                        HandoffState::Prepared => HandoffSnapshotState::Prepared,
                        HandoffState::Committed(receipt) => {
                            HandoffSnapshotState::Committed(receipt)
                        }
                        HandoffState::Aborted => HandoffSnapshotState::Aborted,
                    },
                })
                .collect(),
        }
    }

    /// Restores a mesh from a provider snapshot after rechecking all bounds.
    pub fn from_snapshot(snapshot: ServerMeshSnapshot) -> Result<Self, MeshSnapshotError> {
        snapshot.validate()
    }

    /// Replaces this mesh with a validated provider snapshot.
    pub fn restore(&mut self, snapshot: ServerMeshSnapshot) -> Result<(), MeshSnapshotError> {
        *self = Self::from_snapshot(snapshot)?;
        Ok(())
    }

    /// Prepares one command and reserves target capacity.
    ///
    /// Repeating the exact command is idempotent. Reusing its transaction id
    /// with different contents is rejected. A prepared command remains
    /// reserved until [`Self::commit_handoff`] or [`Self::abort_handoff`].
    pub fn prepare_handoff(
        &mut self,
        now: u64,
        command: HandoffCommand,
    ) -> Result<PrepareOutcome, HandoffError> {
        self.prepare_handoff_inner(now, command, false, None)
    }

    /// Prepares a handoff after a host failure detector has declared the
    /// current authority failed.
    ///
    /// The source authority and lease identity must still match the current
    /// record, but the source lease may be expired. The caller must provide a
    /// detector decision separately; this method does not perform probing.
    pub fn prepare_handoff_after_failure(
        &mut self,
        now: u64,
        failed_authority: ActorId,
        command: HandoffCommand,
    ) -> Result<PrepareOutcome, HandoffError> {
        self.prepare_handoff_inner(now, command, true, Some(failed_authority))
    }

    fn prepare_handoff_inner(
        &mut self,
        now: u64,
        command: HandoffCommand,
        recovery: bool,
        failed_authority: Option<ActorId>,
    ) -> Result<PrepareOutcome, HandoffError> {
        if let Some(record) = self.transactions.get(&command.transaction).copied() {
            if record.command != command || record.recovery != recovery {
                return Err(HandoffError::ConflictingTransaction(command.transaction));
            }
            return match record.state {
                HandoffState::Prepared => {
                    Ok(PrepareOutcome::AlreadyPrepared(Self::ticket_for(command)))
                }
                HandoffState::Committed(receipt) => Ok(PrepareOutcome::AlreadyCommitted(receipt)),
                HandoffState::Aborted => Err(HandoffError::TransactionAborted(command.transaction)),
            };
        }

        let current = self.current_partition(command.partition)?;
        if let Some(failed_authority) = failed_authority {
            if current.authority.authority != failed_authority {
                return Err(HandoffError::Lease(LeaseError::AuthorityMismatch {
                    expected: current.authority.authority,
                    actual: failed_authority,
                }));
            }
        }
        self.validate_source(now, current.authority, command.source, recovery)?;
        self.validate_target(now, current.authority, command.target)?;

        if self.transactions.len() >= self.config.max_transactions {
            return Err(HandoffError::Admission(AdmissionError::TransactionLimit {
                limit: self.config.max_transactions,
                current: self.transactions.len(),
            }));
        }
        if self.inflight_handoffs >= self.config.max_inflight_handoffs {
            return Err(HandoffError::Admission(AdmissionError::InflightLimit {
                limit: self.config.max_inflight_handoffs,
                current: self.inflight_handoffs,
            }));
        }
        let load = self.authority_load(command.target.authority);
        let current_plus_reserved = load
            .owned_partitions
            .saturating_add(load.reserved_partitions);
        if current_plus_reserved >= self.config.max_partitions_per_authority {
            return Err(HandoffError::Admission(
                AdmissionError::AuthorityPartitionLimit {
                    authority: command.target.authority,
                    limit: self.config.max_partitions_per_authority,
                    current: current_plus_reserved,
                },
            ));
        }

        self.transactions.insert(
            command.transaction,
            HandoffRecord {
                command,
                recovery,
                state: HandoffState::Prepared,
            },
        );
        self.inflight_handoffs += 1;
        *self
            .reserved_by_authority
            .entry(command.target.authority)
            .or_default() += 1;
        Ok(PrepareOutcome::Prepared(Self::ticket_for(command)))
    }

    /// Prepares commands in deterministic command order.
    ///
    /// The input order is intentionally ignored. Sorting includes the
    /// transaction id and all command fields, so conflicting duplicate ids
    /// also receive a stable result order.
    pub fn prepare_batch<I>(&mut self, now: u64, commands: I) -> Vec<BatchPrepareResult>
    where
        I: IntoIterator<Item = HandoffCommand>,
    {
        let mut commands: Vec<_> = commands.into_iter().collect();
        commands.sort_unstable();
        commands
            .into_iter()
            .map(|command| {
                let transaction = command.transaction;
                BatchPrepareResult {
                    transaction,
                    outcome: self.prepare_handoff(now, command),
                }
            })
            .collect()
    }

    /// Commits a prepared handoff after rechecking source fencing.
    ///
    /// Repeating a committed transaction returns the same receipt without
    /// changing mesh state. Revalidation at commit prevents a stale prepared
    /// source from committing after another command changed the partition.
    pub fn commit_handoff(
        &mut self,
        now: u64,
        transaction: HandoffTransactionId,
    ) -> Result<CommitOutcome, HandoffError> {
        let record = self
            .transactions
            .get(&transaction)
            .copied()
            .ok_or(HandoffError::UnknownTransaction(transaction))?;
        match record.state {
            HandoffState::Committed(receipt) => {
                return Ok(CommitOutcome::AlreadyCommitted(receipt))
            }
            HandoffState::Aborted => return Err(HandoffError::AlreadyAborted(transaction)),
            HandoffState::Prepared => {}
        }

        let current = self.current_partition(record.command.partition)?;
        self.validate_source(
            now,
            current.authority,
            record.command.source,
            record.recovery,
        )?;
        self.validate_target(now, current.authority, record.command.target)?;

        let receipt = HandoffReceipt {
            transaction,
            partition: record.command.partition,
            previous: current.authority,
            current: record.command.target,
        };
        self.partitions.insert(
            record.command.partition,
            PartitionRecord {
                partition: record.command.partition,
                authority: record.command.target,
            },
        );
        self.release_reservation(record.command.target.authority);
        self.inflight_handoffs -= 1;
        self.transactions.insert(
            transaction,
            HandoffRecord {
                command: record.command,
                recovery: record.recovery,
                state: HandoffState::Committed(receipt),
            },
        );
        Ok(CommitOutcome::Committed(receipt))
    }

    /// Aborts a prepared handoff and releases its target reservation.
    pub fn abort_handoff(
        &mut self,
        transaction: HandoffTransactionId,
    ) -> Result<AbortOutcome, HandoffError> {
        let record = self
            .transactions
            .get(&transaction)
            .copied()
            .ok_or(HandoffError::UnknownTransaction(transaction))?;
        match record.state {
            HandoffState::Prepared => {
                self.release_reservation(record.command.target.authority);
                self.inflight_handoffs -= 1;
                self.transactions.insert(
                    transaction,
                    HandoffRecord {
                        command: record.command,
                        recovery: record.recovery,
                        state: HandoffState::Aborted,
                    },
                );
                Ok(AbortOutcome::Aborted)
            }
            HandoffState::Committed(_) => Err(HandoffError::AlreadyCommitted(transaction)),
            HandoffState::Aborted => Ok(AbortOutcome::AlreadyAborted),
        }
    }

    /// Aborts every prepared handoff involving an authority reported failed.
    ///
    /// The partition record is intentionally left in place. A replacement
    /// authority still needs a fresh, externally authorized handoff command;
    /// failure detection only releases reservations that can no longer make
    /// progress.
    pub fn abort_prepared_by_authority(&mut self, authority: ActorId) -> usize {
        let transactions: Vec<_> = self
            .transactions
            .iter()
            .filter_map(|(transaction, record)| {
                if record.state == HandoffState::Prepared
                    && (record.command.source.authority == authority
                        || record.command.target.authority == authority)
                {
                    Some(*transaction)
                } else {
                    None
                }
            })
            .collect();
        let mut aborted = 0;
        for transaction in transactions {
            if matches!(self.abort_handoff(transaction), Ok(AbortOutcome::Aborted)) {
                aborted += 1;
            }
        }
        aborted
    }

    /// Performs prepare followed by commit for one atomic host call.
    ///
    /// If commit fails after prepare, the reservation remains prepared so the
    /// host can retry with the same transaction id or explicitly abort it.
    pub fn apply_handoff(
        &mut self,
        now: u64,
        command: HandoffCommand,
    ) -> Result<ApplyOutcome, HandoffError> {
        self.apply_handoff_inner(now, command, false, None)
    }

    /// Performs failure recovery prepare followed by commit.
    pub fn apply_handoff_after_failure(
        &mut self,
        now: u64,
        failed_authority: ActorId,
        command: HandoffCommand,
    ) -> Result<ApplyOutcome, HandoffError> {
        self.apply_handoff_inner(now, command, true, Some(failed_authority))
    }

    fn apply_handoff_inner(
        &mut self,
        now: u64,
        command: HandoffCommand,
        recovery: bool,
        failed_authority: Option<ActorId>,
    ) -> Result<ApplyOutcome, HandoffError> {
        match self.prepare_handoff_inner(now, command, recovery, failed_authority)? {
            PrepareOutcome::AlreadyCommitted(receipt) => {
                Ok(ApplyOutcome::AlreadyCommitted(receipt))
            }
            PrepareOutcome::Prepared(_) | PrepareOutcome::AlreadyPrepared(_) => {
                match self.commit_handoff(now, command.transaction)? {
                    CommitOutcome::Committed(receipt) => Ok(ApplyOutcome::Committed(receipt)),
                    CommitOutcome::AlreadyCommitted(receipt) => {
                        Ok(ApplyOutcome::AlreadyCommitted(receipt))
                    }
                }
            }
        }
    }

    fn current_partition(&self, partition: SectorId) -> Result<PartitionRecord, HandoffError> {
        self.partitions
            .get(&partition)
            .copied()
            .ok_or(HandoffError::UnknownPartition(partition))
    }

    fn transaction_source(&self, transaction: HandoffTransactionId) -> Option<ActorId> {
        self.transactions
            .get(&transaction)
            .map(|record| record.command.source.authority)
    }

    fn validate_source(
        &self,
        now: u64,
        current: AuthorityLease,
        source: AuthorityLease,
        recovery: bool,
    ) -> Result<(), HandoffError> {
        if current.authority != source.authority {
            return Err(HandoffError::Lease(LeaseError::AuthorityMismatch {
                expected: current.authority,
                actual: source.authority,
            }));
        }
        if source.epoch < current.epoch {
            return Err(HandoffError::Lease(LeaseError::StaleEpoch {
                current: current.epoch,
                received: source.epoch,
            }));
        }
        if source.epoch != current.epoch {
            return Err(HandoffError::Lease(LeaseError::EpochMismatch {
                expected: current.epoch,
                received: source.epoch,
            }));
        }
        if source.lease != current.lease {
            return Err(HandoffError::Lease(LeaseError::LeaseMismatch {
                expected: current.lease,
                actual: source.lease,
            }));
        }
        if !recovery && !current.is_valid_at(now) {
            return Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Current,
                expires_at: current.expires_at,
                now,
            }));
        }
        if !recovery && !source.is_valid_at(now) {
            return Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Source,
                expires_at: source.expires_at,
                now,
            }));
        }
        Ok(())
    }

    fn validate_target(
        &self,
        now: u64,
        current: AuthorityLease,
        target: AuthorityLease,
    ) -> Result<(), HandoffError> {
        if target.authority == current.authority {
            return Err(HandoffError::SameAuthority(target.authority));
        }
        let expected = current
            .epoch
            .next()
            .map_err(|_| HandoffError::EpochExhausted(current.epoch))?;
        if target.epoch != expected {
            return Err(HandoffError::Lease(LeaseError::EpochMismatch {
                expected,
                received: target.epoch,
            }));
        }
        if target.lease == current.lease {
            return Err(HandoffError::Lease(LeaseError::LeaseReuse(target.lease)));
        }
        if !target.is_valid_at(now) {
            return Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Target,
                expires_at: target.expires_at,
                now,
            }));
        }
        Ok(())
    }

    fn current_owned(&self, authority: ActorId) -> usize {
        self.partitions
            .values()
            .filter(|record| record.authority.authority == authority)
            .count()
    }

    fn release_reservation(&mut self, authority: ActorId) {
        if let Some(count) = self.reserved_by_authority.get_mut(&authority) {
            *count -= 1;
            if *count == 0 {
                self.reserved_by_authority.remove(&authority);
            }
        }
    }

    const fn ticket_for(command: HandoffCommand) -> HandoffTicket {
        HandoffTicket {
            transaction: command.transaction,
            partition: command.partition,
            target: command.target,
        }
    }
}

/// Current and reserved partition load for an authority.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AuthorityLoad {
    /// Number of partitions currently owned.
    pub owned_partitions: usize,
    /// Number of target partition slots reserved by prepared handoffs.
    pub reserved_partitions: usize,
}

/// Result of applying a handoff command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    /// The command committed the authority change.
    Committed(HandoffReceipt),
    /// The command was an exact replay of a committed transaction.
    AlreadyCommitted(HandoffReceipt),
}

/// Serializable-by-provider state for one retained handoff transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandoffSnapshot {
    command: HandoffCommand,
    recovery: bool,
    state: HandoffSnapshotState,
}

impl HandoffSnapshot {
    /// Returns the retained handoff command.
    pub const fn command(self) -> HandoffCommand {
        self.command
    }

    /// Returns whether this transaction used failure-authority recovery mode.
    pub const fn recovery(self) -> bool {
        self.recovery
    }

    /// Returns the retained transaction state.
    pub const fn state(self) -> HandoffSnapshotState {
        self.state
    }
}

/// Retained state for one handoff transaction in a mesh snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffSnapshotState {
    /// The command has reserved capacity but has not committed.
    Prepared,
    /// The command committed with the supplied receipt.
    Committed(HandoffReceipt),
    /// The command was explicitly aborted.
    Aborted,
}

/// Complete bounded state needed to recover a [`ServerMesh`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerMeshSnapshot {
    config: ServerMeshConfig,
    partitions: Vec<PartitionRecord>,
    transactions: Vec<HandoffSnapshot>,
}

impl ServerMeshSnapshot {
    /// Returns the mesh limits captured by this snapshot.
    pub const fn config(&self) -> ServerMeshConfig {
        self.config
    }

    /// Returns partition records in deterministic sector order.
    pub fn partitions(&self) -> &[PartitionRecord] {
        &self.partitions
    }

    /// Returns retained handoff records in deterministic transaction order.
    pub fn transactions(&self) -> &[HandoffSnapshot] {
        &self.transactions
    }

    fn validate(self) -> Result<ServerMesh, MeshSnapshotError> {
        let mut mesh = ServerMesh::new(self.config).map_err(MeshSnapshotError::Mesh)?;
        if self.partitions.len() > mesh.config.max_partitions {
            return Err(MeshSnapshotError::PartitionLimit {
                limit: mesh.config.max_partitions,
                current: self.partitions.len(),
            });
        }
        if self.transactions.len() > mesh.config.max_transactions {
            return Err(MeshSnapshotError::TransactionLimit {
                limit: mesh.config.max_transactions,
                current: self.transactions.len(),
            });
        }

        for record in self.partitions {
            if mesh.partitions.insert(record.partition, record).is_some() {
                return Err(MeshSnapshotError::DuplicatePartition(record.partition));
            }
            let load = mesh.current_owned(record.authority.authority);
            if load > mesh.config.max_partitions_per_authority {
                return Err(MeshSnapshotError::AuthorityPartitionLimit {
                    authority: record.authority.authority,
                    limit: mesh.config.max_partitions_per_authority,
                    current: load,
                });
            }
        }

        for retained in self.transactions {
            let transaction = retained.command.transaction;
            if mesh.transactions.contains_key(&transaction) {
                return Err(MeshSnapshotError::DuplicateTransaction(transaction));
            }
            if !mesh.partitions.contains_key(&retained.command.partition) {
                return Err(MeshSnapshotError::UnknownPartition(
                    retained.command.partition,
                ));
            }
            let state = match retained.state {
                HandoffSnapshotState::Prepared => {
                    if mesh.inflight_handoffs >= mesh.config.max_inflight_handoffs {
                        return Err(MeshSnapshotError::InflightLimit {
                            limit: mesh.config.max_inflight_handoffs,
                            current: mesh.inflight_handoffs,
                        });
                    }
                    let load = mesh.authority_load(retained.command.target.authority);
                    let current = load
                        .owned_partitions
                        .saturating_add(load.reserved_partitions);
                    if current >= mesh.config.max_partitions_per_authority {
                        return Err(MeshSnapshotError::AuthorityPartitionLimit {
                            authority: retained.command.target.authority,
                            limit: mesh.config.max_partitions_per_authority,
                            current,
                        });
                    }
                    mesh.inflight_handoffs += 1;
                    *mesh
                        .reserved_by_authority
                        .entry(retained.command.target.authority)
                        .or_default() += 1;
                    HandoffState::Prepared
                }
                HandoffSnapshotState::Committed(receipt) => {
                    if receipt.transaction != transaction
                        || receipt.partition != retained.command.partition
                        || receipt.previous != retained.command.source
                        || receipt.current != retained.command.target
                    {
                        return Err(MeshSnapshotError::InvalidReceipt(transaction));
                    }
                    HandoffState::Committed(receipt)
                }
                HandoffSnapshotState::Aborted => HandoffState::Aborted,
            };
            mesh.transactions.insert(
                transaction,
                HandoffRecord {
                    command: retained.command,
                    recovery: retained.recovery,
                    state,
                },
            );
        }
        Ok(mesh)
    }
}

/// Failures found while validating a recovered mesh snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshSnapshotError {
    /// The embedded mesh configuration is invalid.
    Mesh(MeshError),
    /// The snapshot contains too many partition records.
    PartitionLimit {
        /// Configured partition bound.
        limit: usize,
        /// Number of records in the snapshot.
        current: usize,
    },
    /// The snapshot contains too many retained transaction records.
    TransactionLimit {
        /// Configured transaction bound.
        limit: usize,
        /// Number of records in the snapshot.
        current: usize,
    },
    /// A partition identity occurs more than once.
    DuplicatePartition(SectorId),
    /// A transaction identity occurs more than once.
    DuplicateTransaction(HandoffTransactionId),
    /// A retained transaction names no registered partition.
    UnknownPartition(SectorId),
    /// The prepared transaction window would exceed its configured bound.
    InflightLimit {
        /// Configured prepared handoff bound.
        limit: usize,
        /// Number of prepared records already reconstructed.
        current: usize,
    },
    /// A current or reserved authority load exceeds its configured bound.
    AuthorityPartitionLimit {
        /// Authority whose load is invalid.
        authority: ActorId,
        /// Configured per-authority bound.
        limit: usize,
        /// Reconstructed current plus reserved load.
        current: usize,
    },
    /// A committed receipt does not match its retained command.
    InvalidReceipt(HandoffTransactionId),
}

/// A request sent through a node transport provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshRequest {
    /// Applies prepare and commit as one idempotent operation.
    Apply(HandoffCommand),
    /// Reserves a handoff without committing it.
    Prepare(HandoffCommand),
    /// Commits a previously prepared handoff.
    Commit(HandoffTransactionId),
    /// Aborts a previously prepared handoff.
    Abort(HandoffTransactionId),
    /// Applies a handoff after the requester has confirmed the source
    /// authority failed.
    Recover {
        /// Authority whose lease is being fenced by recovery.
        failed_authority: ActorId,
        /// Handoff command carrying the current source fence and next target
        /// lease.
        command: HandoffCommand,
    },
}

/// A response returned by a node for one handoff request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshReply {
    /// Result of [`MeshRequest::Apply`].
    Applied(Result<ApplyOutcome, HandoffError>),
    /// Result of [`MeshRequest::Prepare`].
    Prepared(Result<PrepareOutcome, HandoffError>),
    /// Result of [`MeshRequest::Commit`].
    Committed(Result<CommitOutcome, HandoffError>),
    /// Result of [`MeshRequest::Abort`].
    Aborted(Result<AbortOutcome, HandoffError>),
    /// The request was rejected before entering the mesh state machine.
    Rejected(MeshReject),
}

/// Transport-level rejection produced before a handoff enters the mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshReject {
    /// The source authority has been reported failed by the detector.
    AuthorityUnavailable(ActorId),
    /// The packet source does not match the command's source authority.
    SourceMismatch {
        /// Authority named by the command.
        expected: ActorId,
        /// Packet source identity.
        actual: ActorId,
    },
    /// A recovery request was made before the detector observed failure.
    AuthorityNotFailed(ActorId),
}

/// One packet exchanged by a server-mesh transport provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshPacket {
    /// Sender node identity.
    pub source: ActorId,
    /// Intended receiver node identity.
    pub destination: ActorId,
    /// Request or response payload.
    pub message: MeshMessage,
}

/// A server-mesh request or response payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshMessage {
    /// A handoff control request.
    Request(MeshRequest),
    /// A response to a prior request.
    Reply(MeshReply),
}

/// Replaceable, non-blocking transport provider for mesh control packets.
pub trait MeshTransport {
    /// Provider-specific transport error.
    type Error;

    /// Attempts to enqueue one packet for its destination.
    fn try_send(&mut self, packet: MeshPacket) -> Result<(), Self::Error>;

    /// Removes one received packet, if available.
    fn try_recv(&mut self) -> Option<MeshPacket>;
}

/// Replaceable failure detector polled by a [`ServerMeshNode`].
pub trait MeshFailureDetector {
    /// Provider-specific detector error.
    type Error;

    /// Returns authorities newly observed as failed at the supplied logical tick.
    fn poll_failed(&mut self, now: u64) -> Result<Vec<ActorId>, Self::Error>;
}

/// Replaceable provider for loading and atomically publishing mesh snapshots.
pub trait MeshPersistence {
    /// Provider-specific persistence error.
    type Error;

    /// Loads the latest complete snapshot, if one exists.
    fn load(&mut self) -> Result<Option<ServerMeshSnapshot>, Self::Error>;

    /// Publishes a complete snapshot after a successful state transition.
    fn save(&mut self, snapshot: &ServerMeshSnapshot) -> Result<(), Self::Error>;
}

/// Errors while opening a node from its persistence provider.
#[derive(Debug)]
pub enum MeshNodeOpenError<PE> {
    /// The provider could not load its latest snapshot.
    Persistence(PE),
    /// The snapshot failed bounded state validation.
    Snapshot(MeshSnapshotError),
    /// No snapshot existed and the requested mesh configuration was invalid.
    Mesh(MeshError),
    /// A recovered snapshot uses different bounds than the caller requested.
    ConfigMismatch {
        /// Configuration requested by the host.
        expected: ServerMeshConfig,
        /// Configuration recorded by persistence.
        actual: ServerMeshConfig,
    },
}

/// Errors from node transport, failure detection, persistence, or mesh policy.
#[derive(Debug)]
pub enum MeshNodeError<TE, FE, PE> {
    /// The transport provider rejected a packet.
    Transport(TE),
    /// The failure detector provider failed.
    FailureDetector(FE),
    /// The persistence provider rejected a state publication.
    Persistence(PE),
    /// A local registration failed mesh admission.
    Mesh(MeshError),
    /// A local rollback or recovery snapshot was invalid.
    Snapshot(MeshSnapshotError),
    /// The requested handoff was rejected by mesh policy.
    Handoff(HandoffError),
    /// A send targeted an authority already reported failed.
    AuthorityUnavailable(ActorId),
}

/// Observable work performed by one node poll.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshNodeEvent {
    /// A failure was newly observed and prepared reservations were released.
    FailureDetected {
        /// Authority reported failed.
        authority: ActorId,
        /// Number of prepared handoffs aborted as cleanup.
        aborted_handoffs: usize,
    },
    /// A request was handled and its response was sent.
    RequestHandled {
        /// Request sender.
        source: ActorId,
        /// Request that was handled.
        request: MeshRequest,
        /// Response sent to the requester.
        reply: MeshReply,
    },
    /// A response packet was received from another node.
    ReplyReceived {
        /// Response sender.
        source: ActorId,
        /// Response payload.
        reply: MeshReply,
    },
}

type MeshPacketQueue = Rc<RefCell<VecDeque<MeshPacket>>>;

/// Errors from the deterministic in-memory mesh transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryMeshTransportError {
    /// A zero-capacity queue cannot be constructed.
    InvalidCapacity,
    /// The destination receive queue is full.
    Backpressure,
}

/// A bounded in-memory mesh transport for tests and host integration harnesses.
pub struct MemoryMeshTransport {
    inbound: MeshPacketQueue,
    outbound: MeshPacketQueue,
    capacity: usize,
}

impl MemoryMeshTransport {
    /// Creates two connected bounded endpoints.
    pub fn pair(capacity: usize) -> Result<(Self, Self), MemoryMeshTransportError> {
        if capacity == 0 {
            return Err(MemoryMeshTransportError::InvalidCapacity);
        }
        let left_to_right = Rc::new(RefCell::new(VecDeque::new()));
        let right_to_left = Rc::new(RefCell::new(VecDeque::new()));
        Ok((
            Self {
                inbound: Rc::clone(&right_to_left),
                outbound: Rc::clone(&left_to_right),
                capacity,
            },
            Self {
                inbound: left_to_right,
                outbound: right_to_left,
                capacity,
            },
        ))
    }

    /// Returns the number of packets waiting on this endpoint.
    pub fn queued(&self) -> usize {
        self.inbound.borrow().len()
    }
}

impl MeshTransport for MemoryMeshTransport {
    type Error = MemoryMeshTransportError;

    fn try_send(&mut self, packet: MeshPacket) -> Result<(), Self::Error> {
        let mut outbound = self.outbound.borrow_mut();
        if outbound.len() >= self.capacity {
            return Err(MemoryMeshTransportError::Backpressure);
        }
        outbound.push_back(packet);
        Ok(())
    }

    fn try_recv(&mut self) -> Option<MeshPacket> {
        self.inbound.borrow_mut().pop_front()
    }
}

/// A node coordinator connecting the bounded mesh to host providers.
pub struct ServerMeshNode<T, F, P> {
    local: ActorId,
    mesh: ServerMesh,
    transport: T,
    failure_detector: F,
    persistence: P,
    failed_authorities: BTreeSet<ActorId>,
}

impl<T, F, P> ServerMeshNode<T, F, P>
where
    T: MeshTransport,
    F: MeshFailureDetector,
    P: MeshPersistence,
{
    /// Opens a node from the latest provider snapshot or creates a new mesh.
    pub fn open(
        local: ActorId,
        config: ServerMeshConfig,
        transport: T,
        failure_detector: F,
        mut persistence: P,
    ) -> Result<Self, MeshNodeOpenError<P::Error>> {
        let mesh = match persistence.load().map_err(MeshNodeOpenError::Persistence)? {
            Some(snapshot) => {
                let actual = snapshot.config();
                if actual != config {
                    return Err(MeshNodeOpenError::ConfigMismatch {
                        expected: config,
                        actual,
                    });
                }
                ServerMesh::from_snapshot(snapshot).map_err(MeshNodeOpenError::Snapshot)?
            }
            None => ServerMesh::new(config).map_err(MeshNodeOpenError::Mesh)?,
        };
        Ok(Self {
            local,
            mesh,
            transport,
            failure_detector,
            persistence,
            failed_authorities: BTreeSet::new(),
        })
    }

    /// Returns this node's identity.
    pub const fn local(&self) -> ActorId {
        self.local
    }

    /// Returns the current bounded mesh policy state.
    pub const fn mesh(&self) -> &ServerMesh {
        &self.mesh
    }

    /// Returns whether an authority has been reported failed in this node.
    pub fn is_failed(&self, authority: ActorId) -> bool {
        self.failed_authorities.contains(&authority)
    }

    /// Registers an initial partition and publishes the resulting snapshot.
    pub fn register_partition(
        &mut self,
        now: u64,
        partition: SectorId,
        authority: AuthorityLease,
    ) -> Result<(), MeshNodeError<T::Error, F::Error, P::Error>> {
        let before = self.mesh.snapshot();
        match self.mesh.register_partition(now, partition, authority) {
            Ok(()) => self.publish_or_restore(before),
            Err(error) => {
                self.mesh.restore(before).map_err(MeshNodeError::Snapshot)?;
                Err(MeshNodeError::Mesh(error))
            }
        }
    }

    /// Sends one handoff request through the configured transport provider.
    pub fn send_request(
        &mut self,
        destination: ActorId,
        request: MeshRequest,
    ) -> Result<(), MeshNodeError<T::Error, F::Error, P::Error>> {
        if self.failed_authorities.contains(&destination) {
            return Err(MeshNodeError::AuthorityUnavailable(destination));
        }
        self.transport
            .try_send(MeshPacket {
                source: self.local,
                destination,
                message: MeshMessage::Request(request),
            })
            .map_err(MeshNodeError::Transport)
    }

    /// Polls failure detection and all currently queued transport packets.
    pub fn poll(
        &mut self,
        now: u64,
    ) -> Result<Vec<MeshNodeEvent>, MeshNodeError<T::Error, F::Error, P::Error>> {
        let mut events = Vec::new();
        let failed = self
            .failure_detector
            .poll_failed(now)
            .map_err(MeshNodeError::FailureDetector)?;
        for authority in failed {
            if self.failed_authorities.insert(authority) {
                let before = self.mesh.snapshot();
                let aborted = self.mesh.abort_prepared_by_authority(authority);
                if aborted != 0 {
                    if let Err(error) = self.publish_or_restore(before) {
                        self.failed_authorities.remove(&authority);
                        return Err(error);
                    }
                }
                events.push(MeshNodeEvent::FailureDetected {
                    authority,
                    aborted_handoffs: aborted,
                });
            }
        }

        while let Some(packet) = self.transport.try_recv() {
            if packet.destination != self.local {
                continue;
            }
            match packet.message {
                MeshMessage::Reply(reply) => events.push(MeshNodeEvent::ReplyReceived {
                    source: packet.source,
                    reply,
                }),
                MeshMessage::Request(request) => {
                    let reply = self.handle_request(now, packet.source, request)?;
                    self.transport
                        .try_send(MeshPacket {
                            source: self.local,
                            destination: packet.source,
                            message: MeshMessage::Reply(reply.clone()),
                        })
                        .map_err(MeshNodeError::Transport)?;
                    events.push(MeshNodeEvent::RequestHandled {
                        source: packet.source,
                        request,
                        reply,
                    });
                }
            }
        }
        Ok(events)
    }

    fn handle_request(
        &mut self,
        now: u64,
        source: ActorId,
        request: MeshRequest,
    ) -> Result<MeshReply, MeshNodeError<T::Error, F::Error, P::Error>> {
        if self.failed_authorities.contains(&source) {
            return Ok(MeshReply::Rejected(MeshReject::AuthorityUnavailable(
                source,
            )));
        }
        match request {
            MeshRequest::Apply(command) => {
                if let Some(reply) = self.reject_source_mismatch(source, command.source.authority) {
                    return Ok(reply);
                }
                Ok(MeshReply::Applied(self.persist_handoff(|mesh| {
                    mesh.apply_handoff(now, command)
                })?))
            }
            MeshRequest::Prepare(command) => {
                if let Some(reply) = self.reject_source_mismatch(source, command.source.authority) {
                    return Ok(reply);
                }
                Ok(MeshReply::Prepared(self.persist_handoff(|mesh| {
                    mesh.prepare_handoff(now, command)
                })?))
            }
            MeshRequest::Commit(transaction) => {
                if let Some(expected) = self.mesh.transaction_source(transaction) {
                    if let Some(reply) = self.reject_source_mismatch(source, expected) {
                        return Ok(reply);
                    }
                }
                Ok(MeshReply::Committed(self.persist_handoff(|mesh| {
                    mesh.commit_handoff(now, transaction)
                })?))
            }
            MeshRequest::Abort(transaction) => {
                if let Some(expected) = self.mesh.transaction_source(transaction) {
                    if let Some(reply) = self.reject_source_mismatch(source, expected) {
                        return Ok(reply);
                    }
                }
                Ok(MeshReply::Aborted(
                    self.persist_handoff(|mesh| mesh.abort_handoff(transaction))?,
                ))
            }
            MeshRequest::Recover {
                failed_authority,
                command,
            } => {
                if let Some(reply) = self.reject_source_mismatch(source, command.target.authority) {
                    return Ok(reply);
                }
                if !self.failed_authorities.contains(&failed_authority) {
                    return Ok(MeshReply::Rejected(MeshReject::AuthorityNotFailed(
                        failed_authority,
                    )));
                }
                Ok(MeshReply::Applied(self.persist_handoff(|mesh| {
                    mesh.apply_handoff_after_failure(now, failed_authority, command)
                })?))
            }
        }
    }

    fn reject_source_mismatch(&self, actual: ActorId, expected: ActorId) -> Option<MeshReply> {
        (actual != expected).then_some(MeshReply::Rejected(MeshReject::SourceMismatch {
            expected,
            actual,
        }))
    }

    fn persist_handoff<R>(
        &mut self,
        operation: impl FnOnce(&mut ServerMesh) -> Result<R, HandoffError>,
    ) -> Result<Result<R, HandoffError>, MeshNodeError<T::Error, F::Error, P::Error>> {
        let before = self.mesh.snapshot();
        match operation(&mut self.mesh) {
            Ok(result) => {
                self.publish_or_restore(before)?;
                Ok(Ok(result))
            }
            Err(error) => {
                self.mesh.restore(before).map_err(MeshNodeError::Snapshot)?;
                Ok(Err(error))
            }
        }
    }

    fn publish_or_restore(
        &mut self,
        before: ServerMeshSnapshot,
    ) -> Result<(), MeshNodeError<T::Error, F::Error, P::Error>> {
        let snapshot = self.mesh.snapshot();
        match self.persistence.save(&snapshot) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.mesh.restore(before).map_err(MeshNodeError::Snapshot)?;
                Err(MeshNodeError::Persistence(error))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn config() -> ServerMeshConfig {
        ServerMeshConfig {
            max_partitions: 4,
            max_transactions: 8,
            max_inflight_handoffs: 4,
            max_partitions_per_authority: 4,
        }
    }

    fn partition(x: i64) -> SectorId {
        SectorId::new(0, x, 0, 0)
    }

    fn lease(authority: u128, epoch: u64, id: u128, expires_at: u64) -> AuthorityLease {
        AuthorityLease::new(
            ActorId(authority),
            AuthorityEpoch(epoch),
            LeaseId(id),
            expires_at,
        )
    }

    fn command(
        transaction: u128,
        partition: SectorId,
        source: AuthorityLease,
        target: AuthorityLease,
    ) -> HandoffCommand {
        HandoffCommand::new(HandoffTransactionId(transaction), partition, source, target)
    }

    fn registered_mesh() -> ServerMesh {
        let mut mesh = ServerMesh::new(config()).expect("valid mesh config");
        mesh.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("initial partition authority is valid");
        mesh
    }

    #[derive(Default)]
    struct TestFailureDetector {
        failed: Vec<ActorId>,
    }

    impl MeshFailureDetector for TestFailureDetector {
        type Error = ();

        fn poll_failed(&mut self, _now: u64) -> Result<Vec<ActorId>, Self::Error> {
            Ok(core::mem::take(&mut self.failed))
        }
    }

    struct TestPersistence {
        snapshot: Option<ServerMeshSnapshot>,
        reject_save: bool,
    }

    impl TestPersistence {
        fn empty() -> Self {
            Self {
                snapshot: None,
                reject_save: false,
            }
        }

        fn with_snapshot(snapshot: ServerMeshSnapshot) -> Self {
            Self {
                snapshot: Some(snapshot),
                reject_save: false,
            }
        }
    }

    impl MeshPersistence for TestPersistence {
        type Error = ();

        fn load(&mut self) -> Result<Option<ServerMeshSnapshot>, Self::Error> {
            Ok(self.snapshot.clone())
        }

        fn save(&mut self, snapshot: &ServerMeshSnapshot) -> Result<(), Self::Error> {
            if self.reject_save {
                return Err(());
            }
            self.snapshot = Some(snapshot.clone());
            Ok(())
        }
    }

    #[test]
    fn failure_recovery_accepts_expired_source_and_preserves_recovery_mode() {
        let mut mesh = ServerMesh::new(config()).expect("valid mesh config");
        mesh.register_partition(0, partition(1), lease(1, 0, 11, 10))
            .expect("initial partition authority is valid");
        let source = lease(1, 0, 11, 10);
        let target = lease(2, 1, 22, 100);
        let handoff = command(41, partition(1), source, target);
        assert_eq!(
            mesh.apply_handoff(10, handoff),
            Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Current,
                expires_at: 10,
                now: 10,
            }))
        );

        let receipt = HandoffReceipt {
            transaction: HandoffTransactionId(41),
            partition: partition(1),
            previous: source,
            current: target,
        };
        assert_eq!(
            mesh.apply_handoff_after_failure(10, ActorId(1), handoff),
            Ok(ApplyOutcome::Committed(receipt))
        );
        assert!(mesh.snapshot().transactions()[0].recovery());

        let mut restored = ServerMesh::from_snapshot(mesh.snapshot()).expect("valid snapshot");
        assert_eq!(
            restored.apply_handoff_after_failure(10, ActorId(1), handoff),
            Ok(ApplyOutcome::AlreadyCommitted(receipt))
        );
    }

    #[test]
    fn node_routes_handoff_over_transport_and_reopens_from_persistence() {
        let (mut client, server_transport) = MemoryMeshTransport::pair(8).expect("valid transport");
        let mut node = ServerMeshNode::open(
            ActorId(2),
            config(),
            server_transport,
            TestFailureDetector::default(),
            TestPersistence::empty(),
        )
        .expect("node opens");
        node.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("initial partition is persisted");

        let source = lease(1, 0, 11, 100);
        let target = lease(2, 1, 22, 200);
        let handoff = command(42, partition(1), source, target);
        client
            .try_send(MeshPacket {
                source: ActorId(1),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Prepare(handoff)),
            })
            .expect("prepare request is transported");
        let events = node.poll(1).expect("prepare is handled");
        assert!(matches!(
            events.as_slice(),
            [MeshNodeEvent::RequestHandled {
                reply: MeshReply::Prepared(Ok(PrepareOutcome::Prepared(_))),
                ..
            }]
        ));
        assert!(matches!(
            client.try_recv().map(|packet| packet.message),
            Some(MeshMessage::Reply(MeshReply::Prepared(Ok(
                PrepareOutcome::Prepared(_)
            ))))
        ));

        client
            .try_send(MeshPacket {
                source: ActorId(1),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Commit(HandoffTransactionId(42))),
            })
            .expect("commit request is transported");
        node.poll(1).expect("commit is handled");
        assert!(matches!(
            client.try_recv().map(|packet| packet.message),
            Some(MeshMessage::Reply(MeshReply::Committed(Ok(
                CommitOutcome::Committed(_)
            ))))
        ));
        assert_eq!(
            node.mesh()
                .partition(partition(1))
                .map(|record| record.authority),
            Some(target)
        );

        let snapshot = node.mesh().snapshot();
        let (_replay_client, replay_transport) =
            MemoryMeshTransport::pair(8).expect("valid transport");
        let recovered = ServerMeshNode::open(
            ActorId(2),
            config(),
            replay_transport,
            TestFailureDetector::default(),
            TestPersistence::with_snapshot(snapshot),
        )
        .expect("node reopens from persisted state");
        assert_eq!(
            recovered
                .mesh()
                .partition(partition(1))
                .map(|record| record.authority),
            Some(target)
        );
        assert_eq!(recovered.mesh().transaction_count(), 1);
    }

    #[test]
    fn node_requires_detector_failure_before_expired_lease_recovery() {
        let (mut client, server_transport) = MemoryMeshTransport::pair(8).expect("valid transport");
        let mut node = ServerMeshNode::open(
            ActorId(2),
            config(),
            server_transport,
            TestFailureDetector::default(),
            TestPersistence::empty(),
        )
        .expect("node opens");
        node.register_partition(0, partition(1), lease(1, 0, 11, 10))
            .expect("initial partition is persisted");
        let handoff = command(43, partition(1), lease(1, 0, 11, 10), lease(2, 1, 23, 100));

        client
            .try_send(MeshPacket {
                source: ActorId(2),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Recover {
                    failed_authority: ActorId(1),
                    command: handoff,
                }),
            })
            .expect("recovery request is transported");
        node.poll(10).expect("request is rejected as not failed");
        assert!(matches!(
            client.try_recv().map(|packet| packet.message),
            Some(MeshMessage::Reply(MeshReply::Rejected(
                MeshReject::AuthorityNotFailed(ActorId(1))
            )))
        ));

        node.failure_detector.failed.push(ActorId(1));
        let events = node.poll(10).expect("failure detector advances");
        assert!(matches!(
            events.as_slice(),
            [MeshNodeEvent::FailureDetected {
                authority: ActorId(1),
                aborted_handoffs: 0,
            }]
        ));
        client
            .try_send(MeshPacket {
                source: ActorId(2),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Recover {
                    failed_authority: ActorId(1),
                    command: handoff,
                }),
            })
            .expect("recovery retry is transported");
        node.poll(10).expect("recovery commits");
        assert!(matches!(
            client.try_recv().map(|packet| packet.message),
            Some(MeshMessage::Reply(MeshReply::Applied(Ok(
                ApplyOutcome::Committed(_)
            ))))
        ));
        assert_eq!(
            node.mesh()
                .partition(partition(1))
                .map(|record| record.authority),
            Some(lease(2, 1, 23, 100))
        );
    }

    #[test]
    fn node_rolls_back_memory_when_persistence_rejects_transition() {
        let (mut client, server_transport) = MemoryMeshTransport::pair(8).expect("valid transport");
        let mut node = ServerMeshNode::open(
            ActorId(2),
            config(),
            server_transport,
            TestFailureDetector::default(),
            TestPersistence::empty(),
        )
        .expect("node opens");
        node.persistence.reject_save = true;
        assert!(matches!(
            node.register_partition(0, partition(1), lease(1, 0, 11, 100)),
            Err(MeshNodeError::Persistence(()))
        ));
        assert_eq!(node.mesh().partition_count(), 0);

        node.persistence.reject_save = false;
        node.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("initial partition is persisted");
        node.persistence.reject_save = true;
        let handoff = command(44, partition(1), lease(1, 0, 11, 100), lease(2, 1, 24, 200));
        client
            .try_send(MeshPacket {
                source: ActorId(1),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Apply(handoff)),
            })
            .expect("handoff request is transported");
        assert!(matches!(node.poll(1), Err(MeshNodeError::Persistence(()))));
        assert_eq!(
            node.mesh()
                .partition(partition(1))
                .map(|record| record.authority),
            Some(lease(1, 0, 11, 100))
        );

        node.persistence.reject_save = false;
        client
            .try_send(MeshPacket {
                source: ActorId(1),
                destination: ActorId(2),
                message: MeshMessage::Request(MeshRequest::Apply(handoff)),
            })
            .expect("handoff retry is transported");
        node.poll(1).expect("retry persists");
        assert_eq!(
            node.mesh()
                .partition(partition(1))
                .map(|record| record.authority),
            Some(lease(2, 1, 24, 200))
        );
    }

    #[test]
    fn zero_bounds_are_rejected() {
        let mut limited = config();
        limited.max_partitions = 0;
        assert_eq!(
            ServerMesh::new(limited),
            Err(MeshError::InvalidConfig(MeshConfigError::ZeroPartitions))
        );
        let mut limited = config();
        limited.max_transactions = 0;
        assert_eq!(
            ServerMesh::new(limited),
            Err(MeshError::InvalidConfig(MeshConfigError::ZeroTransactions))
        );
    }

    #[test]
    fn registration_is_bounded_and_duplicate_safe() {
        let mut config = config();
        config.max_partitions = 1;
        let mut mesh = ServerMesh::new(config).expect("valid config");
        mesh.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("first partition");
        assert_eq!(
            mesh.register_partition(0, partition(1), lease(1, 0, 12, 100)),
            Err(MeshError::AlreadyRegistered(partition(1)))
        );
        assert_eq!(
            mesh.register_partition(0, partition(2), lease(1, 0, 13, 100)),
            Err(MeshError::Admission(AdmissionError::PartitionLimit {
                limit: 1,
                current: 1,
            }))
        );
    }

    #[test]
    fn prepare_commit_and_exact_replay_are_idempotent() {
        let mut mesh = registered_mesh();
        let source = lease(1, 0, 11, 100);
        let target = lease(2, 1, 22, 200);
        let command = command(7, partition(1), source, target);
        let ticket = HandoffTicket {
            transaction: HandoffTransactionId(7),
            partition: partition(1),
            target,
        };
        assert_eq!(
            mesh.prepare_handoff(1, command),
            Ok(PrepareOutcome::Prepared(ticket))
        );
        assert_eq!(
            mesh.prepare_handoff(1, command),
            Ok(PrepareOutcome::AlreadyPrepared(ticket))
        );
        let receipt = HandoffReceipt {
            transaction: HandoffTransactionId(7),
            partition: partition(1),
            previous: source,
            current: target,
        };
        assert_eq!(
            mesh.commit_handoff(1, HandoffTransactionId(7)),
            Ok(CommitOutcome::Committed(receipt))
        );
        assert_eq!(
            mesh.commit_handoff(1, HandoffTransactionId(7)),
            Ok(CommitOutcome::AlreadyCommitted(receipt))
        );
        assert_eq!(
            mesh.partition(partition(1)).map(|record| record.authority),
            Some(target)
        );
        assert_eq!(mesh.inflight_handoffs(), 0);
        assert_eq!(mesh.authority_load(ActorId(2)).owned_partitions, 1);
    }

    #[test]
    fn old_authority_and_stale_epoch_are_fenced() {
        let mut mesh = registered_mesh();
        let first = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        mesh.apply_handoff(1, first).expect("first handoff");

        let old_authority = command(2, partition(1), lease(1, 0, 11, 100), lease(3, 2, 33, 200));
        assert_eq!(
            mesh.prepare_handoff(2, old_authority),
            Err(HandoffError::Lease(LeaseError::AuthorityMismatch {
                expected: ActorId(2),
                actual: ActorId(1),
            }))
        );
        let stale_epoch = command(3, partition(1), lease(2, 0, 22, 100), lease(3, 2, 33, 200));
        assert_eq!(
            mesh.prepare_handoff(2, stale_epoch),
            Err(HandoffError::Lease(LeaseError::StaleEpoch {
                current: AuthorityEpoch(1),
                received: AuthorityEpoch(0),
            }))
        );
    }

    #[test]
    fn epoch_and_lease_validation_is_explicit() {
        let mut mesh = registered_mesh();
        let source = lease(1, 0, 11, 100);
        assert_eq!(
            mesh.prepare_handoff(1, command(1, partition(1), source, lease(2, 2, 22, 200)),),
            Err(HandoffError::Lease(LeaseError::EpochMismatch {
                expected: AuthorityEpoch(1),
                received: AuthorityEpoch(2),
            }))
        );
        assert_eq!(
            mesh.prepare_handoff(1, command(2, partition(1), source, lease(2, 1, 11, 200)),),
            Err(HandoffError::Lease(LeaseError::LeaseReuse(LeaseId(11))))
        );
        assert_eq!(
            mesh.prepare_handoff(100, command(3, partition(1), source, lease(2, 1, 22, 200)),),
            Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Current,
                expires_at: 100,
                now: 100,
            }))
        );
        assert_eq!(
            mesh.prepare_handoff(1, command(4, partition(1), source, lease(2, 1, 22, 1)),),
            Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Target,
                expires_at: 1,
                now: 1,
            }))
        );
    }

    #[test]
    fn commit_rechecks_fencing_after_prepare() {
        let mut mesh = registered_mesh();
        let first = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        let second = command(2, partition(1), lease(1, 0, 11, 100), lease(3, 1, 33, 200));
        mesh.prepare_handoff(1, first).expect("first reservation");
        mesh.prepare_handoff(1, second).expect("second reservation");
        mesh.commit_handoff(1, HandoffTransactionId(2))
            .expect("second command wins the race");
        assert_eq!(
            mesh.commit_handoff(1, HandoffTransactionId(1)),
            Err(HandoffError::Lease(LeaseError::AuthorityMismatch {
                expected: ActorId(3),
                actual: ActorId(1),
            }))
        );
        mesh.abort_handoff(HandoffTransactionId(1))
            .expect("stale prepared command can release its reservation");
    }

    #[test]
    fn admission_bounds_inflight_transactions_and_target_load() {
        let mut config = config();
        config.max_inflight_handoffs = 1;
        config.max_partitions_per_authority = 1;
        let mut mesh = ServerMesh::new(config).expect("valid config");
        mesh.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("source partition");
        let first = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        mesh.prepare_handoff(1, first).expect("first reservation");
        let second = command(2, partition(1), lease(1, 0, 11, 100), lease(3, 1, 33, 200));
        assert_eq!(
            mesh.prepare_handoff(1, second),
            Err(HandoffError::Admission(AdmissionError::InflightLimit {
                limit: 1,
                current: 1,
            }))
        );
        mesh.abort_handoff(HandoffTransactionId(1))
            .expect("abort releases inflight capacity");
        let retried = command(3, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        mesh.apply_handoff(1, retried)
            .expect("target slot available again");
        assert_eq!(
            mesh.authority_load(ActorId(2)),
            AuthorityLoad {
                owned_partitions: 1,
                reserved_partitions: 0,
            }
        );
    }

    #[test]
    fn target_partition_capacity_counts_prepared_reservations() {
        let mut config = config();
        config.max_partitions_per_authority = 1;
        let mut mesh = ServerMesh::new(config).expect("valid config");
        mesh.register_partition(0, partition(1), lease(1, 0, 11, 100))
            .expect("first source");
        mesh.register_partition(0, partition(2), lease(3, 0, 12, 100))
            .expect("second source");
        let first = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        let second = command(2, partition(2), lease(3, 0, 12, 100), lease(2, 1, 23, 200));
        mesh.prepare_handoff(1, first)
            .expect("first target reservation");
        assert_eq!(
            mesh.prepare_handoff(1, second),
            Err(HandoffError::Admission(
                AdmissionError::AuthorityPartitionLimit {
                    authority: ActorId(2),
                    limit: 1,
                    current: 1,
                }
            ))
        );
    }

    #[test]
    fn transaction_history_is_bounded_but_replay_remains_idempotent() {
        let mut config = config();
        config.max_transactions = 1;
        let mut mesh = registered_mesh();
        mesh.config = config;
        let first = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        mesh.apply_handoff(1, first).expect("first transaction");
        assert_eq!(
            mesh.apply_handoff(
                1,
                command(2, partition(1), lease(2, 1, 22, 200), lease(3, 2, 33, 300)),
            ),
            Err(HandoffError::Admission(AdmissionError::TransactionLimit {
                limit: 1,
                current: 1,
            }))
        );
        assert!(matches!(
            mesh.apply_handoff(1, first),
            Ok(ApplyOutcome::AlreadyCommitted(_))
        ));
    }

    #[test]
    fn batch_preparation_is_sorted_independent_of_arrival_order() {
        let mut left = ServerMesh::new(config()).expect("valid config");
        let mut right = ServerMesh::new(config()).expect("valid config");
        for mesh in [&mut left, &mut right] {
            mesh.register_partition(0, partition(1), lease(1, 0, 11, 100))
                .expect("first source");
            mesh.register_partition(0, partition(2), lease(1, 0, 12, 100))
                .expect("second source");
        }
        let first = command(2, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        let second = command(1, partition(2), lease(1, 0, 12, 100), lease(2, 1, 23, 200));
        let left_result = left.prepare_batch(1, [first, second]);
        let right_result = right.prepare_batch(1, [second, first]);
        assert_eq!(left_result, right_result);
        assert_eq!(
            left_result
                .iter()
                .map(|item| item.transaction)
                .collect::<Vec<_>>(),
            vec![HandoffTransactionId(1), HandoffTransactionId(2)]
        );
    }

    #[test]
    fn aborted_and_unknown_transactions_have_stable_results() {
        let mut mesh = registered_mesh();
        assert_eq!(
            mesh.commit_handoff(1, HandoffTransactionId(99)),
            Err(HandoffError::UnknownTransaction(HandoffTransactionId(99)))
        );
        let command = command(1, partition(1), lease(1, 0, 11, 100), lease(2, 1, 22, 200));
        mesh.prepare_handoff(1, command).expect("prepare");
        assert_eq!(
            mesh.abort_handoff(HandoffTransactionId(1)),
            Ok(AbortOutcome::Aborted)
        );
        assert_eq!(
            mesh.abort_handoff(HandoffTransactionId(1)),
            Ok(AbortOutcome::AlreadyAborted)
        );
        assert_eq!(
            mesh.prepare_handoff(1, command),
            Err(HandoffError::TransactionAborted(HandoffTransactionId(1)))
        );
    }
}
