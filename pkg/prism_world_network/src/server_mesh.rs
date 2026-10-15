//! Bounded server-mesh and authority-handoff contracts.
//!
//! This module is a deterministic control-plane state machine for partition
//! ownership. It owns only bounded in-memory metadata: partition identities,
//! authority leases, epochs, admission reservations, and handoff outcomes.
//! It does not open sockets, serialize messages, detect failed hosts, or write
//! durable state. A host may persist or transport the public values through a
//! separate provider.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

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
        if let Some(record) = self.transactions.get(&command.transaction).copied() {
            if record.command != command {
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
        self.validate_source(now, current.authority, command.source)?;
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
        self.validate_source(now, current.authority, record.command.source)?;
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
                        state: HandoffState::Aborted,
                    },
                );
                Ok(AbortOutcome::Aborted)
            }
            HandoffState::Committed(_) => Err(HandoffError::AlreadyCommitted(transaction)),
            HandoffState::Aborted => Ok(AbortOutcome::AlreadyAborted),
        }
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
        match self.prepare_handoff(now, command)? {
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

    fn validate_source(
        &self,
        now: u64,
        current: AuthorityLease,
        source: AuthorityLease,
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
        if !current.is_valid_at(now) {
            return Err(HandoffError::Lease(LeaseError::Expired {
                role: LeaseRole::Current,
                expires_at: current.expires_at,
                now,
            }));
        }
        if !source.is_valid_at(now) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
