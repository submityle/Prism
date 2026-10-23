//! Deterministic world-to-network policy contracts for Prism.
//!
//! The crate owns no sockets or world state. It joins the spatial contracts
//! from [`prism_world_stream`] to durable identity and field merge contracts
//! from [`prism_world_persist`], and exposes a host-provided reliable session
//! seam over those messages.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "crypto")]
mod crypto_transport;
#[cfg(feature = "crypto")]
mod psk_handshake;
mod server_mesh;
mod session;
#[cfg(feature = "std")]
mod udp_transport;
mod wire;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;

#[cfg(feature = "crypto")]
pub use crypto_transport::{
    Aes256GcmTransport, CryptoTransportError, CryptoTransportErrorSource, ReplayWindow,
};
pub use prism_world_persist::{
    ActorId, CrdtError, FieldId, LogicalClock, LwwRegister, OverrideDelta, OverrideValue,
    PersistentId, Transaction, TransactionId,
};
pub use prism_world_stream::{AabbI64, GlobalPos, SectorId};
#[cfg(feature = "crypto")]
pub use psk_handshake::{
    ClientHandshake, ClientHello, HandshakeError, SecureSessionConfig, ServerHandshake, ServerHello,
};
pub use server_mesh::{
    AbortOutcome, AdmissionError, ApplyOutcome, AuthorityEpoch, AuthorityLease, AuthorityLoad,
    BatchPrepareResult, CommitOutcome, EpochError, HandoffCommand,
    HandoffError as ServerMeshHandoffError, HandoffReceipt, HandoffSnapshot, HandoffSnapshotState,
    HandoffTicket, HandoffTransactionId, LeaseError, LeaseId, LeaseRole, MemoryMeshTransport,
    MemoryMeshTransportError, MeshConfigError, MeshError, MeshFailureDetector, MeshMessage,
    MeshNodeError, MeshNodeEvent, MeshNodeOpenError, MeshPacket, MeshPersistence, MeshReject,
    MeshReply, MeshRequest, MeshSnapshotError, MeshTransport, PartitionRecord, PrepareOutcome,
    ServerMesh, ServerMeshConfig, ServerMeshNode, ServerMeshSnapshot,
};
pub use session::{
    AuthToken, ByteFlushError, ByteReceiveError, ByteStreamReceiveError, ByteStreamReceiveStatus,
    FlushError, FlushReport, MemoryTransport, MemoryTransportError, ReceiveError, ReceiveOutcome,
    Session, SessionConfig, SessionConfigError, SessionError, SessionId, SessionState, Transport,
    TransportPacket, WorldMessage,
};
#[cfg(feature = "std")]
pub use udp_transport::{UdpByteTransport, UdpByteTransportConfig, UdpByteTransportError};
pub use wire::{
    decode_packet, encode_packet, ByteFrameDecoder, ByteTransport, FrameDecodeStatus,
    FrameDecoderError, MemoryByteTransport, MemoryByteTransportError, WireCodecError,
    MAX_TRANSPORT_PAYLOAD,
};

/// Stable identity for an AOI subscription.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SubscriptionId(pub u128);

/// A spatial subscription owned by one network participant.
///
/// Bounds are inclusive. `min` and `max` are inherited from
/// [`prism_world_stream::AabbI64`], so the subscription uses the same integer
/// world-space coordinate convention as the world system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AoiSubscription {
    id: SubscriptionId,
    subscriber: ActorId,
    bounds: AabbI64,
}

impl AoiSubscription {
    /// Creates a subscription with inclusive world-space bounds.
    pub const fn new(id: SubscriptionId, subscriber: ActorId, bounds: AabbI64) -> Self {
        Self {
            id,
            subscriber,
            bounds,
        }
    }

    /// Returns the subscription identity.
    pub const fn id(self) -> SubscriptionId {
        self.id
    }

    /// Returns the subscriber identity.
    pub const fn subscriber(self) -> ActorId {
        self.subscriber
    }

    /// Returns the inclusive world-space bounds.
    pub const fn bounds(self) -> AabbI64 {
        self.bounds
    }

    /// Returns whether a world position is inside the inclusive AOI.
    pub const fn contains(self, position: GlobalPos) -> bool {
        position.x >= self.bounds.min.x
            && position.x <= self.bounds.max.x
            && position.y >= self.bounds.min.y
            && position.y <= self.bounds.max.y
            && position.z >= self.bounds.min.z
            && position.z <= self.bounds.max.z
    }

    /// Returns whether two inclusive world-space bounds overlap or touch.
    pub const fn intersects(self, bounds: AabbI64) -> bool {
        self.bounds.min.x <= bounds.max.x
            && self.bounds.max.x >= bounds.min.x
            && self.bounds.min.y <= bounds.max.y
            && self.bounds.max.y >= bounds.min.y
            && self.bounds.min.z <= bounds.max.z
            && self.bounds.max.z >= bounds.min.z
    }

    /// Returns whether an object is visible to this subscription at a position.
    pub const fn observes(self, _object: PersistentId, position: GlobalPos) -> bool {
        self.contains(position)
    }
}

/// A validated persistence transaction annotated with its world sector.
///
/// The transaction remains the durable source of object and field identity.
/// The sector is routing context only and is not used to resolve field
/// conflicts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeltaEnvelope {
    sector: SectorId,
    transaction: Transaction,
}

impl DeltaEnvelope {
    /// Creates an envelope from an already validated persistence transaction.
    pub const fn new(sector: SectorId, transaction: Transaction) -> Self {
        Self {
            sector,
            transaction,
        }
    }

    /// Returns the world sector associated with this envelope.
    pub const fn sector(&self) -> SectorId {
        self.sector
    }

    /// Returns the durable transaction carried by this envelope.
    pub const fn transaction(&self) -> &Transaction {
        &self.transaction
    }

    /// Returns the durable object identity carried by this envelope.
    pub const fn object(&self) -> PersistentId {
        self.transaction.object()
    }

    /// Returns the stable transaction identity.
    pub const fn transaction_id(&self) -> TransactionId {
        self.transaction.id()
    }
}

impl Ord for DeltaEnvelope {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sector
            .cmp(&other.sector)
            .then_with(|| {
                self.transaction
                    .sequence()
                    .cmp(&other.transaction.sequence())
            })
            .then_with(|| self.transaction.id().cmp(&other.transaction.id()))
            .then_with(|| self.transaction.actor().cmp(&other.transaction.actor()))
            .then_with(|| self.transaction.object().cmp(&other.transaction.object()))
    }
}

impl PartialOrd for DeltaEnvelope {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Replication behavior for one authority boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicationPolicy {
    /// Only the configured authority may publish a delta.
    AuthorityOnly,
    /// Any actor may publish a delta; persistence resolves field conflicts by
    /// its deterministic LWW ordering.
    SharedLww,
    /// No actor may publish a delta.
    ReadOnly,
}

/// Authority and replication policy for one world/network boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorityPolicy {
    authority: ActorId,
    replication: ReplicationPolicy,
}

impl AuthorityPolicy {
    /// Creates a policy with one current authority.
    pub const fn new(authority: ActorId, replication: ReplicationPolicy) -> Self {
        Self {
            authority,
            replication,
        }
    }

    /// Returns the actor currently allowed to perform authority operations.
    pub const fn authority(self) -> ActorId {
        self.authority
    }

    /// Returns the configured replication behavior.
    pub const fn replication(self) -> ReplicationPolicy {
        self.replication
    }

    /// Checks whether an actor may publish the envelope's transaction.
    pub fn authorize_delta(&self, envelope: &DeltaEnvelope) -> Result<(), PermissionError> {
        match self.replication {
            ReplicationPolicy::ReadOnly => Err(PermissionError::ReadOnly),
            ReplicationPolicy::AuthorityOnly
                if envelope.transaction().actor() != self.authority =>
            {
                Err(PermissionError::NotAuthority {
                    expected: self.authority,
                    actual: envelope.transaction().actor(),
                })
            }
            ReplicationPolicy::AuthorityOnly | ReplicationPolicy::SharedLww => Ok(()),
        }
    }

    /// Checks whether an actor may commit a sector handoff.
    pub fn authorize_handoff(
        &self,
        requester: ActorId,
        handoff: &SectorHandoff,
    ) -> Result<(), PermissionError> {
        if requester != self.authority {
            return Err(PermissionError::NotAuthority {
                expected: self.authority,
                actual: requester,
            });
        }
        if handoff.previous_authority() != self.authority {
            return Err(PermissionError::NotAuthority {
                expected: self.authority,
                actual: handoff.previous_authority(),
            });
        }
        Ok(())
    }
}

/// Permission failures at the world/network seam.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermissionError {
    /// The policy is read-only.
    ReadOnly,
    /// The caller does not match the policy's current authority.
    NotAuthority {
        /// Actor required by the policy.
        expected: ActorId,
        /// Actor that attempted the operation.
        actual: ActorId,
    },
}

/// An immutable request to transfer one object between sectors and authorities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectorHandoff {
    object: PersistentId,
    from: SectorId,
    to: SectorId,
    previous_authority: ActorId,
    next_authority: ActorId,
    epoch: u64,
}

impl SectorHandoff {
    /// Creates a handoff request. The source and destination must differ, as
    /// must the old and new authority.
    pub fn new(
        object: PersistentId,
        from: SectorId,
        to: SectorId,
        previous_authority: ActorId,
        next_authority: ActorId,
        epoch: u64,
    ) -> Result<Self, HandoffError> {
        if from == to {
            return Err(HandoffError::SameSector);
        }
        if previous_authority == next_authority {
            return Err(HandoffError::SameAuthority);
        }
        Ok(Self {
            object,
            from,
            to,
            previous_authority,
            next_authority,
            epoch,
        })
    }

    /// Returns the object being handed off.
    pub const fn object(self) -> PersistentId {
        self.object
    }

    /// Returns the source sector.
    pub const fn from(self) -> SectorId {
        self.from
    }

    /// Returns the destination sector.
    pub const fn to(self) -> SectorId {
        self.to
    }

    /// Returns the authority before the handoff.
    pub const fn previous_authority(self) -> ActorId {
        self.previous_authority
    }

    /// Returns the authority after the handoff.
    pub const fn next_authority(self) -> ActorId {
        self.next_authority
    }

    /// Returns the strictly increasing handoff epoch supplied by the caller.
    pub const fn epoch(self) -> u64 {
        self.epoch
    }
}

/// A placement and authority record tracked by [`HandoffLedger`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectPlacement {
    /// Current sector.
    pub sector: SectorId,
    /// Current authority.
    pub authority: ActorId,
    /// Last committed handoff epoch.
    pub epoch: u64,
}

/// Handoff validation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffError {
    /// The request does not move the object to another sector.
    SameSector,
    /// The request does not change authority.
    SameAuthority,
    /// A placement was registered twice.
    AlreadyRegistered(PersistentId),
    /// The object has no current placement.
    UnknownObject(PersistentId),
    /// The request starts from a sector other than the ledger's current sector.
    SourceMismatch {
        /// Sector recorded by the ledger.
        expected: SectorId,
        /// Sector supplied by the request.
        actual: SectorId,
    },
    /// The request uses an epoch that is not newer than the current epoch.
    StaleEpoch {
        /// Current ledger epoch.
        current: u64,
        /// Epoch supplied by the request.
        received: u64,
    },
    /// The request names a different current authority.
    AuthorityMismatch {
        /// Authority recorded by the ledger.
        expected: ActorId,
        /// Authority supplied by the request.
        actual: ActorId,
    },
    /// The caller failed the configured authority policy.
    Permission(PermissionError),
}

/// Deterministic placement ledger for sector handoffs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HandoffLedger {
    placements: BTreeMap<PersistentId, ObjectPlacement>,
}

impl HandoffLedger {
    /// Creates an empty handoff ledger.
    pub const fn new() -> Self {
        Self {
            placements: BTreeMap::new(),
        }
    }

    /// Registers an object's initial world placement.
    pub fn register(
        &mut self,
        object: PersistentId,
        placement: ObjectPlacement,
    ) -> Result<(), HandoffError> {
        if self.placements.contains_key(&object) {
            return Err(HandoffError::AlreadyRegistered(object));
        }
        self.placements.insert(object, placement);
        Ok(())
    }

    /// Returns the current placement of an object.
    pub fn placement(&self, object: PersistentId) -> Option<ObjectPlacement> {
        self.placements.get(&object).copied()
    }

    /// Applies an authorized, newer handoff to the current placement.
    pub fn apply(
        &mut self,
        requester: ActorId,
        policy: &AuthorityPolicy,
        handoff: SectorHandoff,
    ) -> Result<(), HandoffError> {
        let current = self
            .placements
            .get(&handoff.object)
            .copied()
            .ok_or(HandoffError::UnknownObject(handoff.object))?;
        policy
            .authorize_handoff(requester, &handoff)
            .map_err(HandoffError::Permission)?;
        if current.sector != handoff.from {
            return Err(HandoffError::SourceMismatch {
                expected: current.sector,
                actual: handoff.from,
            });
        }
        if current.authority != handoff.previous_authority {
            return Err(HandoffError::AuthorityMismatch {
                expected: current.authority,
                actual: handoff.previous_authority,
            });
        }
        if handoff.epoch <= current.epoch {
            return Err(HandoffError::StaleEpoch {
                current: current.epoch,
                received: handoff.epoch,
            });
        }
        self.placements.insert(
            handoff.object,
            ObjectPlacement {
                sector: handoff.to,
                authority: handoff.next_authority,
                epoch: handoff.epoch,
            },
        );
        Ok(())
    }
}

/// Errors returned while merging envelopes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MergeError {
    /// A transaction ID was reused with different envelope contents.
    ConflictingTransaction(TransactionId),
    /// The persistence CRDT rejected a transaction batch.
    Crdt(CrdtError),
}

/// Object-indexed, order-independent merge state.
///
/// Envelopes are indexed by [`PersistentId`] while field arbitration is
/// delegated to [`prism_world_persist::CrdtState`]. Repeating the exact same
/// envelope is idempotent; reusing a transaction ID with different contents
/// is rejected.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PersistentMergeState {
    envelopes: BTreeMap<PersistentId, BTreeMap<TransactionId, DeltaEnvelope>>,
    transaction_objects: BTreeMap<TransactionId, PersistentId>,
    crdt: prism_world_persist::CrdtState,
}

impl PersistentMergeState {
    /// Creates an empty object-indexed merge state.
    pub const fn new() -> Self {
        Self {
            envelopes: BTreeMap::new(),
            transaction_objects: BTreeMap::new(),
            crdt: prism_world_persist::CrdtState::new(),
        }
    }

    /// Merges a batch of envelopes without depending on arrival order.
    pub fn merge<'a, I>(&mut self, envelopes: I) -> Result<(), MergeError>
    where
        I: IntoIterator<Item = &'a DeltaEnvelope>,
    {
        let mut pending: BTreeMap<TransactionId, &DeltaEnvelope> = BTreeMap::new();
        for envelope in envelopes {
            let id = envelope.transaction_id();
            if let Some(existing) = self.find_envelope(id) {
                if existing != envelope {
                    return Err(MergeError::ConflictingTransaction(id));
                }
                continue;
            }
            if let Some(existing) = pending.get(&id) {
                if *existing != envelope {
                    return Err(MergeError::ConflictingTransaction(id));
                }
                continue;
            }
            pending.insert(id, envelope);
        }

        let mut next_crdt = self.crdt.clone();
        next_crdt
            .merge(pending.values().map(|envelope| envelope.transaction()))
            .map_err(MergeError::Crdt)?;
        for envelope in pending.into_values() {
            self.envelopes
                .entry(envelope.object())
                .or_default()
                .insert(envelope.transaction_id(), envelope.clone());
            self.transaction_objects
                .insert(envelope.transaction_id(), envelope.object());
        }
        self.crdt = next_crdt;
        Ok(())
    }

    /// Returns the materialized value for one object field.
    pub fn get(&self, object: PersistentId, field: FieldId) -> Option<&OverrideValue> {
        self.crdt.get(object, field)
    }

    /// Returns the winning register for one object field.
    pub fn register(&self, object: PersistentId, field: FieldId) -> Option<&LwwRegister> {
        self.crdt.register(object, field)
    }

    /// Returns all winning field registers for one object in field order.
    pub fn object_registers(&self, object: PersistentId) -> Vec<(FieldId, &LwwRegister)> {
        self.crdt
            .iter()
            .filter(|(key, _)| key.object == object)
            .map(|(key, register)| (key.field, register))
            .collect()
    }

    /// Returns the envelopes associated with one object in transaction order.
    pub fn object_envelopes(&self, object: PersistentId) -> Vec<&DeltaEnvelope> {
        self.envelopes
            .get(&object)
            .map(|envelopes| envelopes.values().collect())
            .unwrap_or_default()
    }

    /// Returns the number of unique accepted transactions.
    pub fn len(&self) -> usize {
        self.envelopes.values().map(BTreeMap::len).sum()
    }

    /// Returns whether no envelopes have been accepted.
    pub fn is_empty(&self) -> bool {
        self.envelopes.is_empty()
    }

    fn find_envelope(&self, id: TransactionId) -> Option<&DeltaEnvelope> {
        let object = self.transaction_objects.get(&id)?;
        self.envelopes.get(object)?.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_world_persist::{decode_log, encode_log, AppendOnlyLog};

    fn bounds() -> AabbI64 {
        AabbI64::new(GlobalPos::new(-10, -10, -10), GlobalPos::new(10, 10, 10))
            .expect("test bounds are ordered")
    }

    fn transaction(
        id: u128,
        actor: u128,
        clock: u64,
        sequence: u64,
        object: u128,
        value: i64,
    ) -> Transaction {
        Transaction::new(
            TransactionId(id),
            ActorId(actor),
            LogicalClock(clock),
            sequence,
            PersistentId(object),
            OverrideDelta::single(FieldId(1), OverrideValue::Integer(value))
                .expect("test delta is valid"),
            OverrideDelta::single(FieldId(1), OverrideValue::Delete)
                .expect("test inverse is valid"),
        )
        .expect("test transaction is valid")
    }

    #[test]
    fn aoi_uses_inclusive_min_and_max_boundaries() {
        let subscription = AoiSubscription::new(SubscriptionId(1), ActorId(7), bounds());
        assert!(subscription.contains(GlobalPos::new(-10, 0, 0)));
        assert!(subscription.contains(GlobalPos::new(10, 10, -10)));
        assert!(!subscription.contains(GlobalPos::new(11, 0, 0)));
        assert!(subscription.intersects(
            AabbI64::new(GlobalPos::new(10, 10, 10), GlobalPos::new(12, 12, 12))
                .expect("test bounds are ordered")
        ));
    }

    #[test]
    fn authority_policy_rejects_unauthorized_delta() {
        let envelope =
            DeltaEnvelope::new(SectorId::new(0, 0, 0, 0), transaction(1, 99, 1, 0, 5, 10));
        let policy = AuthorityPolicy::new(ActorId(7), ReplicationPolicy::AuthorityOnly);
        assert_eq!(
            policy.authorize_delta(&envelope),
            Err(PermissionError::NotAuthority {
                expected: ActorId(7),
                actual: ActorId(99),
            })
        );
    }

    #[test]
    fn handoff_requires_current_authority_and_new_epoch() {
        let object = PersistentId(5);
        let first = SectorId::new(0, 0, 0, 0);
        let second = SectorId::new(0, 1, 0, 0);
        let mut ledger = HandoffLedger::new();
        ledger
            .register(
                object,
                ObjectPlacement {
                    sector: first,
                    authority: ActorId(7),
                    epoch: 0,
                },
            )
            .expect("initial placement is unique");
        let handoff = SectorHandoff::new(object, first, second, ActorId(7), ActorId(8), 1)
            .expect("handoff changes sector and authority");
        let policy = AuthorityPolicy::new(ActorId(7), ReplicationPolicy::AuthorityOnly);
        assert_eq!(
            ledger.apply(ActorId(9), &policy, handoff),
            Err(HandoffError::Permission(PermissionError::NotAuthority {
                expected: ActorId(7),
                actual: ActorId(9),
            }))
        );
        ledger
            .apply(ActorId(7), &policy, handoff)
            .expect("current authority may hand off");
        assert_eq!(
            ledger.placement(object),
            Some(ObjectPlacement {
                sector: second,
                authority: ActorId(8),
                epoch: 1,
            })
        );
        let stale = SectorHandoff::new(object, second, first, ActorId(8), ActorId(7), 1)
            .expect("stale handoff keeps the current source and authority");
        let next_policy = AuthorityPolicy::new(ActorId(8), ReplicationPolicy::AuthorityOnly);
        assert_eq!(
            ledger.apply(ActorId(8), &next_policy, stale),
            Err(HandoffError::StaleEpoch {
                current: 1,
                received: 1,
            })
        );
    }

    #[test]
    fn merge_is_order_independent_and_indexed_by_persistent_id() {
        let sector = SectorId::new(0, 0, 0, 0);
        let first = DeltaEnvelope::new(sector, transaction(1, 2, 1, 0, 42, 10));
        let second = DeltaEnvelope::new(sector, transaction(2, 3, 2, 1, 42, 20));
        let unrelated = DeltaEnvelope::new(sector, transaction(3, 2, 1, 2, 99, 30));
        let mut left = PersistentMergeState::new();
        let mut right = PersistentMergeState::new();
        left.merge([&first, &second, &unrelated])
            .expect("left merge is valid");
        right
            .merge([&unrelated, &second, &first])
            .expect("right merge is valid");
        assert_eq!(left, right);
        assert_eq!(
            left.get(PersistentId(42), FieldId(1)),
            Some(&OverrideValue::Integer(20))
        );
        assert_eq!(left.object_envelopes(PersistentId(42)).len(), 2);
        assert_eq!(left.object_registers(PersistentId(42)).len(), 1);
        left.merge(vec![&first])
            .expect("repeating an envelope is idempotent");
        assert_eq!(left.len(), 3);
    }

    #[test]
    fn durable_persist_log_delta_round_trips_through_session_and_crdt_merge() {
        let sector = SectorId::new(0, 3, -2, 1);
        let original = transaction(0xabc, 7, 10, 0, 0xdef, 1234);
        let mut source_log = AppendOnlyLog::new();
        source_log
            .append(original)
            .expect("source persistence accepts the transaction");
        let wire_log = encode_log(&source_log).expect("canonical persist log encodes");
        let recovered_log = decode_log(&wire_log).expect("canonical persist log decodes");
        assert_eq!(recovered_log, source_log);
        let envelope = DeltaEnvelope::new(
            sector,
            recovered_log
                .get(0)
                .expect("one durable transaction")
                .clone(),
        );

        let config = |actor| SessionConfig {
            session: SessionId(0x55),
            actor: ActorId(actor),
            token: AuthToken(0xfeed),
            authority: AuthorityPolicy::new(ActorId(7), ReplicationPolicy::AuthorityOnly),
            outbound_capacity: 2,
            inbound_capacity: 2,
            max_attempts: 3,
            retry_after_ticks: 1,
        };
        let mut sender = Session::new(config(7)).expect("valid sender");
        let mut receiver = Session::new(config(8)).expect("valid receiver");
        sender
            .authenticate(AuthToken(0xfeed))
            .expect("sender authenticated");
        receiver
            .authenticate(AuthToken(0xfeed))
            .expect("receiver authenticated");
        sender
            .send(WorldMessage::Delta(envelope.clone()))
            .expect("authority queues durable delta");
        let (mut sender_transport, mut receiver_transport) =
            MemoryTransport::pair(4).expect("bounded transport");
        assert_eq!(sender.flush(&mut sender_transport, 0).unwrap().sent, 1);
        let received = receiver_transport.try_recv().expect("data packet");
        let delivered = receiver
            .receive(received)
            .expect("receiver validates delta");
        let ReceiveOutcome::Delivered { messages, .. } = delivered else {
            panic!("first reliable data packet must be delivered");
        };
        let [WorldMessage::Delta(received_envelope)] = messages.as_slice() else {
            panic!("one persistence delta must be delivered");
        };
        assert_eq!(received_envelope, &envelope);

        let mut merge = PersistentMergeState::new();
        merge
            .merge([received_envelope])
            .expect("received transaction merges into persistence CRDT");
        assert_eq!(
            merge.get(PersistentId(0xdef), FieldId(1)),
            Some(&OverrideValue::Integer(1234))
        );

        receiver
            .flush(&mut receiver_transport, 0)
            .expect("receiver emits cumulative acknowledgement");
        let ack = sender_transport.try_recv().expect("ack packet");
        sender.receive(ack).expect("sender accepts ack");
        assert_eq!(sender.pending(), 0);
        assert_eq!(recovered_log.get(0), Some(received_envelope.transaction()));
    }
}
