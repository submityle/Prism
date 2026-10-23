//! Bounded multi-peer authority orchestration.
//!
//! [`ServerReplication`] owns one authenticated transport and one replicated
//! record. [`MultiPeerAuthority`] owns the server-side peer identity table and
//! connects it to [`AuthorityAoi`]. AOI changes are encoded as versioned event
//! payloads and retained in a bounded per-peer queue until the corresponding
//! server connection accepts them.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use std::sync::Mutex;

use prism_app::{App, First, Plugin};
use prism_ecs::resource::Resource;

use crate::authority_aoi::{
    AuthorityAoi, AuthorityAoiConfig, AuthorityAoiDelta, AuthorityAoiError,
};
use crate::delta::Schema;
use crate::multiplex::MuxSendError;
use crate::session::SessionError;
use crate::transport::Transport;

use super::{ExternalCommandModeError, ReplicationConfig, ServerReplication, ServerStrategy};

/// The stable identity used by the authority peer table and AOI events.
pub type PeerId = u64;

const AOI_EVENT_MAGIC: u8 = 0xa1;
const AOI_EVENT_VERSION: u8 = 1;
const AOI_EVENT_HEADER_LEN: usize = 2 + 8 + 8 + 4 + 4 + 4;
const AOI_REPLICATION_MAGIC: u8 = 0xa2;
const AOI_REPLICATION_VERSION_V1: u8 = 1;
const AOI_REPLICATION_VERSION: u8 = 2;
const AOI_REPLICATION_V1_HEADER_LEN: usize = 2 + 8 + 8 + 4 + 4;
const AOI_REPLICATION_HEADER_LEN: usize = AOI_REPLICATION_V1_HEADER_LEN + 4;

/// A peer-specific projection of an entity from the shared authority world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntitySnapshotPayload {
    /// Stable entity id used by the AOI index.
    pub entity_id: u32,
    /// Provider-serialized entity state. The provider owns the schema/version.
    pub bytes: Vec<u8>,
}

/// One decoded reliable command removed from a peer connection for a shared
/// authority executor. Ordering within a peer is preserved; across peers the
/// coordinator emits a deterministic round-robin order for the currently
/// queued commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundAuthorityCommand {
    /// Stable identity of the peer that submitted the command.
    pub peer_id: PeerId,
    /// Command payload after RPC framing has been removed.
    pub payload: Vec<u8>,
}

/// A reliable command temporarily reserved for an external authority consumer.
/// The command remains in its peer's FIFO until acknowledged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeasedAuthorityCommand {
    /// Stable identity of the peer that submitted the command.
    pub peer_id: PeerId,
    /// Connection-local identity used to acknowledge or release this lease.
    pub command_id: u128,
    /// Command payload after RPC framing has been removed.
    pub payload: Vec<u8>,
}

/// A generic authority-side entity projection operation. It carries only
/// stable network identity and caller-serialized state; gameplay components and
/// ECS entity types remain application-owned.
#[derive(Clone, Debug, PartialEq)]
pub enum AuthorityEntityProjection {
    /// Insert or move an entity in the AOI index and publish entered snapshots.
    Upsert {
        /// Stable replication identity, never an ECS allocation slot.
        entity_id: u32,
        /// Authoritative spatial position.
        position: (f32, f32),
        /// Canonical provider payload for newly-visible peers.
        snapshot: Vec<u8>,
    },
    /// Remove an entity from the AOI index and publish leave edges.
    Remove {
        /// Stable replication identity.
        entity_id: u32,
    },
    /// Publish a changed snapshot to the current viewers without changing AOI.
    Update {
        /// Stable replication identity.
        entity_id: u32,
        /// Canonical provider payload for current viewers.
        snapshot: Vec<u8>,
    },
}

/// One peer's AOI update with real entity snapshots and lifecycle removals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityAoiReplicationPayload {
    /// Stable peer identity.
    pub peer_id: PeerId,
    /// Monotonic subscription revision.
    pub revision: u64,
    /// Full snapshots for newly visible entities.
    pub entered: Vec<EntitySnapshotPayload>,
    /// Latest provider-produced states for entities that remain visible.
    pub updated: Vec<EntitySnapshotPayload>,
    /// Entity ids to despawn from this peer's replica.
    pub left: Vec<u32>,
}

/// A provider failure while projecting the shared authoritative world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityReplicationProviderError {
    /// The entity ceased to exist or is unavailable in the authoritative world.
    EntityUnavailable,
    /// The provider could not serialize the entity within its own schema.
    SerializationFailed,
}

impl core::fmt::Display for AuthorityReplicationProviderError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::EntityUnavailable => "authoritative AOI entity is unavailable",
            Self::SerializationFailed => "authoritative AOI entity serialization failed",
        })
    }
}

/// Project AOI edges from one shared authority world into peer-specific wire
/// payloads. Implementations must read the same canonical world for every peer;
/// snapshots are not sourced from per-peer `ServerSession` records. Calls may
/// be retried while transport backpressure persists, so serialization must be
/// deterministic for a given `(peer_id, entity_id, AOI revision)`.
pub trait AuthorityReplicationProvider: Send + Sync + 'static {
    /// Serialize one entity from the shared authoritative state for this peer.
    fn snapshot_entity(
        &self,
        peer_id: PeerId,
        entity_id: u32,
        aoi_revision: u64,
    ) -> Result<Vec<u8>, AuthorityReplicationProviderError>;
}

/// Configuration for [`MultiPeerAuthority`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiPeerAuthorityConfig {
    /// Bounds for admitted peers and indexed entities.
    pub aoi: AuthorityAoiConfig,
    /// Maximum number of AOI event payloads retained for one peer while its
    /// handshake or reliable event lane is not accepting new messages.
    pub max_pending_aoi_events_per_peer: usize,
    /// Maximum encoded provider payload size admitted to one reliable event.
    pub max_aoi_payload_bytes: usize,
}

impl Default for MultiPeerAuthorityConfig {
    fn default() -> Self {
        Self {
            aoi: AuthorityAoiConfig::default(),
            max_pending_aoi_events_per_peer: 64,
            max_aoi_payload_bytes: 64 * 1024,
        }
    }
}

/// Errors raised while constructing or mutating the multi-peer authority.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiPeerAuthorityError {
    /// A multi-peer queue bound or AOI bound is invalid for the wire format.
    InvalidConfig,
    /// The underlying bounded AOI rejected an operation.
    Aoi(AuthorityAoiError),
    /// A peer's replicated server could not be constructed.
    Session(SessionError),
    /// A peer's initial AOI event could not be accepted by its transport.
    Send(MuxSendError),
    /// A generated AOI event exceeded the configured pending queue bound.
    PendingQueueFull {
        /// Peer whose retained update queue is full.
        peer_id: PeerId,
    },
    /// A peer id was not found in the authority table.
    UnknownPeer {
        /// Peer that was not admitted.
        peer_id: PeerId,
    },
    /// A wire event cannot represent one of the supplied collection lengths.
    EventCountOverflow,
    /// The provider could not project a retained AOI update; it remains queued.
    Provider(AuthorityReplicationProviderError),
    /// A provider snapshot exceeds the configured byte bound.
    AoiPayloadTooLarge {
        /// Encoded payload size.
        actual: usize,
        /// Configured encoded payload bound.
        max: usize,
    },
    /// A peer cannot transfer command ownership from its record authority.
    ExternalCommandMode {
        /// Peer whose authority state prevents the transition.
        peer_id: PeerId,
        /// The state that makes the transition unsafe.
        reason: ExternalCommandModeError,
    },
    /// The requested command lease is stale or is not the peer's current head.
    InvalidCommandLease {
        /// Peer that owns the command lease.
        peer_id: PeerId,
        /// Connection-local command identity supplied by the consumer.
        command_id: u128,
    },
    /// The authority exhausted its never-reused command lease identity space.
    CommandLeaseIdExhausted,
}

impl core::fmt::Display for MultiPeerAuthorityError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => {
                formatter.write_str("multi-peer authority configuration is invalid")
            }
            Self::Aoi(error) => error.fmt(formatter),
            Self::Session(error) => write!(formatter, "{error:?}"),
            Self::Send(error) => write!(formatter, "{error:?}"),
            Self::PendingQueueFull { peer_id } => write!(
                formatter,
                "multi-peer AOI pending queue is full for peer {peer_id}"
            ),
            Self::UnknownPeer { peer_id } => write!(formatter, "unknown authority peer {peer_id}"),
            Self::EventCountOverflow => {
                formatter.write_str("AOI event collection length exceeds u32")
            }
            Self::Provider(error) => error.fmt(formatter),
            Self::AoiPayloadTooLarge { actual, max } => write!(
                formatter,
                "AOI payload length {actual} exceeds configured maximum {max}"
            ),
            Self::ExternalCommandMode { peer_id, reason } => {
                write!(
                    formatter,
                    "peer {peer_id} cannot enter external command mode: {reason:?}"
                )
            }
            Self::InvalidCommandLease {
                peer_id,
                command_id,
            } => write!(
                formatter,
                "command lease {command_id} is not active for peer {peer_id}"
            ),
            Self::CommandLeaseIdExhausted => {
                formatter.write_str("authority command lease identity space exhausted")
            }
        }
    }
}

/// Encode provider-produced AOI snapshots, state updates, and despawns for
/// reliable delivery. Version 2 adds a distinct `updated` collection; the
/// decoder continues to accept version 1 payloads.
pub fn encode_authority_aoi_replication_payload(
    payload: &AuthorityAoiReplicationPayload,
) -> Result<Vec<u8>, MultiPeerAuthorityError> {
    let entered = u32::try_from(payload.entered.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let updated = u32::try_from(payload.updated.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let left = u32::try_from(payload.left.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let mut capacity = AOI_REPLICATION_HEADER_LEN;
    for snapshot in payload.entered.iter().chain(payload.updated.iter()) {
        let len = u32::try_from(snapshot.bytes.len())
            .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
        capacity = capacity
            .checked_add(8)
            .and_then(|capacity| capacity.checked_add(len as usize))
            .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
    }
    capacity = capacity
        .checked_add(
            payload
                .left
                .len()
                .checked_mul(size_of::<u32>())
                .ok_or(MultiPeerAuthorityError::EventCountOverflow)?,
        )
        .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.push(AOI_REPLICATION_MAGIC);
    bytes.push(AOI_REPLICATION_VERSION);
    bytes.extend_from_slice(&payload.peer_id.to_le_bytes());
    bytes.extend_from_slice(&payload.revision.to_le_bytes());
    bytes.extend_from_slice(&entered.to_le_bytes());
    bytes.extend_from_slice(&left.to_le_bytes());
    bytes.extend_from_slice(&updated.to_le_bytes());
    for snapshot in payload.entered.iter().chain(payload.updated.iter()) {
        let len = u32::try_from(snapshot.bytes.len())
            .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
        bytes.extend_from_slice(&snapshot.entity_id.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&snapshot.bytes);
    }
    for entity_id in &payload.left {
        bytes.extend_from_slice(&entity_id.to_le_bytes());
    }
    Ok(bytes)
}

/// Decode a provider-produced AOI replication payload under explicit aggregate
/// entity-operation and byte limits. `max_entities` bounds the sum of enter,
/// update, and leave entries in one payload.
pub fn decode_authority_aoi_replication_payload(
    bytes: &[u8],
    max_entities: usize,
    max_payload_bytes: usize,
) -> Result<AuthorityAoiReplicationPayload, AuthorityAoiEventError> {
    if bytes.len() > max_payload_bytes {
        return Err(AuthorityAoiEventError::BoundExceeded);
    }
    if bytes.len() < AOI_REPLICATION_V1_HEADER_LEN || bytes[0] != AOI_REPLICATION_MAGIC {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let version = bytes[1];
    let (header_len, updated) = match version {
        AOI_REPLICATION_VERSION_V1 => (AOI_REPLICATION_V1_HEADER_LEN, 0usize),
        AOI_REPLICATION_VERSION => {
            if bytes.len() < AOI_REPLICATION_HEADER_LEN {
                return Err(AuthorityAoiEventError::Malformed);
            }
            (
                AOI_REPLICATION_HEADER_LEN,
                u32::from_le_bytes(bytes[26..30].try_into().unwrap()) as usize,
            )
        }
        _ => return Err(AuthorityAoiEventError::Malformed),
    };
    let peer_id = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
    let revision = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    let entered = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    let left = u32::from_le_bytes(bytes[22..26].try_into().unwrap()) as usize;
    let entity_operations = entered
        .checked_add(updated)
        .and_then(|count| count.checked_add(left))
        .ok_or(AuthorityAoiEventError::Malformed)?;
    if entity_operations > max_entities {
        return Err(AuthorityAoiEventError::BoundExceeded);
    }
    let minimum_body = entered
        .checked_mul(8)
        .and_then(|size| size.checked_add(updated.checked_mul(8)?))
        .and_then(|size| size.checked_add(left.checked_mul(size_of::<u32>())?))
        .ok_or(AuthorityAoiEventError::Malformed)?;
    if minimum_body > bytes.len() - header_len {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let mut offset = header_len;
    let mut read_snapshots =
        |count: usize| -> Result<Vec<EntitySnapshotPayload>, AuthorityAoiEventError> {
            let mut snapshots = Vec::with_capacity(count);
            for _ in 0..count {
                if bytes.len().saturating_sub(offset) < 8 {
                    return Err(AuthorityAoiEventError::Malformed);
                }
                let entity_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                let len =
                    u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
                offset += 8;
                if bytes.len().saturating_sub(offset) < len {
                    return Err(AuthorityAoiEventError::Malformed);
                }
                snapshots.push(EntitySnapshotPayload {
                    entity_id,
                    bytes: bytes[offset..offset + len].to_vec(),
                });
                offset += len;
            }
            Ok(snapshots)
        };
    let snapshots = read_snapshots(entered)?;
    let updates = read_snapshots(updated)?;
    let left_bytes = left
        .checked_mul(4)
        .ok_or(AuthorityAoiEventError::Malformed)?;
    if bytes.len().saturating_sub(offset) != left_bytes {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let mut despawns = Vec::with_capacity(left);
    for _ in 0..left {
        despawns.push(u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ));
        offset += 4;
    }
    Ok(AuthorityAoiReplicationPayload {
        peer_id,
        revision,
        entered: snapshots,
        updated: updates,
        left: despawns,
    })
}

impl From<AuthorityAoiError> for MultiPeerAuthorityError {
    fn from(error: AuthorityAoiError) -> Self {
        Self::Aoi(error)
    }
}

impl From<SessionError> for MultiPeerAuthorityError {
    fn from(error: SessionError) -> Self {
        Self::Session(error)
    }
}

/// Errors raised while decoding an AOI event payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityAoiEventError {
    /// The event header, version, counts, or total length is invalid.
    Malformed,
    /// A collection exceeds the caller's explicit decode bound.
    BoundExceeded,
}

impl core::fmt::Display for AuthorityAoiEventError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::Malformed => "malformed authority AOI event",
            Self::BoundExceeded => "authority AOI event exceeds decode bound",
        })
    }
}

/// Encode one AOI enter/leave update for a peer's reliable event lane.
pub fn encode_authority_aoi_event(
    delta: &AuthorityAoiDelta,
) -> Result<Vec<u8>, MultiPeerAuthorityError> {
    let entered = u32::try_from(delta.entered.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let left =
        u32::try_from(delta.left.len()).map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let visible = u32::try_from(delta.visible.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let total = delta
        .entered
        .len()
        .checked_add(delta.left.len())
        .and_then(|length| length.checked_add(delta.visible.len()))
        .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
    let capacity = AOI_EVENT_HEADER_LEN
        .checked_add(
            total
                .checked_mul(size_of::<u32>())
                .ok_or(MultiPeerAuthorityError::EventCountOverflow)?,
        )
        .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.push(AOI_EVENT_MAGIC);
    bytes.push(AOI_EVENT_VERSION);
    bytes.extend_from_slice(&delta.peer_id.to_le_bytes());
    bytes.extend_from_slice(&delta.revision.to_le_bytes());
    bytes.extend_from_slice(&entered.to_le_bytes());
    bytes.extend_from_slice(&left.to_le_bytes());
    bytes.extend_from_slice(&visible.to_le_bytes());
    for id in delta
        .entered
        .iter()
        .chain(delta.left.iter())
        .chain(delta.visible.iter())
    {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    Ok(bytes)
}

/// Decode one AOI enter/leave update with an explicit per-collection bound.
pub fn decode_authority_aoi_event(
    bytes: &[u8],
    max_entities_per_peer: usize,
) -> Result<AuthorityAoiDelta, AuthorityAoiEventError> {
    if bytes.len() < AOI_EVENT_HEADER_LEN
        || bytes[0] != AOI_EVENT_MAGIC
        || bytes[1] != AOI_EVENT_VERSION
    {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let peer_id = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
    let revision = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    let entered = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    let left = u32::from_le_bytes(bytes[22..26].try_into().unwrap()) as usize;
    let visible = u32::from_le_bytes(bytes[26..30].try_into().unwrap()) as usize;
    if entered > max_entities_per_peer
        || left > max_entities_per_peer
        || visible > max_entities_per_peer
    {
        return Err(AuthorityAoiEventError::BoundExceeded);
    }
    let total = entered
        .checked_add(left)
        .and_then(|length| length.checked_add(visible))
        .ok_or(AuthorityAoiEventError::Malformed)?;
    let expected = AOI_EVENT_HEADER_LEN
        .checked_add(
            total
                .checked_mul(size_of::<u32>())
                .ok_or(AuthorityAoiEventError::Malformed)?,
        )
        .ok_or(AuthorityAoiEventError::Malformed)?;
    if expected != bytes.len() {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let mut offset = AOI_EVENT_HEADER_LEN;
    let mut read_ids = |count: usize| {
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }
        ids
    };
    Ok(AuthorityAoiDelta {
        peer_id,
        revision,
        entered: read_ids(entered),
        left: read_ids(left),
        visible: read_ids(visible),
    })
}

struct PeerAuthority<S: ServerStrategy> {
    server: ServerReplication<S>,
    leased_command: Option<(u128, u64)>,
    snapshot_overrides: BTreeMap<(u32, u64), VecDeque<Vec<u8>>>,
    pending_aoi_events: VecDeque<PendingAoiEvent>,
    last_aoi_send_error: Option<MuxSendError>,
    last_provider_error: Option<AuthorityReplicationProviderError>,
}

/// Fully validated AOI/outbox replacement produced by projection preflight.
/// It stays private to the atomic projection method and cannot be retained
/// across another authority mutation.
struct PreparedAuthorityProjection {
    aoi: AuthorityAoi,
    deltas: Vec<AuthorityAoiDelta>,
    overrides: Vec<(PeerId, u32, u64, Vec<u8>)>,
    events: Vec<(PeerId, PendingAoiEvent)>,
    touched: BTreeSet<PeerId>,
}

#[derive(Clone, Debug)]
enum PendingAoiEvent {
    Delta(AuthorityAoiDelta),
    EntityUpdate { entity_id: u32, revision: u64 },
}

/// A bounded authority coordinator for multiple authenticated peers.
///
/// The coordinator owns peer identity and AOI state. Each peer retains its own
/// [`ServerReplication`] and therefore its own transport, handshake, reliable
/// windows, and record. AOI updates are dispatched in deterministic peer-id
/// order and retained until the target server accepts them. This is an AOI
/// projection/transport primitive; it does not merge per-peer records into one
/// shared authoritative command executor.
pub struct MultiPeerAuthority<S: ServerStrategy> {
    config: MultiPeerAuthorityConfig,
    aoi: AuthorityAoi,
    peers: BTreeMap<PeerId, PeerAuthority<S>>,
    provider: Box<dyn AuthorityReplicationProvider>,
    command_cursor: Option<PeerId>,
    next_command_lease_id: Option<u128>,
}

impl<S: ServerStrategy> Resource for MultiPeerAuthority<S> {}

impl<S: ServerStrategy> MultiPeerAuthority<S> {
    /// Construct an empty bounded authority coordinator.
    pub fn new(
        config: MultiPeerAuthorityConfig,
        provider: impl AuthorityReplicationProvider,
    ) -> Result<Self, MultiPeerAuthorityError> {
        if config.max_pending_aoi_events_per_peer == 0
            || config.max_aoi_payload_bytes == 0
            || config.max_aoi_payload_bytes < AOI_REPLICATION_HEADER_LEN
            || config.aoi.max_entities_per_peer > u32::MAX as usize
        {
            return Err(MultiPeerAuthorityError::InvalidConfig);
        }
        Ok(Self {
            aoi: AuthorityAoi::new(config.aoi)?,
            config,
            peers: BTreeMap::new(),
            provider: Box::new(provider),
            command_cursor: None,
            next_command_lease_id: Some(0),
        })
    }

    /// The immutable coordinator configuration.
    #[must_use]
    pub const fn config(&self) -> MultiPeerAuthorityConfig {
        self.config
    }

    /// The current AOI index, exposed read-only so its invariants remain owned
    /// by this coordinator.
    #[must_use]
    pub fn aoi(&self) -> &AuthorityAoi {
        &self.aoi
    }

    /// Number of admitted peers.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return admitted peer ids in deterministic ascending order.
    #[must_use]
    pub fn peer_ids(&self) -> Vec<PeerId> {
        self.peers.keys().copied().collect()
    }

    /// Add an already constructed per-peer server and send its initial AOI
    /// snapshot. This is the escape hatch for durable or otherwise specialized
    /// [`ServerReplication`] construction.
    pub fn admit_server(
        &mut self,
        peer_id: PeerId,
        center: (f32, f32),
        radius: f32,
        server: ServerReplication<S>,
    ) -> Result<AuthorityAoiDelta, MultiPeerAuthorityError> {
        if self.peers.contains_key(&peer_id) {
            return Err(AuthorityAoiError::PeerAlreadyAdmitted.into());
        }
        let delta = self.aoi.admit_peer(peer_id, center, radius)?;
        let mut peer = PeerAuthority {
            server,
            leased_command: None,
            snapshot_overrides: BTreeMap::new(),
            pending_aoi_events: VecDeque::new(),
            last_aoi_send_error: None,
            last_provider_error: None,
        };
        peer.pending_aoi_events
            .push_back(PendingAoiEvent::Delta(delta.clone()));
        self.peers.insert(peer_id, peer);
        self.flush_aoi_events(peer_id);
        Ok(delta)
    }

    /// Construct and admit one per-peer [`ServerReplication`] with the default
    /// replication configuration derived from its snapshot capacity.
    pub fn admit_peer(
        &mut self,
        peer_id: PeerId,
        center: (f32, f32),
        radius: f32,
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
    ) -> Result<AuthorityAoiDelta, MultiPeerAuthorityError> {
        self.admit_peer_with_config(
            peer_id,
            center,
            radius,
            schema,
            capacity,
            snapshot,
            strategy,
            transport,
            ReplicationConfig::for_capacity(capacity),
        )
    }

    /// Construct and admit one per-peer server with explicit transport limits.
    pub fn admit_peer_with_config(
        &mut self,
        peer_id: PeerId,
        center: (f32, f32),
        radius: f32,
        schema: Schema,
        capacity: usize,
        snapshot: &[u8],
        strategy: S,
        transport: Box<dyn Transport>,
        replication_config: ReplicationConfig,
    ) -> Result<AuthorityAoiDelta, MultiPeerAuthorityError> {
        let server = ServerReplication::new_with_config(
            schema,
            capacity,
            snapshot,
            strategy,
            transport,
            replication_config,
        )?;
        self.admit_server(peer_id, center, radius, server)
    }

    /// Remove a peer and return its server, allowing the caller to inspect or
    /// close the transport explicitly.
    pub fn remove_peer(
        &mut self,
        peer_id: PeerId,
    ) -> Result<ServerReplication<S>, MultiPeerAuthorityError> {
        if !self.peers.contains_key(&peer_id) {
            return Err(MultiPeerAuthorityError::UnknownPeer { peer_id });
        }
        self.aoi.remove_peer(peer_id)?;
        Ok(self
            .peers
            .remove(&peer_id)
            .expect("peer presence checked before AOI removal")
            .server)
    }

    /// Borrow one peer's replication server.
    #[must_use]
    pub fn peer(&self, peer_id: PeerId) -> Option<&ServerReplication<S>> {
        self.peers.get(&peer_id).map(|peer| &peer.server)
    }

    /// Mutably borrow one peer's replication server for application payloads.
    pub fn peer_mut(&mut self, peer_id: PeerId) -> Option<&mut ServerReplication<S>> {
        self.peers.get_mut(&peer_id).map(|peer| &mut peer.server)
    }

    /// Atomically select external World authority for every admitted peer.
    /// Validate all peer authority/journal states before mutating any connection.
    pub fn enable_external_world_authority(&mut self) -> Result<(), MultiPeerAuthorityError> {
        let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
        for &peer_id in &peer_ids {
            if let Some(peer) = self.peers.get(&peer_id) {
                peer.server
                    .validate_external_command_mode()
                    .map_err(|reason| MultiPeerAuthorityError::ExternalCommandMode {
                        peer_id,
                        reason,
                    })?;
            }
        }
        for peer_id in peer_ids {
            if let Some(peer) = self.peers.get_mut(&peer_id) {
                peer.server
                    .select_external_command_mode()
                    .expect("all peer external authority states were prevalidated");
            }
        }
        Ok(())
    }

    /// Number of AOI events retained for one peer.
    #[must_use]
    pub fn pending_aoi_events(&self, peer_id: PeerId) -> Option<usize> {
        self.peers
            .get(&peer_id)
            .map(|peer| peer.pending_aoi_events.len())
    }

    /// Take the last AOI event send refusal observed for one peer.
    pub fn take_aoi_send_error(&mut self, peer_id: PeerId) -> Option<MuxSendError> {
        self.peers
            .get_mut(&peer_id)
            .and_then(|peer| peer.last_aoi_send_error.take())
    }

    /// Take the most recent provider projection error. The update remains
    /// queued and is retried by a later frame.
    pub fn take_provider_error(
        &mut self,
        peer_id: PeerId,
    ) -> Option<AuthorityReplicationProviderError> {
        self.peers
            .get_mut(&peer_id)
            .and_then(|peer| peer.last_provider_error.take())
    }

    /// Pump every admitted peer once and return the number of applied commands.
    /// AOI events are flushed before each server pump so accepted events enter
    /// the transport's normal retransmit/ack path on that same frame.
    pub fn pump(&mut self) -> usize {
        let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
        let mut applied: usize = 0;
        for peer_id in peer_ids {
            self.flush_aoi_events(peer_id);
            if let Some(peer) = self.peers.get_mut(&peer_id) {
                applied = applied.saturating_add(peer.server.pump());
            }
        }
        applied
    }

    /// Poll all peer transports without applying their commands to per-peer
    /// `Record` state, then remove up to `limit` commands in bounded
    /// round-robin order. The first call permanently selects external command
    /// consumption for each connection. Commands retain each peer's reliable
    /// FIFO order; cross-peer order is a local scheduling policy, not a claim
    /// of cross-machine deterministic arrival ordering.
    ///
    /// # Errors
    /// Returns [`MultiPeerAuthorityError::ExternalCommandMode`] without changing
    /// any peer mode if a record journal is installed or record authority has
    /// already advanced for any admitted peer.
    pub fn poll_inbound_commands(
        &mut self,
        limit: usize,
    ) -> Result<Vec<InboundAuthorityCommand>, MultiPeerAuthorityError> {
        let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
        self.enable_external_world_authority()?;
        for &peer_id in &peer_ids {
            self.flush_aoi_events(peer_id);
            if let Some(peer) = self.peers.get_mut(&peer_id) {
                peer.server.poll_external_commands();
            }
        }
        if limit == 0 || peer_ids.is_empty() {
            return Ok(Vec::new());
        }

        let start = self
            .command_cursor
            .and_then(|cursor| peer_ids.iter().position(|peer_id| *peer_id == cursor))
            .map_or(0, |index| (index + 1) % peer_ids.len());
        let mut commands = Vec::new();
        let mut next_index = start;
        while commands.len() < limit {
            let mut found = false;
            for offset in 0..peer_ids.len() {
                let index = (next_index + offset) % peer_ids.len();
                let peer_id = peer_ids[index];
                let command = self
                    .peers
                    .get_mut(&peer_id)
                    .and_then(|peer| peer.server.take_inbound_command());
                if let Some(payload) = command {
                    commands.push(InboundAuthorityCommand { peer_id, payload });
                    self.command_cursor = Some(peer_id);
                    next_index = (index + 1) % peer_ids.len();
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        Ok(commands)
    }

    /// Poll peer transports and lease up to `limit` commands without removing
    /// them from the reliable queues. A connection may have only one outstanding
    /// lease, so its FIFO head cannot be bypassed. Consumers must acknowledge a
    /// lease only after committing its World mutation, or release it to retry.
    /// Leases remain bounded by the admitted peer count and each peer's bounded
    /// receive queue.
    pub fn lease_inbound_commands(
        &mut self,
        limit: usize,
    ) -> Result<Vec<LeasedAuthorityCommand>, MultiPeerAuthorityError> {
        self.poll_inbound_commands(0)?;
        if limit == 0 || self.peers.is_empty() {
            return Ok(Vec::new());
        }
        let peer_ids: Vec<PeerId> = self.peers.keys().copied().collect();
        let start = self
            .command_cursor
            .and_then(|cursor| peer_ids.iter().position(|peer_id| *peer_id == cursor))
            .map_or(0, |index| (index + 1) % peer_ids.len());
        let mut commands = Vec::new();
        let mut next_index = start;
        while commands.len() < limit {
            let mut found = false;
            for offset in 0..peer_ids.len() {
                let index = (next_index + offset) % peer_ids.len();
                let peer_id = peer_ids[index];
                let Some(peer) = self.peers.get_mut(&peer_id) else {
                    continue;
                };
                if peer.leased_command.is_some() {
                    continue;
                }
                let Some((server_command_id, payload)) = peer.server.lease_inbound_command() else {
                    continue;
                };
                let Some(command_id) = self.next_command_lease_id else {
                    peer.server.release_inbound_command(server_command_id);
                    return Err(MultiPeerAuthorityError::CommandLeaseIdExhausted);
                };
                self.next_command_lease_id = command_id.checked_add(1);
                peer.leased_command = Some((command_id, server_command_id));
                {
                    commands.push(LeasedAuthorityCommand {
                        peer_id,
                        command_id,
                        payload,
                    });
                    self.command_cursor = Some(peer_id);
                    next_index = (index + 1) % peer_ids.len();
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        Ok(commands)
    }

    /// Acknowledge a leased command after its authoritative World mutation has
    /// committed. A stale or out-of-order lease is rejected without dequeuing.
    pub fn acknowledge_inbound_command(
        &mut self,
        command: &LeasedAuthorityCommand,
    ) -> Result<(), MultiPeerAuthorityError> {
        let peer =
            self.peers
                .get_mut(&command.peer_id)
                .ok_or(MultiPeerAuthorityError::UnknownPeer {
                    peer_id: command.peer_id,
                })?;
        let Some((lease_id, server_command_id)) = peer.leased_command else {
            return Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            });
        };
        if lease_id != command.command_id {
            return Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            });
        }
        if peer.server.acknowledge_inbound_command(server_command_id) {
            peer.leased_command = None;
            Ok(())
        } else {
            Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            })
        }
    }

    /// Release a leased command without consuming it, allowing the next
    /// executor pass to retry the same reliable FIFO head.
    pub fn release_inbound_command(
        &mut self,
        command: &LeasedAuthorityCommand,
    ) -> Result<(), MultiPeerAuthorityError> {
        let peer =
            self.peers
                .get_mut(&command.peer_id)
                .ok_or(MultiPeerAuthorityError::UnknownPeer {
                    peer_id: command.peer_id,
                })?;
        let Some((lease_id, server_command_id)) = peer.leased_command else {
            return Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            });
        };
        if lease_id != command.command_id {
            return Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            });
        }
        if peer.server.release_inbound_command(server_command_id) {
            peer.leased_command = None;
            Ok(())
        } else {
            Err(MultiPeerAuthorityError::InvalidCommandLease {
                peer_id: command.peer_id,
                command_id: command.command_id,
            })
        }
    }

    /// Index or move an entity and dispatch all resulting enter/leave updates.
    pub fn upsert_entity(
        &mut self,
        entity_id: u32,
        position: (f32, f32),
    ) -> Result<Vec<AuthorityAoiDelta>, MultiPeerAuthorityError> {
        self.ensure_capacity_for_all_peers()?;
        let deltas = self.aoi.upsert_entity(entity_id, position)?;
        self.dispatch_deltas(deltas)
    }

    /// Preflight a bounded batch of entity projection operations. This clones
    /// only the bounded AOI index/state, simulates operations in caller order,
    /// and verifies exact per-peer outbox demand before changing the live AOI.
    /// The returned plan is tied to this authority's current peer/outbox state
    /// and commit rejects it if any intervening authority mutation occurred.
    fn prepare_entity_projection(
        &self,
        operations: &[AuthorityEntityProjection],
    ) -> Result<PreparedAuthorityProjection, MultiPeerAuthorityError> {
        let mut aoi = self.aoi.clone();
        let mut deltas = Vec::new();
        let mut overrides = Vec::new();
        let mut events = Vec::new();
        let mut required = BTreeMap::<PeerId, usize>::new();
        let mut touched = BTreeSet::new();
        for operation in operations {
            match operation {
                AuthorityEntityProjection::Upsert {
                    entity_id,
                    position,
                    snapshot,
                } => {
                    let changed = aoi.upsert_entity(*entity_id, *position)?;
                    for delta in changed {
                        touched.insert(delta.peer_id);
                        *required.entry(delta.peer_id).or_default() += 1;
                        if delta.entered.contains(entity_id) {
                            // The enter delta itself carries the snapshot;
                            // record an override without enqueueing a duplicate
                            // standalone update event.
                            overrides.push((
                                delta.peer_id,
                                *entity_id,
                                delta.revision,
                                snapshot.clone(),
                            ));
                        }
                        events.push((delta.peer_id, PendingAoiEvent::Delta(delta.clone())));
                        deltas.push(delta);
                    }
                }
                AuthorityEntityProjection::Remove { entity_id } => {
                    let changed = aoi.remove_entity(*entity_id)?;
                    for delta in changed {
                        touched.insert(delta.peer_id);
                        *required.entry(delta.peer_id).or_default() += 1;
                        events.push((delta.peer_id, PendingAoiEvent::Delta(delta.clone())));
                        deltas.push(delta);
                    }
                }
                AuthorityEntityProjection::Update {
                    entity_id,
                    snapshot,
                } => {
                    for peer_id in aoi.viewers(*entity_id) {
                        touched.insert(peer_id);
                        *required.entry(peer_id).or_default() += 1;
                        let revision = aoi.peer_revision(peer_id)?;
                        overrides.push((peer_id, *entity_id, revision, snapshot.clone()));
                        events.push((
                            peer_id,
                            PendingAoiEvent::EntityUpdate {
                                entity_id: *entity_id,
                                revision,
                            },
                        ));
                    }
                }
            }
        }
        for (peer_id, needed) in required {
            let peer = self
                .peers
                .get(&peer_id)
                .ok_or(MultiPeerAuthorityError::UnknownPeer { peer_id })?;
            if peer
                .pending_aoi_events
                .len()
                .checked_add(needed)
                .is_none_or(|length| length > self.config.max_pending_aoi_events_per_peer)
            {
                return Err(MultiPeerAuthorityError::PendingQueueFull { peer_id });
            }
        }
        for delta in &deltas {
            let mut encoded_len = AOI_REPLICATION_HEADER_LEN
                .checked_add(
                    delta
                        .left
                        .len()
                        .checked_mul(size_of::<u32>())
                        .ok_or(MultiPeerAuthorityError::EventCountOverflow)?,
                )
                .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
            for entity_id in &delta.entered {
                let snapshot = overrides
                    .iter()
                    .find(|(peer_id, candidate, revision, _)| {
                        *peer_id == delta.peer_id
                            && *candidate == *entity_id
                            && *revision == delta.revision
                    })
                    .ok_or(MultiPeerAuthorityError::AoiPayloadTooLarge {
                        actual: usize::MAX,
                        max: self.config.max_aoi_payload_bytes,
                    })?;
                if u32::try_from(snapshot.3.len()).is_err() {
                    return Err(MultiPeerAuthorityError::AoiPayloadTooLarge {
                        actual: usize::MAX,
                        max: self.config.max_aoi_payload_bytes,
                    });
                }
                encoded_len = encoded_len
                    .checked_add(8)
                    .and_then(|length| length.checked_add(snapshot.3.len()))
                    .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
            }
            if encoded_len > self.config.max_aoi_payload_bytes {
                return Err(MultiPeerAuthorityError::AoiPayloadTooLarge {
                    actual: encoded_len,
                    max: self.config.max_aoi_payload_bytes,
                });
            }
        }
        for (_, _, _, snapshot) in &overrides {
            if u32::try_from(snapshot.len()).is_err() {
                return Err(MultiPeerAuthorityError::AoiPayloadTooLarge {
                    actual: usize::MAX,
                    max: self.config.max_aoi_payload_bytes,
                });
            }
            let encoded_len = AOI_REPLICATION_HEADER_LEN
                .checked_add(8)
                .and_then(|length| length.checked_add(snapshot.len()))
                .ok_or(MultiPeerAuthorityError::EventCountOverflow)?;
            if encoded_len > self.config.max_aoi_payload_bytes {
                return Err(MultiPeerAuthorityError::AoiPayloadTooLarge {
                    actual: encoded_len,
                    max: self.config.max_aoi_payload_bytes,
                });
            }
        }
        Ok(PreparedAuthorityProjection {
            aoi,
            deltas,
            overrides,
            events,
            touched,
        })
    }

    /// Commit a preflighted entity projection batch and enqueue all resulting
    /// enter/leave/update snapshots. No capacity or AOI validation remains.
    fn commit_entity_projection(
        &mut self,
        prepared: PreparedAuthorityProjection,
    ) -> Vec<AuthorityAoiDelta> {
        let PreparedAuthorityProjection {
            aoi,
            deltas,
            overrides,
            events,
            touched,
        } = prepared;
        self.aoi = aoi;
        for (peer_id, entity_id, revision, snapshot) in overrides {
            let peer = self
                .peers
                .get_mut(&peer_id)
                .expect("prepared update peer remains admitted");
            peer.snapshot_overrides
                .entry((entity_id, revision))
                .or_default()
                .push_back(snapshot);
        }
        // Preserve caller operation order in the per-peer FIFO.
        for (peer_id, event) in events {
            self.peers
                .get_mut(&peer_id)
                .expect("prepared AOI peers remain admitted")
                .pending_aoi_events
                .push_back(event);
        }
        for peer_id in touched {
            self.flush_aoi_events(peer_id);
        }
        deltas
    }

    /// Atomically preflight and enqueue an entity projection batch. On any
    /// AOI validation or per-peer outbox capacity error, neither the AOI index
    /// nor its queues change. Successful preflight and commit happen under one
    /// exclusive borrow of this coordinator, so the candidate cannot go stale.
    pub fn project_entities_atomic(
        &mut self,
        operations: &[AuthorityEntityProjection],
    ) -> Result<Vec<AuthorityAoiDelta>, MultiPeerAuthorityError> {
        let prepared = self.prepare_entity_projection(operations)?;
        Ok(self.commit_entity_projection(prepared))
    }

    /// Remove an entity and dispatch all resulting leave updates.
    pub fn remove_entity(
        &mut self,
        entity_id: u32,
    ) -> Result<Vec<AuthorityAoiDelta>, MultiPeerAuthorityError> {
        self.ensure_capacity_for_all_peers()?;
        let deltas = self.aoi.remove_entity(entity_id)?;
        self.dispatch_deltas(deltas)
    }

    /// Move one peer's view and dispatch its enter/leave update.
    pub fn update_peer_view(
        &mut self,
        peer_id: PeerId,
        center: (f32, f32),
        radius: f32,
    ) -> Result<AuthorityAoiDelta, MultiPeerAuthorityError> {
        self.ensure_capacity_for_peer(peer_id)?;
        let delta = self.aoi.update_peer(peer_id, center, radius)?;
        let mut deltas = self.dispatch_deltas(vec![delta])?;
        Ok(deltas
            .pop()
            .expect("one peer view update produces one delta"))
    }

    /// Recompute one peer's view against the current indexed entities.
    pub fn refresh_peer(
        &mut self,
        peer_id: PeerId,
    ) -> Result<AuthorityAoiDelta, MultiPeerAuthorityError> {
        self.ensure_capacity_for_peer(peer_id)?;
        let delta = self.aoi.refresh_peer(peer_id)?;
        let mut deltas = self.dispatch_deltas(vec![delta])?;
        Ok(deltas.pop().expect("one peer refresh produces one delta"))
    }

    /// Publish the latest provider state for an entity to its current viewers.
    /// Viewer selection is deterministic, and each peer's update is retained
    /// in the same bounded FIFO as AOI enter/leave edges.
    pub fn publish_entity_update(
        &mut self,
        entity_id: u32,
    ) -> Result<Vec<PeerId>, MultiPeerAuthorityError> {
        let viewers = self.aoi.viewers(entity_id);
        for &peer_id in &viewers {
            self.ensure_capacity_for_peer(peer_id)?;
        }
        for &peer_id in &viewers {
            let revision = self.aoi.peer_revision(peer_id)?;
            self.peers
                .get_mut(&peer_id)
                .ok_or(MultiPeerAuthorityError::UnknownPeer { peer_id })?
                .pending_aoi_events
                .push_back(PendingAoiEvent::EntityUpdate {
                    entity_id,
                    revision,
                });
        }
        for &peer_id in &viewers {
            self.flush_aoi_events(peer_id);
        }
        Ok(viewers)
    }

    fn ensure_capacity_for_peer(&self, peer_id: PeerId) -> Result<(), MultiPeerAuthorityError> {
        let peer = self
            .peers
            .get(&peer_id)
            .ok_or(MultiPeerAuthorityError::UnknownPeer { peer_id })?;
        if peer.pending_aoi_events.len() >= self.config.max_pending_aoi_events_per_peer {
            return Err(MultiPeerAuthorityError::PendingQueueFull { peer_id });
        }
        Ok(())
    }

    fn ensure_capacity_for_all_peers(&self) -> Result<(), MultiPeerAuthorityError> {
        for &peer_id in self.peers.keys() {
            self.ensure_capacity_for_peer(peer_id)?;
        }
        Ok(())
    }

    fn dispatch_deltas(
        &mut self,
        deltas: Vec<AuthorityAoiDelta>,
    ) -> Result<Vec<AuthorityAoiDelta>, MultiPeerAuthorityError> {
        for delta in &deltas {
            let peer =
                self.peers
                    .get_mut(&delta.peer_id)
                    .ok_or(MultiPeerAuthorityError::UnknownPeer {
                        peer_id: delta.peer_id,
                    })?;
            peer.pending_aoi_events
                .push_back(PendingAoiEvent::Delta(delta.clone()));
        }
        for delta in &deltas {
            self.flush_aoi_events(delta.peer_id);
        }
        Ok(deltas)
    }

    fn flush_aoi_events(&mut self, peer_id: PeerId) {
        loop {
            let Some(event) = self
                .peers
                .get(&peer_id)
                .and_then(|peer| peer.pending_aoi_events.front())
                .cloned()
            else {
                return;
            };
            let (revision, entered_ids, updated_id, left) = match event {
                PendingAoiEvent::Delta(delta) => (delta.revision, delta.entered, None, delta.left),
                PendingAoiEvent::EntityUpdate {
                    entity_id,
                    revision,
                } => (revision, Vec::new(), Some(entity_id), Vec::new()),
            };
            let mut payload = AuthorityAoiReplicationPayload {
                peer_id,
                revision,
                entered: Vec::with_capacity(entered_ids.len()),
                updated: Vec::with_capacity(usize::from(updated_id.is_some())),
                left,
            };
            let mut encoded_len = AOI_REPLICATION_HEADER_LEN
                .checked_add(
                    payload
                        .left
                        .len()
                        .checked_mul(size_of::<u32>())
                        .unwrap_or(usize::MAX),
                )
                .unwrap_or(usize::MAX);
            if encoded_len > self.config.max_aoi_payload_bytes {
                if let Some(peer) = self.peers.get_mut(&peer_id) {
                    peer.last_aoi_send_error = Some(MuxSendError::TooLarge {
                        len: encoded_len,
                        max: self.config.max_aoi_payload_bytes,
                    });
                }
                return;
            }
            let mut provider_failed = None;
            let mut consumed_overrides = Vec::new();
            for entity_id in entered_ids {
                let provided = self
                    .peers
                    .get(&peer_id)
                    .and_then(|peer| peer.snapshot_overrides.get(&(entity_id, revision)))
                    .and_then(|snapshots| snapshots.front())
                    .cloned();
                if provided.is_some() {
                    consumed_overrides.push((entity_id, revision));
                }
                match provided
                    .map(Ok)
                    .unwrap_or_else(|| self.provider.snapshot_entity(peer_id, entity_id, revision))
                {
                    Ok(bytes) => {
                        let next_len = encoded_len
                            .checked_add(8)
                            .and_then(|length| length.checked_add(bytes.len()));
                        if next_len.is_none_or(|length| {
                            length > self.config.max_aoi_payload_bytes
                                || u32::try_from(bytes.len()).is_err()
                        }) {
                            let actual = next_len.unwrap_or(usize::MAX);
                            if let Some(peer) = self.peers.get_mut(&peer_id) {
                                peer.last_aoi_send_error = Some(MuxSendError::TooLarge {
                                    len: actual,
                                    max: self.config.max_aoi_payload_bytes,
                                });
                            }
                            return;
                        }
                        encoded_len = next_len.expect("length checked above");
                        payload
                            .entered
                            .push(EntitySnapshotPayload { entity_id, bytes });
                    }
                    Err(error) => {
                        provider_failed = Some(error);
                        break;
                    }
                }
            }
            if let Some(entity_id) = updated_id {
                let provided = self
                    .peers
                    .get(&peer_id)
                    .and_then(|peer| peer.snapshot_overrides.get(&(entity_id, revision)))
                    .and_then(|snapshots| snapshots.front())
                    .cloned();
                if provided.is_some() {
                    consumed_overrides.push((entity_id, revision));
                }
                match provided
                    .map(Ok)
                    .unwrap_or_else(|| self.provider.snapshot_entity(peer_id, entity_id, revision))
                {
                    Ok(bytes) => {
                        let next_len = encoded_len
                            .checked_add(8)
                            .and_then(|length| length.checked_add(bytes.len()));
                        if next_len.is_none_or(|length| {
                            length > self.config.max_aoi_payload_bytes
                                || u32::try_from(bytes.len()).is_err()
                        }) {
                            let actual = next_len.unwrap_or(usize::MAX);
                            if let Some(peer) = self.peers.get_mut(&peer_id) {
                                peer.last_aoi_send_error = Some(MuxSendError::TooLarge {
                                    len: actual,
                                    max: self.config.max_aoi_payload_bytes,
                                });
                            }
                            return;
                        }
                        encoded_len = next_len.expect("length checked above");
                        payload
                            .updated
                            .push(EntitySnapshotPayload { entity_id, bytes });
                    }
                    Err(error) => {
                        if let Some(peer) = self.peers.get_mut(&peer_id) {
                            peer.last_provider_error = Some(error);
                        }
                        return;
                    }
                }
            }
            if let Some(error) = provider_failed {
                if let Some(peer) = self.peers.get_mut(&peer_id) {
                    peer.last_provider_error = Some(error);
                }
                return;
            }
            let encoded = match encode_authority_aoi_replication_payload(&payload) {
                Ok(encoded) if encoded.len() == encoded_len => encoded,
                Ok(encoded) => {
                    if let Some(peer) = self.peers.get_mut(&peer_id) {
                        peer.last_aoi_send_error = Some(MuxSendError::TooLarge {
                            len: encoded.len(),
                            max: self.config.max_aoi_payload_bytes,
                        });
                    }
                    return;
                }
                Err(_) => return,
            };
            let send_result = match self.peers.get_mut(&peer_id) {
                Some(peer) => peer.server.send_event(&encoded),
                None => return,
            };
            match send_result {
                Ok(_) => {
                    if let Some(peer) = self.peers.get_mut(&peer_id) {
                        peer.pending_aoi_events.pop_front();
                        for key in consumed_overrides {
                            if let Some(snapshots) = peer.snapshot_overrides.get_mut(&key) {
                                snapshots.pop_front();
                                if snapshots.is_empty() {
                                    peer.snapshot_overrides.remove(&key);
                                }
                            }
                        }
                        peer.last_provider_error = None;
                        peer.last_aoi_send_error = None;
                    }
                }
                Err(error) => {
                    if let Some(peer) = self.peers.get_mut(&peer_id) {
                        peer.last_aoi_send_error = Some(error);
                    }
                    return;
                }
            }
        }
    }
}

/// App plugin that installs and pumps a [`MultiPeerAuthority`] resource.
pub struct MultiPeerServerPlugin<S: ServerStrategy> {
    authority: Mutex<Option<MultiPeerAuthority<S>>>,
}

impl<S: ServerStrategy> MultiPeerServerPlugin<S> {
    /// Create an empty multi-peer authority plugin.
    pub fn new(
        config: MultiPeerAuthorityConfig,
        provider: impl AuthorityReplicationProvider,
    ) -> Result<Self, MultiPeerAuthorityError> {
        Ok(Self {
            authority: Mutex::new(Some(MultiPeerAuthority::new(config, provider)?)),
        })
    }

    /// Install a previously initialized authority, useful when the caller has
    /// admitted peers before app construction.
    #[must_use]
    pub fn from_authority(authority: MultiPeerAuthority<S>) -> Self {
        Self {
            authority: Mutex::new(Some(authority)),
        }
    }
}

impl<S: ServerStrategy> Plugin for MultiPeerServerPlugin<S> {
    fn build(&self, app: &mut App) {
        let authority = self
            .authority
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("MultiPeerServerPlugin::build ran twice");
        app.insert_resource(authority);
        app.add_systems(First, pump_multi_peer_server::<S>);
    }
}

fn pump_multi_peer_server<S: ServerStrategy>(
    mut authority: prism_ecs::system::ResMut<MultiPeerAuthority<S>>,
) {
    authority.pump();
}

#[cfg(test)]
#[path = "multi_server_tests.rs"]
mod tests;
