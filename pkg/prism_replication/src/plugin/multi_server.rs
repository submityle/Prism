//! Bounded multi-peer authority orchestration.
//!
//! [`ServerReplication`] owns one authenticated transport and one replicated
//! record. [`MultiPeerAuthority`] owns the server-side peer identity table and
//! connects it to [`AuthorityAoi`]. AOI changes are encoded as versioned event
//! payloads and retained in a bounded per-peer queue until the corresponding
//! server connection accepts them.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
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

use super::{ReplicationConfig, ServerReplication, ServerStrategy};

/// The stable identity used by the authority peer table and AOI events.
pub type PeerId = u64;

const AOI_EVENT_MAGIC: u8 = 0xa1;
const AOI_EVENT_VERSION: u8 = 1;
const AOI_EVENT_HEADER_LEN: usize = 2 + 8 + 8 + 4 + 4 + 4;
const AOI_REPLICATION_MAGIC: u8 = 0xa2;
const AOI_REPLICATION_VERSION: u8 = 1;
const AOI_REPLICATION_HEADER_LEN: usize = 2 + 8 + 8 + 4 + 4;

/// A peer-specific projection of an entity from the shared authority world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntitySnapshotPayload {
    /// Stable entity id used by the AOI index.
    pub entity_id: u32,
    /// Provider-serialized entity state. The provider owns the schema/version.
    pub bytes: Vec<u8>,
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
        }
    }
}

/// Encode a provider-produced snapshot/despawn update for reliable delivery.
pub fn encode_authority_aoi_replication_payload(
    payload: &AuthorityAoiReplicationPayload,
) -> Result<Vec<u8>, MultiPeerAuthorityError> {
    let entered = u32::try_from(payload.entered.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let left = u32::try_from(payload.left.len())
        .map_err(|_| MultiPeerAuthorityError::EventCountOverflow)?;
    let mut capacity = AOI_REPLICATION_HEADER_LEN;
    for snapshot in &payload.entered {
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
    for snapshot in &payload.entered {
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

/// Decode a provider-produced AOI replication payload under explicit entity
/// count and byte limits.
pub fn decode_authority_aoi_replication_payload(
    bytes: &[u8],
    max_entities: usize,
    max_payload_bytes: usize,
) -> Result<AuthorityAoiReplicationPayload, AuthorityAoiEventError> {
    if bytes.len() > max_payload_bytes {
        return Err(AuthorityAoiEventError::BoundExceeded);
    }
    if bytes.len() < AOI_REPLICATION_HEADER_LEN
        || bytes[0] != AOI_REPLICATION_MAGIC
        || bytes[1] != AOI_REPLICATION_VERSION
    {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let peer_id = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
    let revision = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    let entered = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    let left = u32::from_le_bytes(bytes[22..26].try_into().unwrap()) as usize;
    if entered > max_entities || left > max_entities {
        return Err(AuthorityAoiEventError::BoundExceeded);
    }
    let minimum_body = entered
        .checked_mul(8)
        .and_then(|size| size.checked_add(left.checked_mul(size_of::<u32>())?))
        .ok_or(AuthorityAoiEventError::Malformed)?;
    if minimum_body > bytes.len() - AOI_REPLICATION_HEADER_LEN {
        return Err(AuthorityAoiEventError::Malformed);
    }
    let mut offset = AOI_REPLICATION_HEADER_LEN;
    let mut snapshots = Vec::with_capacity(entered);
    for _ in 0..entered {
        if bytes.len().saturating_sub(offset) < 8 {
            return Err(AuthorityAoiEventError::Malformed);
        }
        let entity_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
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
    pending_aoi_events: VecDeque<AuthorityAoiDelta>,
    last_aoi_send_error: Option<MuxSendError>,
    last_provider_error: Option<AuthorityReplicationProviderError>,
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
            pending_aoi_events: VecDeque::new(),
            last_aoi_send_error: None,
            last_provider_error: None,
        };
        peer.pending_aoi_events.push_back(delta.clone());
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
            peer.pending_aoi_events.push_back(delta.clone());
        }
        for delta in &deltas {
            self.flush_aoi_events(delta.peer_id);
        }
        Ok(deltas)
    }

    fn flush_aoi_events(&mut self, peer_id: PeerId) {
        loop {
            let Some(delta) = self
                .peers
                .get(&peer_id)
                .and_then(|peer| peer.pending_aoi_events.front())
                .cloned()
            else {
                return;
            };
            let mut payload = AuthorityAoiReplicationPayload {
                peer_id: delta.peer_id,
                revision: delta.revision,
                entered: Vec::with_capacity(delta.entered.len()),
                left: delta.left.clone(),
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
            for &entity_id in &delta.entered {
                match self
                    .provider
                    .snapshot_entity(delta.peer_id, entity_id, delta.revision)
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
