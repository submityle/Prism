//! Bounded authority-side AOI subscription state.
//!
//! [`crate::aoi::AoiGrid`] answers spatial queries, but a server still needs a
//! per-peer subscription state to turn those queries into deterministic
//! enter/leave edges. [`AuthorityAoi`] is that transport-independent seam. It
//! owns only stable numeric peer/entity ids and positions; scene/world
//! lifecycle, component serialization, authority handoff, and packet delivery
//! remain caller/provider responsibilities.
//!
//! The implementation is deliberately bounded: peer count, entity count,
//! per-peer visible count, query radius, and revision space are all explicit.
//! A changed entity recomputes the admitted peer views and returns changed
//! deltas in ascending peer order. This is suitable for an authority tick or
//! an app-owned provider, but it is not a claim that AOI is wired into a real
//! ECS/world producer or a cross-process server mesh.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::aoi::{interest_delta, select_within_budget, AoiError, AoiGrid};

const AUTHORITY_AOI_SNAPSHOT_MAGIC: [u8; 4] = *b"PAOI";
const AUTHORITY_AOI_SNAPSHOT_VERSION: u8 = 1;
const AUTHORITY_AOI_SNAPSHOT_HEADER_LEN: usize = 4 + 1 + 8 * 3 + 4 * 2 + 8 * 2;
const AUTHORITY_AOI_ENTITY_RECORD_LEN: usize = 4 + 4 * 2;
const AUTHORITY_AOI_PEER_RECORD_LEN: usize = 8 + 4 * 3 + 8 + 8;

/// Bounded authority-side AOI configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuthorityAoiConfig {
    /// Maximum number of peers with an active AOI subscription.
    pub max_peers: usize,
    /// Maximum number of entities tracked by the authority AOI index.
    pub max_entities: usize,
    /// Maximum number of visible entity ids emitted for one peer.
    pub max_entities_per_peer: usize,
    /// Maximum accepted peer query radius.
    pub max_radius: f32,
    /// Cell edge length used by the underlying deterministic grid.
    pub cell_size: f32,
}

impl Default for AuthorityAoiConfig {
    fn default() -> Self {
        Self {
            max_peers: 256,
            max_entities: 65_536,
            max_entities_per_peer: 4_096,
            max_radius: 100_000.0,
            cell_size: 100.0,
        }
    }
}

/// A rejected authority AOI operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityAoiError {
    /// A configured capacity was zero or the configured maximum radius invalid.
    InvalidConfig,
    /// The configured grid cell size is invalid.
    BadCellSize,
    /// The requested radius is invalid or exceeds the configured bound.
    InvalidRadius,
    /// A position or query centre is not finite.
    NonFinitePosition,
    /// No additional peer can be admitted.
    PeerCapacityExceeded,
    /// No additional entity can be indexed.
    EntityCapacityExceeded,
    /// The peer id is already admitted.
    PeerAlreadyAdmitted,
    /// The peer id is not admitted.
    UnknownPeer,
    /// A per-peer subscription revision cannot advance further.
    RevisionExhausted,
    /// A persisted AOI snapshot is malformed, inconsistent, or exceeds bounds.
    InvalidSnapshot,
}

impl core::fmt::Display for AuthorityAoiError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::InvalidConfig => "authority AOI configuration is invalid",
            Self::BadCellSize => "authority AOI cell size is invalid",
            Self::InvalidRadius => "authority AOI radius is invalid or exceeds its bound",
            Self::NonFinitePosition => "authority AOI position must be finite",
            Self::PeerCapacityExceeded => "authority AOI peer capacity is full",
            Self::EntityCapacityExceeded => "authority AOI entity capacity is full",
            Self::PeerAlreadyAdmitted => "authority AOI peer is already admitted",
            Self::UnknownPeer => "authority AOI peer is not admitted",
            Self::RevisionExhausted => "authority AOI peer revision is exhausted",
            Self::InvalidSnapshot => "authority AOI snapshot is invalid",
        };
        formatter.write_str(message)
    }
}

/// Stable, transport-independent snapshot of an authority AOI index.
///
/// The vectors are emitted in ascending ID order. Restore validates the saved
/// visible sets against the saved entity positions and view parameters instead
/// of silently recomputing a potentially different subscription state.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityAoiSnapshot {
    /// Configuration that bounded and shaped this AOI state.
    pub config: AuthorityAoiConfig,
    /// Tracked entities as `(stable_id, (x, z))` pairs.
    pub entities: Vec<(u32, (f32, f32))>,
    /// Active peer subscriptions in ascending peer-id order.
    pub peers: Vec<AuthorityAoiPeerSnapshot>,
}

/// One peer's saved AOI view.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityAoiPeerSnapshot {
    /// Stable caller-owned peer identity.
    pub peer_id: u64,
    /// Saved view centre.
    pub center: (f32, f32),
    /// Saved view radius.
    pub radius: f32,
    /// Saved visible entity IDs in ascending order.
    pub visible: Vec<u32>,
    /// Monotonic subscription revision, never zero for an admitted peer.
    pub revision: u64,
}

impl AuthorityAoiSnapshot {
    /// Encode this snapshot in the canonical, versioned little-endian format.
    ///
    /// `max_bytes` is checked before any output allocation. Snapshot encoding
    /// is intended for checkpoints, not per-frame hot paths.
    pub fn encode(&self, max_bytes: usize) -> Result<Vec<u8>, AuthorityAoiError> {
        // Reuse restore's invariant checks so callers cannot persist malformed
        // public snapshot values as if they were canonical authority state.
        AuthorityAoi::restore(self.clone(), self.config)?;
        let entity_count =
            u64::try_from(self.entities.len()).map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        let peer_count =
            u64::try_from(self.peers.len()).map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        let mut encoded_len = AUTHORITY_AOI_SNAPSHOT_HEADER_LEN;
        encoded_len = encoded_len
            .checked_add(
                self.entities
                    .len()
                    .checked_mul(AUTHORITY_AOI_ENTITY_RECORD_LEN)
                    .ok_or(AuthorityAoiError::InvalidSnapshot)?,
            )
            .ok_or(AuthorityAoiError::InvalidSnapshot)?;
        for peer in &self.peers {
            encoded_len = encoded_len
                .checked_add(AUTHORITY_AOI_PEER_RECORD_LEN)
                .and_then(|length| length.checked_add(peer.visible.len().checked_mul(4)?))
                .ok_or(AuthorityAoiError::InvalidSnapshot)?;
        }
        if encoded_len > max_bytes {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(encoded_len)
            .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        bytes.extend_from_slice(&AUTHORITY_AOI_SNAPSHOT_MAGIC);
        bytes.push(AUTHORITY_AOI_SNAPSHOT_VERSION);
        for limit in [
            self.config.max_peers,
            self.config.max_entities,
            self.config.max_entities_per_peer,
        ] {
            bytes.extend_from_slice(
                &u64::try_from(limit)
                    .map_err(|_| AuthorityAoiError::InvalidSnapshot)?
                    .to_le_bytes(),
            );
        }
        bytes.extend_from_slice(&self.config.max_radius.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.config.cell_size.to_bits().to_le_bytes());
        bytes.extend_from_slice(&entity_count.to_le_bytes());
        bytes.extend_from_slice(&peer_count.to_le_bytes());
        for &(entity_id, (x, z)) in &self.entities {
            bytes.extend_from_slice(&entity_id.to_le_bytes());
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
            bytes.extend_from_slice(&z.to_bits().to_le_bytes());
        }
        for peer in &self.peers {
            bytes.extend_from_slice(&peer.peer_id.to_le_bytes());
            bytes.extend_from_slice(&peer.center.0.to_bits().to_le_bytes());
            bytes.extend_from_slice(&peer.center.1.to_bits().to_le_bytes());
            bytes.extend_from_slice(&peer.radius.to_bits().to_le_bytes());
            bytes.extend_from_slice(&peer.revision.to_le_bytes());
            bytes.extend_from_slice(
                &u64::try_from(peer.visible.len())
                    .map_err(|_| AuthorityAoiError::InvalidSnapshot)?
                    .to_le_bytes(),
            );
            for entity_id in &peer.visible {
                bytes.extend_from_slice(&entity_id.to_le_bytes());
            }
        }
        debug_assert_eq!(bytes.len(), encoded_len);
        Ok(bytes)
    }

    /// Decode a canonical AOI snapshot with strict byte and collection bounds.
    /// The persisted configuration must match `expected_config` exactly.
    pub fn decode(
        bytes: &[u8],
        expected_config: AuthorityAoiConfig,
        max_bytes: usize,
    ) -> Result<Self, AuthorityAoiError> {
        if bytes.len() > max_bytes || bytes.len() < AUTHORITY_AOI_SNAPSHOT_HEADER_LEN {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        let mut cursor = SnapshotCursor::new(bytes);
        if cursor.take(4)? != AUTHORITY_AOI_SNAPSHOT_MAGIC
            || cursor.byte()? != AUTHORITY_AOI_SNAPSHOT_VERSION
        {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        let config = AuthorityAoiConfig {
            max_peers: cursor.usize_u64()?,
            max_entities: cursor.usize_u64()?,
            max_entities_per_peer: cursor.usize_u64()?,
            max_radius: f32::from_bits(cursor.u32()?),
            cell_size: f32::from_bits(cursor.u32()?),
        };
        if config != expected_config {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        let entity_count = cursor.bounded_count(config.max_entities)?;
        let peer_count = cursor.bounded_count(config.max_peers)?;
        cursor.ensure_records_fit(entity_count, AUTHORITY_AOI_ENTITY_RECORD_LEN)?;
        let mut entities = Vec::new();
        entities
            .try_reserve_exact(entity_count)
            .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        for _ in 0..entity_count {
            entities.push((
                cursor.u32()?,
                (f32::from_bits(cursor.u32()?), f32::from_bits(cursor.u32()?)),
            ));
        }

        // Every peer has a fixed-size header. Check the entire minimum peer
        // section before allocating any peer or visible-id vectors.
        cursor.ensure_records_fit(peer_count, AUTHORITY_AOI_PEER_RECORD_LEN)?;
        let mut peers = Vec::new();
        peers
            .try_reserve_exact(peer_count)
            .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        for _ in 0..peer_count {
            let peer_id = cursor.u64()?;
            let center = (f32::from_bits(cursor.u32()?), f32::from_bits(cursor.u32()?));
            let radius = f32::from_bits(cursor.u32()?);
            let revision = cursor.u64()?;
            let visible_count = cursor.bounded_count(config.max_entities_per_peer)?;
            cursor.ensure_records_fit(visible_count, 4)?;
            let mut visible = Vec::new();
            visible
                .try_reserve_exact(visible_count)
                .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
            for _ in 0..visible_count {
                visible.push(cursor.u32()?);
            }
            peers.push(AuthorityAoiPeerSnapshot {
                peer_id,
                center,
                radius,
                visible,
                revision,
            });
        }
        if !cursor.is_empty() {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        let snapshot = Self {
            config,
            entities,
            peers,
        };
        AuthorityAoi::restore(snapshot.clone(), expected_config)?;
        Ok(snapshot)
    }
}

struct SnapshotCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SnapshotCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], AuthorityAoiError> {
        let end = self
            .offset
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(AuthorityAoiError::InvalidSnapshot)?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8, AuthorityAoiError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, AuthorityAoiError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| AuthorityAoiError::InvalidSnapshot)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, AuthorityAoiError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| AuthorityAoiError::InvalidSnapshot)?,
        ))
    }

    fn usize_u64(&mut self) -> Result<usize, AuthorityAoiError> {
        usize::try_from(self.u64()?).map_err(|_| AuthorityAoiError::InvalidSnapshot)
    }

    fn bounded_count(&mut self, max: usize) -> Result<usize, AuthorityAoiError> {
        let count = self.usize_u64()?;
        if count > max {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        Ok(count)
    }

    fn ensure_records_fit(&self, count: usize, record_len: usize) -> Result<(), AuthorityAoiError> {
        let required = count
            .checked_mul(record_len)
            .ok_or(AuthorityAoiError::InvalidSnapshot)?;
        if required > self.bytes.len().saturating_sub(self.offset) {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// The deterministic subscription delta for one authority peer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorityAoiDelta {
    /// Stable caller-owned peer identity.
    pub peer_id: u64,
    /// Monotonic subscription revision for this peer.
    pub revision: u64,
    /// Entity ids newly visible to the peer, ascending.
    pub entered: Vec<u32>,
    /// Entity ids no longer visible to the peer, ascending.
    pub left: Vec<u32>,
    /// The complete bounded visible set after this update, ascending.
    pub visible: Vec<u32>,
}

#[derive(Clone)]
struct PeerInterest {
    center: (f32, f32),
    radius: f32,
    visible: BTreeSet<u32>,
    revision: u64,
}

/// Bounded authority-side AOI index and per-peer subscription state.
#[derive(Clone)]
pub struct AuthorityAoi {
    config: AuthorityAoiConfig,
    grid: AoiGrid,
    peers: BTreeMap<u64, PeerInterest>,
}

impl AuthorityAoi {
    /// Create a bounded authority AOI index.
    pub fn new(config: AuthorityAoiConfig) -> Result<Self, AuthorityAoiError> {
        if config.max_peers == 0
            || config.max_entities == 0
            || config.max_entities_per_peer == 0
            || !config.max_radius.is_finite()
            || config.max_radius <= 0.0
        {
            return Err(AuthorityAoiError::InvalidConfig);
        }
        let grid = AoiGrid::new(config.cell_size).map_err(|error| match error {
            AoiError::BadCellSize => AuthorityAoiError::BadCellSize,
            AoiError::NonFinite => AuthorityAoiError::NonFinitePosition,
        })?;
        Ok(Self {
            config,
            grid,
            peers: BTreeMap::new(),
        })
    }

    /// The configured authority-side bounds.
    #[must_use]
    pub const fn config(&self) -> AuthorityAoiConfig {
        self.config
    }

    /// Number of admitted AOI peers.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Number of indexed entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.grid.len()
    }

    /// Export the complete bounded AOI state in deterministic ID order.
    #[must_use]
    pub fn snapshot(&self) -> AuthorityAoiSnapshot {
        AuthorityAoiSnapshot {
            config: self.config,
            entities: self.grid.entries().collect(),
            peers: self
                .peers
                .iter()
                .map(|(&peer_id, peer)| AuthorityAoiPeerSnapshot {
                    peer_id,
                    center: peer.center,
                    radius: peer.radius,
                    visible: peer.visible.iter().copied().collect(),
                    revision: peer.revision,
                })
                .collect(),
        }
    }

    /// Restore a saved AOI state only when its configuration matches the
    /// caller's current limits and every persisted view is internally
    /// consistent with the saved spatial index.
    pub fn restore(
        snapshot: AuthorityAoiSnapshot,
        expected_config: AuthorityAoiConfig,
    ) -> Result<Self, AuthorityAoiError> {
        if snapshot.config != expected_config
            || snapshot.entities.len() > expected_config.max_entities
            || snapshot.peers.len() > expected_config.max_peers
        {
            return Err(AuthorityAoiError::InvalidSnapshot);
        }

        let mut restored = Self::new(expected_config)?;
        let mut previous_entity = None;
        for (entity_id, position) in snapshot.entities {
            if previous_entity.is_some_and(|previous| previous >= entity_id)
                || !position.0.is_finite()
                || !position.1.is_finite()
            {
                return Err(AuthorityAoiError::InvalidSnapshot);
            }
            previous_entity = Some(entity_id);
            restored
                .grid
                .insert(entity_id, position.0, position.1)
                .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
        }

        let mut previous_peer = None;
        for peer_snapshot in snapshot.peers {
            if previous_peer.is_some_and(|previous| previous >= peer_snapshot.peer_id)
                || peer_snapshot.revision == 0
                || peer_snapshot.visible.len() > expected_config.max_entities_per_peer
                || peer_snapshot.visible.windows(2).any(|ids| ids[0] >= ids[1])
            {
                return Err(AuthorityAoiError::InvalidSnapshot);
            }
            previous_peer = Some(peer_snapshot.peer_id);
            restored
                .validate_view(peer_snapshot.center, peer_snapshot.radius)
                .map_err(|_| AuthorityAoiError::InvalidSnapshot)?;
            let visible: BTreeSet<u32> = peer_snapshot.visible.into_iter().collect();
            if visible != restored.visible_set(peer_snapshot.center, peer_snapshot.radius) {
                return Err(AuthorityAoiError::InvalidSnapshot);
            }
            restored.peers.insert(
                peer_snapshot.peer_id,
                PeerInterest {
                    center: peer_snapshot.center,
                    radius: peer_snapshot.radius,
                    visible,
                    revision: peer_snapshot.revision,
                },
            );
        }
        Ok(restored)
    }

    /// Return the current viewers of an indexed entity in deterministic
    /// ascending peer-id order.
    #[must_use]
    pub fn viewers(&self, entity_id: u32) -> Vec<u64> {
        self.peers
            .iter()
            .filter_map(|(&peer_id, peer)| peer.visible.contains(&entity_id).then_some(peer_id))
            .collect()
    }

    /// Return a peer's current AOI subscription revision.
    pub fn peer_revision(&self, peer_id: u64) -> Result<u64, AuthorityAoiError> {
        self.peers
            .get(&peer_id)
            .map(|peer| peer.revision)
            .ok_or(AuthorityAoiError::UnknownPeer)
    }

    /// Admit a peer and return its initial bounded visible set.
    pub fn admit_peer(
        &mut self,
        peer_id: u64,
        center: (f32, f32),
        radius: f32,
    ) -> Result<AuthorityAoiDelta, AuthorityAoiError> {
        if self.peers.contains_key(&peer_id) {
            return Err(AuthorityAoiError::PeerAlreadyAdmitted);
        }
        if self.peers.len() >= self.config.max_peers {
            return Err(AuthorityAoiError::PeerCapacityExceeded);
        }
        self.validate_view(center, radius)?;
        let visible = self.visible_set(center, radius);
        let peer = PeerInterest {
            center,
            radius,
            visible,
            revision: 1,
        };
        let delta = AuthorityAoiDelta {
            peer_id,
            revision: peer.revision,
            entered: peer.visible.iter().copied().collect(),
            left: Vec::new(),
            visible: peer.visible.iter().copied().collect(),
        };
        self.peers.insert(peer_id, peer);
        Ok(delta)
    }

    /// Remove a peer subscription. The peer id can be admitted again later.
    pub fn remove_peer(&mut self, peer_id: u64) -> Result<(), AuthorityAoiError> {
        self.peers
            .remove(&peer_id)
            .map(|_| ())
            .ok_or(AuthorityAoiError::UnknownPeer)
    }

    /// Insert or move one entity and return the peers whose visible set changed.
    pub fn upsert_entity(
        &mut self,
        entity_id: u32,
        position: (f32, f32),
    ) -> Result<Vec<AuthorityAoiDelta>, AuthorityAoiError> {
        self.validate_position(position)?;
        if self.grid.position(entity_id).is_none() && self.grid.len() >= self.config.max_entities {
            return Err(AuthorityAoiError::EntityCapacityExceeded);
        }
        self.grid
            .insert(entity_id, position.0, position.1)
            .map_err(|error| match error {
                AoiError::BadCellSize => AuthorityAoiError::BadCellSize,
                AoiError::NonFinite => AuthorityAoiError::NonFinitePosition,
            })?;
        self.refresh_changed_peers()
    }

    /// Remove one entity and return the peers whose visible set changed.
    pub fn remove_entity(
        &mut self,
        entity_id: u32,
    ) -> Result<Vec<AuthorityAoiDelta>, AuthorityAoiError> {
        if !self.grid.remove(entity_id) {
            return Ok(Vec::new());
        }
        self.refresh_changed_peers()
    }

    /// Move a peer's AOI centre/radius and return its new visible set delta.
    pub fn update_peer(
        &mut self,
        peer_id: u64,
        center: (f32, f32),
        radius: f32,
    ) -> Result<AuthorityAoiDelta, AuthorityAoiError> {
        self.validate_view(center, radius)?;
        let visible = self.visible_set(center, radius);
        let peer = self
            .peers
            .get_mut(&peer_id)
            .ok_or(AuthorityAoiError::UnknownPeer)?;
        peer.center = center;
        peer.radius = radius;
        refresh_peer(peer_id, peer, visible)
    }

    /// Refresh one peer against the current entity index without changing its view.
    pub fn refresh_peer(&mut self, peer_id: u64) -> Result<AuthorityAoiDelta, AuthorityAoiError> {
        let (center, radius) = self
            .peers
            .get(&peer_id)
            .map(|peer| (peer.center, peer.radius))
            .ok_or(AuthorityAoiError::UnknownPeer)?;
        let visible = self.visible_set(center, radius);
        let peer = self
            .peers
            .get_mut(&peer_id)
            .expect("peer was checked above");
        refresh_peer(peer_id, peer, visible)
    }

    fn refresh_changed_peers(&mut self) -> Result<Vec<AuthorityAoiDelta>, AuthorityAoiError> {
        let peer_ids: Vec<u64> = self.peers.keys().copied().collect();
        let mut updates = Vec::new();
        for peer_id in peer_ids {
            let delta = self.refresh_peer(peer_id)?;
            if !delta.entered.is_empty() || !delta.left.is_empty() {
                updates.push(delta);
            }
        }
        Ok(updates)
    }

    fn visible_set(&self, center: (f32, f32), radius: f32) -> BTreeSet<u32> {
        let ranked = self.grid.query_circle_ranked(center.0, center.1, radius);
        select_within_budget(&ranked, self.config.max_entities_per_peer)
            .into_iter()
            .collect()
    }

    fn validate_position(&self, position: (f32, f32)) -> Result<(), AuthorityAoiError> {
        if position.0.is_finite() && position.1.is_finite() {
            Ok(())
        } else {
            Err(AuthorityAoiError::NonFinitePosition)
        }
    }

    fn validate_view(&self, center: (f32, f32), radius: f32) -> Result<(), AuthorityAoiError> {
        self.validate_position(center)?;
        if !radius.is_finite() || radius <= 0.0 || radius > self.config.max_radius {
            return Err(AuthorityAoiError::InvalidRadius);
        }
        Ok(())
    }
}

fn refresh_peer(
    peer_id: u64,
    peer: &mut PeerInterest,
    visible: BTreeSet<u32>,
) -> Result<AuthorityAoiDelta, AuthorityAoiError> {
    let before: Vec<u32> = peer.visible.iter().copied().collect();
    let after: Vec<u32> = visible.iter().copied().collect();
    let edges = interest_delta(&before, &after);
    if !edges.is_empty() {
        peer.revision = peer
            .revision
            .checked_add(1)
            .ok_or(AuthorityAoiError::RevisionExhausted)?;
        peer.visible = visible;
    }
    Ok(AuthorityAoiDelta {
        peer_id,
        revision: peer.revision,
        entered: edges.entered,
        left: edges.left,
        visible: peer.visible.iter().copied().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn aoi() -> AuthorityAoi {
        AuthorityAoi::new(AuthorityAoiConfig {
            max_peers: 2,
            max_entities: 3,
            max_entities_per_peer: 2,
            max_radius: 20.0,
            cell_size: 10.0,
        })
        .expect("valid authority AOI config")
    }

    #[test]
    fn admission_and_entity_updates_emit_deterministic_bounded_edges() {
        let mut aoi = aoi();
        let initial = aoi.admit_peer(20, (0.0, 0.0), 5.0).unwrap();
        assert_eq!(initial.revision, 1);
        assert!(initial.visible.is_empty());

        assert!(aoi.upsert_entity(2, (30.0, 0.0)).unwrap().is_empty());
        let updates = aoi.upsert_entity(1, (1.0, 0.0)).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].peer_id, 20);
        assert_eq!(updates[0].entered, vec![1]);
        assert!(updates[0].left.is_empty());
        assert_eq!(updates[0].visible, vec![1]);
        assert_eq!(updates[0].revision, 2);

        let moved = aoi.update_peer(20, (100.0, 0.0), 5.0).unwrap();
        assert!(moved.entered.is_empty());
        assert_eq!(moved.left, vec![1]);
        assert!(moved.visible.is_empty());
        assert_eq!(moved.revision, 3);
    }

    #[test]
    fn peer_budget_is_nearest_first_and_entity_capacity_is_hard() {
        let mut aoi = aoi();
        aoi.admit_peer(1, (0.0, 0.0), 20.0).unwrap();
        aoi.upsert_entity(30, (10.0, 0.0)).unwrap();
        aoi.upsert_entity(10, (1.0, 0.0)).unwrap();
        let update = aoi.upsert_entity(20, (2.0, 0.0)).unwrap();
        assert_eq!(update[0].visible, vec![10, 20]);
        assert_eq!(
            aoi.upsert_entity(40, (3.0, 0.0)),
            Err(AuthorityAoiError::EntityCapacityExceeded)
        );
    }

    #[test]
    fn peer_admission_and_view_bounds_fail_closed() {
        let mut aoi = aoi();
        assert_eq!(
            aoi.admit_peer(1, (0.0, 0.0), 21.0),
            Err(AuthorityAoiError::InvalidRadius)
        );
        assert_eq!(
            aoi.admit_peer(1, (f32::NAN, 0.0), 1.0),
            Err(AuthorityAoiError::NonFinitePosition)
        );
        aoi.admit_peer(1, (0.0, 0.0), 1.0).unwrap();
        assert_eq!(
            aoi.admit_peer(1, (0.0, 0.0), 1.0),
            Err(AuthorityAoiError::PeerAlreadyAdmitted)
        );
        aoi.admit_peer(2, (0.0, 0.0), 1.0).unwrap();
        assert_eq!(
            aoi.admit_peer(3, (0.0, 0.0), 1.0),
            Err(AuthorityAoiError::PeerCapacityExceeded)
        );
        assert_eq!(aoi.refresh_peer(99), Err(AuthorityAoiError::UnknownPeer));
    }

    #[test]
    fn removing_entities_clears_visible_state_without_unbounded_history() {
        let mut aoi = aoi();
        aoi.admit_peer(7, (0.0, 0.0), 5.0).unwrap();
        aoi.upsert_entity(1, (0.0, 0.0)).unwrap();
        let updates = aoi.remove_entity(1).unwrap();
        assert_eq!(updates[0].left, vec![1]);
        assert!(updates[0].visible.is_empty());
        assert!(aoi.remove_entity(1).unwrap().is_empty());
        assert_eq!(aoi.entity_count(), 0);
    }

    #[test]
    fn snapshot_round_trips_exact_bounded_spatial_and_subscription_state() {
        let mut original = aoi();
        original.admit_peer(20, (0.0, 0.0), 5.0).unwrap();
        original.admit_peer(10, (30.0, 0.0), 5.0).unwrap();
        original.upsert_entity(9, (1.0, 0.0)).unwrap();
        original.upsert_entity(3, (31.0, 0.0)).unwrap();
        original.update_peer(20, (1.0, 0.0), 6.0).unwrap();

        let snapshot = original.snapshot();
        assert_eq!(snapshot.entities, vec![(3, (31.0, 0.0)), (9, (1.0, 0.0))]);
        assert_eq!(
            snapshot
                .peers
                .iter()
                .map(|peer| peer.peer_id)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        let restored = AuthorityAoi::restore(snapshot, original.config()).unwrap();
        assert_eq!(restored.snapshot(), original.snapshot());
        assert_eq!(restored.viewers(9), vec![20]);
        assert_eq!(restored.viewers(3), vec![10]);
    }

    #[test]
    fn restore_rejects_configuration_or_visible_set_mismatch() {
        let mut original = aoi();
        original.admit_peer(1, (0.0, 0.0), 5.0).unwrap();
        original.upsert_entity(7, (1.0, 0.0)).unwrap();
        let canonical = original.snapshot();

        let mut mismatched_config = canonical.clone();
        mismatched_config.config.cell_size = 5.0;
        assert_eq!(
            AuthorityAoi::restore(mismatched_config, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );

        let mut mismatched_view = canonical.clone();
        mismatched_view.peers[0].visible.clear();
        assert_eq!(
            AuthorityAoi::restore(mismatched_view, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );

        let mut duplicate_entity = canonical.clone();
        duplicate_entity.entities.push(duplicate_entity.entities[0]);
        assert_eq!(
            AuthorityAoi::restore(duplicate_entity, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );

        let mut non_finite_entity = canonical.clone();
        non_finite_entity.entities[0].1 .0 = f32::NAN;
        assert_eq!(
            AuthorityAoi::restore(non_finite_entity, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );

        let mut zero_revision = canonical.clone();
        zero_revision.peers[0].revision = 0;
        assert_eq!(
            AuthorityAoi::restore(zero_revision, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );

        let mut duplicate_peer = canonical;
        duplicate_peer.peers.push(duplicate_peer.peers[0].clone());
        assert_eq!(
            AuthorityAoi::restore(duplicate_peer, original.config()).err(),
            Some(AuthorityAoiError::InvalidSnapshot)
        );
    }

    #[test]
    fn snapshot_codec_is_canonical_bounded_and_rejects_corruption() {
        let mut original = aoi();
        original.admit_peer(4, (0.0, 0.0), 8.0).unwrap();
        original.upsert_entity(2, (1.0, 0.0)).unwrap();
        original.upsert_entity(7, (7.0, 0.0)).unwrap();
        let snapshot = original.snapshot();
        let encoded = snapshot.encode(4096).unwrap();
        let decoded = AuthorityAoiSnapshot::decode(&encoded, original.config(), 4096).unwrap();
        assert_eq!(decoded, snapshot);
        assert_eq!(decoded.encode(4096).unwrap(), encoded);
        assert_eq!(
            snapshot.encode(encoded.len() - 1),
            Err(AuthorityAoiError::InvalidSnapshot)
        );

        assert_eq!(
            AuthorityAoiSnapshot::decode(&encoded[..encoded.len() - 1], original.config(), 4096),
            Err(AuthorityAoiError::InvalidSnapshot)
        );
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert_eq!(
            AuthorityAoiSnapshot::decode(&trailing, original.config(), 4096),
            Err(AuthorityAoiError::InvalidSnapshot)
        );
        let mut unsupported_version = encoded.clone();
        unsupported_version[4] = AUTHORITY_AOI_SNAPSHOT_VERSION + 1;
        assert_eq!(
            AuthorityAoiSnapshot::decode(&unsupported_version, original.config(), 4096),
            Err(AuthorityAoiError::InvalidSnapshot)
        );
        let mut oversized_count = encoded;
        let count_offset = 4 + 1 + 8 * 3 + 4 * 2;
        oversized_count[count_offset..count_offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            AuthorityAoiSnapshot::decode(&oversized_count, original.config(), 4096),
            Err(AuthorityAoiError::InvalidSnapshot)
        );
    }
}
