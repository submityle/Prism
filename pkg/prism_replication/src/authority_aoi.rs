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
        };
        formatter.write_str(message)
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
}
