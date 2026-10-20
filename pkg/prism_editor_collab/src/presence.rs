use crate::command::ActorId;
use core::fmt;
use std::collections::BTreeMap;

/// Advisory state shown for an active actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresenceStatus {
    /// Actor is actively editing.
    Active,
    /// Actor is connected but not actively editing.
    Away,
    /// Actor is connected and asks clients to highlight activity.
    Busy,
}

/// Capability to renew or leave one presence session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresenceLease {
    actor: ActorId,
    epoch: u64,
}

impl PresenceLease {
    /// Returns the actor represented by the lease.
    #[must_use]
    pub fn actor(&self) -> &ActorId {
        &self.actor
    }
}

/// Current presence record with explicit expiry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresenceRecord {
    /// Actor identity.
    pub actor: ActorId,
    /// User-facing label, without imposing an account system.
    pub label: String,
    /// Current activity status.
    pub status: PresenceStatus,
    /// Last accepted heartbeat time.
    pub last_seen: u64,
    /// Exclusive expiration time.
    pub expires_at: u64,
    /// Adapter-defined stable metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Presence lifecycle failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PresenceError {
    /// The actor already has a live session.
    AlreadyPresent,
    /// The session token does not match the current session.
    InvalidLease,
    /// The lease has expired.
    Expired,
    /// TTL zero would make the session immediately invalid.
    InvalidTtl,
    /// The configured participant bound is full.
    Capacity,
}

impl fmt::Display for PresenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "presence lifecycle error: {self:?}")
    }
}

impl std::error::Error for PresenceError {}

/// Deterministic, bounded presence registry.
#[derive(Clone, Debug)]
pub struct PresenceRegistry {
    capacity: usize,
    next_epoch: BTreeMap<ActorId, u64>,
    records: BTreeMap<ActorId, (PresenceLease, PresenceRecord)>,
}

impl PresenceRegistry {
    /// Creates a registry with a participant bound.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_epoch: BTreeMap::new(),
            records: BTreeMap::new(),
        }
    }

    /// Starts a presence session.
    pub fn join(
        &mut self,
        actor: ActorId,
        label: impl Into<String>,
        now: u64,
        ttl: u64,
        metadata: BTreeMap<String, String>,
    ) -> Result<PresenceLease, PresenceError> {
        if ttl == 0 {
            return Err(PresenceError::InvalidTtl);
        }
        self.expire(now);
        if self.records.contains_key(&actor) {
            return Err(PresenceError::AlreadyPresent);
        }
        if self.records.len() >= self.capacity {
            return Err(PresenceError::Capacity);
        }
        let epoch = self
            .next_epoch
            .get(&actor)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        self.next_epoch.insert(actor.clone(), epoch);
        let lease = PresenceLease {
            actor: actor.clone(),
            epoch,
        };
        let record = PresenceRecord {
            actor: actor.clone(),
            label: label.into(),
            status: PresenceStatus::Active,
            last_seen: now,
            expires_at: now.saturating_add(ttl),
            metadata,
        };
        self.records.insert(actor, (lease.clone(), record));
        Ok(lease)
    }

    /// Renews a live session and optionally updates its status.
    pub fn heartbeat(
        &mut self,
        lease: &PresenceLease,
        now: u64,
        ttl: u64,
        status: PresenceStatus,
    ) -> Result<(), PresenceError> {
        if ttl == 0 {
            return Err(PresenceError::InvalidTtl);
        }
        let Some((current, record)) = self.records.get_mut(&lease.actor) else {
            return Err(PresenceError::InvalidLease);
        };
        if current != lease {
            return Err(PresenceError::InvalidLease);
        }
        if now >= record.expires_at {
            self.records.remove(&lease.actor);
            return Err(PresenceError::Expired);
        }
        record.last_seen = now;
        record.expires_at = now.saturating_add(ttl);
        record.status = status;
        Ok(())
    }

    /// Ends a live session.
    pub fn leave(&mut self, lease: &PresenceLease) -> Result<PresenceRecord, PresenceError> {
        let Some((current, _)) = self.records.get(&lease.actor) else {
            return Err(PresenceError::InvalidLease);
        };
        if current != lease {
            return Err(PresenceError::InvalidLease);
        }
        Ok(self
            .records
            .remove(&lease.actor)
            .expect("presence record was checked above")
            .1)
    }

    /// Removes sessions whose exclusive expiry is at or before `now`.
    pub fn expire(&mut self, now: u64) -> Vec<ActorId> {
        let expired: Vec<_> = self
            .records
            .iter()
            .filter(|(_, (_, record))| now >= record.expires_at)
            .map(|(actor, _)| actor.clone())
            .collect();
        for actor in &expired {
            self.records.remove(actor);
        }
        expired
    }

    /// Returns records in stable actor order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<PresenceRecord> {
        self.records
            .values()
            .map(|(_, record)| record.clone())
            .collect()
    }
}
