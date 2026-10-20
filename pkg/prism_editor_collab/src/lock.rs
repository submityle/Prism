use crate::command::ActorId;
use core::fmt;
use std::collections::BTreeMap;

/// Generic resource name protected by an advisory soft lock.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ResourceId(String);

impl ResourceId {
    /// Creates a non-empty resource identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, LockError> {
        let value = value.into();
        if value.is_empty() {
            return Err(LockError::EmptyResource);
        }
        Ok(Self(value))
    }

    /// Creates a resource identifier for trusted static data.
    #[must_use]
    pub fn from_static(value: &'static str) -> Self {
        assert!(
            !value.is_empty(),
            "static resource identifiers must not be empty"
        );
        Self(value.to_owned())
    }

    /// Returns the resource text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Capability to renew or release one soft lock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockLease {
    resource: ResourceId,
    owner: ActorId,
    token: u64,
}

impl LockLease {
    /// Returns the lock resource.
    #[must_use]
    pub fn resource(&self) -> &ResourceId {
        &self.resource
    }

    /// Returns the lock owner.
    #[must_use]
    pub fn owner(&self) -> &ActorId {
        &self.owner
    }
}

/// Current advisory lock record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockRecord {
    /// Resource held by the lock.
    pub resource: ResourceId,
    /// Actor holding the lease.
    pub owner: ActorId,
    /// Monotonic local token for stale-operation rejection.
    pub token: u64,
    /// Acquisition time supplied by the caller.
    pub acquired_at: u64,
    /// Exclusive expiration time.
    pub expires_at: u64,
}

/// Soft-lock lifecycle failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LockError {
    /// Resource identifier was empty.
    EmptyResource,
    /// A zero TTL is invalid.
    InvalidTtl,
    /// Resource is currently held by another live lease.
    Held(LockRecord),
    /// Lease token/owner does not match the current lock.
    InvalidLease,
    /// Lease expired before the operation.
    Expired,
}

impl fmt::Display for LockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "soft-lock lifecycle error: {self:?}")
    }
}

impl std::error::Error for LockError {}

/// Deterministic advisory soft-lock registry.
#[derive(Clone, Debug)]
pub struct SoftLockRegistry {
    next_token: u64,
    locks: BTreeMap<ResourceId, LockRecord>,
}

impl SoftLockRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_token: 0,
            locks: BTreeMap::new(),
        }
    }

    /// Acquires a resource or reports the current live holder.
    pub fn acquire(
        &mut self,
        owner: ActorId,
        resource: ResourceId,
        now: u64,
        ttl: u64,
    ) -> Result<LockLease, LockError> {
        if ttl == 0 {
            return Err(LockError::InvalidTtl);
        }
        self.expire(now);
        if let Some(record) = self.locks.get(&resource) {
            return Err(LockError::Held(record.clone()));
        }
        self.next_token = self.next_token.saturating_add(1);
        let record = LockRecord {
            resource: resource.clone(),
            owner: owner.clone(),
            token: self.next_token,
            acquired_at: now,
            expires_at: now.saturating_add(ttl),
        };
        let lease = LockLease {
            resource,
            owner,
            token: record.token,
        };
        self.locks.insert(record.resource.clone(), record);
        Ok(lease)
    }

    /// Renews a live lease.
    pub fn renew(&mut self, lease: &LockLease, now: u64, ttl: u64) -> Result<(), LockError> {
        if ttl == 0 {
            return Err(LockError::InvalidTtl);
        }
        let Some(record) = self.locks.get_mut(&lease.resource) else {
            return Err(LockError::InvalidLease);
        };
        if record.owner != lease.owner || record.token != lease.token {
            return Err(LockError::InvalidLease);
        }
        if now >= record.expires_at {
            self.locks.remove(&lease.resource);
            return Err(LockError::Expired);
        }
        record.expires_at = now.saturating_add(ttl);
        Ok(())
    }

    /// Releases a live lease.
    pub fn release(&mut self, lease: &LockLease) -> Result<LockRecord, LockError> {
        let Some(record) = self.locks.get(&lease.resource) else {
            return Err(LockError::InvalidLease);
        };
        if record.owner != lease.owner || record.token != lease.token {
            return Err(LockError::InvalidLease);
        }
        Ok(self
            .locks
            .remove(&lease.resource)
            .expect("lock record was checked above"))
    }

    /// Expires all locks whose exclusive deadline has passed.
    pub fn expire(&mut self, now: u64) -> Vec<LockRecord> {
        let expired: Vec<_> = self
            .locks
            .values()
            .filter(|record| now >= record.expires_at)
            .cloned()
            .collect();
        for record in &expired {
            self.locks.remove(&record.resource);
        }
        expired
    }

    /// Returns all live lock records in stable resource order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<LockRecord> {
        self.locks.values().cloned().collect()
    }
}

impl Default for SoftLockRegistry {
    fn default() -> Self {
        Self::new()
    }
}
