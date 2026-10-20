//! Deterministic cell edit locks with explicit expiry.

use std::{collections::BTreeMap, fmt};

use prism_asset::StableGuid;

use super::{CellCoord, CellKey};

/// Capability to mutate one loaded cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellLockToken {
    cell: CellKey,
    owner: StableGuid,
    token: u64,
}

impl CellLockToken {
    /// Returns the protected cell.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        self.cell.coord()
    }

    /// Returns the lock owner.
    #[must_use]
    pub const fn owner(&self) -> StableGuid {
        self.owner
    }

    /// Returns the monotonic lease token.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.token
    }
}

/// Current live lock information.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellLockRecord {
    /// Protected cell.
    pub cell: CellCoord,
    /// Owner identity.
    pub owner: StableGuid,
    /// Monotonic lease token.
    pub token: u64,
    /// Acquisition timestamp supplied by the caller.
    pub acquired_at: u64,
    /// Exclusive expiry timestamp.
    pub expires_at: u64,
}

/// Lock lifecycle failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CellLockError {
    /// The requested cell is already held by another live owner.
    Held(CellLockRecord),
    /// A zero TTL would never protect an edit.
    InvalidTtl,
    /// The caller's expiry time cannot be represented.
    ExpiryOverflow,
    /// The token does not identify the current lock.
    InvalidToken,
    /// The token was valid but expired before the operation.
    Expired,
    /// The monotonic lock sequence cannot advance.
    TokenExhausted,
}

impl fmt::Display for CellLockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Held(record) => write!(formatter, "cell lock held by {}", record.owner),
            Self::InvalidTtl => formatter.write_str("cell lock TTL must be non-zero"),
            Self::ExpiryOverflow => formatter.write_str("cell lock expiry exceeds clock range"),
            Self::InvalidToken => formatter.write_str("cell lock token is not current"),
            Self::Expired => formatter.write_str("cell lock token expired"),
            Self::TokenExhausted => formatter.write_str("cell lock token space exhausted"),
        }
    }
}

impl std::error::Error for CellLockError {}

/// Bounded-by-live-state registry of cell locks.
#[derive(Clone, Debug, Default)]
pub struct CellLockRegistry {
    next_token: u64,
    locks: BTreeMap<CellKey, CellLockRecord>,
}

impl CellLockRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_token: 0,
            locks: BTreeMap::new(),
        }
    }

    /// Acquires a cell lock or returns the current live holder.
    pub fn acquire(
        &mut self,
        cell: CellCoord,
        owner: StableGuid,
        now: u64,
        ttl: u64,
    ) -> Result<CellLockToken, CellLockError> {
        if ttl == 0 {
            return Err(CellLockError::InvalidTtl);
        }
        let expires_at = now.checked_add(ttl).ok_or(CellLockError::ExpiryOverflow)?;
        let key = CellKey::new(cell);
        self.expire(now);
        if let Some(record) = self.locks.get(&key) {
            return Err(CellLockError::Held(record.clone()));
        }
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(CellLockError::TokenExhausted)?;
        let record = CellLockRecord {
            cell,
            owner,
            token: self.next_token,
            acquired_at: now,
            expires_at,
        };
        let token = CellLockToken {
            cell: key,
            owner,
            token: record.token,
        };
        self.locks.insert(key, record);
        Ok(token)
    }

    /// Renews a live token using a caller-supplied clock.
    pub fn renew(
        &mut self,
        token: &CellLockToken,
        now: u64,
        ttl: u64,
    ) -> Result<(), CellLockError> {
        if ttl == 0 {
            return Err(CellLockError::InvalidTtl);
        }
        let expires_at = now.checked_add(ttl).ok_or(CellLockError::ExpiryOverflow)?;
        let Some(record) = self.locks.get_mut(&token.cell) else {
            return Err(CellLockError::InvalidToken);
        };
        if record.owner != token.owner || record.token != token.token {
            return Err(CellLockError::InvalidToken);
        }
        if now >= record.expires_at {
            self.locks.remove(&token.cell);
            return Err(CellLockError::Expired);
        }
        record.expires_at = expires_at;
        Ok(())
    }

    /// Releases a current token.
    pub fn release(&mut self, token: &CellLockToken) -> Result<CellLockRecord, CellLockError> {
        let Some(record) = self.locks.get(&token.cell) else {
            return Err(CellLockError::InvalidToken);
        };
        if record.owner != token.owner || record.token != token.token {
            return Err(CellLockError::InvalidToken);
        }
        Ok(self
            .locks
            .remove(&token.cell)
            .expect("current cell lock is registered"))
    }

    /// Checks and expires all locks at `now`.
    pub fn expire(&mut self, now: u64) -> Vec<CellLockRecord> {
        let expired: Vec<_> = self
            .locks
            .values()
            .filter(|record| now >= record.expires_at)
            .cloned()
            .collect();
        for record in &expired {
            self.locks.remove(&CellKey::new(record.cell));
        }
        expired
    }

    /// Returns whether `token` is still the live token for its cell.
    #[must_use]
    pub fn is_current(&self, token: &CellLockToken, now: u64) -> bool {
        self.locks.get(&token.cell).is_some_and(|record| {
            record.owner == token.owner
                && record.token == token.token
                && now >= record.acquired_at
                && now < record.expires_at
        })
    }

    /// Returns whether any live edit lease protects `cell` at caller time.
    #[must_use]
    pub fn is_locked(&self, cell: CellCoord, now: u64) -> bool {
        self.locks
            .get(&CellKey::new(cell))
            .is_some_and(|record| now >= record.acquired_at && now < record.expires_at)
    }

    /// Returns live locks in deterministic cell order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<CellLockRecord> {
        self.locks.values().cloned().collect()
    }
}
