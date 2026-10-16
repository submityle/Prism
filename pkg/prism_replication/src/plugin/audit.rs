//! Bounded audit of the retained, pre-input ECS snapshot timeline.
//!
//! Hashes use `WorldSnapshot::state_hash`, not the record protocol's values.
//! Only registered semantic hash glue covers component values; clone-only
//! registrations do not. Resources require explicit snapshot/hash registration.
//! Peers must agree on schemas, entity IDs, change ticks and storage layout.
//! Equality is probabilistic and is never a claim about unregistered state.

use alloc::collections::BTreeMap;
use prism_app::SnapshotHistory;
use prism_ecs::world::snapshot::{FrameHash, FrameHashLog, TickDivergence};

/// One pre-input snapshot hash. `sequence` increases for every locally changed
/// audit entry, including a corrected tick after rollback. It never wraps and
/// is scoped to a connection; reconnecting requires a new audit instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHashUpdate {
    /// Zero-based input tick (not the one-based authority revision).
    pub frame: FrameHash,
    /// Sender's monotonic update sequence.
    pub sequence: u64,
}

impl FrameHashUpdate {
    pub(super) fn encode(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[..8].copy_from_slice(&self.frame.tick.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.frame.hash.to_le_bytes());
        bytes[16..].copy_from_slice(&self.sequence.to_le_bytes());
        bytes
    }

    pub(super) fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 24 {
            return None;
        }
        Some(Self {
            frame: FrameHash {
                tick: u64::from_le_bytes(bytes[..8].try_into().ok()?),
                hash: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            },
            sequence: u64::from_le_bytes(bytes[16..].try_into().ok()?),
        })
    }
}

/// Audit failures are separate from state divergence (a comparison result).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameHashAuditError {
    /// This tick has left the retained local window.
    Expired {
        /// Rejected tick.
        tick: u64,
        /// Oldest locally retained tick.
        oldest: u64,
    },
    /// A peer is too far ahead of the local timeline.
    OutsideWindow {
        /// Rejected tick.
        tick: u64,
        /// Exclusive upper bound for admitted future ticks.
        end_exclusive: u64,
    },
    /// Retained remote entries already occupy the entire bounded window.
    RemoteWindowFull {
        /// Tick refused without evicting another admitted entry.
        tick: u64,
    },
    /// The same tick/sequence was reused for a different hash.
    ConflictingDuplicate {
        /// Reused tick.
        tick: u64,
        /// Reused sequence.
        sequence: u64,
    },
    /// No safe update sequence remains; reconnect instead of wrapping.
    SequenceExhausted,
    /// Backpressure outlived retention; this update never entered the reliable
    /// lane. The frame is removed but the loss is reported to the caller.
    UnsentFrameExpired {
        /// First expired unsent tick in ascending order.
        tick: u64,
    },
}

/// Result of receiving one well-formed hash envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameHashReceipt {
    /// First admitted update for this tick.
    Inserted,
    /// A higher sequence replaced an older hash for this tick.
    Replaced,
    /// Exact idempotent retransmission.
    Duplicate,
    /// Lower sequence ignored without rolling back a correction.
    Stale,
}

/// Comparison over retained common ticks, sorted by tick, not arrival order.
/// Missing packets never imply agreement. `first_divergence` is the earliest
/// *known* divergent tick in this window, not proof that missing/expired ticks
/// agreed. It clears when a later rollback correction makes the hashes equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHashComparison {
    /// Number of retained ticks with both local and remote hashes.
    pub compared: usize,
    /// Retained snapshots not recorded because an audit operation failed.
    pub missing_local: usize,
    /// Locally recorded ticks with no remote hash yet.
    pub missing_remote: usize,
    /// Remote ticks waiting for a local snapshot.
    pub pending_remote: usize,
    /// Earliest known divergence, not proof about absent/expired ticks.
    pub first_divergence: Option<TickDivergence>,
}

impl FrameHashComparison {
    /// A nonempty retained local window's recorded hashes are fully covered
    /// and agree. This does not establish semantic hash coverage or equality.
    #[must_use]
    pub fn retained_window_matches(&self) -> bool {
        self.compared != 0
            && self.missing_local == 0
            && self.missing_remote == 0
            && self.pending_remote == 0
            && self.first_divergence.is_none()
    }
}

/// Local, remote and unsent maps each hold at most `capacity` entries. Outgoing
/// corrections coalesce per tick until admitted to the reliable transport.
/// No raw component bytes or shared mutable snapshot buffers are retained here.
pub struct FrameHashAudit {
    capacity: usize,
    oldest: u64,
    next_tick: u64,
    sequence: u64,
    missing_local: usize,
    local: BTreeMap<u64, FrameHashUpdate>,
    remote: BTreeMap<u64, FrameHashUpdate>,
    pending: BTreeMap<u64, FrameHashUpdate>,
}

impl FrameHashAudit {
    /// A positive retention capacity, normally the ECS snapshot-ring capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "audit capacity must be positive");
        Self {
            capacity,
            oldest: 0,
            next_tick: 0,
            sequence: 0,
            missing_local: 0,
            local: BTreeMap::new(),
            remote: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    #[must_use]
    /// Maximum entries retained in each of the three maps.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    /// Local captured hash at this retained tick.
    pub fn local(&self, tick: u64) -> Option<FrameHashUpdate> {
        self.local.get(&tick).copied()
    }

    #[must_use]
    /// Most recent accepted remote update for this tick.
    pub fn remote(&self, tick: u64) -> Option<FrameHashUpdate> {
        self.remote.get(&tick).copied()
    }

    #[must_use]
    /// Number of coalesced updates not yet admitted to the reliable lane.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Abandon a timeline after a full snapshot has committed. An empty history
    /// starts at `next_tick`; no old local, remote or unsent entry belongs to it,
    /// even if the new baseline reuses the same tick numbers. Keep the outgoing
    /// sequence monotonic for the lifetime of this connection.
    pub fn reset_branch(&mut self, next_tick: u64) {
        self.oldest = next_tick;
        self.next_tick = next_tick;
        self.missing_local = 0;
        self.local.clear();
        self.remote.clear();
        self.pending.clear();
    }

    /// Export the bounded local window in ECS replay's native comparison type.
    #[must_use]
    pub fn local_log(&self) -> FrameHashLog {
        let mut log = FrameHashLog::with_capacity(self.local.len());
        for update in self.local.values() {
            log.record(update.frame.tick, update.frame.hash);
        }
        log
    }

    /// Reconcile audit retention with the actual snapshot ring, including a
    /// truncated/partially replayed future. Only ticks at/after `from` are hashed;
    /// normal captures pass the one new tick, rollbacks pass the authority
    /// revision (the unchanged pre-input anchor is revision minus one).
    /// Continue reconciling after a loss so no abandoned audit branch survives.
    pub fn sync_history(
        &mut self,
        history: &SnapshotHistory,
        from: u64,
    ) -> Result<(), FrameHashAuditError> {
        self.next_tick = history.next_frame();
        self.oldest = history
            .oldest_frame()
            .unwrap_or(self.next_tick)
            .max(self.next_tick.saturating_sub(self.capacity as u64));
        let mut error = None;
        self.local.retain(|tick, _| {
            *tick >= self.oldest && *tick < self.next_tick && history.contains(*tick)
        });
        self.pending.retain(|tick, _| {
            let retain = *tick >= self.oldest && *tick < self.next_tick && history.contains(*tick);
            if !retain && *tick < self.oldest && error.is_none() {
                error = Some(FrameHashAuditError::UnsentFrameExpired { tick: *tick });
            }
            retain
        });
        let end = self.next_tick.saturating_add(self.capacity as u64);
        self.remote
            .retain(|tick, _| *tick >= self.oldest && *tick < end);
        for tick in from.max(self.oldest)..self.next_tick {
            if let Some(hash) = history.state_hash(tick)
                && let Err(record_error) = self.record_retained(tick, hash)
            {
                error.get_or_insert(record_error);
            }
        }
        self.missing_local = (self.oldest..self.next_tick)
            .filter(|tick| history.contains(*tick) && !self.local.contains_key(tick))
            .count();
        error.map_or(Ok(()), Err)
    }

    fn record_retained(&mut self, tick: u64, hash: u64) -> Result<(), FrameHashAuditError> {
        if self
            .local
            .get(&tick)
            .is_some_and(|old| old.frame.hash == hash)
        {
            return Ok(());
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            // Never compare or transmit the old branch as if it were the newly
            // captured value when a correction cannot get a fresh sequence.
            self.local.remove(&tick);
            self.pending.remove(&tick);
            return Err(FrameHashAuditError::SequenceExhausted);
        };
        self.sequence = sequence;
        let update = FrameHashUpdate {
            frame: FrameHash { tick, hash },
            sequence,
        };
        self.local.insert(tick, update);
        self.pending.insert(tick, update);
        Ok(())
    }

    /// Admit remote hashes in any tick order, but only within a bounded window.
    /// A future tick waits for its local capture; it is not a mismatch yet.
    pub fn receive(
        &mut self,
        update: FrameHashUpdate,
    ) -> Result<FrameHashReceipt, FrameHashAuditError> {
        let tick = update.frame.tick;
        if tick < self.oldest {
            return Err(FrameHashAuditError::Expired {
                tick,
                oldest: self.oldest,
            });
        }
        let end = self.next_tick.saturating_add(self.capacity as u64);
        if tick >= end {
            return Err(FrameHashAuditError::OutsideWindow {
                tick,
                end_exclusive: end,
            });
        }
        if let Some(old) = self.remote.get(&tick) {
            if update.sequence < old.sequence {
                return Ok(FrameHashReceipt::Stale);
            }
            if update.sequence == old.sequence {
                return if update == *old {
                    Ok(FrameHashReceipt::Duplicate)
                } else {
                    Err(FrameHashAuditError::ConflictingDuplicate {
                        tick,
                        sequence: update.sequence,
                    })
                };
            }
            self.remote.insert(tick, update);
            return Ok(FrameHashReceipt::Replaced);
        }
        if self.remote.len() == self.capacity {
            return Err(FrameHashAuditError::RemoteWindowFull { tick });
        }
        self.remote.insert(tick, update);
        Ok(FrameHashReceipt::Inserted)
    }

    #[must_use]
    /// Align shared ticks before using ECS's positional replay-log comparator.
    pub fn comparison(&self) -> FrameHashComparison {
        let mut local = FrameHashLog::with_capacity(self.local.len());
        let mut remote = FrameHashLog::with_capacity(self.local.len());
        let mut missing_remote = 0;
        for (&tick, update) in &self.local {
            if let Some(peer) = self.remote.get(&tick) {
                local.record(tick, update.frame.hash);
                remote.record(tick, peer.frame.hash);
            } else {
                missing_remote += 1;
            }
        }
        FrameHashComparison {
            compared: local.len(),
            missing_local: self.missing_local,
            missing_remote,
            pending_remote: self
                .remote
                .keys()
                .filter(|tick| !self.local.contains_key(*tick))
                .count(),
            first_divergence: local.first_divergence(&remote),
        }
    }

    pub(super) fn next_pending(&self) -> Option<FrameHashUpdate> {
        self.pending.first_key_value().map(|(_, update)| *update)
    }

    pub(super) fn sent(&mut self, update: FrameHashUpdate) {
        if self.pending.get(&update.frame.tick) == Some(&update) {
            self.pending.remove(&update.frame.tick);
        }
    }
}

#[cfg(test)]
mod tests;
