//! Deterministic shade-page residency state machine.

use alloc::collections::BTreeMap;

use super::key::ShadePageKey;

/// Lifecycle of a writable shade-cache page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadePageState {
    /// No physical page is currently available.
    Missing,
    /// The page was requested and is eligible for a shade plan.
    Requested,
    /// A compute writer owns the page and it is not sampleable yet.
    Writing { generation: u64, epoch: u64 },
    /// The page is safe for sampling.
    Ready { generation: u64, epoch: u64 },
    /// The page is retained as a fallback candidate but must be re-shaded.
    Stale { generation: u64, epoch: u64 },
    /// The old physical allocation was released.
    Evicted { generation: u64 },
}

impl ShadePageState {
    /// Returns whether a page can be sampled for `epoch`.
    #[must_use]
    pub const fn is_ready_for(self, epoch: u64) -> bool {
        matches!(self, Self::Ready { epoch: page_epoch, .. } if page_epoch == epoch)
    }

    /// Returns the generation carried by this state, if any.
    #[must_use]
    pub const fn generation(self) -> Option<u64> {
        match self {
            Self::Writing { generation, .. }
            | Self::Ready { generation, .. }
            | Self::Stale { generation, .. }
            | Self::Evicted { generation } => Some(generation),
            Self::Missing | Self::Requested => None,
        }
    }
}

/// A request fed to the shade planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadePageRequest {
    /// Requested page identity.
    pub key: ShadePageKey,
    /// Fixed-point priority; larger values are planned first.
    pub priority: u64,
    /// Physical byte cost of the page.
    pub byte_cost: u64,
    /// Frame in which the page was observed.
    pub frame: u64,
}

/// A committed write token. It cannot commit twice or after its page has been
/// invalidated because the table checks generation and epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeWriteTicket {
    /// Page identity.
    pub key: ShadePageKey,
    /// Monotonic write generation.
    pub generation: u64,
    /// Material epoch being shaded.
    pub epoch: u64,
}

/// A page lookup result used by the screen-space fallback path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadePageLookup {
    /// Exact current-epoch page is sampleable.
    Ready { generation: u64 },
    /// A page exists but needs a refresh before it can be used as an exact hit.
    Stale { generation: u64 },
    /// No usable exact page exists.
    Fallback,
}

/// Errors from invalid residency transitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeResidencyError {
    /// The key is not tracked by the table.
    UnknownPage,
    /// A write token does not match the currently owned write.
    InvalidWrite,
    /// An operation would violate the state machine.
    InvalidTransition { from: ShadePageState },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Entry {
    state: ShadePageState,
    priority: u64,
    byte_cost: u64,
    last_used_frame: u64,
}

/// CPU-side source of truth for writable shade-page residency.
///
/// It owns no GPU handles. A render backend can serialize the sorted entries
/// into a page table while the exact transition protocol and generation checks
/// remain backend-independent and testable.
#[derive(Clone, Debug)]
pub struct ShadeResidencyTable {
    entries: BTreeMap<ShadePageKey, Entry>,
    next_generation: u64,
}

impl Default for ShadeResidencyTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ShadeResidencyTable {
    /// Creates an empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_generation: 1,
        }
    }

    /// Records a visible page request without making it sampleable.
    pub fn request(&mut self, request: ShadePageRequest, epoch: u64) {
        let entry = self.entries.entry(request.key).or_insert(Entry {
            state: ShadePageState::Missing,
            priority: 0,
            byte_cost: request.byte_cost,
            last_used_frame: request.frame,
        });
        entry.priority = entry.priority.max(request.priority);
        entry.byte_cost = request.byte_cost;
        entry.last_used_frame = entry.last_used_frame.max(request.frame);
        if let ShadePageState::Ready {
            generation,
            epoch: ready_epoch,
        } = entry.state
        {
            if ready_epoch != epoch {
                entry.state = ShadePageState::Stale {
                    generation,
                    epoch: ready_epoch,
                };
            }
        } else if matches!(
            entry.state,
            ShadePageState::Missing | ShadePageState::Evicted { .. }
        ) {
            entry.state = ShadePageState::Requested;
        }
    }

    /// Begins a write for a requested or stale page.
    pub fn begin_write(
        &mut self,
        key: ShadePageKey,
        epoch: u64,
    ) -> Result<ShadeWriteTicket, ShadeResidencyError> {
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or(ShadeResidencyError::UnknownPage)?;
        if !matches!(
            entry.state,
            ShadePageState::Requested | ShadePageState::Stale { .. }
        ) {
            return Err(ShadeResidencyError::InvalidTransition { from: entry.state });
        }
        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1).max(1);
        entry.state = ShadePageState::Writing { generation, epoch };
        Ok(ShadeWriteTicket {
            key,
            generation,
            epoch,
        })
    }

    /// Publishes a completed write into the table.
    pub fn commit_write(
        &mut self,
        ticket: ShadeWriteTicket,
        frame: u64,
    ) -> Result<(), ShadeResidencyError> {
        let entry = self
            .entries
            .get_mut(&ticket.key)
            .ok_or(ShadeResidencyError::UnknownPage)?;
        if entry.state
            != (ShadePageState::Writing {
                generation: ticket.generation,
                epoch: ticket.epoch,
            })
        {
            return Err(ShadeResidencyError::InvalidWrite);
        }
        entry.state = ShadePageState::Ready {
            generation: ticket.generation,
            epoch: ticket.epoch,
        };
        entry.last_used_frame = entry.last_used_frame.max(frame);
        Ok(())
    }

    /// Marks a ready page stale after a light/material/geometry epoch change.
    pub fn invalidate(&mut self, key: ShadePageKey, epoch: u64) -> Result<(), ShadeResidencyError> {
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or(ShadeResidencyError::UnknownPage)?;
        let ShadePageState::Ready {
            generation,
            epoch: old_epoch,
        } = entry.state
        else {
            return Err(ShadeResidencyError::InvalidTransition { from: entry.state });
        };
        entry.state = ShadePageState::Stale {
            generation,
            epoch: old_epoch.max(epoch),
        };
        Ok(())
    }

    /// Releases a physical page while retaining its generation history.
    pub fn evict(&mut self, key: ShadePageKey) -> Result<(), ShadeResidencyError> {
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or(ShadeResidencyError::UnknownPage)?;
        let generation = entry.state.generation().unwrap_or(0);
        entry.state = ShadePageState::Evicted { generation };
        Ok(())
    }

    /// Looks up a page for sampling at the requested material epoch.
    #[must_use]
    pub fn lookup(&self, key: ShadePageKey, epoch: u64) -> ShadePageLookup {
        match self.entries.get(&key).map(|entry| entry.state) {
            Some(ShadePageState::Ready {
                generation,
                epoch: page_epoch,
            }) if page_epoch == epoch => ShadePageLookup::Ready { generation },
            Some(state @ (ShadePageState::Ready { .. } | ShadePageState::Stale { .. })) => {
                ShadePageLookup::Stale {
                    generation: state.generation().unwrap_or(0),
                }
            }
            _ => ShadePageLookup::Fallback,
        }
    }

    /// Returns the exact state for diagnostics and deterministic planning.
    #[must_use]
    pub fn state(&self, key: ShadePageKey) -> ShadePageState {
        self.entries
            .get(&key)
            .map_or(ShadePageState::Missing, |entry| entry.state)
    }

    /// Returns tracked page metadata in key order.
    pub fn iter(&self) -> impl Iterator<Item = (ShadePageKey, ShadePageState, u64, u64)> + '_ {
        self.entries
            .iter()
            .map(|(key, entry)| (*key, entry.state, entry.priority, entry.byte_cost))
    }
}
