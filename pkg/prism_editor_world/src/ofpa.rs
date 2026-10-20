//! Deterministic one-file-per-actor metadata and write planning.
//!
//! OFPA here is a storage-granularity contract only. The catalog never emits
//! or parses a file format; callers provide opaque actor bytes to a provider.

use std::{collections::BTreeMap, fmt};

use prism_asset::StableGuid;

use super::{CellCoord, CellKey};

/// Stable path identity for one actor payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActorFileIdentity {
    actor: StableGuid,
    cell: CellCoord,
    relative_path: String,
}

impl ActorFileIdentity {
    /// Creates a numeric path identity; all signed i32 coordinates are valid.
    pub fn new(actor: StableGuid, cell: CellCoord) -> Result<Self, OfpaError> {
        if actor.is_nil() {
            return Err(OfpaError::NilActor);
        }
        Ok(Self::derive(actor, cell))
    }

    fn derive(actor: StableGuid, cell: CellCoord) -> Self {
        Self {
            actor,
            cell,
            relative_path: format!(
                "cells/{}_{}_{}/actors/{}",
                cell.x,
                cell.y,
                cell.z,
                actor.to_u128()
            ),
        }
    }

    /// Stable actor identity.
    #[must_use]
    pub const fn actor(&self) -> StableGuid {
        self.actor
    }

    /// Owning cell.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        self.cell
    }

    /// Relative path key; the provider decides how to materialize it.
    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }
}

/// Metadata for one actor in the OFPA catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActorRecord {
    identity: ActorFileIdentity,
    revision: u64,
    dirty: bool,
    persisted: Option<u64>,
}

impl ActorRecord {
    /// Last acknowledged durable revision; None means create-only.
    pub const fn persisted_revision(&self) -> Option<u64> {
        self.persisted
    }

    /// Stable file identity.
    #[must_use]
    pub const fn identity(&self) -> &ActorFileIdentity {
        &self.identity
    }

    /// Revision last observed by the catalog.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Whether the actor needs an OFPA write.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }
}

/// OFPA catalog validation or lifecycle failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OfpaError {
    /// The actor identity is the reserved nil GUID.
    NilActor,
    /// The actor already exists in the catalog.
    DuplicateActor(StableGuid),
    /// The catalog reached its configured actor capacity.
    CapacityExceeded {
        /// Configured actor capacity.
        limit: usize,
    },
    /// The actor was not registered.
    UnknownActor(StableGuid),
    /// The operation belongs to a different actor-file identity.
    IdentityMismatch(StableGuid),
    /// A clean operation cannot be applied to a newer revision.
    StaleRevision {
        /// Revision expected by the plan.
        expected: u64,
        /// Current actor revision.
        actual: u64,
    },
}

impl fmt::Display for OfpaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NilActor => formatter.write_str("actor identity must not be nil"),
            Self::DuplicateActor(actor) => write!(formatter, "actor {actor} already exists"),
            Self::CapacityExceeded { limit } => {
                write!(formatter, "OFPA actor capacity {limit} is full")
            }
            Self::UnknownActor(actor) => write!(formatter, "actor {actor} does not exist"),
            Self::IdentityMismatch(actor) => {
                write!(formatter, "actor {actor} file identity differs")
            }
            Self::StaleRevision { expected, actual } => {
                write!(
                    formatter,
                    "OFPA revision {expected} is stale; current revision is {actual}"
                )
            }
        }
    }
}

impl std::error::Error for OfpaError {}

/// One deterministic actor write operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfpaOperation {
    identity: ActorFileIdentity,
    revision: u64,
}

impl OfpaOperation {
    /// Actor file identity to write.
    #[must_use]
    pub const fn identity(&self) -> &ActorFileIdentity {
        &self.identity
    }

    /// Revision included in the plan.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}

/// Deterministic OFPA write plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OfpaPlan {
    epoch: u64,
    operations: Vec<OfpaOperation>,
}

impl OfpaPlan {
    /// Catalog epoch captured by this plan.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Actor writes in stable GUID order.
    #[must_use]
    pub fn operations(&self) -> &[OfpaOperation] {
        &self.operations
    }
}

/// Bounded deterministic actor catalog.
#[derive(Clone, Debug)]
pub struct OfpaCatalog {
    actors: BTreeMap<StableGuid, ActorRecord>,
    max_actors: usize,
    epoch: u64,
}

impl OfpaCatalog {
    /// Creates an empty catalog with a fixed actor count bound.
    pub const fn new(max_actors: usize) -> Self {
        Self {
            actors: BTreeMap::new(),
            max_actors,
            epoch: 0,
        }
    }

    /// Returns the configured actor capacity.
    #[must_use]
    pub const fn max_actors(&self) -> usize {
        self.max_actors
    }

    /// Registers an actor in its owning cell.
    pub fn register(
        &mut self,
        actor: StableGuid,
        cell: CellCoord,
    ) -> Result<&ActorRecord, OfpaError> {
        if actor.is_nil() {
            return Err(OfpaError::NilActor);
        }
        if self.actors.contains_key(&actor) {
            return Err(OfpaError::DuplicateActor(actor));
        }
        if self.actors.len() >= self.max_actors {
            return Err(OfpaError::CapacityExceeded {
                limit: self.max_actors,
            });
        }
        self.actors.insert(
            actor,
            ActorRecord {
                identity: ActorFileIdentity::derive(actor, cell),
                revision: 0,
                dirty: false,
                persisted: None,
            },
        );
        self.bump_epoch();
        Ok(self
            .actors
            .get(&actor)
            .expect("inserted actor is registered"))
    }

    /// Returns a registered actor.
    #[must_use]
    pub fn actor(&self, actor: StableGuid) -> Option<&ActorRecord> {
        self.actors.get(&actor)
    }

    /// Returns actors in stable GUID order.
    pub fn actors(&self) -> impl Iterator<Item = &ActorRecord> {
        self.actors.values()
    }

    /// Marks an actor dirty and advances its revision.
    pub fn mark_dirty(&mut self, actor: StableGuid) -> Result<(), OfpaError> {
        let record = self
            .actors
            .get_mut(&actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        record.revision = record
            .revision
            .checked_add(1)
            .ok_or(OfpaError::StaleRevision {
                expected: u64::MAX,
                actual: record.revision,
            })?;
        record.dirty = true;
        self.bump_epoch();
        Ok(())
    }

    /// Plans all dirty actor writes without changing catalog state.
    #[must_use]
    pub fn plan(&self) -> OfpaPlan {
        OfpaPlan {
            epoch: self.epoch,
            operations: self
                .actors
                .values()
                .filter(|record| record.dirty)
                .map(|record| OfpaOperation {
                    identity: record.identity.clone(),
                    revision: record.revision,
                })
                .collect(),
        }
    }

    /// Acknowledges external metadata only, without claiming any disk I/O.
    /// The world facade instead requires an actual actor save receipt.
    pub fn commit(&mut self, operation: &OfpaOperation) -> Result<(), OfpaError> {
        let actor = operation.identity.actor();
        let record = self
            .actors
            .get_mut(&actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        if record.identity != operation.identity {
            return Err(OfpaError::IdentityMismatch(actor));
        }
        if record.revision != operation.revision {
            return Err(OfpaError::StaleRevision {
                expected: operation.revision,
                actual: record.revision,
            });
        }
        record.dirty = false;
        self.bump_epoch();
        Ok(())
    }

    /// Whether any actor in this cell still has an uncommitted write.
    #[must_use]
    pub fn is_cell_dirty(&self, cell: CellCoord) -> bool {
        self.actors
            .values()
            .any(|record| record.identity.cell() == cell && record.dirty)
    }

    /// Returns whether an actor belongs to a cell.
    #[must_use]
    pub fn contains_in_cell(&self, actor: StableGuid, cell: CellKey) -> bool {
        self.actors
            .get(&actor)
            .is_some_and(|record| CellKey::new(record.identity.cell()) == cell)
    }

    pub(crate) fn note_persisted(
        &mut self,
        actor: StableGuid,
        revision: u64,
    ) -> Result<(), OfpaError> {
        let record = self
            .actors
            .get_mut(&actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        record.persisted = Some(record.persisted.map_or(revision, |old| old.max(revision)));
        Ok(())
    }

    pub(crate) fn restore(&mut self, actor: StableGuid, revision: u64) -> Result<(), OfpaError> {
        let record = self
            .actors
            .get_mut(&actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        record.revision = revision;
        record.persisted = Some(revision);
        record.dirty = false;
        self.bump_epoch();
        Ok(())
    }

    pub(crate) fn operation(&self, actor: StableGuid) -> Result<OfpaOperation, OfpaError> {
        let record = self
            .actors
            .get(&actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        Ok(OfpaOperation {
            identity: record.identity.clone(),
            revision: record.revision,
        })
    }

    fn bump_epoch(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
    }
}
