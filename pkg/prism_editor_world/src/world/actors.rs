//! Lease-protected reconciliation of real actor storage results.
use super::*;
use crate::{
    ActorRevisionBase, ActorSaveReceipt, ActorSaveRequest, ActorSnapshot, FileActorIoProvider,
    OfpaOperation,
};
use std::{path::PathBuf, sync::Arc};

/// Captured actor write, cell revision, storage root and authorizing lease.
#[derive(Clone, Debug)]
pub struct PendingActorSave {
    request: ActorSaveRequest,
    operation: OfpaOperation,
    payload: Arc<[u8]>,
    cell_revision: u64,
    lock: CellLockToken,
    root: PathBuf,
    residency: TransitionToken,
}
impl PendingActorSave {
    /// Metadata to submit to the actor provider.
    pub fn request(&self) -> &ActorSaveRequest {
        &self.request
    }
    /// Shares immutable bytes without making a second payload copy.
    pub fn payload(&self) -> Arc<[u8]> {
        self.payload.clone()
    }
}

/// Captured clean actor and cell revisions for a guarded load.
#[derive(Clone, Debug)]
pub struct PendingActorLoad {
    identity: crate::ActorFileIdentity,
    actor_revision: u64,
    cell_revision: u64,
    lock: CellLockToken,
    root: PathBuf,
    residency: TransitionToken,
}
impl PendingActorLoad {
    /// Identity to submit to the actor provider.
    pub fn identity(&self) -> &crate::ActorFileIdentity {
        &self.identity
    }
}

impl EditorWorldData {
    /// Returns opaque resident actor bytes, independently of the cell payload.
    pub fn actor_payload(&self, actor: StableGuid) -> Option<&[u8]> {
        self.actor_payloads.get(&actor).map(AsRef::as_ref)
    }

    fn actor_cell_revision(
        &self,
        actor: StableGuid,
        lock: &CellLockToken,
        now: u64,
    ) -> Result<(CellCoord, u64), EditorWorldDataError> {
        let cell = self
            .ofpa
            .actor(actor)
            .ok_or(OfpaError::UnknownActor(actor))?
            .identity()
            .cell();
        self.require_lock(cell, lock, now)?;
        self.require_editable(cell)?;
        let record = self.model.cell(cell).ok_or(WorldError::UnknownCell(cell))?;
        if record.residency() != CellResidency::Loaded {
            return Err(WorldError::InvalidTransition(cell).into());
        }
        Ok((cell, record.revision()))
    }

    fn check_actor_bytes(
        &self,
        actor: StableGuid,
        cell: CellCoord,
        bytes: usize,
    ) -> Result<(), EditorWorldDataError> {
        // One additional bounded actor-byte pool per resident cell. Cell bytes
        // retain their existing independent limit for backwards compatibility.
        let mut total = bytes;
        for (id, payload) in &self.actor_payloads {
            if *id != actor && self.ofpa.contains_in_cell(*id, CellKey::new(cell)) {
                total = total.saturating_add(payload.len());
            }
        }
        if total > self.limits.max_cell_bytes() {
            return Err(CellPayloadError::TooLarge {
                actual: total,
                limit: self.limits.max_cell_bytes(),
            }
            .into());
        }
        Ok(())
    }

    /// Replaces opaque bytes under a live lease and advances actor and cell
    /// revisions atomically. The sum of actor bytes per cell is bounded.
    pub fn replace_actor_payload(
        &mut self,
        actor: StableGuid,
        lock: &CellLockToken,
        now: u64,
        payload: Arc<[u8]>,
    ) -> Result<(), EditorWorldDataError> {
        let (cell, _) = self.actor_cell_revision(actor, lock, now)?;
        self.check_actor_bytes(actor, cell, payload.len())?;
        self.mark_actor_dirty(actor, lock, now)?;
        self.actor_payloads.insert(actor, payload);
        Ok(())
    }

    /// Captures a dirty actor for a real compare-and-save in this provider root.
    pub fn prepare_actor_save(
        &self,
        actor: StableGuid,
        lock: &CellLockToken,
        now: u64,
        provider: &FileActorIoProvider,
    ) -> Result<PendingActorSave, EditorWorldDataError> {
        let (cell, cell_revision) = self.actor_cell_revision(actor, lock, now)?;
        let record = self
            .ofpa
            .actor(actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        if !record.is_dirty() {
            return Err(WorldError::InvalidTransition(record.identity().cell()).into());
        }
        let payload = self
            .actor_payloads
            .get(&actor)
            .ok_or(CellPayloadError::Missing)?
            .clone();
        let base = record
            .persisted_revision()
            .map_or(ActorRevisionBase::Missing, ActorRevisionBase::Revision);
        let request = ActorSaveRequest::new(record.identity().clone(), record.revision(), base)
            .map_err(|_| EditorWorldDataError::UnexpectedCompletion)?;
        Ok(PendingActorSave {
            residency: self.resident_tokens[&CellKey::new(cell)],
            request,
            operation: self.ofpa.operation(actor)?,
            payload,
            cell_revision,
            lock: lock.clone(),
            root: provider.root().to_owned(),
        })
    }

    /// Reconciles an actual durable receipt. Stale edits or expired leases keep
    /// actor/cell dirty, but the observed durable base is retained for retry.
    /// Actor success never by itself marks the enclosing cell clean.
    pub fn commit_ofpa(
        &mut self,
        pending: &PendingActorSave,
        receipt: &ActorSaveReceipt,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        if receipt.request() != &pending.request
            || receipt.payload() != &*pending.payload
            || receipt.root() != pending.root
        {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        let actor = pending.request.identity().actor();
        self.ofpa
            .note_persisted(actor, pending.request.revision())?;
        let (_, revision) = self.actor_cell_revision(actor, &pending.lock, now)?;
        if self
            .resident_tokens
            .get(&CellKey::new(pending.request.identity().cell()))
            != Some(&pending.residency)
            || revision != pending.cell_revision
            || self.actor_payload(actor) != Some(&*pending.payload)
        {
            return Err(WorldError::StalePlan {
                expected: pending.cell_revision,
                actual: revision,
            }
            .into());
        }
        self.ofpa.commit(&pending.operation)?;
        Ok(())
    }

    /// Captures a clean actor before asynchronous disk loading. Dirty actor
    /// bytes must be saved or explicitly resolved before loading a replacement.
    pub fn prepare_actor_load(
        &self,
        actor: StableGuid,
        lock: &CellLockToken,
        now: u64,
        provider: &FileActorIoProvider,
    ) -> Result<PendingActorLoad, EditorWorldDataError> {
        let (cell, cell_revision) = self.actor_cell_revision(actor, lock, now)?;
        let record = self
            .ofpa
            .actor(actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        if record.is_dirty() {
            return Err(EditorWorldDataError::DirtyActors(cell));
        }
        Ok(PendingActorLoad {
            residency: self.resident_tokens[&CellKey::new(cell)],
            identity: record.identity().clone(),
            actor_revision: record.revision(),
            cell_revision,
            lock: lock.clone(),
            root: provider.root().to_owned(),
        })
    }

    /// Installs a validated snapshot only if cell, actor and lease still match
    /// the captured load. No mutation occurs when rejecting a stale result.
    pub fn apply_actor_load(
        &mut self,
        pending: &PendingActorLoad,
        snapshot: ActorSnapshot,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        if snapshot.identity() != &pending.identity || snapshot.root() != pending.root {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        let actor = pending.identity.actor();
        let (cell, revision) = self.actor_cell_revision(actor, &pending.lock, now)?;
        let record = self
            .ofpa
            .actor(actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        if self.resident_tokens.get(&CellKey::new(cell)) != Some(&pending.residency)
            || revision != pending.cell_revision
            || record.revision() != pending.actor_revision
            || record.is_dirty()
            || record
                .persisted_revision()
                .is_some_and(|base| snapshot.revision() < base)
        {
            return Err(WorldError::StalePlan {
                expected: pending.cell_revision,
                actual: revision,
            }
            .into());
        }
        self.check_actor_bytes(actor, cell, snapshot.bytes().len())?;
        self.ofpa.restore(actor, snapshot.revision())?;
        self.actor_payloads
            .insert(actor, snapshot.into_bytes().into());
        Ok(())
    }
}
