//! Captured local intent and all-or-nothing receipt reconciliation.
use super::*;
use crate::{
    ActorFileIdentity, ActorRecord, ActorRevisionBase, FileWorldTransactionProvider,
    WorldFileChange, WorldFileTarget, WorldTransactionReceipt, WorldTransactionRequest,
    WorldTransactionSnapshot,
};
use std::{collections::BTreeSet, path::PathBuf, sync::Arc};

/// Opaque local edits to publish together. Moves require explicit updated
/// payloads for both owner cells; the engine never edits membership schemas.
#[derive(Clone, Debug)]
pub enum WorldTransactionEdit {
    /// Replace one resident cell's opaque bytes.
    Cell {
        /// Owning coordinate.
        cell: CellCoord,
        /// New opaque bytes.
        payload: Arc<[u8]>,
    },
    /// Save a registered actor, optionally moving it to another resident cell.
    Actor {
        /// Registered actor GUID.
        actor: StableGuid,
        /// Destination owner.
        cell: CellCoord,
        /// New opaque bytes.
        payload: Arc<[u8]>,
    },
    /// Delete a registered, durably existing actor.
    DeleteActor(StableGuid),
}

#[derive(Clone, Debug)]
struct CellGuard {
    cell: CellCoord,
    revision: u64,
    residency: TransitionToken,
    lease: CellLockToken,
}

/// Immutable write intent with captured leases, residency and local revisions.
/// Preparation pins affected cells dirty. Failed/stale receipts never clear
/// them; explicit retry or discard-and-restore resolves uncertain outcomes.
#[derive(Clone, Debug)]
pub struct PendingWorldTransaction {
    request: WorldTransactionRequest,
    root: PathBuf,
    cells: Vec<CellGuard>,
    actors: Vec<(ActorRecord, Option<Arc<[u8]>>)>,
}
impl PendingWorldTransaction {
    /// Exact immutable request to submit to the transaction provider.
    pub fn request(&self) -> &WorldTransactionRequest {
        &self.request
    }
}

fn unexpected(_: crate::FileCellIoError) -> EditorWorldDataError {
    EditorWorldDataError::UnexpectedCompletion
}
fn base(snapshot: &WorldTransactionSnapshot, target: &WorldFileTarget) -> ActorRevisionBase {
    snapshot
        .files()
        .iter()
        .find(|f| f.target() == target)
        .map_or(ActorRevisionBase::Missing, |f| {
            ActorRevisionBase::Revision(f.revision())
        })
}

impl EditorWorldData {
    fn transaction_actors(
        &self,
        cells: &BTreeSet<CellKey>,
    ) -> Vec<(ActorRecord, Option<Arc<[u8]>>)> {
        self.ofpa
            .actors()
            .filter(|a| cells.contains(&CellKey::new(a.identity().cell())))
            .map(|a| {
                (
                    a.clone(),
                    self.actor_payloads.get(&a.identity().actor()).cloned(),
                )
            })
            .collect()
    }

    /// Builds real CAS changes from a durable snapshot and explicit opaque
    /// edits. Every touched owner must have a cell edit and a live lease.
    /// On failure nothing changes; on success cells advance once and stay
    /// dirty until the complete durable receipt passes every captured guard.
    pub fn prepare_world_transaction(
        &mut self,
        id: StableGuid,
        snapshot: &WorldTransactionSnapshot,
        edits: &[WorldTransactionEdit],
        leases: &[CellLockToken],
        now: u64,
        provider: &FileWorldTransactionProvider,
    ) -> Result<PendingWorldTransaction, EditorWorldDataError> {
        if snapshot.root() != provider.root() {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        match &self.transaction_root {
            Some((root, generation))
                if root == snapshot.root() && *generation == snapshot.generation() => {}
            None if snapshot.generation() == 0 => {}
            _ => return Err(EditorWorldDataError::UnexpectedCompletion),
        }
        let mut cells = BTreeMap::new();
        let mut actor_ids = BTreeSet::new();
        for edit in edits {
            if let WorldTransactionEdit::Cell { cell, payload } = edit {
                if payload.len() > self.limits.max_cell_bytes() {
                    return Err(CellPayloadError::TooLarge {
                        actual: payload.len(),
                        limit: self.limits.max_cell_bytes(),
                    }
                    .into());
                }
                if cells.insert(CellKey::new(*cell), payload.clone()).is_some() {
                    return Err(EditorWorldDataError::UnexpectedCompletion);
                }
            }
        }
        let mut guards = Vec::with_capacity(cells.len());
        let mut changes = Vec::new();
        for (key, payload) in &cells {
            let cell = key.coord();
            self.require_editable(cell)?;
            let lease = leases
                .iter()
                .find(|l| self.require_lock(cell, l, now).is_ok())
                .ok_or(EditorWorldDataError::UnexpectedCompletion)?;
            let record = self.model.cell(cell).ok_or(WorldError::UnknownCell(cell))?;
            let revision = record
                .revision()
                .checked_add(1)
                .ok_or(WorldError::RevisionExhausted(cell))?;
            let target = WorldFileTarget::Cell(record.file().clone());
            let old = base(snapshot, &target);
            if matches!(old,ActorRevisionBase::Revision(r) if r>=revision) {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
            guards.push(CellGuard {
                cell,
                revision,
                residency: *self
                    .resident_tokens
                    .get(key)
                    .ok_or(EditorWorldDataError::UnexpectedCompletion)?,
                lease: lease.clone(),
            });
            changes.push(
                WorldFileChange::put(target, old, revision, payload.clone()).map_err(unexpected)?,
            );
        }
        for edit in edits {
            let (actor, destination, payload) = match edit {
                WorldTransactionEdit::Cell { .. } => continue,
                WorldTransactionEdit::Actor {
                    actor,
                    cell,
                    payload,
                } => (*actor, Some(*cell), Some(payload.clone())),
                WorldTransactionEdit::DeleteActor(actor) => (*actor, None, None),
            };
            if !actor_ids.insert(actor) {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
            let record = self
                .ofpa
                .actor(actor)
                .ok_or(OfpaError::UnknownActor(actor))?;
            let source = record.identity().cell();
            if !cells.contains_key(&CellKey::new(source))
                || destination.is_some_and(|d| !cells.contains_key(&CellKey::new(d)))
            {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
            // A snapshot containing this GUID in another cell signals a stale
            // local catalog. Do not turn a stale move into an implicit overwrite.
            if snapshot.files().iter().any(|f|matches!(f.target(),WorldFileTarget::Actor(a) if a.actor()==actor&&a.cell()!=source)) {return Err(EditorWorldDataError::UnexpectedCompletion);}
            let source_target = WorldFileTarget::Actor(record.identity().clone());
            let old = base(snapshot, &source_target);
            if destination != Some(source) {
                let ActorRevisionBase::Revision(r) = old else {
                    return Err(EditorWorldDataError::UnexpectedCompletion);
                };
                changes
                    .push(WorldFileChange::delete(source_target.clone(), r).map_err(unexpected)?);
            }
            if let (Some(cell), Some(payload)) = (destination, payload) {
                let revision = record
                    .revision()
                    .checked_add(1)
                    .ok_or(EditorWorldDataError::UnexpectedCompletion)?;
                let target = WorldFileTarget::Actor(ActorFileIdentity::new(actor, cell)?);
                changes.push(
                    WorldFileChange::put(
                        target,
                        if cell == source {
                            old
                        } else {
                            ActorRevisionBase::Missing
                        },
                        revision,
                        payload,
                    )
                    .map_err(unexpected)?,
                );
            }
        }
        // A complete cell publication must not silently omit an existing
        // catalog actor or bless an uncommitted actor edit as durable.
        for record in self
            .ofpa
            .actors()
            .filter(|a| cells.contains_key(&CellKey::new(a.identity().cell())))
        {
            if actor_ids.contains(&record.identity().actor()) {
                continue;
            }
            let target = WorldFileTarget::Actor(record.identity().clone());
            let durable = snapshot
                .files()
                .iter()
                .find(|f| f.target() == &target)
                .ok_or(EditorWorldDataError::UnexpectedCompletion)?;
            if record.is_dirty()
                || record.revision() != durable.revision()
                || self
                    .actor_payload(record.identity().actor())
                    .is_some_and(|b| b != durable.bytes())
            {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
        }
        for file in snapshot.files() {
            if let WorldFileTarget::Actor(a) = file.target()
                && cells.contains_key(&CellKey::new(a.cell()))
                && self.ofpa.actor(a.actor()).is_none()
            {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
        }
        // Validate final actor payload totals before mutating local state.
        self.validate_transaction_actor_totals(&changes)?;
        let request =
            WorldTransactionRequest::new(id, snapshot.generation(), changes).map_err(unexpected)?;
        let keys = cells.keys().copied().collect();
        let actors = self.transaction_actors(&keys);
        for guard in &guards {
            self.model.mark_dirty(guard.cell)?;
        }
        self.transaction_root = Some((provider.root().to_owned(), snapshot.generation()));
        Ok(PendingWorldTransaction {
            request,
            root: provider.root().to_owned(),
            cells: guards,
            actors,
        })
    }

    fn validate_transaction_actor_totals(
        &self,
        changes: &[WorldFileChange],
    ) -> Result<(), EditorWorldDataError> {
        let mut sizes: BTreeMap<StableGuid, (CellKey, usize)> = self
            .ofpa
            .actors()
            .map(|a| {
                (
                    a.identity().actor(),
                    (
                        CellKey::new(a.identity().cell()),
                        self.actor_payloads
                            .get(&a.identity().actor())
                            .map_or(0, |b| b.len()),
                    ),
                )
            })
            .collect();
        // Apply deletions before puts so lexicographic move direction is irrelevant.
        for c in changes {
            if let WorldFileTarget::Actor(a) = c.target()
                && c.value().is_none()
            {
                sizes.remove(&a.actor());
            }
        }
        for c in changes {
            if let (WorldFileTarget::Actor(a), Some((_, b))) = (c.target(), c.value()) {
                sizes.insert(a.actor(), (CellKey::new(a.cell()), b.len()));
            }
        }
        let mut totals = BTreeMap::<CellKey, usize>::new();
        for (cell, size) in sizes.values() {
            let total = totals.entry(*cell).or_default();
            *total = total.saturating_add(*size);
            if *total > self.limits.max_cell_bytes() {
                return Err(CellPayloadError::TooLarge {
                    actual: *total,
                    limit: self.limits.max_cell_bytes(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Acknowledges all mutations together only after matching a provider-made
    /// durable receipt and every local guard. No partial clean/catalog update
    /// occurs on stale edits, changed residency, expired leases or wrong roots.
    pub fn commit_world_transaction(
        &mut self,
        pending: &PendingWorldTransaction,
        receipt: &WorldTransactionReceipt,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        if receipt.request() != &pending.request || receipt.root() != pending.root {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        if self.transaction_root.as_ref()
            != Some(&(pending.root.clone(), pending.request.base_generation()))
        {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        let keys = pending.cells.iter().map(|g| CellKey::new(g.cell)).collect();
        for guard in &pending.cells {
            self.require_lock(guard.cell, &guard.lease, now)?;
            self.require_editable(guard.cell)?;
            let record = self
                .model
                .cell(guard.cell)
                .ok_or(WorldError::UnknownCell(guard.cell))?;
            if record.revision() != guard.revision
                || self.resident_tokens.get(&CellKey::new(guard.cell)) != Some(&guard.residency)
            {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
        }
        if self.transaction_actors(&keys) != pending.actors {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        self.validate_transaction_actor_totals(pending.request.changes())?;
        self.install_transaction_changes(pending.request.changes());
        for guard in &pending.cells {
            // Untouched dirty actors are not covered by a cell-only receipt.
            if !self.ofpa.is_cell_dirty(guard.cell) {
                self.model.mark_clean(guard.cell)?;
            }
        }
        self.transaction_root = Some((pending.root.clone(), receipt.generation()));
        Ok(())
    }

    fn install_transaction_changes(&mut self, changes: &[WorldFileChange]) {
        for c in changes {
            if let WorldFileTarget::Actor(a) = c.target()
                && c.value().is_none()
            {
                self.ofpa.remove_transaction_actor(a.actor());
                self.actor_payloads.remove(&a.actor());
            }
        }
        for c in changes {
            match (c.target(), c.value()) {
                (WorldFileTarget::Cell(f), Some((r, bytes))) => {
                    let key = CellKey::new(f.coord());
                    self.payloads.insert(key, bytes.to_vec());
                    self.persisted_revisions.insert(key, r);
                    self.model
                        .cells
                        .get_mut(&key)
                        .expect("prevalidated resident cell")
                        .revision = r;
                }
                (WorldFileTarget::Actor(a), Some((r, bytes))) => {
                    self.ofpa.install_transaction_actor(a.clone(), r);
                    self.actor_payloads.insert(a.actor(), Arc::from(bytes));
                }
                _ => {}
            }
        }
    }

    /// Explicitly discards resident edits and restores a complete published
    /// snapshot after a stale/uncertain receipt. All registered cells must be
    /// loaded/editable and leased, with exact identity coverage; this never
    /// silently merges a moved durable actor into an old local catalog.
    pub fn discard_edits_and_restore_transaction_snapshot(
        &mut self,
        snapshot: &WorldTransactionSnapshot,
        leases: &[CellLockToken],
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        if self
            .transaction_root
            .as_ref()
            .is_some_and(|(root, generation)| {
                root != snapshot.root() || *generation > snapshot.generation()
            })
        {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        let cells: Vec<_> = self.model.cells.values().map(|c| c.coord()).collect();
        if cells.len()
            != snapshot
                .files()
                .iter()
                .filter(|f| matches!(f.target(), WorldFileTarget::Cell(_)))
                .count()
        {
            return Err(EditorWorldDataError::UnexpectedCompletion);
        }
        let mut changes = Vec::new();
        let mut actor_count = 0;
        for cell in &cells {
            self.require_editable(*cell)?;
            if !leases
                .iter()
                .any(|l| self.require_lock(*cell, l, now).is_ok())
            {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
            let target = WorldFileTarget::Cell(
                self.model
                    .cell(*cell)
                    .expect("registered cell")
                    .file()
                    .clone(),
            );
            if !snapshot.files().iter().any(|f| f.target() == &target) {
                return Err(EditorWorldDataError::UnexpectedCompletion);
            }
        }
        for f in snapshot.files() {
            if f.bytes().len() > self.limits.max_cell_bytes() {
                return Err(CellPayloadError::TooLarge {
                    actual: f.bytes().len(),
                    limit: self.limits.max_cell_bytes(),
                }
                .into());
            }
            if matches!(f.target(), WorldFileTarget::Actor(_)) {
                actor_count += 1;
            }
            changes.push(
                WorldFileChange::put(
                    f.target().clone(),
                    ActorRevisionBase::Missing,
                    f.revision(),
                    Arc::from(f.bytes()),
                )
                .map_err(unexpected)?,
            );
        }
        if actor_count > self.limits.max_actors() {
            return Err(OfpaError::CapacityExceeded {
                limit: self.limits.max_actors(),
            }
            .into());
        }
        let mut totals = BTreeMap::<CellKey, usize>::new();
        for f in snapshot.files() {
            if let WorldFileTarget::Actor(a) = f.target() {
                let total = totals.entry(CellKey::new(a.cell())).or_default();
                *total = total.saturating_add(f.bytes().len());
                if *total > self.limits.max_cell_bytes() {
                    return Err(CellPayloadError::TooLarge {
                        actual: *total,
                        limit: self.limits.max_cell_bytes(),
                    }
                    .into());
                }
            }
        }
        self.ofpa = OfpaCatalog::new(self.limits.max_actors());
        self.actor_payloads.clear();
        self.install_transaction_changes(&changes);
        for cell in cells {
            self.model.mark_clean(cell)?;
        }
        self.transaction_root = Some((snapshot.root().to_owned(), snapshot.generation()));
        Ok(())
    }
}
