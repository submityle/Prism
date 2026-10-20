//! Executable editor-world data facade.

use std::{collections::BTreeMap, fmt};

use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};

use super::{
    CellFileIdentity, CellKey, CellLockError, CellLockRegistry, CellLockToken, CellRecord,
    CellResidency, EditScope, EditorWorld, IoCompletion, OfpaCatalog, OfpaError, OfpaPlan,
    ResidencyDemand, ResidencyPlan, TransitionKind, TransitionToken, WorldError,
};
use super::{CellIoProvider, CellSaveRequest, HlodPlanner};

/// Bounds applied to editor-world metadata and resident payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditorWorldLimits {
    max_cells: usize,
    max_cell_bytes: usize,
    max_actors: usize,
    max_hlod_layers: usize,
    max_hlod_clusters: usize,
}

impl EditorWorldLimits {
    /// Creates non-zero bounded world limits.
    pub const fn new(
        max_cells: usize,
        max_cell_bytes: usize,
        max_actors: usize,
        max_hlod_layers: usize,
        max_hlod_clusters: usize,
    ) -> Result<Self, EditorWorldDataError> {
        if max_cells == 0
            || max_cell_bytes == 0
            || max_actors == 0
            || max_hlod_layers == 0
            || max_hlod_clusters == 0
        {
            return Err(EditorWorldDataError::InvalidLimits);
        }
        Ok(Self {
            max_cells,
            max_cell_bytes,
            max_actors,
            max_hlod_layers,
            max_hlod_clusters,
        })
    }

    /// A practical bounded editor default.
    #[must_use]
    pub const fn editor_default() -> Self {
        Self {
            max_cells: 1_000_000,
            max_cell_bytes: 64 * 1024 * 1024,
            max_actors: 10_000_000,
            max_hlod_layers: 16,
            max_hlod_clusters: 1_000_000,
        }
    }

    /// Maximum registered cells.
    #[must_use]
    pub const fn max_cells(self) -> usize {
        self.max_cells
    }

    /// Maximum resident payload bytes per cell.
    #[must_use]
    pub const fn max_cell_bytes(self) -> usize {
        self.max_cell_bytes
    }

    /// Maximum OFPA actor records.
    #[must_use]
    pub const fn max_actors(self) -> usize {
        self.max_actors
    }

    /// Maximum HLOD layers.
    #[must_use]
    pub const fn max_hlod_layers(self) -> usize {
        self.max_hlod_layers
    }

    /// Maximum HLOD clusters.
    #[must_use]
    pub const fn max_hlod_clusters(self) -> usize {
        self.max_hlod_clusters
    }
}

/// Payload validation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CellPayloadError {
    /// A payload is larger than the configured per-cell bound.
    TooLarge {
        /// Actual payload size.
        actual: usize,
        /// Configured payload limit.
        limit: usize,
    },
    /// A loaded cell has no payload available for saving.
    Missing,
}

impl fmt::Display for CellPayloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { actual, limit } => {
                write!(formatter, "cell payload {actual} exceeds limit {limit}")
            }
            Self::Missing => formatter.write_str("loaded cell has no resident payload"),
        }
    }
}

impl std::error::Error for CellPayloadError {}

/// Errors returned by the executable editor-world facade.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditorWorldDataError {
    /// Limits would allow an unbounded or unusable model.
    InvalidLimits,
    /// The cell catalog reached its configured capacity.
    CellCapacityExceeded,
    /// The underlying deterministic model rejected an operation.
    Model(WorldError),
    /// Model validation rejected an operation before mutation.
    Validation(super::ValidationError),
    /// A lock token was invalid or expired.
    Lock(CellLockError),
    /// Resident payload validation failed.
    Payload(CellPayloadError),
    /// OFPA metadata rejected an operation.
    Ofpa(OfpaError),
    /// A cell save cannot clear dirtiness while actor writes remain pending.
    DirtyActors(CellCoord),
    /// A terminal I/O event does not match the operation being reconciled.
    UnexpectedCompletion,
}

impl fmt::Display for EditorWorldDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("editor-world limits must be non-zero"),
            Self::CellCapacityExceeded => {
                formatter.write_str("editor-world cell capacity exceeded")
            }
            Self::Model(error) => error.fmt(formatter),
            Self::Validation(error) => error.fmt(formatter),
            Self::Lock(error) => error.fmt(formatter),
            Self::Payload(error) => error.fmt(formatter),
            Self::Ofpa(error) => error.fmt(formatter),
            Self::DirtyActors(cell) => write!(
                formatter,
                "cell ({},{},{}) still has dirty actor writes",
                cell.x, cell.y, cell.z
            ),
            Self::UnexpectedCompletion => {
                formatter.write_str("I/O completion does not match request")
            }
        }
    }
}

impl std::error::Error for EditorWorldDataError {}

impl From<WorldError> for EditorWorldDataError {
    fn from(error: WorldError) -> Self {
        Self::Model(error)
    }
}

impl From<super::ValidationError> for EditorWorldDataError {
    fn from(error: super::ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl From<CellLockError> for EditorWorldDataError {
    fn from(error: CellLockError) -> Self {
        Self::Lock(error)
    }
}

impl From<CellPayloadError> for EditorWorldDataError {
    fn from(error: CellPayloadError) -> Self {
        Self::Payload(error)
    }
}

impl From<OfpaError> for EditorWorldDataError {
    fn from(error: OfpaError) -> Self {
        Self::Ofpa(error)
    }
}

/// Failure to validate or enqueue a cell load.
#[derive(Debug)]
pub enum LoadSubmitError<E> {
    /// World validation or recovery failed.
    World(EditorWorldDataError),
    /// The driver rejected the request. New submissions roll back their load
    /// transition; duplicate submissions leave the already accepted load intact.
    Io(super::IoSubmitError<E>),
}

impl<E: fmt::Display> fmt::Display for LoadSubmitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::World(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for LoadSubmitError<E> {}

/// A save request retained with the lock that authorized the snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingCellSave {
    request: CellSaveRequest,
    payload: Vec<u8>,
    lock: CellLockToken,
}

impl PendingCellSave {
    /// Provider request metadata.
    #[must_use]
    pub const fn request(&self) -> &CellSaveRequest {
        &self.request
    }

    /// Opaque payload snapshot to pass to a provider.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Lock that authorized the snapshot.
    #[must_use]
    pub const fn lock(&self) -> &CellLockToken {
        &self.lock
    }
}

/// Visibility projection for one editor frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisibilityPlan {
    visible: Vec<CellCoord>,
    hidden: Vec<CellCoord>,
    editable: Vec<CellCoord>,
}

impl VisibilityPlan {
    /// Cells visible under the active data-layer policy.
    #[must_use]
    pub fn visible(&self) -> &[CellCoord] {
        &self.visible
    }

    /// Cells hidden by at least one assigned data layer.
    #[must_use]
    pub fn hidden(&self) -> &[CellCoord] {
        &self.hidden
    }

    /// Loaded cells eligible under scope and layer policy.
    /// Mutating them additionally requires a live edit lease.
    #[must_use]
    pub fn editable(&self) -> &[CellCoord] {
        &self.editable
    }
}

/// Complete executable editor-world data layer.
pub struct EditorWorldData {
    model: EditorWorld,
    limits: EditorWorldLimits,
    locks: CellLockRegistry,
    payloads: BTreeMap<CellKey, Vec<u8>>,
    ofpa: OfpaCatalog,
    hlod: HlodPlanner,
}

impl EditorWorldData {
    /// Creates an empty world with explicit bounds.
    pub fn new(
        identity: StableGuid,
        limits: EditorWorldLimits,
    ) -> Result<Self, EditorWorldDataError> {
        Ok(Self {
            model: EditorWorld::new(identity)?,
            limits,
            locks: CellLockRegistry::new(),
            payloads: BTreeMap::new(),
            ofpa: OfpaCatalog::new(limits.max_actors()),
            hlod: HlodPlanner::new(limits.max_hlod_layers(), limits.max_hlod_clusters()),
        })
    }

    /// Returns the immutable deterministic model.
    #[must_use]
    pub const fn model(&self) -> &EditorWorld {
        &self.model
    }

    /// Returns configured bounds.
    #[must_use]
    pub const fn limits(&self) -> EditorWorldLimits {
        self.limits
    }

    /// Adds a data layer.
    pub fn add_layer(&mut self, layer: super::DataLayer) -> Result<(), EditorWorldDataError> {
        self.model.add_layer(layer)?;
        Ok(())
    }

    /// Reparents a data layer after validating its ancestor chain.
    pub fn set_layer_parent(
        &mut self,
        id: DataLayerId,
        parent: Option<DataLayerId>,
    ) -> Result<(), EditorWorldDataError> {
        self.model.set_layer_parent(id, parent)?;
        Ok(())
    }

    /// Sets data-layer visibility without changing its editability.
    pub fn set_layer_visible(
        &mut self,
        id: DataLayerId,
        visible: bool,
    ) -> Result<(), EditorWorldDataError> {
        self.model.set_layer_visible(id, visible)?;
        Ok(())
    }

    /// Sets data-layer editability without changing its visibility.
    pub fn set_layer_editable(
        &mut self,
        id: DataLayerId,
        editable: bool,
    ) -> Result<(), EditorWorldDataError> {
        self.model.set_layer_editable(id, editable)?;
        Ok(())
    }

    /// Registers a cell before any I/O transition can reference it.
    pub fn add_cell(
        &mut self,
        coord: CellCoord,
        layers: impl IntoIterator<Item = DataLayerId>,
    ) -> Result<&CellRecord, EditorWorldDataError> {
        if self.model.cell_count() >= self.limits.max_cells() {
            return Err(EditorWorldDataError::CellCapacityExceeded);
        }
        Ok(self.model.add_cell(coord, layers)?)
    }

    /// Replaces a cell's direct data-layer memberships atomically.
    pub fn set_cell_layers(
        &mut self,
        coord: CellCoord,
        layers: impl IntoIterator<Item = DataLayerId>,
    ) -> Result<(), EditorWorldDataError> {
        self.model.set_cell_layers(coord, layers)?;
        Ok(())
    }

    /// Replaces the active edit scope.
    pub fn set_edit_scope(&mut self, scope: EditScope) {
        self.model.set_edit_scope(scope);
    }

    /// Acquires a cell edit lock.
    pub fn acquire_lock(
        &mut self,
        cell: CellCoord,
        owner: StableGuid,
        now: u64,
        ttl: u64,
    ) -> Result<CellLockToken, EditorWorldDataError> {
        let record = self.model.cell(cell).ok_or(WorldError::UnknownCell(cell))?;
        if matches!(record.residency(), CellResidency::Unloading(_)) {
            return Err(WorldError::InvalidTransition(cell).into());
        }
        Ok(self.locks.acquire(cell, owner, now, ttl)?)
    }

    /// Renews a cell edit lock.
    pub fn renew_lock(
        &mut self,
        token: &CellLockToken,
        now: u64,
        ttl: u64,
    ) -> Result<(), EditorWorldDataError> {
        Ok(self.locks.renew(token, now, ttl)?)
    }

    /// Releases a cell edit lock.
    pub fn release_lock(&mut self, token: &CellLockToken) -> Result<(), EditorWorldDataError> {
        self.locks.release(token)?;
        Ok(())
    }

    /// Marks a loaded cell dirty when its lock is current.
    pub fn mark_dirty(
        &mut self,
        cell: CellCoord,
        lock: &CellLockToken,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        self.require_lock(cell, lock, now)?;
        self.require_editable(cell)?;
        self.model.mark_dirty(cell)?;
        Ok(())
    }

    /// Replaces resident opaque bytes and marks the cell dirty atomically.
    pub fn replace_payload(
        &mut self,
        cell: CellCoord,
        lock: &CellLockToken,
        now: u64,
        payload: Vec<u8>,
    ) -> Result<(), EditorWorldDataError> {
        self.require_lock(cell, lock, now)?;
        self.require_editable(cell)?;
        if payload.len() > self.limits.max_cell_bytes() {
            return Err(CellPayloadError::TooLarge {
                actual: payload.len(),
                limit: self.limits.max_cell_bytes(),
            }
            .into());
        }
        if self
            .model
            .cell(cell)
            .is_none_or(|record| record.residency() != CellResidency::Loaded)
        {
            return Err(WorldError::InvalidTransition(cell).into());
        }
        self.model.mark_dirty(cell)?;
        self.payloads.insert(CellKey::new(cell), payload);
        Ok(())
    }

    /// Returns resident opaque bytes, if the cell is loaded.
    #[must_use]
    pub fn payload(&self, cell: CellCoord) -> Option<&[u8]> {
        self.payloads.get(&CellKey::new(cell)).map(Vec::as_slice)
    }

    /// Plans deterministic residency, retaining dirty and live-locked cells.
    pub fn plan_residency(
        &self,
        demand: &ResidencyDemand,
        now: u64,
    ) -> Result<ResidencyPlan, EditorWorldDataError> {
        let mut plan = self.model.plan(demand)?;
        plan.operations.retain(|operation| {
            if matches!(operation, super::PlanOperation::Unload { .. })
                && self.locks.is_locked(operation.cell(), now)
            {
                plan.blocked
                    .push(super::PlanBlock::LockedCell(operation.cell()));
                false
            } else {
                true
            }
        });
        plan.blocked.sort_by_key(|block| match block {
            super::PlanBlock::DirtyCell(cell) | super::PlanBlock::LockedCell(cell) => {
                CellKey::new(*cell)
            }
        });
        Ok(plan)
    }

    /// Plans and starts safe residency operations using a caller-supplied clock.
    pub fn begin_residency(
        &mut self,
        demand: &ResidencyDemand,
        now: u64,
    ) -> Result<(ResidencyPlan, Vec<TransitionToken>), EditorWorldDataError> {
        let plan = self.plan_residency(demand, now)?;
        let tokens = self.model.begin_plan(&plan)?;
        Ok((plan, tokens))
    }

    /// Completes a load and installs its opaque payload.
    pub fn complete_load(
        &mut self,
        token: TransitionToken,
        payload: Vec<u8>,
    ) -> Result<(), EditorWorldDataError> {
        if token.kind() != TransitionKind::Load {
            return Err(WorldError::InvalidTransition(token.cell()).into());
        }
        if payload.len() > self.limits.max_cell_bytes() {
            self.model.fail_load(token)?;
            self.payloads.remove(&CellKey::new(token.cell()));
            return Err(CellPayloadError::TooLarge {
                actual: payload.len(),
                limit: self.limits.max_cell_bytes(),
            }
            .into());
        }
        self.model.complete_load(token)?;
        self.payloads.insert(CellKey::new(token.cell()), payload);
        Ok(())
    }

    /// Rolls a failed load back to unloaded without retaining bytes.
    pub fn fail_load(&mut self, token: TransitionToken) -> Result<(), EditorWorldDataError> {
        self.model.fail_load(token)?;
        self.payloads.remove(&CellKey::new(token.cell()));
        Ok(())
    }

    /// Completes a clean unload and drops its resident payload.
    pub fn complete_unload(&mut self, token: TransitionToken) -> Result<(), EditorWorldDataError> {
        self.model.complete_unload(token)?;
        self.payloads.remove(&CellKey::new(token.cell()));
        Ok(())
    }

    /// Rolls a failed unload back to loaded and retains its payload.
    pub fn fail_unload(&mut self, token: TransitionToken) -> Result<(), EditorWorldDataError> {
        self.model.fail_unload(token)?;
        Ok(())
    }

    /// Captures a lock-protected dirty payload for an asynchronous save.
    pub fn prepare_save(
        &self,
        cell: CellCoord,
        lock: &CellLockToken,
        now: u64,
    ) -> Result<PendingCellSave, EditorWorldDataError> {
        self.require_lock(cell, lock, now)?;
        let record = self.model.cell(cell).ok_or(WorldError::UnknownCell(cell))?;
        if record.residency() != CellResidency::Loaded || !record.is_dirty() {
            return Err(WorldError::InvalidTransition(cell).into());
        }
        let payload = self
            .payload(cell)
            .ok_or(CellPayloadError::Missing)?
            .to_vec();
        Ok(PendingCellSave {
            request: CellSaveRequest::new(cell, record.file().clone(), record.revision()),
            payload,
            lock: lock.clone(),
        })
    }

    /// Commits a successful save only while its authorizing lock remains live.
    pub fn commit_save(
        &mut self,
        pending: &PendingCellSave,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        self.require_lock(pending.request.cell(), &pending.lock, now)?;
        let record = self
            .model
            .cell(pending.request.cell())
            .ok_or(WorldError::UnknownCell(pending.request.cell()))?;
        if record.revision() != pending.request.revision()
            || self.payload(pending.request.cell()) != Some(pending.payload())
        {
            return Err(WorldError::StalePlan {
                expected: pending.request.revision(),
                actual: record.revision(),
            }
            .into());
        }
        if self.ofpa.is_cell_dirty(pending.request.cell()) {
            return Err(EditorWorldDataError::DirtyActors(pending.request.cell()));
        }
        self.model.mark_clean(pending.request.cell())?;
        Ok(())
    }

    /// Registers one OFPA actor under an existing cell.
    pub fn register_actor(
        &mut self,
        actor: StableGuid,
        cell: CellCoord,
    ) -> Result<&super::ActorRecord, EditorWorldDataError> {
        if self.model.cell(cell).is_none() {
            return Err(WorldError::UnknownCell(cell).into());
        }
        Ok(self.ofpa.register(actor, cell)?)
    }

    /// Marks an actor dirty under its cell lock.
    pub fn mark_actor_dirty(
        &mut self,
        actor: StableGuid,
        lock: &CellLockToken,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        let record = self
            .ofpa
            .actor(actor)
            .ok_or(OfpaError::UnknownActor(actor))?;
        let cell = record.identity().cell();
        self.require_lock(cell, lock, now)?;
        self.require_editable(cell)?;
        if record.revision() == u64::MAX {
            return Err(OfpaError::StaleRevision {
                expected: u64::MAX,
                actual: u64::MAX,
            }
            .into());
        }
        self.model.mark_dirty(cell)?;
        self.ofpa.mark_dirty(actor)?;
        Ok(())
    }

    /// Returns and clears no state while building the deterministic OFPA plan.
    #[must_use]
    pub fn plan_ofpa(&self) -> OfpaPlan {
        self.ofpa.plan()
    }

    /// Commits an OFPA actor write.
    pub fn commit_ofpa(
        &mut self,
        operation: &super::OfpaOperation,
        lock: &CellLockToken,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        self.require_lock(operation.identity().cell(), lock, now)?;
        self.ofpa.commit(operation)?;
        Ok(())
    }

    /// Returns a mutable HLOD declaration/planner boundary.
    pub fn hlod_mut(&mut self) -> &mut HlodPlanner {
        &mut self.hlod
    }

    /// Computes visible, hidden, and editable cell sets in coordinate order.
    #[must_use]
    pub fn visibility_plan(&self) -> VisibilityPlan {
        let mut visible = Vec::new();
        let mut hidden = Vec::new();
        let mut editable = Vec::new();
        for record in self.model.cells() {
            if self.model.is_cell_visible(record.coord()) {
                visible.push(record.coord());
            } else {
                hidden.push(record.coord());
            }
            if self.model.is_cell_editable(record.coord()) {
                editable.push(record.coord());
            }
        }
        VisibilityPlan {
            visible,
            hidden,
            editable,
        }
    }

    /// Applies a load completion from the bounded I/O driver.
    pub fn apply_load_completion<E>(
        &mut self,
        completion: IoCompletion<E>,
    ) -> Result<bool, EditorWorldDataError> {
        match completion {
            IoCompletion::Loaded { request, bytes, .. } => {
                if self.cell_file(request.cell()) != Some(request.file()) {
                    return Err(EditorWorldDataError::UnexpectedCompletion);
                }
                self.complete_load(request.token(), bytes)?;
                Ok(true)
            }
            IoCompletion::Failed {
                operation: super::IoOperation::Load(request),
                ..
            }
            | IoCompletion::Cancelled {
                operation: super::IoOperation::Load(request),
                ..
            } => {
                if self.cell_file(request.cell()) != Some(request.file()) {
                    return Err(EditorWorldDataError::UnexpectedCompletion);
                }
                self.fail_load(request.token())?;
                Ok(false)
            }
            IoCompletion::Saved { .. }
            | IoCompletion::Failed { .. }
            | IoCompletion::Cancelled { .. } => Err(EditorWorldDataError::UnexpectedCompletion),
        }
    }

    /// Applies a save completion. A failed or cancelled save preserves dirty
    /// state and returns `false`; only a successful save clears it.
    pub fn apply_save_completion<E>(
        &mut self,
        pending: &PendingCellSave,
        completion: IoCompletion<E>,
        now: u64,
    ) -> Result<bool, EditorWorldDataError> {
        match completion {
            IoCompletion::Saved { request, .. } if request == pending.request => {
                self.commit_save(pending, now)?;
                Ok(true)
            }
            IoCompletion::Failed {
                operation: super::IoOperation::Save(request),
                ..
            }
            | IoCompletion::Cancelled {
                operation: super::IoOperation::Save(request),
                ..
            } if request == pending.request => Ok(false),
            _ => Err(EditorWorldDataError::UnexpectedCompletion),
        }
    }

    fn require_editable(&self, cell: CellCoord) -> Result<(), EditorWorldDataError> {
        if !self.model.is_cell_editable(cell) {
            return Err(WorldError::CellNotEditable(cell).into());
        }
        Ok(())
    }

    fn require_lock(
        &self,
        cell: CellCoord,
        token: &CellLockToken,
        now: u64,
    ) -> Result<(), EditorWorldDataError> {
        if token.cell() != cell {
            return Err(CellLockError::InvalidToken.into());
        }
        if !self.locks.is_current(token, now) {
            return Err(CellLockError::Expired.into());
        }
        Ok(())
    }
}

impl EditorWorldData {
    /// Returns a cell's stable file identity.
    #[must_use]
    pub fn cell_file(&self, cell: CellCoord) -> Option<&CellFileIdentity> {
        self.model.cell(cell).map(CellRecord::file)
    }
}

impl EditorWorldData {
    /// Submits a live load transition and rolls it back on admission/startup failure.
    /// Repeating an accepted request returns `CellBusy` without invalidating it.
    pub fn submit_load<P: CellIoProvider>(
        &mut self,
        driver: &mut super::WorldIoDriver<P>,
        token: TransitionToken,
    ) -> Result<super::IoRequestId, LoadSubmitError<P::Error>> {
        let request = self.load_request(token).map_err(LoadSubmitError::World)?;
        if driver.contains_load(&request) {
            return Err(LoadSubmitError::Io(super::IoSubmitError::CellBusy(
                token.cell(),
            )));
        }
        match driver.submit_load(request) {
            Ok(id) => Ok(id),
            Err(error) => {
                self.fail_load(token).map_err(LoadSubmitError::World)?;
                Err(LoadSubmitError::Io(error))
            }
        }
    }

    /// Builds a load request for a token using the configured payload bound.
    pub fn load_request(
        &self,
        token: TransitionToken,
    ) -> Result<super::CellLoadRequest, EditorWorldDataError> {
        if token.kind() != TransitionKind::Load
            || self
                .model
                .cell(token.cell())
                .is_none_or(|record| record.residency() != CellResidency::Loading(token.id()))
        {
            return Err(WorldError::StaleToken(token.id()).into());
        }
        let file = self
            .cell_file(token.cell())
            .ok_or(WorldError::UnknownCell(token.cell()))?
            .clone();
        Ok(super::CellLoadRequest::new(
            token,
            file,
            self.limits.max_cell_bytes(),
        ))
    }

    /// Builds a provider driver without moving the world model.
    #[must_use]
    pub fn make_io_driver<P: CellIoProvider>(
        &self,
        provider: P,
        budget: super::IoBudget,
    ) -> super::WorldIoDriver<P> {
        super::WorldIoDriver::new(provider, budget)
    }
}
