//! Deterministic editor-side world partition state.
//!
//! This crate owns authoring policy, not runtime world mutation. It executes
//! cells, data-layer visibility/editability, edit scopes, stable per-cell
//! identity, and failure-safe residency transitions. A plan can be adapted to
//! [`prism_ecs`] / [`prism_scene`] partition contracts, but this crate never
//! creates entities or assumes a renderer exists. Opaque asynchronous storage
//! is supplied through [`CellIoProvider`], including [`FileCellIoProvider`],
//! and bounded OFPA actor storage through [`FileActorIoProvider`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Path, PathBuf},
};

use prism_asset::StableGuid;
use prism_ecs::partition::{
    cell::{CellCoord, StreamingDelta},
    data_layer::DataLayerId,
};

mod filesystem;
mod hlod;
mod io;
mod locks;
mod ofpa;
mod world;

pub use filesystem::{
    ActorIoFuture, ActorIoLimits, ActorRevisionBase, ActorSaveReceipt, ActorSaveRequest,
    ActorSnapshot, FileActorIoProvider, FileCellIoError, FileCellIoLimits, FileCellIoProvider,
};
pub use hlod::{
    HlodCluster, HlodError, HlodLayer, HlodPlan, HlodPlanOperation, HlodPlanner, HlodView,
};
pub use io::{
    CellIoProvider, CellLoadFuture, CellLoadRequest, CellSaveFuture, CellSaveRequest, CellSnapshot,
    CellSnapshotFuture, IoBudget, IoCompletion, IoFailure, IoOperation, IoRequestId, IoSubmitError,
    WorldIoDriver,
};
pub use locks::{CellLockError, CellLockRecord, CellLockRegistry, CellLockToken};
pub use ofpa::{ActorFileIdentity, ActorRecord, OfpaCatalog, OfpaError, OfpaOperation, OfpaPlan};
pub use world::{
    CellPayloadError, EditorWorldData, EditorWorldDataError, EditorWorldLimits, LoadSubmitError,
    PendingActorLoad, PendingActorSave, PendingCellSave, VisibilityPlan,
};

/// A total-order wrapper around the ECS cell coordinate.
///
/// `prism_ecs::CellCoord` intentionally only provides equality and hashing.
/// The editor needs a stable order for maps, plans, diagnostics, and file
/// manifests, so it uses lexicographic `(x, y, z)` ordering here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CellKey(CellCoord);

impl Ord for CellKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.x, self.0.y, self.0.z).cmp(&(other.0.x, other.0.y, other.0.z))
    }
}

impl PartialOrd for CellKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl CellKey {
    /// Creates an ordered key for `coord`.
    #[must_use]
    pub const fn new(coord: CellCoord) -> Self {
        Self(coord)
    }

    /// Returns the underlying partition coordinate.
    #[must_use]
    pub const fn coord(self) -> CellCoord {
        self.0
    }
}

/// A stable identity and path for one world-partition cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellFileIdentity {
    coord: CellCoord,
    stable_id: StableGuid,
    relative_path: String,
}

impl CellFileIdentity {
    fn derive(world: StableGuid, coord: CellCoord) -> Self {
        let label = format!("cell/{}/{}/{}", coord.x, coord.y, coord.z);
        let stable_id = StableGuid::derive_sub(world, &label);
        let relative_path = format!("cells/{}_{}_{}.prismcell", coord.x, coord.y, coord.z);
        Self {
            coord,
            stable_id,
            relative_path,
        }
    }

    /// Stable content-independent identity for this cell file.
    #[must_use]
    pub const fn stable_id(&self) -> StableGuid {
        self.stable_id
    }

    /// Stable path relative to the owning world's cell directory.
    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// Partition coordinate encoded by this identity.
    #[must_use]
    pub const fn coord(&self) -> CellCoord {
        self.coord
    }

    /// Resolves the identity through the same filename contract as
    /// `prism_scene::FileCellStore`.
    #[must_use]
    pub fn scene_store_path(&self, root: impl AsRef<Path>) -> PathBuf {
        prism_scene::FileCellStore::new(root.as_ref()).cell_path(self.coord)
    }
}

/// Stable editor residency state for one cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CellResidency {
    /// No editor payload is resident.
    Unloaded,
    /// A load request has been accepted and is awaiting completion.
    Loading(TransitionId),
    /// The cell payload is resident and may be edited if policy permits.
    Loaded,
    /// An unload request has been accepted and is awaiting completion.
    Unloading(TransitionId),
}

impl CellResidency {
    /// Whether the cell has a resident editor payload.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        matches!(self, Self::Loaded | Self::Unloading(_))
    }

    /// Whether the cell is waiting for an asynchronous operation.
    #[must_use]
    pub const fn is_pending(self) -> bool {
        matches!(self, Self::Loading(_) | Self::Unloading(_))
    }
}

/// Authoring metadata for one cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellRecord {
    key: CellKey,
    file: CellFileIdentity,
    layers: BTreeSet<DataLayerId>,
    residency: CellResidency,
    dirty: bool,
    revision: u64,
}

impl CellRecord {
    /// The ordered cell key.
    #[must_use]
    pub const fn key(&self) -> CellKey {
        self.key
    }

    /// The underlying partition coordinate.
    #[must_use]
    pub const fn coord(&self) -> CellCoord {
        self.key.coord()
    }

    /// Stable file identity.
    #[must_use]
    pub const fn file(&self) -> &CellFileIdentity {
        &self.file
    }

    /// Data layers assigned to this cell in deterministic order.
    pub fn layers(&self) -> impl Iterator<Item = DataLayerId> + '_ {
        self.layers.iter().copied()
    }

    /// Current residency state.
    #[must_use]
    pub const fn residency(&self) -> CellResidency {
        self.residency
    }

    /// Whether this cell has unsaved authoring changes.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Monotonic authoring revision, advanced even for byte-identical edits.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}

/// Authoring visibility/editability policy for one data layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataLayer {
    id: DataLayerId,
    name: String,
    parent: Option<DataLayerId>,
    visible: bool,
    editable: bool,
}

impl DataLayer {
    /// Creates a visible and editable data layer.
    pub fn new(id: DataLayerId, name: impl Into<String>) -> Result<Self, ValidationError> {
        let name = name.into();
        validate_layer_name(&name)?;
        Ok(Self {
            id,
            name,
            parent: None,
            visible: true,
            editable: true,
        })
    }

    /// Creates a visible and editable child data layer.
    ///
    /// Parent existence and cycle safety are validated when the layer is
    /// registered in an [`EditorWorld`].
    pub fn with_parent(
        id: DataLayerId,
        name: impl Into<String>,
        parent: DataLayerId,
    ) -> Result<Self, ValidationError> {
        let mut layer = Self::new(id, name)?;
        layer.parent = Some(parent);
        Ok(layer)
    }

    /// Layer identity.
    #[must_use]
    pub const fn id(&self) -> DataLayerId {
        self.id
    }

    /// Stable authoring name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Optional parent layer whose policy also applies to this layer.
    #[must_use]
    pub const fn parent(&self) -> Option<DataLayerId> {
        self.parent
    }

    /// Whether the layer contributes visible content in the editor.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Whether the layer accepts edits under an allowed edit scope.
    #[must_use]
    pub const fn is_editable(&self) -> bool {
        self.editable
    }
}

/// An edit scope that constrains both cells and data layers.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EditScope {
    cells: Option<BTreeSet<CellKey>>,
    layers: Option<BTreeSet<DataLayerId>>,
}

impl EditScope {
    /// Creates an unrestricted scope.
    #[must_use]
    pub fn unrestricted() -> Self {
        Self::default()
    }

    /// Creates a scope limited to the supplied cells.
    #[must_use]
    pub fn for_cells(cells: impl IntoIterator<Item = CellCoord>) -> Self {
        Self {
            cells: Some(cells.into_iter().map(CellKey::new).collect()),
            layers: None,
        }
    }

    /// Adds a cell restriction, preserving deterministic ordering.
    pub fn allow_cell(&mut self, cell: CellCoord) {
        self.cells
            .get_or_insert_with(BTreeSet::new)
            .insert(CellKey::new(cell));
    }

    /// Adds a data-layer restriction.
    pub fn allow_layer(&mut self, layer: DataLayerId) {
        self.layers.get_or_insert_with(BTreeSet::new).insert(layer);
    }

    fn permits(&self, cell: CellKey, layers: &BTreeSet<DataLayerId>) -> bool {
        let cell_allowed = self
            .cells
            .as_ref()
            .is_none_or(|allowed| allowed.contains(&cell));
        let layers_allowed = self
            .layers
            .as_ref()
            .is_none_or(|allowed| layers.iter().all(|layer| allowed.contains(layer)));
        cell_allowed && layers_allowed
    }
}

/// A desired editor residency set.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResidencyDemand {
    desired: BTreeSet<CellKey>,
}

impl ResidencyDemand {
    /// Creates an empty demand.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a cell to the desired resident set.
    pub fn require(&mut self, cell: CellCoord) -> bool {
        self.desired.insert(CellKey::new(cell))
    }

    /// Removes a cell from the desired resident set.
    pub fn release(&mut self, cell: CellCoord) -> bool {
        self.desired.remove(&CellKey::new(cell))
    }

    /// Returns desired cells in deterministic order.
    pub fn cells(&self) -> impl Iterator<Item = CellCoord> + '_ {
        self.desired.iter().map(|key| key.coord())
    }

    fn contains(&self, cell: CellKey) -> bool {
        self.desired.contains(&cell)
    }
}

/// A planned editor residency operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanOperation {
    /// Read a cell payload and make it available to the editor.
    Load {
        /// Cell to load.
        cell: CellCoord,
        /// Stable file identity to read.
        file: CellFileIdentity,
    },
    /// Save or otherwise commit a clean cell before removing it from memory.
    Unload {
        /// Cell to unload.
        cell: CellCoord,
        /// Stable file identity to commit.
        file: CellFileIdentity,
    },
}

impl PlanOperation {
    /// Returns the operation's cell.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        match self {
            Self::Load { cell, .. } | Self::Unload { cell, .. } => *cell,
        }
    }

    /// Returns the operation's stable file identity.
    #[must_use]
    pub const fn file(&self) -> &CellFileIdentity {
        match self {
            Self::Load { file, .. } | Self::Unload { file, .. } => file,
        }
    }
}

/// A planned operation that could not be emitted safely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanBlock {
    /// A cell cannot be unloaded until its changes are committed.
    DirtyCell(CellCoord),
    /// A live edit lease protects this cell from eviction.
    LockedCell(CellCoord),
}

/// Deterministic result of reconciling current residency with a demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyPlan {
    epoch: u64,
    operations: Vec<PlanOperation>,
    blocked: Vec<PlanBlock>,
}

impl ResidencyPlan {
    /// Model revision against which this plan was produced.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Operations sorted by cell key, with unloads before loads for equal keys.
    #[must_use]
    pub fn operations(&self) -> &[PlanOperation] {
        &self.operations
    }

    /// Deterministic blockers encountered while planning.
    #[must_use]
    pub fn blocked(&self) -> &[PlanBlock] {
        &self.blocked
    }

    /// Converts the plan's operations to the scene/ECS streaming contract.
    #[must_use]
    pub fn as_streaming_delta(&self) -> StreamingDelta {
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();
        for operation in &self.operations {
            match operation {
                PlanOperation::Load { cell, .. } => to_load.push(*cell),
                PlanOperation::Unload { cell, .. } => to_unload.push(*cell),
            }
        }
        StreamingDelta { to_load, to_unload }
    }
}

/// Operation kind carried by a transition token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionKind {
    /// A load is in flight.
    Load,
    /// An unload is in flight.
    Unload,
}

/// Monotonic token used to commit or fail one residency transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionToken {
    id: TransitionId,
    cell: CellKey,
    kind: TransitionKind,
}

impl TransitionToken {
    /// Token identity.
    #[must_use]
    pub const fn id(self) -> TransitionId {
        self.id
    }

    /// Cell associated with the token.
    #[must_use]
    pub const fn cell(self) -> CellCoord {
        self.cell.coord()
    }

    /// Operation associated with the token.
    #[must_use]
    pub const fn kind(self) -> TransitionKind {
        self.kind
    }
}

/// Opaque monotonic transition identifier.
pub type TransitionId = u64;

/// Validation failures found before mutating the model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// The world identity cannot be the reserved nil GUID.
    NilWorldIdentity,
    /// A data-layer name is empty or contains a path/control separator.
    InvalidLayerName,
    /// A layer ID is already registered.
    DuplicateLayer(DataLayerId),
    /// A cell assignment references an unknown data layer.
    UnknownLayer(DataLayerId),
    /// A layer's parent does not exist in this world.
    UnknownLayerParent(DataLayerId),
    /// A parent assignment would introduce a data-layer cycle.
    LayerParentCycle {
        /// Child layer that would become part of the cycle.
        child: DataLayerId,
        /// Parent layer requested for the child.
        parent: DataLayerId,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NilWorldIdentity => formatter.write_str("world identity must not be nil"),
            Self::InvalidLayerName => formatter.write_str("data-layer name is invalid"),
            Self::DuplicateLayer(id) => write!(formatter, "data layer {} already exists", id.get()),
            Self::UnknownLayer(id) => write!(formatter, "data layer {} does not exist", id.get()),
            Self::UnknownLayerParent(id) => {
                write!(formatter, "data-layer parent {} does not exist", id.get())
            }
            Self::LayerParentCycle { child, parent } => write!(
                formatter,
                "data-layer parent {} would create a cycle for layer {}",
                parent.get(),
                child.get()
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Errors returned by model operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldError {
    /// The requested cell is not registered.
    UnknownCell(CellCoord),
    /// A cell with the same coordinate is already registered.
    DuplicateCell(CellCoord),
    /// A transition or completion was not valid for the current state.
    InvalidTransition(CellCoord),
    /// A plan was produced from an older model revision.
    StalePlan {
        /// Epoch captured by the plan.
        expected: u64,
        /// Current model epoch.
        actual: u64,
    },
    /// A transition token does not match the current pending operation.
    StaleToken(TransitionId),
    /// The model exhausted its transition ID space.
    TransitionIdExhausted,
    /// A cell exhausted its authoring revision counter.
    RevisionExhausted(CellCoord),
    /// Authoring policy or edit scope currently prohibits the edit.
    CellNotEditable(CellCoord),
    /// Validation failed before a mutation could be committed.
    Validation(ValidationError),
    /// The requested cell set contains an unregistered cell.
    DemandContainsUnknownCell(CellCoord),
}

impl fmt::Display for WorldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCell(cell) => {
                write!(formatter, "unknown cell ({},{},{})", cell.x, cell.y, cell.z)
            }
            Self::DuplicateCell(cell) => {
                write!(
                    formatter,
                    "cell ({},{},{}) already exists",
                    cell.x, cell.y, cell.z
                )
            }
            Self::InvalidTransition(cell) => write!(
                formatter,
                "invalid transition for cell ({},{},{})",
                cell.x, cell.y, cell.z
            ),
            Self::StalePlan { expected, actual } => write!(
                formatter,
                "stale residency plan: expected epoch {expected}, model is at {actual}"
            ),
            Self::StaleToken(id) => write!(formatter, "stale transition token {id}"),
            Self::TransitionIdExhausted => formatter.write_str("transition ID space exhausted"),
            Self::RevisionExhausted(cell) => write!(
                formatter,
                "cell ({},{},{}) exhausted its authoring revision",
                cell.x, cell.y, cell.z
            ),
            Self::CellNotEditable(cell) => write!(
                formatter,
                "cell ({},{},{}) is not editable",
                cell.x, cell.y, cell.z
            ),
            Self::Validation(error) => error.fmt(formatter),
            Self::DemandContainsUnknownCell(cell) => write!(
                formatter,
                "demand contains unknown cell ({},{},{})",
                cell.x, cell.y, cell.z
            ),
        }
    }
}

impl std::error::Error for WorldError {}

impl From<ValidationError> for WorldError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

/// The complete deterministic editor-world model.
#[derive(Clone, Debug)]
pub struct EditorWorld {
    identity: StableGuid,
    layers: BTreeMap<DataLayerId, DataLayer>,
    cells: BTreeMap<CellKey, CellRecord>,
    scope: EditScope,
    epoch: u64,
    next_transition: TransitionId,
}

impl EditorWorld {
    /// Creates a model for one stable world identity.
    pub fn new(identity: StableGuid) -> Result<Self, ValidationError> {
        if identity.is_nil() {
            return Err(ValidationError::NilWorldIdentity);
        }
        Ok(Self {
            identity,
            layers: BTreeMap::new(),
            cells: BTreeMap::new(),
            scope: EditScope::unrestricted(),
            epoch: 0,
            next_transition: 0,
        })
    }

    /// Stable world identity.
    #[must_use]
    pub const fn identity(&self) -> StableGuid {
        self.identity
    }

    /// Current mutation epoch used to reject stale plans.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Adds a data layer after validating its identity and name.
    pub fn add_layer(&mut self, layer: DataLayer) -> Result<(), ValidationError> {
        if self.layers.contains_key(&layer.id) {
            return Err(ValidationError::DuplicateLayer(layer.id));
        }
        validate_layer_name(&layer.name)?;
        if self.layers.values().any(|other| other.name == layer.name) {
            return Err(ValidationError::InvalidLayerName);
        }
        self.validate_layer_parent(layer.id, layer.parent)?;
        self.layers.insert(layer.id, layer);
        self.bump_epoch();
        Ok(())
    }

    /// Returns a data layer by ID.
    #[must_use]
    pub fn layer(&self, id: DataLayerId) -> Option<&DataLayer> {
        self.layers.get(&id)
    }

    /// Reparents a layer after validating the complete ancestor chain.
    ///
    /// The model is unchanged when the requested parent is unknown or would
    /// introduce a cycle.
    pub fn set_layer_parent(
        &mut self,
        id: DataLayerId,
        parent: Option<DataLayerId>,
    ) -> Result<(), ValidationError> {
        if !self.layers.contains_key(&id) {
            return Err(ValidationError::UnknownLayer(id));
        }
        self.validate_layer_parent(id, parent)?;
        self.layers
            .get_mut(&id)
            .expect("validated data layer must exist")
            .parent = parent;
        self.bump_epoch();
        Ok(())
    }

    /// Sets layer visibility without changing editability or residency.
    pub fn set_layer_visible(&mut self, id: DataLayerId, visible: bool) -> Result<(), WorldError> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or(ValidationError::UnknownLayer(id))?;
        layer.visible = visible;
        self.bump_epoch();
        Ok(())
    }

    /// Sets layer editability without changing visibility or residency.
    pub fn set_layer_editable(
        &mut self,
        id: DataLayerId,
        editable: bool,
    ) -> Result<(), WorldError> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or(ValidationError::UnknownLayer(id))?;
        layer.editable = editable;
        self.bump_epoch();
        Ok(())
    }

    /// Registers a cell and its data-layer memberships atomically.
    pub fn add_cell(
        &mut self,
        coord: CellCoord,
        layers: impl IntoIterator<Item = DataLayerId>,
    ) -> Result<&CellRecord, WorldError> {
        let key = CellKey::new(coord);
        if self.cells.contains_key(&key) {
            return Err(WorldError::DuplicateCell(coord));
        }
        let memberships: BTreeSet<_> = layers.into_iter().collect();
        if let Some(layer) = memberships.iter().find(|id| !self.layers.contains_key(id)) {
            return Err(ValidationError::UnknownLayer(*layer).into());
        }
        let record = CellRecord {
            key,
            file: CellFileIdentity::derive(self.identity, coord),
            layers: memberships,
            residency: CellResidency::Unloaded,
            dirty: false,
            revision: 0,
        };
        self.cells.insert(key, record);
        self.bump_epoch();
        Ok(self.cells.get(&key).expect("inserted cell must exist"))
    }

    /// Returns a cell record by coordinate.
    #[must_use]
    pub fn cell(&self, coord: CellCoord) -> Option<&CellRecord> {
        self.cells.get(&CellKey::new(coord))
    }

    /// Returns cells in deterministic coordinate order.
    pub fn cells(&self) -> impl Iterator<Item = &CellRecord> {
        self.cells.values()
    }

    /// Number of registered cells without traversing the catalog.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Validates the complete model without changing it.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.identity.is_nil() {
            return Err(ValidationError::NilWorldIdentity);
        }
        let mut names = BTreeSet::new();
        for (&id, layer) in &self.layers {
            validate_layer_name(&layer.name)?;
            if !names.insert(&layer.name) {
                return Err(ValidationError::InvalidLayerName);
            }
            self.validate_layer_parent(id, layer.parent)?;
        }
        for cell in self.cells.values() {
            if let Some(layer) = cell.layers.iter().find(|id| !self.layers.contains_key(id)) {
                return Err(ValidationError::UnknownLayer(*layer));
            }
        }
        Ok(())
    }

    /// Replaces a cell's layer membership atomically.
    pub fn set_cell_layers(
        &mut self,
        coord: CellCoord,
        layers: impl IntoIterator<Item = DataLayerId>,
    ) -> Result<(), WorldError> {
        let memberships: BTreeSet<_> = layers.into_iter().collect();
        if let Some(layer) = memberships.iter().find(|id| !self.layers.contains_key(id)) {
            return Err(ValidationError::UnknownLayer(*layer).into());
        }
        let record = self
            .cells
            .get_mut(&CellKey::new(coord))
            .ok_or(WorldError::UnknownCell(coord))?;
        record.layers = memberships;
        self.bump_epoch();
        Ok(())
    }

    /// Replaces the active edit scope.
    pub fn set_edit_scope(&mut self, scope: EditScope) {
        self.scope = scope;
        self.bump_epoch();
    }

    /// Returns the active edit scope.
    #[must_use]
    pub const fn edit_scope(&self) -> &EditScope {
        &self.scope
    }

    /// Computes visibility from layer policy only. Hidden content may remain resident.
    #[must_use]
    pub fn is_cell_visible(&self, coord: CellCoord) -> bool {
        let Some(cell) = self.cell(coord) else {
            return false;
        };
        cell.layers
            .iter()
            .all(|id| self.layer_ancestors_allow(*id, DataLayer::is_visible))
    }

    /// Computes editability from residency, scope, and every assigned layer.
    #[must_use]
    pub fn is_cell_editable(&self, coord: CellCoord) -> bool {
        let Some(cell) = self.cell(coord) else {
            return false;
        };
        cell.residency == CellResidency::Loaded
            && self.scope.permits(cell.key, &cell.layers)
            && cell
                .layers
                .iter()
                .all(|id| self.layer_ancestors_allow(*id, DataLayer::is_editable))
    }

    /// Marks a loaded cell dirty, preventing unsafe unload planning.
    pub fn mark_dirty(&mut self, coord: CellCoord) -> Result<(), WorldError> {
        let cell = self
            .cells
            .get_mut(&CellKey::new(coord))
            .ok_or(WorldError::UnknownCell(coord))?;
        if cell.residency != CellResidency::Loaded {
            return Err(WorldError::InvalidTransition(coord));
        }
        cell.revision = cell
            .revision
            .checked_add(1)
            .ok_or(WorldError::RevisionExhausted(coord))?;
        cell.dirty = true;
        self.bump_epoch();
        Ok(())
    }

    /// Marks a loaded cell clean after its bytes have been committed.
    pub fn mark_clean(&mut self, coord: CellCoord) -> Result<(), WorldError> {
        let cell = self
            .cells
            .get_mut(&CellKey::new(coord))
            .ok_or(WorldError::UnknownCell(coord))?;
        if cell.residency != CellResidency::Loaded {
            return Err(WorldError::InvalidTransition(coord));
        }
        cell.dirty = false;
        self.bump_epoch();
        Ok(())
    }

    /// Plans the minimum deterministic load/unload set for `demand`.
    pub fn plan(&self, demand: &ResidencyDemand) -> Result<ResidencyPlan, WorldError> {
        for key in &demand.desired {
            if !self.cells.contains_key(key) {
                return Err(WorldError::DemandContainsUnknownCell(key.coord()));
            }
        }
        let mut operations = Vec::new();
        let mut blocked = Vec::new();
        for cell in self.cells.values() {
            match (demand.contains(cell.key), cell.residency) {
                (true, CellResidency::Unloaded) => operations.push(PlanOperation::Load {
                    cell: cell.coord(),
                    file: cell.file.clone(),
                }),
                (false, CellResidency::Loaded) if cell.dirty => {
                    blocked.push(PlanBlock::DirtyCell(cell.coord()));
                }
                (false, CellResidency::Loaded) => operations.push(PlanOperation::Unload {
                    cell: cell.coord(),
                    file: cell.file.clone(),
                }),
                _ => {}
            }
        }
        Ok(ResidencyPlan {
            epoch: self.epoch,
            operations,
            blocked,
        })
    }

    /// Starts every operation in a plan atomically, returning transition tokens.
    pub fn begin_plan(&mut self, plan: &ResidencyPlan) -> Result<Vec<TransitionToken>, WorldError> {
        if plan.epoch != self.epoch {
            return Err(WorldError::StalePlan {
                expected: plan.epoch,
                actual: self.epoch,
            });
        }
        self.validate_plan_operations(plan)?;
        let count =
            u64::try_from(plan.operations.len()).map_err(|_| WorldError::TransitionIdExhausted)?;
        let last = self
            .next_transition
            .checked_add(count)
            .ok_or(WorldError::TransitionIdExhausted)?;
        let mut tokens = Vec::with_capacity(plan.operations.len());
        for (offset, operation) in plan.operations.iter().enumerate() {
            let id =
                self.next_transition + u64::try_from(offset).expect("plan length fits u64") + 1;
            let kind = match operation {
                PlanOperation::Load { .. } => TransitionKind::Load,
                PlanOperation::Unload { .. } => TransitionKind::Unload,
            };
            let key = CellKey::new(operation.cell());
            let cell = self.cells.get_mut(&key).expect("validated plan cell");
            cell.residency = match kind {
                TransitionKind::Load => CellResidency::Loading(id),
                TransitionKind::Unload => CellResidency::Unloading(id),
            };
            tokens.push(TransitionToken {
                id,
                cell: key,
                kind,
            });
        }
        self.next_transition = last;
        if !tokens.is_empty() {
            self.bump_epoch();
        }
        Ok(tokens)
    }

    /// Completes a load transition and makes the cell editable if policy allows.
    pub fn complete_load(&mut self, token: TransitionToken) -> Result<(), WorldError> {
        self.finish_transition(token, TransitionKind::Load, CellResidency::Loaded)
    }

    /// Completes an unload transition and removes the cell from residency.
    pub fn complete_unload(&mut self, token: TransitionToken) -> Result<(), WorldError> {
        self.finish_transition(token, TransitionKind::Unload, CellResidency::Unloaded)
    }

    /// Fails a load safely; the cell returns to `Unloaded`.
    pub fn fail_load(&mut self, token: TransitionToken) -> Result<(), WorldError> {
        self.finish_transition(token, TransitionKind::Load, CellResidency::Unloaded)
    }

    /// Fails an unload safely; the cell remains `Loaded` and retains dirtiness.
    pub fn fail_unload(&mut self, token: TransitionToken) -> Result<(), WorldError> {
        self.finish_transition(token, TransitionKind::Unload, CellResidency::Loaded)
    }

    fn validate_plan_operations(&self, plan: &ResidencyPlan) -> Result<(), WorldError> {
        for operation in &plan.operations {
            let cell = self
                .cells
                .get(&CellKey::new(operation.cell()))
                .ok_or(WorldError::UnknownCell(operation.cell()))?;
            match operation {
                PlanOperation::Load { .. } if cell.residency != CellResidency::Unloaded => {
                    return Err(WorldError::InvalidTransition(cell.coord()))
                }
                PlanOperation::Unload { .. }
                    if cell.residency != CellResidency::Loaded || cell.dirty =>
                {
                    return Err(WorldError::InvalidTransition(cell.coord()))
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn validate_layer_parent(
        &self,
        child: DataLayerId,
        parent: Option<DataLayerId>,
    ) -> Result<(), ValidationError> {
        let mut current = parent;
        while let Some(id) = current {
            if id == child {
                return Err(ValidationError::LayerParentCycle {
                    child,
                    parent: parent.expect("ancestor traversal requires a parent"),
                });
            }
            let layer = self
                .layers
                .get(&id)
                .ok_or(ValidationError::UnknownLayerParent(id))?;
            current = layer.parent;
        }
        Ok(())
    }

    fn layer_ancestors_allow(&self, id: DataLayerId, policy: fn(&DataLayer) -> bool) -> bool {
        let mut current = Some(id);
        while let Some(layer_id) = current {
            let Some(layer) = self.layers.get(&layer_id) else {
                return false;
            };
            if !policy(layer) {
                return false;
            }
            current = layer.parent;
        }
        true
    }

    fn finish_transition(
        &mut self,
        token: TransitionToken,
        expected: TransitionKind,
        final_state: CellResidency,
    ) -> Result<(), WorldError> {
        if token.kind != expected {
            return Err(WorldError::StaleToken(token.id));
        }
        let cell = self
            .cells
            .get_mut(&token.cell)
            .ok_or(WorldError::UnknownCell(token.cell.coord()))?;
        let valid = match (expected, cell.residency) {
            (TransitionKind::Load, CellResidency::Loading(id))
            | (TransitionKind::Unload, CellResidency::Unloading(id)) => id == token.id,
            _ => false,
        };
        if !valid {
            return Err(WorldError::StaleToken(token.id));
        }
        cell.residency = final_state;
        if final_state == CellResidency::Unloaded {
            cell.dirty = false;
        }
        self.bump_epoch();
        Ok(())
    }

    fn bump_epoch(&mut self) {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("editor world epoch exhausted");
    }
}

fn validate_layer_name(name: &str) -> Result<(), ValidationError> {
    if name.is_empty()
        || name
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
    {
        return Err(ValidationError::InvalidLayerName);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> EditorWorld {
        EditorWorld::new(StableGuid::from_path("worlds/test.world")).expect("world")
    }

    fn cell(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    fn layer(raw: u32, name: &str) -> DataLayer {
        DataLayer::new(DataLayerId::new(raw), name).expect("layer")
    }

    #[test]
    fn visibility_and_editability_are_independent() {
        let mut model = world();
        model.add_layer(layer(1, "day")).expect("add layer");
        model
            .add_cell(cell(2, 0, -1), [DataLayerId::new(1)])
            .expect("add cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(2, 0, -1));
        let plan = model.plan(&demand).expect("plan");
        let [load] = model
            .begin_plan(&plan)
            .expect("begin")
            .try_into()
            .expect("one token");
        model.complete_load(load).expect("load");

        assert!(model.is_cell_visible(cell(2, 0, -1)));
        assert!(model.is_cell_editable(cell(2, 0, -1)));
        model
            .set_layer_visible(DataLayerId::new(1), false)
            .expect("hide");
        assert!(!model.is_cell_visible(cell(2, 0, -1)));
        assert!(model.is_cell_editable(cell(2, 0, -1)));
        model
            .set_layer_editable(DataLayerId::new(1), false)
            .expect("lock");
        assert!(!model.is_cell_editable(cell(2, 0, -1)));
        model
            .set_layer_visible(DataLayerId::new(1), true)
            .expect("show");
        assert!(!model.is_cell_editable(cell(2, 0, -1)));
    }

    #[test]
    fn ancestor_layer_policy_controls_child_assigned_cells() {
        let mut model = world();
        let parent = DataLayerId::new(1);
        let child = DataLayerId::new(2);
        model.add_layer(layer(1, "parent")).expect("parent");
        model
            .add_layer(DataLayer::with_parent(child, "child", parent).expect("child"))
            .expect("add child");
        let coord = cell(4, 0, 0);
        model.add_cell(coord, [child]).expect("add cell");
        let mut demand = ResidencyDemand::new();
        demand.require(coord);
        let plan = model.plan(&demand).expect("plan");
        let tokens = model.begin_plan(&plan).expect("begin");
        model.complete_load(tokens[0]).expect("load");

        model.set_layer_visible(parent, false).expect("hide parent");
        assert!(!model.is_cell_visible(coord));
        assert!(model.is_cell_editable(coord));

        model.set_layer_visible(parent, true).expect("show parent");
        model
            .set_layer_editable(parent, false)
            .expect("lock parent");
        assert!(model.is_cell_visible(coord));
        assert!(!model.is_cell_editable(coord));
    }

    #[test]
    fn invalid_layer_parent_changes_are_rejected_atomically() {
        let mut model = world();
        let parent = DataLayerId::new(1);
        let child = DataLayerId::new(2);
        model.add_layer(layer(1, "parent")).expect("parent");
        model
            .add_layer(DataLayer::with_parent(child, "child", parent).expect("child"))
            .expect("add child");
        let epoch = model.epoch();

        assert_eq!(
            model.add_layer(
                DataLayer::with_parent(DataLayerId::new(3), "orphan", DataLayerId::new(99))
                    .expect("orphan")
            ),
            Err(ValidationError::UnknownLayerParent(DataLayerId::new(99)))
        );
        assert!(model.layer(DataLayerId::new(3)).is_none());
        assert_eq!(model.epoch(), epoch);

        assert_eq!(
            model.set_layer_parent(parent, Some(child)),
            Err(ValidationError::LayerParentCycle {
                child: parent,
                parent: child
            })
        );
        assert_eq!(model.layer(parent).expect("parent").parent(), None);
        assert_eq!(model.layer(child).expect("child").parent(), Some(parent));
        assert_eq!(model.epoch(), epoch);
    }

    #[test]
    fn edit_scope_limits_cells_and_layers() {
        let mut model = world();
        model.add_layer(layer(1, "day")).expect("add layer");
        model.add_layer(layer(2, "night")).expect("add layer");
        model
            .add_cell(cell(0, 0, 0), [DataLayerId::new(1)])
            .expect("cell");
        model
            .add_cell(cell(1, 0, 0), [DataLayerId::new(2)])
            .expect("cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(0, 0, 0));
        demand.require(cell(1, 0, 0));
        let plan = model.plan(&demand).expect("plan");
        let tokens = model.begin_plan(&plan).expect("begin");
        for token in tokens {
            model.complete_load(token).expect("load");
        }

        let mut scope = EditScope::for_cells([cell(0, 0, 0)]);
        scope.allow_layer(DataLayerId::new(1));
        model.set_edit_scope(scope);
        assert!(model.is_cell_editable(cell(0, 0, 0)));
        assert!(!model.is_cell_editable(cell(1, 0, 0)));
    }

    #[test]
    fn plans_are_deterministic_and_adapt_to_scene_delta() {
        let mut model = world();
        model.add_cell(cell(1, 0, 0), []).expect("cell");
        model.add_cell(cell(-1, 0, 0), []).expect("cell");
        model.add_cell(cell(0, 0, 0), []).expect("cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(1, 0, 0));
        demand.require(cell(-1, 0, 0));
        let plan = model.plan(&demand).expect("plan");
        assert_eq!(
            plan.operations()
                .iter()
                .map(PlanOperation::cell)
                .collect::<Vec<_>>(),
            [cell(-1, 0, 0), cell(1, 0, 0)]
        );
        assert_eq!(
            plan.as_streaming_delta().to_load,
            [cell(-1, 0, 0), cell(1, 0, 0)]
        );
        assert!(plan.as_streaming_delta().to_unload.is_empty());
    }

    #[test]
    fn dirty_cells_block_unload_without_mutating_state() {
        let mut model = world();
        model.add_cell(cell(0, 0, 0), []).expect("cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(0, 0, 0));
        let plan = model.plan(&demand).expect("load plan");
        let token = model.begin_plan(&plan).expect("begin")[0];
        model.complete_load(token).expect("load");
        model.mark_dirty(cell(0, 0, 0)).expect("dirty");

        let empty = ResidencyDemand::new();
        let unload_plan = model.plan(&empty).expect("unload plan");
        assert!(unload_plan.operations().is_empty());
        assert_eq!(
            unload_plan.blocked(),
            &[PlanBlock::DirtyCell(cell(0, 0, 0))]
        );
        assert_eq!(
            model.cell(cell(0, 0, 0)).expect("cell").residency(),
            CellResidency::Loaded
        );
    }

    #[test]
    fn failed_transitions_restore_safe_states_and_reject_stale_tokens() {
        let mut model = world();
        model.add_cell(cell(0, 0, 0), []).expect("cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(0, 0, 0));
        let plan = model.plan(&demand).expect("plan");
        let token = model.begin_plan(&plan).expect("begin")[0];
        model.fail_load(token).expect("fail load");
        assert_eq!(
            model.cell(cell(0, 0, 0)).expect("cell").residency(),
            CellResidency::Unloaded
        );
        assert_eq!(
            model.complete_load(token),
            Err(WorldError::StaleToken(token.id()))
        );

        let plan = model.plan(&demand).expect("retry plan");
        let token = model.begin_plan(&plan).expect("retry begin")[0];
        model.complete_load(token).expect("retry load");
        let empty = ResidencyDemand::new();
        let plan = model.plan(&empty).expect("unload plan");
        let token = model.begin_plan(&plan).expect("unload begin")[0];
        model.fail_unload(token).expect("fail unload");
        assert_eq!(
            model.cell(cell(0, 0, 0)).expect("cell").residency(),
            CellResidency::Loaded
        );
    }

    #[test]
    fn file_identity_is_stable_and_scene_compatible() {
        let mut first = world();
        let mut second = world();
        first.add_cell(cell(-3, 4, 5), []).expect("cell");
        second.add_cell(cell(-3, 4, 5), []).expect("cell");
        let a = first.cell(cell(-3, 4, 5)).expect("cell").file();
        let b = second.cell(cell(-3, 4, 5)).expect("cell").file();
        assert_eq!(a, b);
        assert_eq!(a.relative_path(), "cells/-3_4_5.prismcell");
        assert_eq!(
            a.scene_store_path("/tmp/world"),
            Path::new("/tmp/world/-3_4_5.prismcell")
        );
    }

    #[test]
    fn stale_plans_cannot_partially_start() {
        let mut model = world();
        model.add_cell(cell(0, 0, 0), []).expect("cell");
        let mut demand = ResidencyDemand::new();
        demand.require(cell(0, 0, 0));
        let plan = model.plan(&demand).expect("plan");
        model.set_edit_scope(EditScope::unrestricted());
        assert!(matches!(
            model.begin_plan(&plan),
            Err(WorldError::StalePlan { .. })
        ));
        assert_eq!(
            model.cell(cell(0, 0, 0)).expect("cell").residency(),
            CellResidency::Unloaded
        );
    }
}
