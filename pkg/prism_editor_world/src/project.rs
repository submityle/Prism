//! World-project membership, separate from resident bytes and OFPA transactions.
//!
//! A project is the durable identity/layer/cell/actor catalog. Residency, leases,
//! payload revisions, edit scope and HLOD build products remain runtime state.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};

use crate::{CellKey, DataLayer, EditorWorldData, EditorWorldLimits};

#[path = "project_codec.rs"]
mod codec;
#[path = "project_history.rs"]
mod history;
#[path = "project_storage.rs"]
mod storage;
pub use history::{ProjectChange, ProjectEdit, ProjectHistory};
pub use storage::{
    WorldProjectCodec, WorldProjectLoadCandidate, WorldProjectSaveReceipt,
    WorldProjectStorageError, WorldProjectStorageRequest, WorldProjectStorageResult,
    WorldProjectStorageService, WorldProjectToken,
};

/// Admission limits checked before decoding or retaining authoring history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectLimits {
    /// Maximum complete encoded project, including header and checksum.
    pub max_bytes: usize,
    /// Maximum data-layer records.
    pub max_layers: usize,
    /// Maximum registered cells.
    pub max_cells: usize,
    /// Maximum actor-to-cell relationships.
    pub max_actors: usize,
    /// Maximum total direct cell-to-layer relationships.
    pub max_memberships: usize,
    /// Maximum UTF-8 bytes in one data-layer name.
    pub max_name_bytes: usize,
    /// Maximum distinct records in one atomic authoring command.
    pub max_edits: usize,
}

impl Default for ProjectLimits {
    fn default() -> Self {
        Self {
            max_bytes: 8 * 1024 * 1024,
            max_layers: 4096,
            max_cells: 100_000,
            max_actors: 200_000,
            max_memberships: 400_000,
            max_name_bytes: 1024,
            max_edits: 4096,
        }
    }
}

impl ProjectLimits {
    /// Rejects zero and excessive limits, including on trusted encoder inputs.
    pub fn validate(self) -> Result<(), ProjectError> {
        if self.max_bytes < 64
            || self.max_bytes > 64 * 1024 * 1024
            || self.max_layers == 0
            || self.max_layers > 65_536
            || self.max_cells == 0
            || self.max_cells > 1_000_000
            || self.max_actors == 0
            || self.max_actors > 10_000_000
            || self.max_memberships == 0
            || self.max_memberships > 4_000_000
            || self.max_name_bytes == 0
            || self.max_name_bytes > 65_536
            || self.max_edits == 0
            || self.max_edits > 65_536
        {
            return Err(ProjectError::InvalidLimits);
        }
        Ok(())
    }
}

/// Complete durable data-layer definition. Parent policy remains inherited.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectLayer {
    /// Stable data-layer ID; zero is valid in the existing world contract.
    pub id: DataLayerId,
    /// Unique, nonempty authoring name.
    pub name: String,
    /// Optional parent in this same project.
    pub parent: Option<DataLayerId>,
    /// Local visibility policy.
    pub visible: bool,
    /// Local editability policy.
    pub editable: bool,
}

impl From<&DataLayer> for ProjectLayer {
    fn from(layer: &DataLayer) -> Self {
        Self {
            id: layer.id(),
            name: layer.name().to_owned(),
            parent: layer.parent(),
            visible: layer.is_visible(),
            editable: layer.is_editable(),
        }
    }
}

/// Failures leave the externally visible project and history unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectError {
    /// A configured bound is zero or exceeds the supported ceiling.
    InvalidLimits,
    /// A count, encoded length, command or history exceeds its budget.
    Budget(&'static str),
    /// Wire bytes fail exact length, version, order, tag or checksum checks.
    Malformed(&'static str),
    /// Identity, layer hierarchy or membership violates the world contract.
    Invalid(&'static str),
    /// One atomic command mentions the same record twice.
    DuplicateEdit,
    /// A command has no observable membership change.
    NoChange,
    /// An inverse/redo no longer matches its touched records or world identity.
    Conflict,
    /// The monotonic project revision cannot advance.
    RevisionExhausted,
    /// Undo or redo was requested with an empty corresponding stack.
    EmptyHistory,
    /// The existing executable world rejected materialization.
    World(crate::EditorWorldDataError),
}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => f.write_str("invalid project limits"),
            Self::Budget(s) => write!(f, "project budget exceeded: {s}"),
            Self::Malformed(s) => write!(f, "malformed project: {s}"),
            Self::Invalid(s) => write!(f, "invalid project: {s}"),
            Self::DuplicateEdit => f.write_str("duplicate project edit target"),
            Self::NoChange => f.write_str("project command makes no change"),
            Self::Conflict => f.write_str("project history conflict"),
            Self::RevisionExhausted => f.write_str("project revision exhausted"),
            Self::EmptyHistory => f.write_str("project history is empty"),
            Self::World(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for ProjectError {}
impl From<crate::EditorWorldDataError> for ProjectError {
    fn from(e: crate::EditorWorldDataError) -> Self {
        Self::World(e)
    }
}

/// Canonical membership catalog; maps can only be changed by atomic commands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldProject {
    identity: StableGuid,
    revision: u64,
    stamp: std::sync::Arc<()>,
    layers: BTreeMap<DataLayerId, ProjectLayer>,
    cells: BTreeMap<CellKey, BTreeSet<DataLayerId>>,
    actors: BTreeMap<StableGuid, CellKey>,
}

impl WorldProject {
    /// Creates an empty membership catalog with a non-nil real world identity.
    pub fn new(identity: StableGuid) -> Result<Self, ProjectError> {
        if identity.is_nil() {
            return Err(ProjectError::Invalid("nil world identity"));
        }
        Ok(Self {
            identity,
            revision: 0,
            stamp: std::sync::Arc::new(()),
            layers: BTreeMap::new(),
            cells: BTreeMap::new(),
            actors: BTreeMap::new(),
        })
    }

    /// Captures every registered layer/cell/actor, including clean OFPA actors.
    /// Does not copy resident bytes, leases or transient residency states.
    pub fn capture(world: &EditorWorldData, limits: ProjectLimits) -> Result<Self, ProjectError> {
        limits.validate()?;
        let mut project = Self::new(world.model().identity())?;
        // Charge wire bytes before copying each borrowed record, so a small
        // byte budget also limits capture admission on a much larger world.
        let mut encoded = 64usize;
        let mut admit = |bytes: usize| -> Result<(), ProjectError> {
            encoded = encoded
                .checked_add(bytes)
                .ok_or(ProjectError::Budget("encoded bytes"))?;
            if encoded > limits.max_bytes {
                return Err(ProjectError::Budget("encoded bytes"));
            }
            Ok(())
        };
        for layer in world.model().layers.values() {
            if project.layers.len() >= limits.max_layers {
                return Err(ProjectError::Budget("layers"));
            }
            if layer.name().len() > limits.max_name_bytes {
                return Err(ProjectError::Budget("name"));
            }
            admit(10 + layer.name().len() + usize::from(layer.parent().is_some()) * 4)?;
            project.layers.insert(layer.id(), ProjectLayer::from(layer));
        }
        let mut links = 0usize;
        for cell in world.model().cells() {
            if project.cells.len() >= limits.max_cells {
                return Err(ProjectError::Budget("cells"));
            }
            admit(16)?;
            let mut layers = BTreeSet::new();
            for layer in cell.layers() {
                links += 1;
                if links > limits.max_memberships {
                    return Err(ProjectError::Budget("memberships"));
                }
                admit(4)?;
                layers.insert(layer);
            }
            project.cells.insert(cell.key(), layers);
        }
        for actor in world.actor_records() {
            if project.actors.len() >= limits.max_actors {
                return Err(ProjectError::Budget("actors"));
            }
            admit(28)?;
            project.actors.insert(
                actor.identity().actor(),
                CellKey::new(actor.identity().cell()),
            );
        }
        project.validate(limits)?;
        Ok(project)
    }

    /// Stable identity used by cell identity derivation and history fences.
    pub const fn identity(&self) -> StableGuid {
        self.identity
    }
    /// Membership authoring revision; undo and redo also advance it.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// All layers in numeric ID order.
    pub fn layers(&self) -> impl ExactSizeIterator<Item = &ProjectLayer> {
        self.layers.values()
    }
    /// All cells and direct memberships in signed lexicographic coordinate order.
    pub fn cells(&self) -> impl ExactSizeIterator<Item = (CellCoord, &BTreeSet<DataLayerId>)> {
        self.cells.iter().map(|(key, layers)| (key.coord(), layers))
    }
    /// All actors and owner cells in stable GUID order.
    pub fn actors(&self) -> impl ExactSizeIterator<Item = (StableGuid, CellCoord)> + '_ {
        self.actors.iter().map(|(actor, key)| (*actor, key.coord()))
    }
    /// Layer by stable ID.
    pub fn layer(&self, id: DataLayerId) -> Option<&ProjectLayer> {
        self.layers.get(&id)
    }
    /// Direct memberships of a registered cell (empty is different from absent).
    pub fn cell_layers(&self, cell: CellCoord) -> Option<&BTreeSet<DataLayerId>> {
        self.cells.get(&CellKey::new(cell))
    }
    /// Current unique owner of a registered actor.
    pub fn actor_cell(&self, actor: StableGuid) -> Option<CellCoord> {
        self.actors.get(&actor).map(|key| key.coord())
    }

    /// Rebuilds a fresh executable world with real derived cell/OFPA identities.
    /// All cells start unloaded. This never replaces a live world with dirty
    /// bytes or in-flight operations; the host controls installation.
    pub fn materialize(
        &self,
        limits: EditorWorldLimits,
        project_limits: ProjectLimits,
    ) -> Result<EditorWorldData, ProjectError> {
        self.validate(project_limits)?;
        if self.cells.len() > limits.max_cells() || self.actors.len() > limits.max_actors() {
            return Err(ProjectError::Budget("executable world"));
        }
        let mut world = EditorWorldData::new(self.identity, limits)?;
        for layer in self.layers.values() {
            world.add_layer(
                DataLayer::new(layer.id, &layer.name).map_err(|e| ProjectError::World(e.into()))?,
            )?;
        }
        for layer in self.layers.values() {
            world.set_layer_parent(layer.id, layer.parent)?;
            world.set_layer_visible(layer.id, layer.visible)?;
            world.set_layer_editable(layer.id, layer.editable)?;
        }
        for (cell, layers) in self.cells() {
            world.add_cell(cell, layers.iter().copied())?;
        }
        for (actor, cell) in self.actors() {
            world.register_actor(actor, cell)?;
        }
        Ok(world)
    }

    /// Checks complete relationships, hierarchy, names, counts and encoded size.
    pub fn validate(&self, limits: ProjectLimits) -> Result<(), ProjectError> {
        limits.validate()?;
        if self.identity.is_nil() {
            return Err(ProjectError::Invalid("nil world identity"));
        }
        if self.layers.len() > limits.max_layers {
            return Err(ProjectError::Budget("layers"));
        }
        if self.cells.len() > limits.max_cells {
            return Err(ProjectError::Budget("cells"));
        }
        if self.actors.len() > limits.max_actors {
            return Err(ProjectError::Budget("actors"));
        }
        let mut names = BTreeSet::new();
        for (id, layer) in &self.layers {
            if *id != layer.id {
                return Err(ProjectError::Invalid("layer identity"));
            }
            if layer.name.len() > limits.max_name_bytes {
                return Err(ProjectError::Budget("name"));
            }
            crate::validate_layer_name(&layer.name)
                .map_err(|_| ProjectError::Invalid("layer name"))?;
            if !names.insert(layer.name.as_str()) {
                return Err(ProjectError::Invalid("duplicate layer name"));
            }
        }
        // Three-colour traversal: O(layers), bounded heap, no recursive stack.
        let mut done = BTreeSet::new();
        for start in self.layers.keys() {
            let mut path = BTreeSet::new();
            let mut next = Some(*start);
            while let Some(id) = next {
                if done.contains(&id) {
                    break;
                }
                if !path.insert(id) {
                    return Err(ProjectError::Invalid("layer cycle"));
                }
                next = self
                    .layers
                    .get(&id)
                    .ok_or(ProjectError::Invalid("unknown layer parent"))?
                    .parent;
            }
            done.extend(path);
        }
        let mut links = 0usize;
        for layers in self.cells.values() {
            links = links
                .checked_add(layers.len())
                .ok_or(ProjectError::Budget("memberships"))?;
            if links > limits.max_memberships {
                return Err(ProjectError::Budget("memberships"));
            }
            if layers.iter().any(|id| !self.layers.contains_key(id)) {
                return Err(ProjectError::Invalid("unknown cell layer"));
            }
        }
        for (actor, cell) in &self.actors {
            if actor.is_nil() {
                return Err(ProjectError::Invalid("nil actor identity"));
            }
            if !self.cells.contains_key(cell) {
                return Err(ProjectError::Invalid("unknown actor cell"));
            }
        }
        if self.encoded_size()? > limits.max_bytes {
            return Err(ProjectError::Budget("encoded bytes"));
        }
        Ok(())
    }
}
