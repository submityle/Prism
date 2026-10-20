//! Sparse, compare-before-restore authoring transactions.
use super::*;

/// One replacement/removal in an atomic final-state membership transaction.
/// References may point at records added later in the same command. Removals
/// never cascade silently: callers must explicitly repair dependants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectEdit {
    /// Add/replace a complete layer, or remove this exact ID.
    Layer {
        /// Stable target ID, required to match the value's ID.
        id: DataLayerId,
        /// Complete replacement; None removes the layer.
        value: Option<ProjectLayer>,
    },
    /// Add/replace a cell's direct memberships, or remove the cell.
    Cell {
        /// Stable cell coordinate.
        cell: CellCoord,
        /// Complete sorted membership set; None removes the cell.
        layers: Option<BTreeSet<DataLayerId>>,
    },
    /// Register/move an actor, or remove its membership. No payload is edited.
    Actor {
        /// Non-nil stable actor GUID.
        actor: StableGuid,
        /// Unique owner; None removes the actor.
        cell: Option<CellCoord>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Key {
    Layer(DataLayerId),
    Cell(CellKey),
    Actor(StableGuid),
}
#[derive(Clone, Debug, Eq, PartialEq)]
enum Value {
    Layer(ProjectLayer),
    Cell(BTreeSet<DataLayerId>),
    Actor(CellKey),
}
#[derive(Clone, Debug)]
struct Delta {
    key: Key,
    before: Option<Value>,
    after: Option<Value>,
}

/// An opaque sparse inverse: only touched records, never a whole-project copy.
/// Undo/redo compare all touched values and the stable world identity first.
#[derive(Clone, Debug)]
pub struct ProjectChange {
    identity: StableGuid,
    deltas: Vec<Delta>,
    charged_bytes: usize,
}

impl ProjectEdit {
    fn key(&self) -> Key {
        match self {
            Self::Layer { id, .. } => Key::Layer(*id),
            Self::Cell { cell, .. } => Key::Cell(CellKey::new(*cell)),
            Self::Actor { actor, .. } => Key::Actor(*actor),
        }
    }
    fn value(&self) -> Option<Value> {
        match self {
            Self::Layer { value, .. } => value.clone().map(Value::Layer),
            Self::Cell { layers, .. } => layers.clone().map(Value::Cell),
            Self::Actor { cell, .. } => cell.map(|c| Value::Actor(CellKey::new(c))),
        }
    }
    fn validate(&self, limits: ProjectLimits) -> Result<(), ProjectError> {
        match self {
            Self::Layer { id, value: Some(v) } => {
                if *id != v.id {
                    return Err(ProjectError::Invalid("layer identity"));
                }
                if v.name.len() > limits.max_name_bytes {
                    return Err(ProjectError::Budget("name"));
                }
                crate::validate_layer_name(&v.name)
                    .map_err(|_| ProjectError::Invalid("layer name"))?;
            }
            Self::Cell {
                layers: Some(v), ..
            } if v.len() > limits.max_memberships => {
                return Err(ProjectError::Budget("memberships"));
            }
            Self::Actor { actor, .. } if actor.is_nil() => {
                return Err(ProjectError::Invalid("nil actor identity"));
            }
            _ => {}
        }
        Ok(())
    }
    fn charge(&self) -> usize {
        match self {
            Self::Layer { value: Some(v), .. } => 128 + v.name.len(),
            Self::Cell {
                layers: Some(v), ..
            } => 128 + v.len() * 64,
            _ => 128,
        }
    }
}

impl WorldProject {
    fn value(&self, key: Key) -> Option<Value> {
        match key {
            Key::Layer(id) => self.layers.get(&id).cloned().map(Value::Layer),
            Key::Cell(cell) => self.cells.get(&cell).cloned().map(Value::Cell),
            Key::Actor(actor) => self.actors.get(&actor).copied().map(Value::Actor),
        }
    }
    fn matches(&self, key: Key, value: &Option<Value>) -> bool {
        match (key, value) {
            (Key::Layer(id), Some(Value::Layer(v))) => self.layers.get(&id) == Some(v),
            (Key::Cell(cell), Some(Value::Cell(v))) => self.cells.get(&cell) == Some(v),
            (Key::Actor(actor), Some(Value::Actor(v))) => self.actors.get(&actor) == Some(v),
            (Key::Layer(id), None) => !self.layers.contains_key(&id),
            (Key::Cell(cell), None) => !self.cells.contains_key(&cell),
            (Key::Actor(actor), None) => !self.actors.contains_key(&actor),
            _ => false,
        }
    }
    fn old_charge(&self, key: Key) -> usize {
        match key {
            Key::Layer(id) => 128 + self.layers.get(&id).map_or(0, |v| v.name.len()),
            Key::Cell(cell) => 128 + self.cells.get(&cell).map_or(0, |v| v.len() * 64),
            Key::Actor(_) => 128,
        }
    }
    fn install(&mut self, key: Key, value: &Option<Value>) {
        match (key, value) {
            (Key::Layer(id), Some(Value::Layer(v))) => {
                self.layers.insert(id, v.clone());
            }
            (Key::Cell(cell), Some(Value::Cell(v))) => {
                self.cells.insert(cell, v.clone());
            }
            (Key::Actor(actor), Some(Value::Actor(v))) => {
                self.actors.insert(actor, *v);
            }
            (Key::Layer(id), None) => {
                self.layers.remove(&id);
            }
            (Key::Cell(cell), None) => {
                self.cells.remove(&cell);
            }
            (Key::Actor(actor), None) => {
                self.actors.remove(&actor);
            }
            _ => unreachable!("private delta key/value pair"),
        }
    }

    /// Applies a final-state command at an exact project revision and returns
    /// its sparse inverse. Failure leaves bytes and revision unchanged.
    pub fn apply(
        &mut self,
        expected_revision: u64,
        edits: &[ProjectEdit],
        limits: ProjectLimits,
        max_change_bytes: usize,
    ) -> Result<ProjectChange, ProjectError> {
        if self.revision != expected_revision {
            return Err(ProjectError::Conflict);
        }
        let change = ProjectChange::prepare(self, edits, limits, max_change_bytes)?;
        change.redo(self, limits)?;
        Ok(change)
    }
}

impl ProjectChange {
    fn prepare(
        project: &WorldProject,
        edits: &[ProjectEdit],
        limits: ProjectLimits,
        max_bytes: usize,
    ) -> Result<Self, ProjectError> {
        limits.validate()?;
        if max_bytes > 64 * 1024 * 1024 {
            return Err(ProjectError::InvalidLimits);
        }
        if max_bytes == 0 {
            return Err(ProjectError::Budget("history bytes"));
        }
        if edits.is_empty() {
            return Err(ProjectError::NoChange);
        }
        if edits.len() > limits.max_edits {
            return Err(ProjectError::Budget("command edits"));
        }
        let mut keys = BTreeSet::new();
        let mut charged_bytes = 64usize;
        // Admit all retained data before cloning any user-controlled values.
        for edit in edits {
            if !keys.insert(edit.key()) {
                return Err(ProjectError::DuplicateEdit);
            }
            edit.validate(limits)?;
            charged_bytes = charged_bytes
                .checked_add(edit.charge())
                .and_then(|n| n.checked_add(project.old_charge(edit.key())))
                .ok_or(ProjectError::Budget("history bytes"))?;
            if charged_bytes > max_bytes {
                return Err(ProjectError::Budget("history bytes"));
            }
        }
        let mut deltas = Vec::with_capacity(edits.len());
        for edit in edits {
            let key = edit.key();
            let before = project.value(key);
            let after = edit.value();
            if before != after {
                deltas.push(Delta { key, before, after });
            }
        }
        if deltas.is_empty() {
            return Err(ProjectError::NoChange);
        }
        deltas.sort_by_key(|d| d.key);
        Ok(Self {
            identity: project.identity,
            deltas,
            charged_bytes,
        })
    }

    /// Number of changed records retained by this inverse.
    pub fn record_count(&self) -> usize {
        self.deltas.len()
    }
    /// Deterministic retained-data charge, including per-record/node allowance.
    /// This is an admission budget, not an allocator/RSS measurement.
    pub const fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }
    /// Restores before-values if all after-values still match. Unrelated edits
    /// are preserved, provided the complete resulting relationships are valid.
    pub fn undo(
        &self,
        project: &mut WorldProject,
        limits: ProjectLimits,
    ) -> Result<(), ProjectError> {
        self.replay(project, limits, false)
    }
    /// Restores after-values if all before-values still match.
    pub fn redo(
        &self,
        project: &mut WorldProject,
        limits: ProjectLimits,
    ) -> Result<(), ProjectError> {
        self.replay(project, limits, true)
    }
    fn replay(
        &self,
        project: &mut WorldProject,
        limits: ProjectLimits,
        forward: bool,
    ) -> Result<(), ProjectError> {
        limits.validate()?;
        if project.identity != self.identity {
            return Err(ProjectError::Conflict);
        }
        let next = project
            .revision
            .checked_add(1)
            .ok_or(ProjectError::RevisionExhausted)?;
        for d in &self.deltas {
            if !project.matches(d.key, if forward { &d.before } else { &d.after }) {
                return Err(ProjectError::Conflict);
            }
        }
        for d in &self.deltas {
            project.install(d.key, if forward { &d.after } else { &d.before });
        }
        if let Err(error) = project.validate(limits) {
            for d in &self.deltas {
                project.install(d.key, if forward { &d.before } else { &d.after });
            }
            return Err(error);
        }
        project.revision = next;
        project.stamp = std::sync::Arc::new(());
        Ok(())
    }
}

/// Explicitly bounded linear undo/redo history. Full budgets reject commands;
/// they never silently evict older history. A successful branch clears redo.
#[derive(Debug)]
pub struct ProjectHistory {
    undo: Vec<ProjectChange>,
    redo: Vec<ProjectChange>,
    max_entries: usize,
    max_bytes: usize,
    charged_bytes: usize,
    identity: Option<StableGuid>,
}
impl ProjectHistory {
    /// Creates a bounded history with at most 65,536 entries and 64 MiB charge.
    pub fn new(max_entries: usize, max_bytes: usize) -> Result<Self, ProjectError> {
        if max_entries == 0
            || max_entries > 65_536
            || max_bytes == 0
            || max_bytes > 64 * 1024 * 1024
        {
            return Err(ProjectError::InvalidLimits);
        }
        Ok(Self {
            undo: Vec::new(),
            redo: Vec::new(),
            max_entries,
            max_bytes,
            charged_bytes: 0,
            identity: None,
        })
    }
    /// Undoable entry count.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
    /// Redoable entry count.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }
    /// Combined admission charge across both stacks.
    pub const fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }
    /// Discards history and its world fence, without editing any project.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.charged_bytes = 0;
        self.identity = None;
    }
    /// Executes one command with exact revision admission. Failed commands
    /// retain both stacks, including an existing redo branch.
    pub fn execute(
        &mut self,
        project: &mut WorldProject,
        expected_revision: u64,
        edits: &[ProjectEdit],
        limits: ProjectLimits,
    ) -> Result<(), ProjectError> {
        if project.revision != expected_revision
            || self.identity.is_some_and(|id| id != project.identity)
        {
            return Err(ProjectError::Conflict);
        }
        if self.undo.len() >= self.max_entries {
            return Err(ProjectError::Budget("history entries"));
        }
        let kept_bytes: usize = self.undo.iter().map(|c| c.charged_bytes).sum();
        let change = ProjectChange::prepare(project, edits, limits, self.max_bytes - kept_bytes)?;
        change.redo(project, limits)?;
        self.redo.clear();
        self.charged_bytes = kept_bytes + change.charged_bytes;
        self.identity = Some(project.identity);
        self.undo.push(change);
        Ok(())
    }
    /// Atomically undoes the most recent command.
    pub fn undo(
        &mut self,
        project: &mut WorldProject,
        limits: ProjectLimits,
    ) -> Result<(), ProjectError> {
        self.undo
            .last()
            .ok_or(ProjectError::EmptyHistory)?
            .undo(project, limits)?;
        self.redo.push(self.undo.pop().expect("checked entry"));
        Ok(())
    }
    /// Atomically redoes the most recently undone command.
    pub fn redo(
        &mut self,
        project: &mut WorldProject,
        limits: ProjectLimits,
    ) -> Result<(), ProjectError> {
        self.redo
            .last()
            .ok_or(ProjectError::EmptyHistory)?
            .redo(project, limits)?;
        self.undo.push(self.redo.pop().expect("checked entry"));
        Ok(())
    }
}
