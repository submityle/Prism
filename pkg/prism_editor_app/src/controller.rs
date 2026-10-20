//! Engine-neutral `UI` action execution over real reflection editors.
//!
//! Field commands share one document timeline, independent of selection.
//! `Undo`/`Redo` target the object captured when the command was committed. New
//! successful edits discard the whole redo branch. `FieldEditCommand` does not
//! merge, so every effective field submission is a separate history entry.
//! Registration loads clean model data; selection/expansion only change the `UI`
//! revision. Model edits, including `Undo`/`Redo`, advance a separate monotonic
//! revision. Dirty is conservative until that exact model revision is saved.

use alloc::collections::BTreeMap;
use core::fmt;

use prism_editor_core::{Command, Selection, SelectionItem};
use prism_editor_panels::{Outliner, OutlinerError, OutlinerId, OutlinerNode};
use prism_editor_reflect::{
    project_inspector, validate, EditorError, FieldEditCommand, FieldPath, InspectorProjection,
    ReflectEditorContext,
};
use prism_editor_ui::{
    ActionBindings, ActionError, ActionTemplate, ActionValue, EditorAction, SelectionMode, UiEvent,
    MAX_ACTION_PATH_BYTES, MAX_ACTION_TEXT_BYTES,
};
use prism_reflect::{
    reflect_path, Access, ParsedPath, Reflect, ReflectRef, TypeMetadata, TypeRegistration,
    TypeRegistry,
};

/// Resource bounds checked before committing controller state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControllerLimits {
    /// Maximum number of loaded objects.
    pub max_objects: usize,
    /// Maximum number of `Outliner` rows, including collapsed/hidden rows.
    pub max_outliner_nodes: usize,
    /// Maximum `Outliner` depth; a root has depth zero.
    pub max_outliner_depth: usize,
    /// Maximum number of selected objects.
    pub max_selection: usize,
    /// Maximum number of registered command factories.
    pub max_commands: usize,
    /// Maximum command-factory indirections during one execution.
    pub max_command_depth: usize,
    /// Maximum retained document commands (undo plus redo). A full history
    /// rejects a new edit; no entries are silently evicted. On a new branch,
    /// discarded redo entries do not count toward the candidate's capacity.
    pub max_history: usize,
    /// Additional per-object budget within the shared document timeline.
    /// This does not create independently navigable object histories.
    pub max_history_per_object: usize,
    /// Maximum reflected nodes per object, including map keys and hidden fields.
    pub max_inspector_nodes: usize,
    /// Maximum reflected depth; the object root has depth zero.
    pub max_inspector_depth: usize,
}

impl Default for ControllerLimits {
    fn default() -> Self {
        Self {
            max_objects: 4_096,
            max_outliner_nodes: 4_096,
            max_outliner_depth: 64,
            max_selection: 1_024,
            max_commands: 256,
            max_command_depth: 16,
            max_history: 256,
            max_history_per_object: 256,
            max_inspector_nodes: 4_096,
            max_inspector_depth: 64,
        }
    }
}

/// Result of one fully executed action (or a successful no-op).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControllerOutcome {
    /// Whether observable UI/model state changed.
    pub changed: bool,
    /// Whether a field command, `Undo`, or `Redo` changed model data.
    pub model_changed: bool,
    /// Current `UI`/model revision.
    pub revision: u64,
    /// Current model-only revision, used for save acknowledgement.
    pub model_revision: u64,
}

/// A rejected action never advances revisions or changes controller state.
#[derive(Debug)]
pub enum ControllerError {
    /// A semantic event could not be resolved or violates `UI` bounds.
    Action(ActionError),
    /// A reflected field command failed validation/execution.
    Edit(EditorError),
    /// An `Outliner` insertion or lookup failed.
    Outliner(OutlinerError),
    /// The same object identity is already loaded.
    DuplicateObject(SelectionItem),
    /// The requested object is not loaded.
    MissingObject(SelectionItem),
    /// A field action requires a primary object.
    NoPrimary,
    /// The requested row is a non-selectable group.
    UnselectableNode(OutlinerId),
    /// Extend selection requires both its anchor and target to be visible.
    InvisibleRangeEndpoint(OutlinerId),
    /// The requested field does not exist.
    PathNotFound(FieldPath),
    /// The event value cannot safely represent the target's concrete type.
    ValueConversion {
        /// Requested field path.
        path: FieldPath,
        /// Concrete reflected target type.
        target: &'static str,
        /// Conversion failure description.
        reason: &'static str,
    },
    /// A configured resource limit would be exceeded.
    BudgetExceeded {
        /// Resource whose limit was hit.
        resource: &'static str,
        /// Configured limit.
        limit: usize,
    },
    /// A command id is already registered.
    DuplicateCommand(String),
    /// No executable factory is registered for this command ID.
    UnknownCommand(String),
    /// A model revision changed while an asynchronous save was in flight.
    StaleRevision {
        /// Revision acknowledged by the caller.
        saved: u64,
        /// Current model revision.
        current: u64,
    },
    /// A revision counter cannot advance without wrapping.
    RevisionExhausted,
}

impl fmt::Display for ControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Action(error) => error.fmt(f),
            Self::Edit(error) => error.fmt(f),
            Self::Outliner(error) => error.fmt(f),
            Self::DuplicateObject(item) => write!(f, "object {item:?} is already registered"),
            Self::MissingObject(item) => write!(f, "object {item:?} is not registered"),
            Self::NoPrimary => f.write_str("no primary Inspector object is selected"),
            Self::UnselectableNode(id) => write!(f, "Outliner row {} is not selectable", id.raw()),
            Self::InvisibleRangeEndpoint(id) => {
                write!(f, "selection range endpoint {} is not visible", id.raw())
            }
            Self::PathNotFound(path) => write!(f, "field path {path} does not resolve"),
            Self::ValueConversion {
                path,
                target,
                reason,
            } => {
                write!(f, "cannot convert UI value for {path} ({target}): {reason}")
            }
            Self::BudgetExceeded { resource, limit } => {
                write!(f, "{resource} limit {limit} exceeded")
            }
            Self::DuplicateCommand(id) => write!(f, "command {id:?} is already registered"),
            Self::UnknownCommand(id) => write!(f, "command {id:?} is not registered"),
            Self::StaleRevision { saved, current } => {
                write!(
                    f,
                    "saved model revision {saved} does not match current revision {current}"
                )
            }
            Self::RevisionExhausted => f.write_str("controller revision counter exhausted"),
        }
    }
}

impl core::error::Error for ControllerError {}

impl From<ActionError> for ControllerError {
    fn from(error: ActionError) -> Self {
        Self::Action(error)
    }
}

impl From<EditorError> for ControllerError {
    fn from(error: EditorError) -> Self {
        Self::Edit(error)
    }
}

impl From<OutlinerError> for ControllerError {
    fn from(error: OutlinerError) -> Self {
        Self::Outliner(error)
    }
}

type CommandFactory =
    dyn Fn(&EditorController) -> Result<EditorAction, ControllerError> + Send + Sync;

// Capture the target at execution, never resolve it from current selection.
// Payloads are the existing validated, reversible reflection field command.
struct DocumentEdit {
    item: SelectionItem,
    command: FieldEditCommand,
}

/// Owns editor models, selection, projections, and bounded action execution.
///
/// No mutable object/context getters are exposed: every accepted model edit goes
/// through existing reversible reflection field commands. Command
/// factories only receive a shared controller and return one action; controller
/// mutations are therefore still validated and committed by this executor.
pub struct EditorController {
    objects: BTreeMap<SelectionItem, ReflectEditorContext>,
    undo: Vec<DocumentEdit>,
    redo: Vec<DocumentEdit>,
    selection: Selection,
    outliner: Outliner,
    // Parent-before-child insertion order, without the projection's child lists.
    rows: Vec<OutlinerNode>,
    anchor: Option<OutlinerId>,
    commands: BTreeMap<String, Box<CommandFactory>>,
    limits: ControllerLimits,
    revision: u64,
    model_revision: u64,
    saved_model_revision: u64,
}

impl Default for EditorController {
    fn default() -> Self {
        Self::with_limits(ControllerLimits::default())
    }
}

impl EditorController {
    /// Creates a controller with default budgets and no loaded objects.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a controller with explicit budgets. Zero disables that resource.
    #[must_use]
    pub fn with_limits(limits: ControllerLimits) -> Self {
        Self {
            objects: BTreeMap::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            selection: Selection::default(),
            outliner: Outliner::default(),
            rows: Vec::new(),
            anchor: None,
            commands: BTreeMap::new(),
            limits,
            revision: 0,
            model_revision: 0,
            saved_model_revision: 0,
        }
    }

    /// Loads an object without marking authored model data dirty.
    ///
    /// Add an explicit `OutlinerNode` with `set_selection(item)` separately;
    /// object identities are never guessed from `Outliner` numeric row IDs.
    pub fn register_object(
        &mut self,
        item: SelectionItem,
        root: Box<dyn Reflect>,
        registry: TypeRegistry,
    ) -> Result<(), ControllerError> {
        if self.objects.contains_key(&item) {
            return Err(ControllerError::DuplicateObject(item));
        }
        ensure_capacity(self.objects.len(), self.limits.max_objects, "objects")?;
        check_model(&*root, self.limits)?;
        let report = validate(&*root, &registry);
        if !report.is_valid() {
            return Err(EditorError::Validation(report).into());
        }
        self.ensure_revision(false)?;
        self.objects
            .insert(item, ReflectEditorContext::new(root, registry));
        self.advance(false);
        Ok(())
    }

    /// Registers a root/group/object row. Its parent must already be registered.
    /// Incoming child lists are rebuilt from parent identities, never trusted.
    pub fn register_outliner_node(&mut self, node: OutlinerNode) -> Result<(), ControllerError> {
        if self.outliner.node(node.id()).is_some() {
            return Err(OutlinerError::DuplicateNode(node.id()).into());
        }
        ensure_capacity(
            self.rows.len(),
            self.limits.max_outliner_nodes,
            "Outliner nodes",
        )?;
        if let Some(item) = node.selection() {
            self.editor(item)?;
        }
        ensure_bound(
            node.label().len(),
            MAX_ACTION_TEXT_BYTES,
            "Outliner label bytes",
        )?;
        let mut depth = 0;
        let mut parent = node.parent();
        while let Some(id) = parent {
            let ancestor = self
                .outliner
                .node(id)
                .ok_or(OutlinerError::MissingNode(id))?;
            depth += 1;
            ensure_bound(depth, self.limits.max_outliner_depth, "Outliner depth")?;
            parent = ancestor.parent();
        }
        self.ensure_revision(false)?;
        let row = copy_row(&node);
        self.outliner.insert(row.clone())?;
        self.rows.push(row);
        self.advance(false);
        Ok(())
    }

    /// Registers an executable command action factory, not a logging callback.
    ///
    /// Factories may return any action, including another registered command;
    /// indirection is bounded. They must not perform external side effects:
    /// only the returned action is part of this controller's atomic operation.
    pub fn register_command<F>(
        &mut self,
        command_id: impl Into<String>,
        factory: F,
    ) -> Result<(), ControllerError>
    where
        F: Fn(&EditorController) -> Result<EditorAction, ControllerError> + Send + Sync + 'static,
    {
        let command_id = command_id.into();
        ActionTemplate::execute_command(command_id.clone())?;
        if self.commands.contains_key(&command_id) {
            return Err(ControllerError::DuplicateCommand(command_id));
        }
        ensure_capacity(
            self.commands.len(),
            self.limits.max_commands,
            "registered commands",
        )?;
        self.ensure_revision(false)?;
        self.commands.insert(command_id, Box::new(factory));
        self.advance(false);
        Ok(())
    }

    /// Resolves a real semantic binding and executes its resulting action.
    pub fn dispatch(
        &mut self,
        bindings: &ActionBindings,
        target: &str,
        event: &UiEvent,
    ) -> Result<ControllerOutcome, ControllerError> {
        self.execute(bindings.dispatch(target, event)?)
    }

    /// Executes every supported UI action; unknown command ids are errors.
    pub fn execute(
        &mut self,
        mut action: EditorAction,
    ) -> Result<ControllerOutcome, ControllerError> {
        let mut depth = 0;
        while let EditorAction::ExecuteCommand { command_id } = action {
            ActionTemplate::execute_command(command_id.clone())?;
            let factory = self
                .commands
                .get(&command_id)
                .ok_or_else(|| ControllerError::UnknownCommand(command_id.clone()))?;
            ensure_capacity(depth, self.limits.max_command_depth, "command indirections")?;
            depth += 1;
            action = factory(self)?;
        }
        match action {
            EditorAction::SelectOutliner { node, mode } => self.select_outliner(node, mode),
            EditorAction::ToggleOutlinerExpansion { node } => self.toggle_expansion(node),
            EditorAction::EditInspectorField { path, value } => self.edit_field(path, value),
            EditorAction::Undo => self.navigate_history(false),
            EditorAction::Redo => self.navigate_history(true),
            // The loop above resolves every command before any mutation.
            EditorAction::ExecuteCommand { command_id } => {
                Err(ControllerError::UnknownCommand(command_id))
            }
        }
    }

    /// Selects a loaded object directly, independent of Outliner presentation.
    pub fn select(&mut self, item: SelectionItem) -> Result<ControllerOutcome, ControllerError> {
        self.editor(&item)?;
        let mut selection = Selection::default();
        selection.replace(item.clone());
        let anchor = self
            .rows
            .iter()
            .find(|row| row.selection() == Some(&item))
            .map(OutlinerNode::id);
        self.commit_selection(selection, anchor)
    }

    /// Applies one bounded multi-selection operation after validating every
    /// identity. The candidate is committed once, so a GPU box result cannot
    /// expose a partially applied selection or advance the UI revision per hit.
    /// `Extend` uses the supplied ordered items as the replacement range because
    /// a GPU rectangle has no Outliner range endpoints of its own.
    pub fn select_items(
        &mut self,
        items: &[SelectionItem],
        mode: SelectionMode,
    ) -> Result<ControllerOutcome, ControllerError> {
        for item in items {
            self.editor(item)?;
        }
        let mut candidate = self.selection.clone();
        let replace = matches!(mode, SelectionMode::Replace | SelectionMode::Extend);
        if replace {
            candidate.clear();
        }
        for item in items {
            match mode {
                SelectionMode::Replace | SelectionMode::Add | SelectionMode::Extend => {
                    candidate.add(item.clone());
                }
                SelectionMode::Toggle => {
                    if !candidate.remove(item) {
                        candidate.add(item.clone());
                    }
                }
            }
        }
        let anchor = candidate.primary().and_then(|item| {
            self.rows
                .iter()
                .find(|row| row.selection() == Some(item))
                .map(OutlinerNode::id)
        });
        self.commit_selection(candidate, anchor)
    }

    /// Clears the selection and its range anchor.
    pub fn clear_selection(&mut self) -> Result<ControllerOutcome, ControllerError> {
        self.commit_selection(Selection::default(), None)
    }

    /// Returns the authoritative, ordered selection.
    #[must_use]
    pub const fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Returns the authoritative `Outliner` projection; no mutable bypass exists.
    #[must_use]
    pub const fn outliner(&self) -> &Outliner {
        &self.outliner
    }

    /// Borrows every loaded object and its actual schema registry for document
    /// serialization, in deterministic identity order. No reflection clones or
    /// mutable model/history access are exposed.
    pub fn document_objects(
        &self,
    ) -> impl ExactSizeIterator<Item = (&SelectionItem, &dyn Reflect, &TypeRegistry)> {
        self.objects
            .iter()
            .map(|(item, editor)| (item, editor.root(), editor.registry()))
    }

    /// Complete parent-before-child registration order, including collapsed and
    /// hidden rows. Child lists are derived by `Outliner`, not stored here; use
    /// each row's parent identity when restoring a document.
    #[must_use]
    pub fn document_outliner_nodes(&self) -> &[OutlinerNode] {
        &self.rows
    }

    /// Returns the first row representing the primary object, if present.
    #[must_use]
    pub fn selected_outliner(&self) -> Option<OutlinerId> {
        let item = self.selection.primary()?;
        self.rows
            .iter()
            .find(|row| row.selection() == Some(item))
            .map(OutlinerNode::id)
    }

    /// Projects the current primary model with the existing reflection `UI` tree.
    pub fn inspector_projection(&self) -> Result<InspectorProjection, ControllerError> {
        let editor = self.primary_editor()?;
        Ok(project_inspector(editor.root(), editor.registry()))
    }

    /// Alias matching `InspectorSession` terminology.
    pub fn primary_projection(&self) -> Result<InspectorProjection, ControllerError> {
        self.inspector_projection()
    }

    /// Returns the primary root clone for `Reflect::apply` model writeback.
    /// Derived struct clones may be dynamic; do not assume a concrete downcast.
    pub fn primary_snapshot(&self) -> Result<Box<dyn Reflect>, ControllerError> {
        Ok(self.primary_editor()?.root().reflect_clone())
    }

    /// Clones any loaded root without changing selection or its history.
    pub fn snapshot(&self, item: &SelectionItem) -> Result<Box<dyn Reflect>, ControllerError> {
        Ok(self.editor(item)?.root().reflect_clone())
    }

    /// Counts this object's entries within the global undo/redo timeline. These
    /// counts are diagnostic only: navigation always follows document order.
    pub fn history_lengths(&self, item: &SelectionItem) -> Result<(usize, usize), ControllerError> {
        self.editor(item)?;
        Ok((
            self.undo.iter().filter(|entry| &entry.item == item).count(),
            self.redo.iter().filter(|entry| &entry.item == item).count(),
        ))
    }

    /// Number of document commands available to undo and redo, irrespective of
    /// primary selection. Selection/expansion/registration are not model edits.
    #[must_use]
    pub fn document_history_lengths(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }

    /// Revision for rebuilding UI projections, not a save acknowledgement token.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Monotonic authored-model revision; selection/expansion never advance it.
    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        self.model_revision
    }

    /// Whether any model command ran since the last matching save acknowledgement.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.model_revision != self.saved_model_revision
    }

    /// Acknowledges a successful save only if its model revision is still current.
    /// Does not itself save data. Undo back to original values remains dirty.
    pub fn mark_saved(&mut self, model_revision: u64) -> Result<(), ControllerError> {
        if model_revision != self.model_revision {
            return Err(ControllerError::StaleRevision {
                saved: model_revision,
                current: self.model_revision,
            });
        }
        if self.is_dirty() {
            self.ensure_revision(false)?;
            self.saved_model_revision = model_revision;
            self.advance(false);
        }
        Ok(())
    }

    fn editor(&self, item: &SelectionItem) -> Result<&ReflectEditorContext, ControllerError> {
        self.objects
            .get(item)
            .ok_or_else(|| ControllerError::MissingObject(item.clone()))
    }

    fn primary_editor(&self) -> Result<&ReflectEditorContext, ControllerError> {
        self.editor(self.selection.primary().ok_or(ControllerError::NoPrimary)?)
    }

    fn select_outliner(
        &mut self,
        id: OutlinerId,
        mode: SelectionMode,
    ) -> Result<ControllerOutcome, ControllerError> {
        let node = self
            .outliner
            .node(id)
            .ok_or(OutlinerError::MissingNode(id))?;
        let item = node
            .selection()
            .ok_or(ControllerError::UnselectableNode(id))?
            .clone();
        self.editor(&item)?;
        let mut candidate = self.selection.clone();
        let mut anchor = Some(id);
        match mode {
            SelectionMode::Replace => candidate.replace(item),
            SelectionMode::Add => {
                candidate.add(item);
            }
            SelectionMode::Toggle => {
                if !candidate.remove(&item) {
                    candidate.add(item);
                }
            }
            SelectionMode::Extend => {
                let start_id = self.anchor.unwrap_or(id);
                let visible = self
                    .outliner
                    .visible_window(0, self.limits.max_outliner_nodes);
                let start = visible
                    .iter()
                    .position(|node| *node == start_id)
                    .ok_or(ControllerError::InvisibleRangeEndpoint(start_id))?;
                let end = visible
                    .iter()
                    .position(|node| *node == id)
                    .ok_or(ControllerError::InvisibleRangeEndpoint(id))?;
                candidate.clear();
                for row_id in &visible[start.min(end)..=start.max(end)] {
                    if let Some(entry) = self
                        .outliner
                        .node(*row_id)
                        .and_then(OutlinerNode::selection)
                    {
                        self.editor(entry)?;
                        candidate.add(entry.clone());
                        ensure_bound(
                            candidate.items().len(),
                            self.limits.max_selection,
                            "selection",
                        )?;
                    }
                }
                candidate.add(item);
                anchor = Some(start_id);
            }
        }
        self.commit_selection(candidate, anchor)
    }

    fn commit_selection(
        &mut self,
        selection: Selection,
        anchor: Option<OutlinerId>,
    ) -> Result<ControllerOutcome, ControllerError> {
        ensure_bound(
            selection.items().len(),
            self.limits.max_selection,
            "selection",
        )?;
        let changed = self.selection != selection || self.anchor != anchor;
        if changed {
            self.ensure_revision(false)?;
            self.selection = selection;
            self.anchor = anchor;
            self.advance(false);
        }
        Ok(self.outcome(changed, false))
    }

    fn toggle_expansion(&mut self, id: OutlinerId) -> Result<ControllerOutcome, ControllerError> {
        let node = self
            .outliner
            .node(id)
            .ok_or(OutlinerError::MissingNode(id))?;
        let expanded = !node.is_expanded();
        self.ensure_revision(false)?;
        let mut candidate = Outliner::default();
        for row in &self.rows {
            let mut row = copy_row(row);
            if row.id() == id {
                row.set_expanded(expanded);
            }
            candidate.insert(row)?;
        }
        // All fallible work has finished; templates and projection commit together.
        if let Some(row) = self.rows.iter_mut().find(|row| row.id() == id) {
            row.set_expanded(expanded);
        }
        self.outliner = candidate;
        self.advance(false);
        Ok(self.outcome(true, false))
    }

    fn edit_field(
        &mut self,
        path: FieldPath,
        value: ActionValue,
    ) -> Result<ControllerOutcome, ControllerError> {
        ensure_bound(
            path.as_str().len(),
            MAX_ACTION_PATH_BYTES,
            "action path bytes",
        )?;
        ensure_bound(
            path.segments().len(),
            self.limits.max_inspector_depth,
            "action path depth",
        )?;
        let item = self
            .selection
            .primary()
            .ok_or(ControllerError::NoPrimary)?
            .clone();
        let editor = self.editor(&item)?;
        // FieldPath's ParsedPath is crate-private; reparse its bounded canonical text.
        let parsed = ParsedPath::parse(path.as_str())
            .map_err(|_| ControllerError::PathNotFound(path.clone()))?;
        let target = reflect_path(editor.root(), &parsed)
            .ok_or_else(|| ControllerError::PathNotFound(path.clone()))?;
        ensure_editable_ancestors(editor, parsed, &path)?;
        let converted = convert_value(target, &value, &path)?;
        if same_leaf(target, &*converted) {
            // Read-only and validation errors remain errors for identical values.
            editor.prepare_edit(path, converted)?;
            return Ok(self.outcome(false, false));
        }
        // FieldEditCommand explicitly does not merge. Only the applied prefix
        // survives if this commit creates a new document branch.
        ensure_capacity(self.undo.len(), self.limits.max_history, "document history")?;
        ensure_capacity(
            self.undo.iter().filter(|entry| entry.item == item).count(),
            self.limits.max_history_per_object,
            "object commands in document history",
        )?;
        let mut command = editor.prepare_edit(path, converted)?;
        self.ensure_revision(true)?;
        let context = self
            .objects
            .get_mut(&item)
            .ok_or_else(|| ControllerError::MissingObject(item.clone()))?;
        command.apply(context).map_err(EditorError::Command)?;
        // Rejected commands retain both branches. Commit history/revisions only
        // after the existing command has applied and validated atomically.
        self.redo.clear();
        self.undo.push(DocumentEdit { item, command });
        self.advance(true);
        Ok(self.outcome(true, true))
    }

    fn navigate_history(&mut self, redo: bool) -> Result<ControllerOutcome, ControllerError> {
        let source = if redo { &self.redo } else { &self.undo };
        let Some(entry) = source.last() else {
            return Ok(self.outcome(false, false));
        };
        self.editor(&entry.item)?;
        self.ensure_revision(true)?;
        let (source, destination) = if redo {
            (&mut self.redo, &mut self.undo)
        } else {
            (&mut self.undo, &mut self.redo)
        };
        // Keep the entry in place until command success. Failed apply/undo
        // retains its exact position and neither revision counter advances.
        if let Some(entry) = source.last_mut() {
            let context = self
                .objects
                .get_mut(&entry.item)
                .ok_or_else(|| ControllerError::MissingObject(entry.item.clone()))?;
            if redo {
                entry.command.apply(context)
            } else {
                entry.command.undo(context)
            }
            .map_err(EditorError::Command)?;
        }
        if let Some(entry) = source.pop() {
            destination.push(entry);
        }
        self.advance(true);
        Ok(self.outcome(true, true))
    }

    fn ensure_revision(&self, model: bool) -> Result<(), ControllerError> {
        if self.revision == u64::MAX || (model && self.model_revision == u64::MAX) {
            return Err(ControllerError::RevisionExhausted);
        }
        Ok(())
    }

    fn advance(&mut self, model: bool) {
        self.revision += 1;
        if model {
            self.model_revision += 1;
        }
    }

    fn outcome(&self, changed: bool, model_changed: bool) -> ControllerOutcome {
        ControllerOutcome {
            changed,
            model_changed,
            revision: self.revision,
            model_revision: self.model_revision,
        }
    }
}

fn ensure_capacity(
    used: usize,
    limit: usize,
    resource: &'static str,
) -> Result<(), ControllerError> {
    if used >= limit {
        Err(ControllerError::BudgetExceeded { resource, limit })
    } else {
        Ok(())
    }
}

fn ensure_bound(used: usize, limit: usize, resource: &'static str) -> Result<(), ControllerError> {
    if used > limit {
        Err(ControllerError::BudgetExceeded { resource, limit })
    } else {
        Ok(())
    }
}

fn copy_row(source: &OutlinerNode) -> OutlinerNode {
    let mut node = OutlinerNode::new(source.id(), source.parent(), source.label());
    node.set_expanded(source.is_expanded());
    node.set_visible(source.is_visible());
    if let Some(item) = source.selection() {
        node.set_selection(item.clone());
    }
    node
}

fn ensure_editable_ancestors(
    editor: &ReflectEditorContext,
    mut parsed: ParsedPath,
    requested: &FieldPath,
) -> Result<(), ControllerError> {
    while let Some((parent_path, access)) = parsed.split_terminal() {
        if let Access::Field(name) = access
            && let Some(parent) = reflect_path(editor.root(), &parent_path)
            && let Some(metadata) = editor
                .registry()
                .get_with_name(parent.type_name())
                .and_then(TypeRegistration::data::<TypeMetadata>)
            && metadata
                .field(name)
                .is_some_and(prism_reflect::FieldMetadata::is_readonly)
        {
            return Err(EditorError::ReadOnly(requested.clone()).into());
        }
        parsed = parent_path;
    }
    Ok(())
}

// Guard BEFORE recursive inspector projection, validation, or reflect_clone.
// Count all model nodes, even hidden ones, so a hidden subtree cannot evade bounds.
fn check_model(root: &dyn Reflect, limits: ControllerLimits) -> Result<(), ControllerError> {
    fn visit(
        value: &dyn Reflect,
        depth: usize,
        nodes: &mut usize,
        limits: ControllerLimits,
    ) -> Result<(), ControllerError> {
        ensure_bound(depth, limits.max_inspector_depth, "Inspector depth")?;
        ensure_capacity(*nodes, limits.max_inspector_nodes, "Inspector nodes")?;
        *nodes += 1;
        if let Some(text) = value.downcast_ref::<String>() {
            ensure_bound(text.len(), MAX_ACTION_TEXT_BYTES, "model text bytes")?;
        }
        if value.downcast_ref::<f64>().is_some_and(|v| !v.is_finite())
            || value.downcast_ref::<f32>().is_some_and(|v| !v.is_finite())
        {
            return Err(ControllerError::ValueConversion {
                path: FieldPath::root(),
                target: value.type_name(),
                reason: "non-finite model value",
            });
        }
        let mut child = |value: &dyn Reflect| visit(value, depth + 1, nodes, limits);
        match value.reflect_ref() {
            ReflectRef::Struct(v) => {
                for i in 0..v.field_count() {
                    if let Some(v) = v.field_at(i) {
                        child(v)?;
                    }
                }
            }
            ReflectRef::TupleStruct(v) => {
                for i in 0..v.field_count() {
                    if let Some(v) = v.field(i) {
                        child(v)?;
                    }
                }
            }
            ReflectRef::Enum(v) => {
                for i in 0..v.field_count() {
                    if let Some(v) = v.field_at(i) {
                        child(v)?;
                    }
                }
            }
            ReflectRef::List(v) => {
                for v in v.iter_reflect() {
                    child(v)?;
                }
            }
            ReflectRef::Array(v) => {
                for v in v.iter_reflect() {
                    child(v)?;
                }
            }
            ReflectRef::Map(v) => {
                for (key, v) in v.iter_reflect() {
                    child(key)?;
                    child(v)?;
                }
            }
            ReflectRef::Set(v) => {
                for v in v.iter_reflect() {
                    child(v)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    visit(root, 0, &mut 0, limits)
}

// Integers accept checked signed/unsigned integers or exact integer text. Float
// to integer coercion is deliberately rejected. Floats accept finite numbers or
// numeric text; integer -> float requires exact integer representation. f64 ->
// f32 uses normal rounding, but overflow and nonzero -> zero underflow are errors.
fn convert_value(
    target: &dyn Reflect,
    value: &ActionValue,
    path: &FieldPath,
) -> Result<Box<dyn Reflect>, ControllerError> {
    let invalid = |reason| ControllerError::ValueConversion {
        path: path.clone(),
        target: target.type_name(),
        reason,
    };
    macro_rules! integer {
        ($($ty:ty),* $(,)?) => { $(
            if target.is::<$ty>() {
                let result = if let Some(v) = value.as_integer() {
                    <$ty>::try_from(v).ok()
                } else if let Some(v) = value.as_unsigned() {
                    <$ty>::try_from(v).ok()
                } else if let Some(v) = value.as_text() {
                    v.parse::<$ty>().ok()
                } else { None };
                return result.map(|v| Box::new(v) as Box<dyn Reflect>)
                    .ok_or_else(|| invalid("expected an in-range integer or integer text"));
            }
        )* };
    }
    integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
    if target.is::<f32>() || target.is::<f64>() {
        let precision = if target.is::<f32>() { 24 } else { 53 };
        let magnitude = value
            .as_unsigned()
            .or_else(|| value.as_integer().map(i64::unsigned_abs));
        if magnitude
            .is_some_and(|v| v != 0 && 64 - v.leading_zeros() - v.trailing_zeros() > precision)
        {
            return Err(invalid(
                "integer cannot be represented exactly by the target float",
            ));
        }
        let number = if let Some(v) = value.as_float() {
            Some(v)
        } else if let Some(v) = value.as_text() {
            v.parse::<f64>().ok()
        } else if let Some(v) = value.as_integer() {
            Some(v as f64)
        } else {
            value.as_unsigned().map(|v| v as f64)
        }
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid("expected a finite number or numeric text"))?;
        if target.is::<f64>() {
            return Ok(Box::new(number));
        }
        let narrowed = number as f32;
        if !narrowed.is_finite() || (number != 0.0 && narrowed == 0.0) {
            return Err(invalid("number overflows or underflows f32"));
        }
        return Ok(Box::new(narrowed));
    }
    if target.is::<bool>() {
        return value
            .as_bool()
            .or_else(|| value.as_text()?.parse::<bool>().ok())
            .map(|v| Box::new(v) as Box<dyn Reflect>)
            .ok_or_else(|| invalid("expected a boolean or true/false text"));
    }
    if target.is::<String>() {
        return value
            .as_text()
            .map(|v| Box::new(v.to_owned()) as Box<dyn Reflect>)
            .ok_or_else(|| invalid("expected text"));
    }
    if target.is::<char>() {
        return value
            .as_text()
            .and_then(|text| {
                let mut chars = text.chars();
                let first = chars.next()?;
                chars.next().is_none().then_some(first)
            })
            .map(|v| Box::new(v) as Box<dyn Reflect>)
            .ok_or_else(|| invalid("expected exactly one Unicode scalar"));
    }
    Err(invalid("UI values cannot replace this reflected type"))
}

fn same_leaf(left: &dyn Reflect, right: &dyn Reflect) -> bool {
    macro_rules! equal {
        ($($ty:ty),* $(,)?) => { $(
            if let (Some(a), Some(b)) = (left.downcast_ref::<$ty>(), right.downcast_ref::<$ty>()) { return a == b; }
        )* };
    }
    equal!(bool, char, String, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
    if let (Some(a), Some(b)) = (left.downcast_ref::<f32>(), right.downcast_ref::<f32>()) {
        return a.to_bits() == b.to_bits();
    }
    if let (Some(a), Some(b)) = (left.downcast_ref::<f64>(), right.downcast_ref::<f64>()) {
        return a.to_bits() == b.to_bits();
    }
    false
}
