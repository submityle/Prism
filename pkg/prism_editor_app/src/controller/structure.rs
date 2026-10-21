//! Atomic structural authoring with ownership-moving inverse history.

use super::*;
use prism_editor_core::SelectionKind;

/// One reflected object and its explicitly associated primary Outliner row.
pub struct ControllerObject {
    /// Stable authored identity; never inferred from row numbers.
    pub item: SelectionItem,
    /// Authoritative reflected data, moved into the document on success.
    pub root: Box<dyn Reflect>,
    /// Reflection metadata and schema-owned admission rules.
    pub registry: TypeRegistry,
    /// Parent-before-child row whose selection must equal `item`.
    pub row: OutlinerNode,
}

/// A factory result executed on the normal controller command path.
pub enum ControllerAuthoringCommand {
    /// Creates and selects the objects as one document command.
    Create(Vec<ControllerObject>),
    /// Deletes objects, their row subtrees and all affected aliases atomically.
    Delete(Vec<SelectionItem>),
}

pub(super) struct StoredStructure {
    // Desired state: Some installs an absent object; None removes a live one.
    pub(super) objects: BTreeMap<SelectionItem, Option<ReflectEditorContext>>,
    remove_rows: BTreeSet<OutlinerId>,
    insert_rows: Vec<(usize, OutlinerNode)>,
    selection: Selection,
    anchor: Option<OutlinerId>,
}

impl EditorController {
    /// Registers a structural command factory. Like action factories it must
    /// have no external side effects; registration does not edit the document.
    pub fn register_authoring_command<F>(
        &mut self,
        command_id: impl Into<String>,
        factory: F,
    ) -> Result<(), ControllerError>
    where
        F: Fn(&EditorController) -> Result<ControllerAuthoringCommand, ControllerError>
            + Send
            + Sync
            + 'static,
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
        self.commands
            .insert(command_id, CommandFactory::Authoring(Arc::new(factory)));
        self.advance(false);
        Ok(())
    }

    /// Whether a host explicitly installed this executable command.
    #[must_use]
    pub fn has_command(&self, command_id: &str) -> bool {
        self.commands.contains_key(command_id)
    }

    /// Allocates neither data nor an ID: proposes a pair above all accepted
    /// entity/row IDs in this controller's lifetime, including discarded redo.
    /// Reopening a document reconstructs this bound from its loaded identities.
    pub fn next_entity_identity(&self) -> Result<(SelectionItem, OutlinerId), ControllerError> {
        let id = self
            .identity_high_water
            .checked_add(1)
            .ok_or(ControllerError::RevisionExhausted)?;
        Ok((SelectionItem::entity(id), OutlinerId::from_raw(id)))
    }

    /// Creates a bounded, validated batch as one reversible model change.
    /// Empty batches are no-ops. Failed batches leave both histories intact.
    pub fn create_objects(
        &mut self,
        objects: Vec<ControllerObject>,
    ) -> Result<ControllerOutcome, ControllerError> {
        self.ensure_idle()?;
        if objects.is_empty() {
            return Ok(self.outcome(false, false));
        }
        ensure_bound(
            objects.len(),
            self.limits.max_objects.saturating_sub(self.objects.len()),
            "new objects",
        )?;
        ensure_bound(
            objects.len(),
            self.limits
                .max_outliner_nodes
                .saturating_sub(self.rows.len()),
            "new Outliner nodes",
        )?;
        ensure_bound(objects.len(), self.limits.max_selection, "selection")?;
        let mut change = StoredStructure {
            objects: BTreeMap::new(),
            remove_rows: BTreeSet::new(),
            insert_rows: Vec::with_capacity(objects.len()),
            selection: Selection::default(),
            anchor: None,
        };
        for (offset, object) in objects.into_iter().enumerate() {
            let ControllerObject {
                item,
                root,
                registry,
                row,
            } = object;
            if self.objects.contains_key(&item) || change.objects.contains_key(&item) {
                return Err(ControllerError::DuplicateObject(item));
            }
            if row.selection() != Some(&item) {
                return Err(ControllerError::ObjectValidation {
                    item,
                    reason: "created row must select its associated object".into(),
                });
            }
            check_model(&*root, self.limits)?;
            let report = validate(&*root, &registry);
            if !report.is_valid() {
                return Err(EditorError::Validation(report).into());
            }
            validate_object(&item, &*root, &registry, &[])?;
            change.selection.add(item.clone());
            change.anchor = Some(row.id());
            change
                .insert_rows
                .push((self.rows.len() + offset, copy_row(&row)));
            change
                .objects
                .insert(item, Some(ReflectEditorContext::new(root, registry)));
        }
        self.commit_structure(change)
    }

    /// Deletes the alias/subtree closure without cloning any reflected root.
    /// Missing or duplicate requested identities reject the complete command.
    pub fn delete_objects(
        &mut self,
        items: &[SelectionItem],
    ) -> Result<ControllerOutcome, ControllerError> {
        self.ensure_idle()?;
        if items.is_empty() {
            return Ok(self.outcome(false, false));
        }
        ensure_bound(items.len(), self.limits.max_objects, "delete objects")?;
        let mut affected = BTreeSet::new();
        let mut pending_items = VecDeque::new();
        for item in items {
            self.editor(item)?;
            if !affected.insert(item.clone()) {
                return Err(ControllerError::EditConflict);
            }
            pending_items.push_back(item.clone());
        }
        let mut aliases: BTreeMap<&SelectionItem, Vec<OutlinerId>> = BTreeMap::new();
        for row in &self.rows {
            if let Some(item) = row.selection() {
                aliases.entry(item).or_default().push(row.id());
            }
        }
        let mut removed = BTreeSet::new();
        let mut pending_rows = VecDeque::new();
        // Every item/row is queued once, including aliases outside the first
        // subtree. Complexity follows the metadata graph, not repeated scans.
        while !pending_items.is_empty() || !pending_rows.is_empty() {
            while let Some(item) = pending_items.pop_front() {
                if let Some(rows) = aliases.get(&item) {
                    for id in rows {
                        if removed.insert(*id) {
                            pending_rows.push_back(*id);
                        }
                    }
                }
            }
            if let Some(id) = pending_rows.pop_front() {
                let row = self
                    .outliner
                    .node(id)
                    .ok_or(OutlinerError::MissingNode(id))?;
                if let Some(item) = row.selection()
                    && affected.insert(item.clone())
                {
                    pending_items.push_back(item.clone());
                }
                for child in row.children() {
                    if removed.insert(*child) {
                        pending_rows.push_back(*child);
                    }
                }
            }
        }
        // Filter once instead of retaining the complete selection once for
        // each deleted object. Preserve insertion order and an explicit primary
        // that may differ from the last selected item.
        let mut selection = Selection::default();
        for item in self.selection.items() {
            if !affected.contains(item) {
                selection.add(item.clone());
            }
        }
        if let Some(primary) = self.selection.primary().filter(|p| !affected.contains(*p)) {
            selection.add(primary.clone());
        }
        let anchor = self.anchor.filter(|id| !removed.contains(id));
        self.commit_structure(StoredStructure {
            objects: affected.into_iter().map(|item| (item, None)).collect(),
            remove_rows: removed,
            insert_rows: Vec::new(),
            selection,
            anchor,
        })
    }

    fn commit_structure(
        &mut self,
        mut change: StoredStructure,
    ) -> Result<ControllerOutcome, ControllerError> {
        self.ensure_revision(true)?;
        self.ensure_shape_revision()?;
        ensure_capacity(self.undo.len(), self.limits.max_history, "document history")?;
        self.check_structure_history_budget(&change)?;
        self.exchange_structure(&mut change)?;
        self.redo.clear();
        self.undo.push(DocumentEdit {
            fields: Vec::new(),
            structure: Some(change),
            costs: CostPatch::default(),
        });
        self.advance(true);
        self.shape_revision += 1;
        // A field-only delta cannot describe removed roots or changed row
        // topology. Invalidate earlier cursors; subsequent scalar edits can
        // start a new bounded journal at this structural revision.
        self.changes.clear();
        Ok(self.outcome(true, true))
    }

    fn check_structure_history_budget(
        &self,
        change: &StoredStructure,
    ) -> Result<(), ControllerError> {
        // A batch may touch thousands of objects. Do not run `touches` on
        // every retained batch once per candidate object (quadratic in batch
        // size). Count intersections in one pass per retained undo entry.
        // Redo is deliberately excluded: successful branching discards it.
        let mut counts: BTreeMap<&SelectionItem, usize> =
            change.objects.keys().map(|item| (item, 0)).collect();
        ensure_capacity(
            0,
            self.limits.max_history_per_object,
            "object commands in document history",
        )?;
        for entry in &self.undo {
            // Multiple scalar fields in one entry count as ONE object command.
            let touched: BTreeSet<&SelectionItem> = entry
                .fields
                .iter()
                .map(|field| &field.item)
                .chain(entry.structure.iter().flat_map(|s| s.objects.keys()))
                .filter(|item| counts.contains_key(*item))
                .collect();
            for item in touched {
                let count = counts.get_mut(item).expect("candidate intersection");
                *count += 1;
                ensure_capacity(
                    *count,
                    self.limits.max_history_per_object,
                    "object commands in document history",
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn ensure_shape_revision(&self) -> Result<(), ControllerError> {
        if self.shape_revision == u64::MAX {
            Err(ControllerError::RevisionExhausted)
        } else {
            Ok(())
        }
    }

    pub(super) fn observe_identity(&mut self, item: &SelectionItem) {
        if let SelectionKind::Entity(id) = item.kind() {
            self.identity_high_water = self.identity_high_water.max(*id);
        }
    }

    // Validate the complete candidate, then exchange ownership with the inverse
    // record. A rejected undo does not consume or alter its retained entry.
    pub(super) fn exchange_structure(
        &mut self,
        change: &mut StoredStructure,
    ) -> Result<(), ControllerError> {
        let mut object_count = self.objects.len();
        for (item, desired) in &change.objects {
            match (desired.is_some(), self.objects.contains_key(item)) {
                (true, true) => return Err(ControllerError::DuplicateObject(item.clone())),
                (false, false) => return Err(ControllerError::MissingObject(item.clone())),
                (true, false) => object_count += 1,
                (false, true) => object_count -= 1,
            }
        }
        ensure_bound(object_count, self.limits.max_objects, "objects")?;
        for id in &change.remove_rows {
            if self.outliner.node(*id).is_none() {
                return Err(OutlinerError::MissingNode(*id).into());
            }
        }
        let final_len = self.rows.len() - change.remove_rows.len() + change.insert_rows.len();
        ensure_bound(final_len, self.limits.max_outliner_nodes, "Outliner nodes")?;
        let mut previous = None;
        for (index, _) in &change.insert_rows {
            if *index >= final_len || previous.is_some_and(|last| last >= *index) {
                return Err(ControllerError::EditConflict);
            }
            previous = Some(*index);
        }
        let object_exists = |item: &SelectionItem| {
            change
                .objects
                .get(item)
                .map_or_else(|| self.objects.contains_key(item), Option::is_some)
        };
        let mut kept = self
            .rows
            .iter()
            .filter(|row| !change.remove_rows.contains(&row.id()));
        let mut inserts = change.insert_rows.iter().peekable();
        let mut rows = Vec::with_capacity(final_len);
        let mut outliner = Outliner::default();
        let mut depths = BTreeMap::new();
        for index in 0..final_len {
            let row = if inserts
                .peek()
                .is_some_and(|(position, _)| *position == index)
            {
                &inserts.next().expect("peeked insertion").1
            } else {
                kept.next().ok_or(ControllerError::EditConflict)?
            };
            ensure_bound(
                row.label().len(),
                MAX_ACTION_TEXT_BYTES,
                "Outliner label bytes",
            )?;
            if let Some(item) = row.selection()
                && !object_exists(item)
            {
                return Err(ControllerError::MissingObject(item.clone()));
            }
            let depth = match row.parent() {
                Some(parent) => {
                    depths
                        .get(&parent)
                        .copied()
                        .ok_or(OutlinerError::MissingNode(parent))?
                        + 1
                }
                None => 0usize,
            };
            ensure_bound(depth, self.limits.max_outliner_depth, "Outliner depth")?;
            outliner.insert(copy_row(row))?;
            depths.insert(row.id(), depth);
            rows.push(copy_row(row));
        }
        ensure_bound(
            change.selection.items().len(),
            self.limits.max_selection,
            "selection",
        )?;
        for item in change.selection.items() {
            if !object_exists(item) {
                return Err(ControllerError::MissingObject(item.clone()));
            }
        }
        if let Some(id) = change.anchor
            && outliner.node(id).is_none()
        {
            return Err(OutlinerError::MissingNode(id).into());
        }
        let inverse_rows = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| change.remove_rows.contains(&row.id()))
            .map(|(index, row)| (index, copy_row(row)))
            .collect();
        let inverse_remove = change.insert_rows.iter().map(|(_, row)| row.id()).collect();
        let mut costs = self.prepare_costs(
            change
                .objects
                .iter()
                .map(|(item, desired)| (item, desired.as_ref().map(ReflectEditorContext::root))),
        )?;
        let totals = self.check_costs(&costs)?;

        // No fallible operation follows this publication boundary. Roots move,
        // preserving concrete types and allocations through delete/undo/redo.
        for (item, desired) in &mut change.objects {
            if let Some(editor) = desired.take() {
                self.observe_identity(item);
                self.objects.insert(item.clone(), editor);
            } else {
                *desired = self.objects.remove(item);
            }
        }
        for row in &rows {
            self.identity_high_water = self.identity_high_water.max(row.id().raw());
        }
        self.exchange_costs(&mut costs, totals);
        core::mem::swap(&mut self.selection, &mut change.selection);
        core::mem::swap(&mut self.anchor, &mut change.anchor);
        change.insert_rows = inverse_rows;
        change.remove_rows = inverse_remove;
        self.rows = rows;
        self.outliner = outliner;
        Ok(())
    }
}
