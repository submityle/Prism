//! Selection-aware Inspector sessions.

use crate::{project_inspector, EditorError, FieldPath, InspectorProjection, ReflectEditor};
use prism_editor_core::{Selection, SelectionItem};
use prism_reflect::{Reflect, TypeRegistry};
use std::collections::BTreeMap;
use std::fmt;

/// Errors raised while coordinating selection and Inspector edits.
#[derive(Debug)]
pub enum InspectorSessionError {
    /// An object with the same stable selection identity is already registered.
    DuplicateObject(SelectionItem),
    /// The requested stable selection identity is not registered.
    MissingObject(SelectionItem),
    /// No object is currently selected as the primary Inspector target.
    NoPrimary,
    /// A field edit, undo, or redo failed.
    Edit(EditorError),
}

impl fmt::Display for InspectorSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateObject(item) => {
                write!(formatter, "object {item:?} is already registered")
            }
            Self::MissingObject(item) => write!(formatter, "object {item:?} is not registered"),
            Self::NoPrimary => formatter.write_str("no primary Inspector object is selected"),
            Self::Edit(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for InspectorSessionError {}

impl From<EditorError> for InspectorSessionError {
    fn from(error: EditorError) -> Self {
        Self::Edit(error)
    }
}

/// A bounded selection-to-Inspector coordinator.
///
/// Each registered object owns its reflection editor and command history. The
/// session only coordinates stable selection identities; it does not prescribe
/// an ECS, runtime, or persistence implementation.
#[derive(Default)]
pub struct InspectorSession {
    objects: BTreeMap<SelectionItem, ReflectEditor>,
    selection: Selection,
}

impl InspectorSession {
    /// Creates an empty session.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one reflected object under a stable selection identity.
    pub fn register(
        &mut self,
        item: SelectionItem,
        root: Box<dyn Reflect>,
        registry: TypeRegistry,
    ) -> Result<(), InspectorSessionError> {
        if self.objects.contains_key(&item) {
            return Err(InspectorSessionError::DuplicateObject(item));
        }
        self.objects
            .insert(item, ReflectEditor::new(root, registry));
        Ok(())
    }

    /// Removes an object and any matching selection entries.
    pub fn unregister(&mut self, item: &SelectionItem) -> Result<(), InspectorSessionError> {
        self.objects
            .remove(item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?;
        self.selection.remove(item);
        Ok(())
    }

    /// Returns the current selection.
    #[must_use]
    pub const fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Replaces the current selection with one registered object.
    pub fn select(&mut self, item: SelectionItem) -> Result<(), InspectorSessionError> {
        self.ensure_registered(&item)?;
        self.selection.replace(item);
        Ok(())
    }

    /// Adds one registered object to the current multi-selection.
    pub fn add_to_selection(&mut self, item: SelectionItem) -> Result<bool, InspectorSessionError> {
        self.ensure_registered(&item)?;
        Ok(self.selection.add(item))
    }

    /// Clears the current selection.
    pub fn clear_selection(&mut self) -> bool {
        self.selection.clear()
    }

    /// Projects the primary object into the UI-neutral Inspector tree.
    pub fn primary_projection(&self) -> Result<InspectorProjection, InspectorSessionError> {
        let item = self.primary_item()?;
        let editor = self
            .objects
            .get(item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?;
        Ok(project_inspector(
            editor.context().root(),
            editor.context().registry(),
        ))
    }

    /// Replaces a field on the primary object and records the edit for undo.
    pub fn edit_primary(
        &mut self,
        path: FieldPath,
        value: Box<dyn Reflect>,
    ) -> Result<(), InspectorSessionError> {
        let item = self.primary_item()?.clone();
        self.objects
            .get_mut(&item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?
            .edit(path, value)
            .map(|_| ())
            .map_err(Into::into)
    }

    /// Undoes the most recent edit on the primary object.
    pub fn undo_primary(&mut self) -> Result<bool, InspectorSessionError> {
        let item = self.primary_item()?.clone();
        self.objects
            .get_mut(&item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?
            .undo()
            .map_err(Into::into)
    }

    /// Redoes the most recently undone edit on the primary object.
    pub fn redo_primary(&mut self) -> Result<bool, InspectorSessionError> {
        let item = self.primary_item()?.clone();
        self.objects
            .get_mut(&item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?
            .redo()
            .map_err(Into::into)
    }

    /// Clones the primary reflected root for model-layer writeback.
    pub fn primary_snapshot(&self) -> Result<Box<dyn Reflect>, InspectorSessionError> {
        let item = self.primary_item()?;
        let editor = self
            .objects
            .get(item)
            .ok_or_else(|| InspectorSessionError::MissingObject(item.clone()))?;
        Ok(editor.context().root().reflect_clone())
    }

    fn primary_item(&self) -> Result<&SelectionItem, InspectorSessionError> {
        self.selection
            .primary()
            .ok_or(InspectorSessionError::NoPrimary)
    }

    fn ensure_registered(&self, item: &SelectionItem) -> Result<(), InspectorSessionError> {
        if self.objects.contains_key(item) {
            Ok(())
        } else {
            Err(InspectorSessionError::MissingObject(item.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_reflect::{Reflect, TypeRegistry};

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct Model {
        speed: i32,
        label: String,
    }

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::default();
        registry.register::<Model>();
        registry
    }

    #[test]
    fn selection_projection_edit_undo_redo_and_snapshot_form_one_path() {
        let item = SelectionItem::entity(7);
        let mut session = InspectorSession::new();
        session
            .register(
                item.clone(),
                Box::new(Model {
                    speed: 10,
                    label: String::from("ship"),
                }),
                registry(),
            )
            .expect("register");
        session.select(item).expect("select");

        let projection = session.primary_projection().expect("projection");
        assert_eq!(projection.root.children.len(), 2);
        session
            .edit_primary(FieldPath::parse("speed").expect("path"), Box::new(42_i32))
            .expect("edit");
        let snapshot = session.primary_snapshot().expect("snapshot");
        let mut writeback = Model {
            speed: 0,
            label: String::new(),
        };
        writeback.apply(&*snapshot).expect("writeback");
        assert_eq!(writeback.speed, 42);
        assert!(session.undo_primary().expect("undo"));
        let snapshot = session.primary_snapshot().expect("snapshot");
        writeback.apply(&*snapshot).expect("writeback");
        assert_eq!(writeback.speed, 10);
        assert!(session.redo_primary().expect("redo"));
        let snapshot = session.primary_snapshot().expect("snapshot");
        writeback.apply(&*snapshot).expect("writeback");
        assert_eq!(writeback.speed, 42);
    }

    #[test]
    fn selection_rejects_unregistered_objects_and_clears_on_unregister() {
        let item = SelectionItem::entity(9);
        let mut session = InspectorSession::new();
        assert!(matches!(
            session.select(item.clone()),
            Err(InspectorSessionError::MissingObject(_))
        ));
        session
            .register(
                item.clone(),
                Box::new(Model {
                    speed: 1,
                    label: String::new(),
                }),
                registry(),
            )
            .expect("register");
        session.select(item.clone()).expect("select");
        session.unregister(&item).expect("unregister");
        assert!(session.selection().primary().is_none());
    }
}
