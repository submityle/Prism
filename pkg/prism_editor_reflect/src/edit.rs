//! Reversible reflection-backed field commands.

use crate::{validate, FieldPath, FieldPathError, ValidationReport};
use prism_editor_core::{
    AffectSet, Command, CommandError, CommandStack, CommandStatus, EditContext, MergeResult,
};
use prism_reflect::{
    reflect_path, reflect_path_mut, Access, ApplyError, FieldMetadata, Reflect, TypeMetadata,
    TypeRegistry,
};
use std::fmt;

/// Errors raised while preparing or executing reflection-backed edits.
#[derive(Debug)]
pub enum EditorError {
    /// The requested path is syntactically invalid.
    InvalidPath(FieldPathError),
    /// The path does not resolve against the current root shape.
    PathNotFound(FieldPath),
    /// The terminal field is marked read-only in reflection metadata.
    ReadOnly(FieldPath),
    /// The replacement value cannot be applied to the target field.
    Apply { path: FieldPath, error: ApplyError },
    /// The candidate root violates one or more reflection metadata rules.
    Validation(ValidationReport),
    /// The generic editor kernel rejected an already-created command.
    Command(CommandError),
}

impl fmt::Display for EditorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(error) => error.fmt(formatter),
            Self::PathNotFound(path) => write!(formatter, "field path {path} does not resolve"),
            Self::ReadOnly(path) => write!(formatter, "field path {path} is read-only"),
            Self::Apply { path, error } => write!(formatter, "cannot edit {path}: {error}"),
            Self::Validation(report) => write!(formatter, "edit violates validation: {report}"),
            Self::Command(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EditorError {}

/// The command context owned by a reflection editor.
pub struct ReflectEditorContext {
    root: Box<dyn Reflect>,
    registry: TypeRegistry,
}

impl ReflectEditorContext {
    /// Create a context around a reflected root and its type registry.
    #[must_use]
    pub fn new(root: Box<dyn Reflect>, registry: TypeRegistry) -> Self {
        Self { root, registry }
    }

    /// Borrow the current reflected root.
    #[must_use]
    pub fn root(&self) -> &dyn Reflect {
        &*self.root
    }

    /// Mutably borrow the current reflected root for non-command integration.
    pub fn root_mut(&mut self) -> &mut dyn Reflect {
        &mut *self.root
    }

    /// Borrow the reflection type registry.
    #[must_use]
    pub fn registry(&self) -> &TypeRegistry {
        &self.registry
    }

    /// Return the current metadata validation report.
    #[must_use]
    pub fn validation(&self) -> ValidationReport {
        validate(self.root(), &self.registry)
    }

    /// Prepare a field edit by capturing the current value and validating the
    /// candidate root before a command is handed to the editor kernel.
    pub fn prepare_edit(
        &self,
        path: FieldPath,
        value: Box<dyn Reflect>,
    ) -> Result<FieldEditCommand, EditorError> {
        self.ensure_editable(&path)?;
        let before = reflect_path(self.root(), path.parsed())
            .ok_or_else(|| EditorError::PathNotFound(path.clone()))?
            .reflect_clone();
        let mut candidate = self.root.reflect_clone();
        apply_to_root(&mut *candidate, &self.registry, &path, &*value, false)?;
        let report = validate(&*candidate, &self.registry);
        if !report.is_valid() {
            return Err(EditorError::Validation(report));
        }
        Ok(FieldEditCommand::new(path, before, value))
    }

    fn ensure_editable(&self, path: &FieldPath) -> Result<(), EditorError> {
        if reflect_path(self.root(), path.parsed()).is_none() {
            return Err(EditorError::PathNotFound(path.clone()));
        }
        if terminal_metadata(self.root(), &self.registry, path)
            .is_some_and(|metadata| metadata.is_readonly())
        {
            return Err(EditorError::ReadOnly(path.clone()));
        }
        Ok(())
    }

    fn apply_command_value(
        &mut self,
        path: &FieldPath,
        value: &dyn Reflect,
    ) -> Result<(), CommandError> {
        apply_to_root(&mut *self.root, &self.registry, path, value, true)
            .map_err(|error| CommandError::new(error.to_string()))
    }
}

impl EditContext for ReflectEditorContext {}

/// A command that replaces one reflected field and restores it on undo.
pub struct FieldEditCommand {
    path: FieldPath,
    before: Box<dyn Reflect>,
    after: Box<dyn Reflect>,
    label: String,
    affected: AffectSet,
}

impl FieldEditCommand {
    /// Create a command from a captured old value and a replacement value.
    #[must_use]
    pub fn new(path: FieldPath, before: Box<dyn Reflect>, after: Box<dyn Reflect>) -> Self {
        let mut affected = AffectSet::new();
        affected.insert("reflect".to_string());
        affected.insert(format!("reflect:{}", path.as_str()));
        let label = if path.is_root() {
            String::from("Edit root")
        } else {
            format!("Edit {}", path.as_str())
        };
        Self {
            path,
            before,
            after,
            label,
            affected,
        }
    }

    /// Return the command's canonical field path.
    #[must_use]
    pub fn path(&self) -> &FieldPath {
        &self.path
    }
}

impl Command<ReflectEditorContext> for FieldEditCommand {
    fn apply(&mut self, context: &mut ReflectEditorContext) -> Result<(), CommandError> {
        context.apply_command_value(&self.path, &*self.after)
    }

    fn undo(&mut self, context: &mut ReflectEditorContext) -> Result<(), CommandError> {
        context.apply_command_value(&self.path, &*self.before)
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn affected(&self) -> &AffectSet {
        &self.affected
    }

    fn merge_from(&mut self, _next: &dyn Command<ReflectEditorContext>) -> MergeResult {
        MergeResult::NotMergeable
    }
}

/// A reflection editor context plus the generic editor kernel's undo/redo stack.
pub struct ReflectEditor {
    context: ReflectEditorContext,
    commands: CommandStack<ReflectEditorContext>,
}

impl ReflectEditor {
    /// Create an editor with an empty undo/redo history.
    #[must_use]
    pub fn new(root: Box<dyn Reflect>, registry: TypeRegistry) -> Self {
        Self {
            context: ReflectEditorContext::new(root, registry),
            commands: CommandStack::default(),
        }
    }

    /// Borrow the command context.
    #[must_use]
    pub fn context(&self) -> &ReflectEditorContext {
        &self.context
    }

    /// Mutably borrow the command context.
    pub fn context_mut(&mut self) -> &mut ReflectEditorContext {
        &mut self.context
    }

    /// Execute one field replacement and record it for undo.
    pub fn edit(
        &mut self,
        path: FieldPath,
        value: Box<dyn Reflect>,
    ) -> Result<CommandStatus, EditorError> {
        let command = self.context.prepare_edit(path, value)?;
        let mut commands = std::mem::take(&mut self.commands);
        let result = commands
            .execute(Box::new(command), &mut self.context)
            .map_err(EditorError::Command);
        self.commands = commands;
        result
    }

    /// Parse and execute one field replacement.
    pub fn edit_str(
        &mut self,
        path: &str,
        value: Box<dyn Reflect>,
    ) -> Result<CommandStatus, EditorError> {
        let path = FieldPath::parse(path).map_err(EditorError::InvalidPath)?;
        self.edit(path, value)
    }

    /// Undo the most recent successful edit.
    pub fn undo(&mut self) -> Result<bool, EditorError> {
        let mut commands = std::mem::take(&mut self.commands);
        let result = commands
            .undo(&mut self.context)
            .map_err(EditorError::Command);
        self.commands = commands;
        result
    }

    /// Redo the most recently undone edit.
    pub fn redo(&mut self) -> Result<bool, EditorError> {
        let mut commands = std::mem::take(&mut self.commands);
        let result = commands
            .redo(&mut self.context)
            .map_err(EditorError::Command);
        self.commands = commands;
        result
    }

    /// Number of commands currently available to undo.
    #[must_use]
    pub fn undo_len(&self) -> usize {
        self.commands.undo_len()
    }

    /// Number of commands currently available to redo.
    #[must_use]
    pub fn redo_len(&self) -> usize {
        self.commands.redo_len()
    }
}

fn apply_to_root(
    root: &mut dyn Reflect,
    registry: &TypeRegistry,
    path: &FieldPath,
    value: &dyn Reflect,
    validate_after: bool,
) -> Result<(), EditorError> {
    if let Some(metadata) = terminal_metadata(root, registry, path)
        && metadata.is_readonly()
    {
        return Err(EditorError::ReadOnly(path.clone()));
    }

    let previous = reflect_path(root, path.parsed())
        .ok_or_else(|| EditorError::PathNotFound(path.clone()))?
        .reflect_clone();
    let apply_result = reflect_path_mut(root, path.parsed())
        .ok_or_else(|| EditorError::PathNotFound(path.clone()))?
        .apply(value);
    if let Err(error) = apply_result {
        restore(root, path, &*previous);
        return Err(EditorError::Apply {
            path: path.clone(),
            error,
        });
    }

    if validate_after {
        let report = validate(root, registry);
        if !report.is_valid() {
            restore(root, path, &*previous);
            return Err(EditorError::Validation(report));
        }
    }
    Ok(())
}

fn restore(root: &mut dyn Reflect, path: &FieldPath, value: &dyn Reflect) {
    if let Some(slot) = reflect_path_mut(root, path.parsed()) {
        let _ = slot.apply(value);
    }
}

fn terminal_metadata<'a>(
    root: &dyn Reflect,
    registry: &'a TypeRegistry,
    path: &FieldPath,
) -> Option<&'a FieldMetadata> {
    let (prefix, terminal) = path.parsed().split_terminal()?;
    let Access::Field(name) = terminal else {
        return None;
    };
    let parent = reflect_path(root, &prefix)?;
    let metadata = registry
        .get_with_name(parent.type_name())
        .and_then(|registration| registration.data::<TypeMetadata>())?;
    metadata.field(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{project_inspector, validate, ValidationIssueKind};
    use prism_reflect::{Reflect, TypeRegistry};

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct Nested {
        value: i32,
    }

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct Settings {
        #[reflect(range(0.0..=1.0), category = "Display")]
        gain: f32,
        #[reflect(readonly)]
        identifier: u32,
        #[reflect(hidden)]
        internal: u32,
        nested: Nested,
    }

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::new();
        registry.register::<Settings>();
        registry.register::<Nested>();
        registry
    }

    fn editor() -> ReflectEditor {
        ReflectEditor::new(
            Box::new(Settings {
                gain: 0.25,
                identifier: 7,
                internal: 99,
                nested: Nested { value: 4 },
            }),
            registry(),
        )
    }

    #[test]
    fn inspector_projection_uses_reflect_inspector_and_paths() {
        let editor = editor();
        let projection = project_inspector(editor.context().root(), editor.context().registry());
        assert_eq!(projection.root.path, Some(FieldPath::root()));
        assert_eq!(projection.root.children.len(), 3);
        assert_eq!(
            projection.root.children[0].path.as_ref().unwrap().as_str(),
            ".gain"
        );
        assert_eq!(
            projection.root.children[0].node.hints.category.as_deref(),
            Some("Display")
        );
        assert_eq!(
            projection.root.children[2].path.as_ref().unwrap().as_str(),
            ".nested"
        );
        assert_eq!(
            projection.root.children[2].children[0]
                .path
                .as_ref()
                .unwrap()
                .as_str(),
            ".nested.value"
        );
    }

    #[test]
    fn edit_undo_redo_round_trip_through_editor_core() {
        let mut editor = editor();
        editor
            .edit_str("nested.value", Box::new(12_i32))
            .expect("edit");
        assert_eq!(
            editor
                .context()
                .root()
                .downcast_ref::<Settings>()
                .unwrap()
                .nested
                .value,
            12
        );
        assert!(editor.undo().expect("undo"));
        assert_eq!(
            editor
                .context()
                .root()
                .downcast_ref::<Settings>()
                .unwrap()
                .nested
                .value,
            4
        );
        assert!(editor.redo().expect("redo"));
        assert_eq!(
            editor
                .context()
                .root()
                .downcast_ref::<Settings>()
                .unwrap()
                .nested
                .value,
            12
        );
    }

    #[test]
    fn readonly_and_range_errors_are_reported_without_mutation() {
        let mut editor = editor();
        let readonly = editor.edit_str("identifier", Box::new(8_u32));
        assert!(matches!(readonly, Err(EditorError::ReadOnly(_))));
        let out_of_range = editor.edit_str("gain", Box::new(2.0_f32));
        assert!(matches!(out_of_range, Err(EditorError::Validation(_))));
        assert_eq!(
            editor
                .context()
                .root()
                .downcast_ref::<Settings>()
                .unwrap()
                .gain,
            0.25
        );
    }

    #[test]
    fn validation_reports_existing_invalid_state() {
        let mut value = Settings {
            gain: 2.0,
            identifier: 1,
            internal: 0,
            nested: Nested { value: 1 },
        };
        let registry = registry();
        let report = validate(&value, &registry);
        assert_eq!(report.len(), 1);
        assert!(matches!(
            report.issues()[0].kind,
            ValidationIssueKind::OutOfRange { .. }
        ));
        value.gain = 0.5;
        assert!(validate(&value, &registry).is_valid());
    }
}
