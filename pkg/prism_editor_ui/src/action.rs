//! Engine-neutral UI event bindings and deterministic editor actions.

use std::collections::BTreeMap;
use std::fmt;

use prism_editor_panels::OutlinerId;
use prism_editor_reflect::FieldPath;
use prism_ui::Element;

use crate::support::MAX_ACTION_BINDINGS;

/// The maximum number of bytes accepted for one command id.
pub const MAX_ACTION_ID_BYTES: usize = 256;
/// The maximum number of bytes accepted for one inspector path.
pub const MAX_ACTION_PATH_BYTES: usize = 1024;
/// The maximum number of bytes accepted for one semantic event target.
pub const MAX_ACTION_TARGET_BYTES: usize = MAX_ACTION_PATH_BYTES + "field:".len();
/// The maximum number of bytes accepted for one text action value.
pub const MAX_ACTION_TEXT_BYTES: usize = 4096;

/// The selection behavior requested by an Outliner action.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectionMode {
    /// Replace the current selection with the target node.
    Replace,
    /// Add the target node to the current selection.
    Add,
    /// Toggle the target node in the current selection.
    Toggle,
    /// Extend the selection according to the upper layer's anchor policy.
    Extend,
}

/// A bounded value supplied by an engine-neutral UI event.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ActionValue {
    kind: ActionValueKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum ActionValueKind {
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    FloatBits(u64),
    Text(String),
}

impl ActionValue {
    /// Creates a boolean value.
    #[must_use]
    pub const fn bool(value: bool) -> Self {
        Self {
            kind: ActionValueKind::Bool(value),
        }
    }

    /// Creates a signed integer value.
    #[must_use]
    pub const fn integer(value: i64) -> Self {
        Self {
            kind: ActionValueKind::Integer(value),
        }
    }

    /// Creates an unsigned integer value.
    #[must_use]
    pub const fn unsigned(value: u64) -> Self {
        Self {
            kind: ActionValueKind::Unsigned(value),
        }
    }

    /// Creates a floating-point value using its exact IEEE-754 bit pattern.
    #[must_use]
    pub const fn float(value: f64) -> Self {
        Self {
            kind: ActionValueKind::FloatBits(value.to_bits()),
        }
    }

    /// Creates a bounded text value.
    pub fn text(value: impl Into<String>) -> Result<Self, ActionError> {
        let value = value.into();
        ensure_max_bytes("action text", &value, MAX_ACTION_TEXT_BYTES)?;
        Ok(Self {
            kind: ActionValueKind::Text(value),
        })
    }

    /// Returns the contained boolean, if this is a boolean value.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self.kind {
            ActionValueKind::Bool(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the contained signed integer, if this is an integer value.
    #[must_use]
    pub const fn as_integer(&self) -> Option<i64> {
        match self.kind {
            ActionValueKind::Integer(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the contained unsigned integer, if this is an unsigned value.
    #[must_use]
    pub const fn as_unsigned(&self) -> Option<u64> {
        match self.kind {
            ActionValueKind::Unsigned(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the exact floating-point value, if this is a float value.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self.kind {
            ActionValueKind::FloatBits(value) => Some(f64::from_bits(value)),
            _ => None,
        }
    }

    /// Returns the contained text, if this is a text value.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match &self.kind {
            ActionValueKind::Text(value) => Some(value),
            _ => None,
        }
    }
}

/// A semantic event emitted by a Loom backend or an input adapter.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum UiEvent {
    /// Activates a button, row, menu item, or command.
    Activate,
    /// Selects an Outliner row with the requested modifier policy.
    Select {
        /// Selection modifier policy.
        mode: SelectionMode,
    },
    /// Commits a value from an Inspector editor.
    Commit {
        /// Bounded value supplied by the editor control.
        value: ActionValue,
    },
}

impl UiEvent {
    fn trigger(&self) -> UiTrigger {
        match self {
            Self::Activate => UiTrigger::Activate,
            Self::Select { .. } => UiTrigger::Select,
            Self::Commit { .. } => UiTrigger::Commit,
        }
    }
}

/// The event shape accepted by an action binding.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum UiTrigger {
    /// An activation event with no payload.
    Activate,
    /// A selection event carrying a [`SelectionMode`].
    Select,
    /// A committed value event carrying an [`ActionValue`].
    Commit,
}

/// A deterministic action produced after resolving a UI event binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditorAction {
    /// Select an Outliner node.
    SelectOutliner {
        /// Stable Outliner node identity.
        node: OutlinerId,
        /// Selection modifier policy.
        mode: SelectionMode,
    },
    /// Toggle an Outliner node's expansion state.
    ToggleOutlinerExpansion {
        /// Stable Outliner node identity.
        node: OutlinerId,
    },
    /// Replace one reflected Inspector field value.
    EditInspectorField {
        /// Canonical reflected field path.
        path: FieldPath,
        /// Value supplied by the UI adapter.
        value: ActionValue,
    },
    /// Ask the upper layer to execute a registered command.
    ExecuteCommand {
        /// Stable command identifier.
        command_id: String,
    },
    /// Undo the latest editor command.
    Undo,
    /// Redo the latest undone editor command.
    Redo,
}

/// The action recipe stored by a binding before an event supplies its payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionTemplate {
    kind: ActionTemplateKind,
    node: Option<OutlinerId>,
    path: Option<FieldPath>,
    command_id: Option<String>,
}

/// The kind of action produced by an [`ActionTemplate`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionTemplateKind {
    /// Produces [`EditorAction::SelectOutliner`].
    SelectOutliner,
    /// Produces [`EditorAction::ToggleOutlinerExpansion`].
    ToggleOutlinerExpansion,
    /// Produces [`EditorAction::EditInspectorField`].
    EditInspectorField,
    /// Produces [`EditorAction::ExecuteCommand`].
    ExecuteCommand,
    /// Produces [`EditorAction::Undo`].
    Undo,
    /// Produces [`EditorAction::Redo`].
    Redo,
}

impl ActionTemplate {
    /// Creates an Outliner selection recipe.
    #[must_use]
    pub const fn select_outliner(node: OutlinerId) -> Self {
        Self {
            kind: ActionTemplateKind::SelectOutliner,
            node: Some(node),
            path: None,
            command_id: None,
        }
    }

    /// Creates an Outliner expansion recipe.
    #[must_use]
    pub const fn toggle_outliner_expansion(node: OutlinerId) -> Self {
        Self {
            kind: ActionTemplateKind::ToggleOutlinerExpansion,
            node: Some(node),
            path: None,
            command_id: None,
        }
    }

    /// Creates an Inspector field-edit recipe.
    pub fn edit_inspector_field(path: FieldPath) -> Result<Self, ActionError> {
        ensure_max_bytes("field path", path.as_str(), MAX_ACTION_PATH_BYTES)?;
        Ok(Self {
            kind: ActionTemplateKind::EditInspectorField,
            node: None,
            path: Some(path),
            command_id: None,
        })
    }

    /// Creates a command-execution recipe.
    pub fn execute_command(command_id: impl Into<String>) -> Result<Self, ActionError> {
        let command_id = command_id.into();
        ensure_non_empty("command id", &command_id)?;
        ensure_byte_limit("command id", &command_id, MAX_ACTION_ID_BYTES)?;
        Ok(Self {
            kind: ActionTemplateKind::ExecuteCommand,
            node: None,
            path: None,
            command_id: Some(command_id),
        })
    }

    /// Creates an undo recipe.
    #[must_use]
    pub const fn undo() -> Self {
        Self {
            kind: ActionTemplateKind::Undo,
            node: None,
            path: None,
            command_id: None,
        }
    }

    /// Creates a redo recipe.
    #[must_use]
    pub const fn redo() -> Self {
        Self {
            kind: ActionTemplateKind::Redo,
            node: None,
            path: None,
            command_id: None,
        }
    }

    /// Returns the recipe kind.
    #[must_use]
    pub const fn kind(&self) -> ActionTemplateKind {
        self.kind
    }

    /// Returns the target Outliner node for a node recipe.
    #[must_use]
    pub const fn node(&self) -> Option<OutlinerId> {
        self.node
    }

    /// Returns the field path for an Inspector recipe.
    #[must_use]
    pub fn path(&self) -> Option<&FieldPath> {
        self.path.as_ref()
    }

    /// Returns the command identifier for a command recipe.
    #[must_use]
    pub fn command_id(&self) -> Option<&str> {
        self.command_id.as_deref()
    }

    fn resolve(&self, event: &UiEvent) -> Result<EditorAction, ActionError> {
        match (self.kind, event) {
            (ActionTemplateKind::SelectOutliner, UiEvent::Select { mode }) => {
                Ok(EditorAction::SelectOutliner {
                    node: self.node.expect("selection template has a node"),
                    mode: *mode,
                })
            }
            (ActionTemplateKind::ToggleOutlinerExpansion, UiEvent::Activate) => {
                Ok(EditorAction::ToggleOutlinerExpansion {
                    node: self.node.expect("expansion template has a node"),
                })
            }
            (ActionTemplateKind::EditInspectorField, UiEvent::Commit { value }) => {
                Ok(EditorAction::EditInspectorField {
                    path: self.path.clone().expect("edit template has a path"),
                    value: value.clone(),
                })
            }
            (ActionTemplateKind::ExecuteCommand, UiEvent::Activate) => {
                Ok(EditorAction::ExecuteCommand {
                    command_id: self.command_id.clone().expect("command template has an id"),
                })
            }
            (ActionTemplateKind::Undo, UiEvent::Activate) => Ok(EditorAction::Undo),
            (ActionTemplateKind::Redo, UiEvent::Activate) => Ok(EditorAction::Redo),
            _ => Err(ActionError::TriggerMismatch {
                expected: self.trigger(),
                received: event.trigger(),
            }),
        }
    }

    fn trigger(&self) -> UiTrigger {
        match self.kind {
            ActionTemplateKind::SelectOutliner => UiTrigger::Select,
            ActionTemplateKind::EditInspectorField => UiTrigger::Commit,
            ActionTemplateKind::ToggleOutlinerExpansion
            | ActionTemplateKind::ExecuteCommand
            | ActionTemplateKind::Undo
            | ActionTemplateKind::Redo => UiTrigger::Activate,
        }
    }
}

/// A validated semantic target key from a rendered Element tree.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ActionTarget(String);

impl ActionTarget {
    /// Creates a target from an Element's semantic key.
    pub fn new(target: impl Into<String>) -> Result<Self, ActionError> {
        let target = target.into();
        ensure_non_empty("action target", &target)?;
        ensure_max_bytes("action target", &target, MAX_ACTION_TARGET_BYTES)?;
        Ok(Self(target))
    }

    /// Returns the target key used by a backend event.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One target/event to action recipe mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionBinding {
    target: ActionTarget,
    trigger: UiTrigger,
    template: ActionTemplate,
}

impl ActionBinding {
    /// Creates a validated binding.
    #[must_use]
    pub fn new(target: ActionTarget, template: ActionTemplate) -> Self {
        Self {
            target,
            trigger: template.trigger(),
            template,
        }
    }

    /// Creates an activation binding.
    pub fn activate(
        target: impl Into<String>,
        template: ActionTemplate,
    ) -> Result<Self, ActionError> {
        let binding = Self::new(ActionTarget::new(target)?, template);
        if binding.trigger != UiTrigger::Activate {
            return Err(ActionError::TriggerMismatch {
                expected: UiTrigger::Activate,
                received: binding.trigger,
            });
        }
        Ok(binding)
    }

    /// Creates a selection binding.
    pub fn select(
        target: impl Into<String>,
        template: ActionTemplate,
    ) -> Result<Self, ActionError> {
        let binding = Self::new(ActionTarget::new(target)?, template);
        if binding.trigger != UiTrigger::Select {
            return Err(ActionError::TriggerMismatch {
                expected: UiTrigger::Select,
                received: binding.trigger,
            });
        }
        Ok(binding)
    }

    /// Creates a value-commit binding.
    pub fn commit(
        target: impl Into<String>,
        template: ActionTemplate,
    ) -> Result<Self, ActionError> {
        let binding = Self::new(ActionTarget::new(target)?, template);
        if binding.trigger != UiTrigger::Commit {
            return Err(ActionError::TriggerMismatch {
                expected: UiTrigger::Commit,
                received: binding.trigger,
            });
        }
        Ok(binding)
    }

    /// Returns the semantic target key.
    #[must_use]
    pub fn target(&self) -> &ActionTarget {
        &self.target
    }

    /// Returns the accepted event shape.
    #[must_use]
    pub const fn trigger(&self) -> UiTrigger {
        self.trigger
    }

    /// Returns the action recipe.
    #[must_use]
    pub const fn template(&self) -> &ActionTemplate {
        &self.template
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BindingKey {
    target: ActionTarget,
    trigger: UiTrigger,
}

/// A bounded, deterministic set of UI action bindings.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ActionBindings {
    bindings: BTreeMap<BindingKey, ActionTemplate>,
}

impl ActionBindings {
    /// Creates an empty bounded binding set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts one binding, rejecting duplicates and capacity overflow.
    pub fn bind(&mut self, binding: ActionBinding) -> Result<(), ActionError> {
        let key = BindingKey {
            target: binding.target,
            trigger: binding.trigger,
        };
        if self.bindings.contains_key(&key) {
            return Err(ActionError::DuplicateBinding {
                target: key.target.0,
                trigger: key.trigger,
            });
        }
        if self.bindings.len() >= MAX_ACTION_BINDINGS {
            return Err(ActionError::BindingLimit {
                limit: MAX_ACTION_BINDINGS,
            });
        }
        self.bindings.insert(key, binding.template);
        Ok(())
    }

    /// Adds every binding from another set without exceeding the hard cap.
    pub fn extend(&mut self, other: &Self) -> Result<(), ActionError> {
        if self.bindings.len().saturating_add(other.bindings.len()) > MAX_ACTION_BINDINGS {
            return Err(ActionError::BindingLimit {
                limit: MAX_ACTION_BINDINGS,
            });
        }
        if let Some(key) = other
            .bindings
            .keys()
            .find(|key| self.bindings.contains_key(*key))
        {
            return Err(ActionError::DuplicateBinding {
                target: key.target.0.clone(),
                trigger: key.trigger,
            });
        }
        for (key, template) in &other.bindings {
            self.bindings.insert(key.clone(), template.clone());
        }
        Ok(())
    }

    /// Binds an activation event to undo.
    pub fn bind_undo(&mut self, target: impl Into<String>) -> Result<(), ActionError> {
        self.bind(ActionBinding::activate(target, ActionTemplate::undo())?)
    }

    /// Binds an activation event to redo.
    pub fn bind_redo(&mut self, target: impl Into<String>) -> Result<(), ActionError> {
        self.bind(ActionBinding::activate(target, ActionTemplate::redo())?)
    }

    /// Returns the number of registered bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns whether no bindings are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Resolves a backend event into one deterministic editor action.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<EditorAction, ActionError> {
        ensure_non_empty("action target", target)?;
        ensure_max_bytes("action target", target, MAX_ACTION_TARGET_BYTES)?;
        let trigger = event.trigger();
        let key = BindingKey {
            target: ActionTarget(target.to_owned()),
            trigger,
        };
        let template = self
            .bindings
            .get(&key)
            .ok_or_else(|| ActionError::Unbound {
                target: target.to_owned(),
                trigger,
            })?;
        template.resolve(event)
    }

    /// Returns bindings in deterministic target/trigger order.
    pub fn iter(&self) -> impl Iterator<Item = ActionBinding> + '_ {
        self.bindings.iter().map(|(key, template)| ActionBinding {
            target: key.target.clone(),
            trigger: key.trigger,
            template: template.clone(),
        })
    }
}

/// A Loom Element tree paired with its engine-neutral action bindings.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundElement {
    element: Element,
    bindings: ActionBindings,
}

impl BoundElement {
    /// Creates a bound Element tree.
    #[must_use]
    pub const fn new(element: Element, bindings: ActionBindings) -> Self {
        Self { element, bindings }
    }

    /// Returns the rendered Element tree.
    #[must_use]
    pub const fn element(&self) -> &Element {
        &self.element
    }

    /// Returns the action bindings associated with the tree.
    #[must_use]
    pub const fn bindings(&self) -> &ActionBindings {
        &self.bindings
    }

    /// Consumes the wrapper and returns its Element tree.
    #[must_use]
    pub fn into_element(self) -> Element {
        self.element
    }

    /// Consumes the wrapper and returns the tree and bindings separately.
    #[must_use]
    pub fn into_parts(self) -> (Element, ActionBindings) {
        (self.element, self.bindings)
    }

    /// Resolves one backend event against this tree's bindings.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<EditorAction, ActionError> {
        self.bindings.dispatch(target, event)
    }
}

/// Errors raised while validating or resolving a UI action contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActionError {
    /// A required string was empty or whitespace-only.
    EmptyValue {
        /// Logical field name.
        field: &'static str,
    },
    /// A bounded string exceeded its byte limit.
    ValueTooLong {
        /// The logical field that exceeded its limit.
        field: &'static str,
        /// The accepted byte limit.
        limit: usize,
    },
    /// A target/event pair was registered twice.
    DuplicateBinding {
        /// Semantic target key.
        target: String,
        /// Event shape of the duplicate binding.
        trigger: UiTrigger,
    },
    /// The binding set reached its hard capacity.
    BindingLimit {
        /// Maximum number of bindings.
        limit: usize,
    },
    /// The event shape does not match the action recipe.
    TriggerMismatch {
        /// Event shape required by the recipe.
        expected: UiTrigger,
        /// Event shape received by the dispatcher.
        received: UiTrigger,
    },
    /// No binding exists for the target/event pair.
    Unbound {
        /// Semantic target key.
        target: String,
        /// Event shape that was requested.
        trigger: UiTrigger,
    },
}

impl fmt::Display for ActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyValue { field } => write!(formatter, "{field} cannot be empty"),
            Self::ValueTooLong { field, limit } => {
                write!(formatter, "{field} exceeds the {limit}-byte limit")
            }
            Self::DuplicateBinding { target, trigger } => {
                write!(formatter, "duplicate {trigger:?} binding for {target}")
            }
            Self::BindingLimit { limit } => {
                write!(formatter, "action binding limit {limit} exceeded")
            }
            Self::TriggerMismatch { expected, received } => {
                write!(
                    formatter,
                    "binding expects {expected:?}, received {received:?}"
                )
            }
            Self::Unbound { target, trigger } => {
                write!(formatter, "no {trigger:?} binding for {target}")
            }
        }
    }
}

impl std::error::Error for ActionError {}

/// Errors raised while building an action-aware Element tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiBuildError {
    /// The generated action contract is invalid or exceeds a hard bound.
    Action(ActionError),
    /// Inspector nesting exceeded [`crate::MAX_INSPECTOR_DEPTH`].
    InspectorDepthLimit {
        /// Maximum permitted nesting depth.
        limit: usize,
    },
    /// Inspector node count exceeded [`crate::MAX_INSPECTOR_NODES`].
    InspectorNodeLimit {
        /// Maximum permitted node count.
        limit: usize,
    },
}

impl fmt::Display for UiBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Action(error) => error.fmt(formatter),
            Self::InspectorDepthLimit { limit } => {
                write!(formatter, "inspector depth limit {limit} exceeded")
            }
            Self::InspectorNodeLimit { limit } => {
                write!(formatter, "inspector node limit {limit} exceeded")
            }
        }
    }
}

impl std::error::Error for UiBuildError {}

impl From<ActionError> for UiBuildError {
    fn from(error: ActionError) -> Self {
        Self::Action(error)
    }
}

fn ensure_non_empty(field: &'static str, value: &str) -> Result<(), ActionError> {
    if value.trim().is_empty() {
        Err(ActionError::EmptyValue { field })
    } else {
        Ok(())
    }
}

fn ensure_byte_limit(field: &'static str, value: &str, limit: usize) -> Result<(), ActionError> {
    ensure_non_empty(field, value)?;
    ensure_max_bytes(field, value, limit)
}

fn ensure_max_bytes(field: &'static str, value: &str, limit: usize) -> Result<(), ActionError> {
    if value.len() > limit {
        return Err(ActionError::ValueTooLong { field, limit });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_dispatch_deterministically_for_all_editor_actions() -> Result<(), ActionError> {
        let mut bindings = ActionBindings::new();
        bindings
            .bind(ActionBinding::select(
                "outliner:7",
                ActionTemplate::select_outliner(OutlinerId::from_raw(7)),
            )?)
            .expect("selection binding");
        bindings
            .bind(ActionBinding::activate("undo", ActionTemplate::undo())?)
            .expect("undo binding");
        bindings
            .bind(ActionBinding::activate(
                "command:save",
                ActionTemplate::execute_command("scene.save")?,
            )?)
            .expect("command binding");
        let path = FieldPath::parse("speed").expect("path");
        bindings
            .bind(ActionBinding::commit(
                "field:.speed",
                ActionTemplate::edit_inspector_field(path.clone())?,
            )?)
            .expect("field binding");

        assert_eq!(
            bindings.dispatch(
                "outliner:7",
                &UiEvent::Select {
                    mode: SelectionMode::Add,
                },
            )?,
            EditorAction::SelectOutliner {
                node: OutlinerId::from_raw(7),
                mode: SelectionMode::Add,
            }
        );
        assert_eq!(
            bindings.dispatch("undo", &UiEvent::Activate)?,
            EditorAction::Undo
        );
        assert_eq!(
            bindings.dispatch("command:save", &UiEvent::Activate)?,
            EditorAction::ExecuteCommand {
                command_id: String::from("scene.save"),
            }
        );
        let value = ActionValue::integer(42);
        assert_eq!(
            bindings.dispatch(
                "field:.speed",
                &UiEvent::Commit {
                    value: value.clone(),
                },
            )?,
            EditorAction::EditInspectorField { path, value }
        );
        Ok::<(), ActionError>(())
    }

    #[test]
    fn dynamic_inputs_and_binding_count_are_bounded() -> Result<(), ActionError> {
        assert_eq!(ActionValue::text("")?.as_text(), Some(""));
        assert!(matches!(
            ActionValue::text("x".repeat(MAX_ACTION_TEXT_BYTES + 1)),
            Err(ActionError::ValueTooLong { .. })
        ));
        let mut bindings = ActionBindings::new();
        for index in 0..MAX_ACTION_BINDINGS {
            bindings
                .bind(ActionBinding::activate(
                    format!("target:{index}"),
                    ActionTemplate::undo(),
                )?)
                .expect("within binding limit");
        }
        assert!(matches!(
            bindings.bind(ActionBinding::activate("overflow", ActionTemplate::redo())?),
            Err(ActionError::BindingLimit { .. })
        ));
        let mut additional = ActionBindings::new();
        additional.bind(ActionBinding::activate("another", ActionTemplate::redo())?)?;
        assert!(matches!(
            bindings.extend(&additional),
            Err(ActionError::BindingLimit { .. })
        ));
        assert_eq!(bindings.len(), MAX_ACTION_BINDINGS);
        Ok::<(), ActionError>(())
    }

    #[test]
    fn root_field_paths_are_editable_and_merges_are_atomic() -> Result<(), ActionError> {
        let root = FieldPath::root();
        let template = ActionTemplate::edit_inspector_field(root.clone())?;
        let mut bindings = ActionBindings::new();
        bindings.bind(ActionBinding::commit("field:", template)?)?;
        assert_eq!(
            bindings.dispatch(
                "field:",
                &UiEvent::Commit {
                    value: ActionValue::text("")?,
                },
            )?,
            EditorAction::EditInspectorField {
                path: root,
                value: ActionValue::text("")?,
            }
        );

        let mut conflicting = ActionBindings::new();
        conflicting.bind(ActionBinding::activate("undo", ActionTemplate::undo())?)?;
        bindings.bind(ActionBinding::activate("undo", ActionTemplate::undo())?)?;
        let binding_count = bindings.len();
        assert!(matches!(
            bindings.extend(&conflicting),
            Err(ActionError::DuplicateBinding { .. })
        ));
        assert_eq!(bindings.len(), binding_count);
        Ok(())
    }

    #[test]
    fn duplicate_bindings_and_mismatched_events_are_rejected() -> Result<(), ActionError> {
        let mut bindings = ActionBindings::new();
        let binding = ActionBinding::activate("undo", ActionTemplate::undo())?;
        bindings.bind(binding.clone())?;
        assert!(matches!(
            bindings.bind(binding),
            Err(ActionError::DuplicateBinding { .. })
        ));
        assert!(matches!(
            bindings.dispatch(
                "undo",
                &UiEvent::Commit {
                    value: ActionValue::integer(1),
                }
            ),
            Err(ActionError::Unbound { .. })
        ));
        Ok(())
    }
}
