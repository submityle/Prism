//! The accessible node and its builder.

use prism_ui::Key;

use crate::label::Label;
use crate::role::Role;
use crate::state::AriaState;

/// A single node in the accessibility tree.
///
/// A node pairs an element's reconciliation [`Key`] with its semantic
/// [`Role`], accessible [`Label`], dynamic [`AriaState`] and focus metadata.
/// Build one with [`A11yNode::builder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct A11yNode {
    /// The key of the backing element.
    pub key: Key,
    /// The semantic role.
    pub role: Role,
    /// The accessible name source.
    pub label: Label,
    /// The dynamic accessibility state.
    pub state: AriaState,
    /// Whether the node can receive keyboard focus.
    pub focusable: bool,
    /// The authored tab order. Positive values form an explicit sequence;
    /// `0` means "in document order"; negative values are focusable only
    /// programmatically and are skipped by sequential navigation.
    pub tab_index: i32,
}

impl A11yNode {
    /// Starts building a node for `key` with the given `role`.
    ///
    /// Defaults: an empty [`Label`], a default [`AriaState`], `focusable`
    /// derived from the role, and `tab_index` of `0`.
    #[must_use]
    pub fn builder(key: Key, role: Role) -> A11yNodeBuilder {
        A11yNodeBuilder::new(key, role)
    }

    /// Returns `true` when sequential (Tab) navigation should land on this
    /// node: it must be `focusable`, non-negative `tab_index`, and neither
    /// disabled nor hidden.
    #[must_use]
    pub fn is_tab_stop(&self) -> bool {
        self.focusable && self.tab_index >= 0 && !self.state.is_inert()
    }
}

/// A builder for [`A11yNode`].
#[derive(Clone, Debug)]
pub struct A11yNodeBuilder {
    key: Key,
    role: Role,
    label: Label,
    state: AriaState,
    focusable: bool,
    tab_index: i32,
}

impl A11yNodeBuilder {
    fn new(key: Key, role: Role) -> Self {
        let focusable = default_focusable(&role);
        Self {
            key,
            role,
            label: Label::None,
            state: AriaState::new(),
            focusable,
            tab_index: 0,
        }
    }

    /// Sets the accessible name source.
    #[must_use]
    pub fn label(mut self, label: Label) -> Self {
        self.label = label;
        self
    }

    /// Sets the dynamic accessibility state.
    #[must_use]
    pub fn state(mut self, state: AriaState) -> Self {
        self.state = state;
        self
    }

    /// Overrides whether the node can receive keyboard focus.
    #[must_use]
    pub fn focusable(mut self, value: bool) -> Self {
        self.focusable = value;
        self
    }

    /// Sets the authored tab order.
    #[must_use]
    pub fn tab_index(mut self, value: i32) -> Self {
        self.tab_index = value;
        self
    }

    /// Finalises the node.
    #[must_use]
    pub fn build(self) -> A11yNode {
        A11yNode {
            key: self.key,
            role: self.role,
            label: self.label,
            state: self.state,
            focusable: self.focusable,
            tab_index: self.tab_index,
        }
    }
}

/// Interactive roles are focusable by default; structural and presentational
/// roles are not.
fn default_focusable(role: &Role) -> bool {
    matches!(
        role,
        Role::Button
            | Role::Link
            | Role::Textbox
            | Role::Checkbox
            | Role::Radio
            | Role::Tab
            | Role::MenuItem
    )
}

#[cfg(test)]
mod tests {
    use super::A11yNode;
    use crate::label::Label;
    use crate::role::Role;
    use crate::state::AriaState;
    use prism_ui::Key;

    #[test]
    fn builder_defaults() {
        let n = A11yNode::builder(Key::Int(1), Role::Button).build();
        assert_eq!(n.key, Key::Int(1));
        assert_eq!(n.role, Role::Button);
        assert_eq!(n.label, Label::None);
        assert!(n.focusable);
        assert_eq!(n.tab_index, 0);
    }

    #[test]
    fn structural_roles_not_focusable_by_default() {
        assert!(
            !A11yNode::builder(Key::Int(1), Role::Group)
                .build()
                .focusable
        );
        assert!(!A11yNode::builder(Key::Int(2), Role::List).build().focusable);
        assert!(
            !A11yNode::builder(Key::Int(3), Role::Image)
                .build()
                .focusable
        );
    }

    #[test]
    fn tab_stop_rules() {
        let ok = A11yNode::builder(Key::Int(1), Role::Button).build();
        assert!(ok.is_tab_stop());

        let negative = A11yNode::builder(Key::Int(2), Role::Button)
            .tab_index(-1)
            .build();
        assert!(!negative.is_tab_stop());

        let disabled = A11yNode::builder(Key::Int(3), Role::Button)
            .state(AriaState::new().disabled(true))
            .build();
        assert!(!disabled.is_tab_stop());

        let not_focusable = A11yNode::builder(Key::Int(4), Role::Group).build();
        assert!(!not_focusable.is_tab_stop());
    }
}
