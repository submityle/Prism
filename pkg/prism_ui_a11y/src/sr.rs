//! Screen-reader text composition.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Key;

use crate::node::A11yNode;
use crate::state::AriaState;
use crate::tree::A11yTree;

/// Composes the spoken description of the node identified by `key`.
///
/// The result is a deterministic, comma-separated phrase of the form:
///
/// ```text
/// <role>[, <label>][, <state> ...]
/// ```
///
/// where the role comes from [`Role::describe`](crate::Role::describe), the
/// label is resolved via [`A11yTree::label_text`] (following `labelled-by`),
/// and state words are appended in a fixed order: `checked`/`not checked`,
/// `selected`/`not selected`, `expanded`/`collapsed`, `required`, `disabled`.
/// A hidden node yields the empty string, matching its exclusion from the
/// accessibility tree.
///
/// Returns the empty string when `key` is not present in the tree.
///
/// # Examples
///
/// ```
/// use prism_ui::Key;
/// use prism_ui_a11y::{A11yNode, A11yTree, AriaState, Label, Role, screen_reader_text};
///
/// let mut tree = A11yTree::new();
/// tree.insert(
///     A11yNode::builder(Key::Int(1), Role::Checkbox)
///         .label(Label::text("Accept terms"))
///         .state(AriaState::new().checked(true))
///         .build(),
/// );
///
/// assert_eq!(
///     screen_reader_text(&tree, &Key::Int(1)),
///     "checkbox, Accept terms, checked",
/// );
/// ```
#[must_use]
pub fn screen_reader_text(tree: &A11yTree, key: &Key) -> String {
    let Some(node) = tree.get(key) else {
        return String::new();
    };
    if node.state.hidden {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    parts.push(node.role.describe());

    if let Some(label) = tree.label_text(key)
        && !label.is_empty()
    {
        parts.push(label);
    }

    append_state(&node.state, &mut parts);

    join_commas(&parts)
}

/// Describes a node directly (role + inline label + state) without tree
/// resolution. Useful when a node is held outside a tree; `labelled-by`
/// references cannot be resolved here and are treated as unnamed.
#[must_use]
pub fn describe_node(node: &A11yNode) -> String {
    if node.state.hidden {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    parts.push(node.role.describe());
    if let Some(label) = node.label.inline_text()
        && !label.is_empty()
    {
        parts.push(label.into());
    }
    append_state(&node.state, &mut parts);
    join_commas(&parts)
}

fn append_state(state: &AriaState, parts: &mut Vec<String>) {
    if let Some(checked) = state.checked {
        parts.push(String::from(if checked {
            "checked"
        } else {
            "not checked"
        }));
    }
    if let Some(selected) = state.selected {
        parts.push(String::from(if selected {
            "selected"
        } else {
            "not selected"
        }));
    }
    if let Some(expanded) = state.expanded {
        parts.push(String::from(if expanded {
            "expanded"
        } else {
            "collapsed"
        }));
    }
    if state.required {
        parts.push(String::from("required"));
    }
    if state.disabled {
        parts.push(String::from("disabled"));
    }
}

fn join_commas(parts: &[String]) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{describe_node, screen_reader_text};
    use crate::label::Label;
    use crate::node::A11yNode;
    use crate::role::Role;
    use crate::state::AriaState;
    use crate::tree::A11yTree;
    use prism_ui::Key;

    #[test]
    fn checkbox_checked_with_label() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Checkbox)
                .label(Label::text("Accept terms"))
                .state(AriaState::new().checked(true))
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "checkbox, Accept terms, checked"
        );
    }

    #[test]
    fn button_disabled() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Button)
                .label(Label::text("Save"))
                .state(AriaState::new().disabled(true))
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "button, Save, disabled"
        );
    }

    #[test]
    fn heading_level_and_no_state() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Heading { level: 2 })
                .label(Label::text("Settings"))
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "heading level 2, Settings"
        );
    }

    #[test]
    fn tab_selected_and_labelled_by() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Tab)
                .label(Label::labelled_by(Key::Int(2)))
                .state(AriaState::new().selected(true))
                .build(),
        );
        tree.insert(
            A11yNode::builder(Key::Int(2), Role::Presentation)
                .label(Label::text("Profile"))
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "tab, Profile, selected"
        );
    }

    #[test]
    fn textbox_required_and_state_order() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Textbox)
                .label(Label::text("Email"))
                .state(AriaState::new().required(true))
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "text field, Email, required"
        );
    }

    #[test]
    fn combobox_like_expanded_order_is_fixed() {
        // selected + expanded + required + disabled must appear in that order.
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::MenuItem)
                .label(Label::text("Options"))
                .state(
                    AriaState::new()
                        .selected(false)
                        .expanded(true)
                        .required(true)
                        .disabled(true),
                )
                .build(),
        );
        assert_eq!(
            screen_reader_text(&tree, &Key::Int(1)),
            "menu item, Options, not selected, expanded, required, disabled"
        );
    }

    #[test]
    fn hidden_and_missing_yield_empty() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Button)
                .label(Label::text("x"))
                .state(AriaState::new().hidden(true))
                .build(),
        );
        assert_eq!(screen_reader_text(&tree, &Key::Int(1)), "");
        assert_eq!(screen_reader_text(&tree, &Key::Int(2)), "");
    }

    #[test]
    fn describe_node_without_tree() {
        let node = A11yNode::builder(Key::Int(1), Role::Radio)
            .label(Label::text("Male"))
            .state(AriaState::new().checked(false))
            .build();
        assert_eq!(describe_node(&node), "radio button, Male, not checked");

        // A labelled-by reference cannot be resolved standalone.
        let node2 = A11yNode::builder(Key::Int(2), Role::Checkbox)
            .label(Label::labelled_by(Key::Int(9)))
            .build();
        assert_eq!(describe_node(&node2), "checkbox");
    }
}
