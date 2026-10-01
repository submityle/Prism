//! Role-aware keyboard navigation.
//!
//! [`KeyboardNav`] maps a [`KeyInput`] plus the currently focused element to a
//! [`NavAction`], applying conventional desktop/web keyboard semantics based on
//! the focused node's [`Role`](crate::Role).

use prism_ui::Key;

use crate::role::Role;
use crate::tree::A11yTree;

/// A direction for an arrow-key press.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Arrow {
    /// The up arrow.
    Up,
    /// The down arrow.
    Down,
    /// The left arrow.
    Left,
    /// The right arrow.
    Right,
}

/// A single logical keyboard input relevant to accessibility navigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyInput {
    /// Move to the next focus stop.
    Tab,
    /// Move to the previous focus stop.
    ShiftTab,
    /// An arrow key in the given direction.
    Arrow(Arrow),
    /// The Enter/Return key.
    Enter,
    /// The Space bar.
    Space,
    /// The Escape key.
    Escape,
    /// The Home key.
    Home,
    /// The End key.
    End,
}

/// Which focus stop a [`NavAction::MoveFocus`] targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FocusMove {
    /// The next stop (wrapping).
    Next,
    /// The previous stop (wrapping).
    Prev,
    /// The first stop.
    First,
    /// The last stop.
    Last,
}

/// The resolved outcome of a key press.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NavAction {
    /// Move focus to another stop.
    MoveFocus(FocusMove),
    /// Activate the focused control (click/open/submit).
    Activate,
    /// Toggle the focused control's checked state.
    Toggle,
    /// Dismiss the current context (e.g. close a dialog).
    Dismiss,
    /// The key has no effect for this role.
    None,
}

/// Resolves keyboard input into [`NavAction`]s against a tree.
///
/// The resolver is stateless; it reads the focused node's role from the tree.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyboardNav;

impl KeyboardNav {
    /// Creates a resolver.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Resolves `input` for the node currently focused (`current`) within
    /// `tree`.
    ///
    /// Tab/Shift+Tab, Home and End are role-independent focus moves. Enter,
    /// Space, Escape and the arrow keys are interpreted according to the
    /// focused node's [`Role`](crate::Role). If `current` is not in the tree,
    /// only the role-independent moves apply.
    #[must_use]
    pub fn resolve(&self, tree: &A11yTree, current: &Key, input: KeyInput) -> NavAction {
        // Role-independent sequential navigation.
        match input {
            KeyInput::Tab => return NavAction::MoveFocus(FocusMove::Next),
            KeyInput::ShiftTab => return NavAction::MoveFocus(FocusMove::Prev),
            KeyInput::Home => return NavAction::MoveFocus(FocusMove::First),
            KeyInput::End => return NavAction::MoveFocus(FocusMove::Last),
            _ => {}
        }

        let Some(node) = tree.get(current) else {
            return NavAction::None;
        };
        Self::resolve_for_role(&node.role, input)
    }

    /// Resolves a role-dependent `input` for a known `role`. Exposed for
    /// callers that already have a role in hand.
    #[must_use]
    pub fn resolve_for_role(role: &Role, input: KeyInput) -> NavAction {
        match input {
            KeyInput::Tab => NavAction::MoveFocus(FocusMove::Next),
            KeyInput::ShiftTab => NavAction::MoveFocus(FocusMove::Prev),
            KeyInput::Home => NavAction::MoveFocus(FocusMove::First),
            KeyInput::End => NavAction::MoveFocus(FocusMove::Last),
            KeyInput::Enter => match role {
                Role::Button | Role::Link | Role::MenuItem | Role::Tab | Role::ListItem => {
                    NavAction::Activate
                }
                _ => NavAction::None,
            },
            KeyInput::Space => match role {
                Role::Checkbox | Role::Radio => NavAction::Toggle,
                Role::Button | Role::MenuItem => NavAction::Activate,
                _ => NavAction::None,
            },
            KeyInput::Escape => match role {
                Role::Dialog | Role::Menu | Role::MenuItem => NavAction::Dismiss,
                _ => NavAction::None,
            },
            KeyInput::Arrow(dir) => {
                if role.is_arrow_navigable() {
                    match dir {
                        Arrow::Up | Arrow::Left => NavAction::MoveFocus(FocusMove::Prev),
                        Arrow::Down | Arrow::Right => NavAction::MoveFocus(FocusMove::Next),
                    }
                } else {
                    NavAction::None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Arrow, FocusMove, KeyInput, KeyboardNav, NavAction};
    use crate::node::A11yNode;
    use crate::role::Role;
    use crate::tree::A11yTree;
    use prism_ui::Key;

    fn tree_with(role: Role) -> (A11yTree, Key) {
        let mut tree = A11yTree::new();
        let key = Key::Int(1);
        tree.insert(A11yNode::builder(key.clone(), role).build());
        (tree, key)
    }

    #[test]
    fn tab_and_shift_tab_are_role_independent() {
        let (tree, key) = tree_with(Role::Group);
        let nav = KeyboardNav::new();
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Tab),
            NavAction::MoveFocus(FocusMove::Next)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::ShiftTab),
            NavAction::MoveFocus(FocusMove::Prev)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Home),
            NavAction::MoveFocus(FocusMove::First)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::End),
            NavAction::MoveFocus(FocusMove::Last)
        );
    }

    #[test]
    fn button_enter_and_space_activate() {
        let (tree, key) = tree_with(Role::Button);
        let nav = KeyboardNav::new();
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Enter),
            NavAction::Activate
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Space),
            NavAction::Activate
        );
    }

    #[test]
    fn checkbox_space_toggles_enter_does_nothing() {
        let (tree, key) = tree_with(Role::Checkbox);
        let nav = KeyboardNav::new();
        assert_eq!(nav.resolve(&tree, &key, KeyInput::Space), NavAction::Toggle);
        assert_eq!(nav.resolve(&tree, &key, KeyInput::Enter), NavAction::None);
    }

    #[test]
    fn list_arrows_move_focus() {
        let (tree, key) = tree_with(Role::ListItem);
        let nav = KeyboardNav::new();
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Arrow(Arrow::Down)),
            NavAction::MoveFocus(FocusMove::Next)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Arrow(Arrow::Up)),
            NavAction::MoveFocus(FocusMove::Prev)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Arrow(Arrow::Right)),
            NavAction::MoveFocus(FocusMove::Next)
        );
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Arrow(Arrow::Left)),
            NavAction::MoveFocus(FocusMove::Prev)
        );
    }

    #[test]
    fn button_ignores_arrows() {
        let (tree, key) = tree_with(Role::Button);
        let nav = KeyboardNav::new();
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Arrow(Arrow::Down)),
            NavAction::None
        );
    }

    #[test]
    fn dialog_escape_dismisses() {
        let (tree, key) = tree_with(Role::Dialog);
        let nav = KeyboardNav::new();
        assert_eq!(
            nav.resolve(&tree, &key, KeyInput::Escape),
            NavAction::Dismiss
        );

        let (tree2, key2) = tree_with(Role::Textbox);
        assert_eq!(
            nav.resolve(&tree2, &key2, KeyInput::Escape),
            NavAction::None
        );
    }

    #[test]
    fn unknown_focus_only_moves() {
        let tree = A11yTree::new();
        let nav = KeyboardNav::new();
        let missing = Key::Int(42);
        assert_eq!(
            nav.resolve(&tree, &missing, KeyInput::Tab),
            NavAction::MoveFocus(FocusMove::Next)
        );
        assert_eq!(
            nav.resolve(&tree, &missing, KeyInput::Enter),
            NavAction::None
        );
    }
}
