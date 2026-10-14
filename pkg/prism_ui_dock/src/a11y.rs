//! Accessibility projection for a solved dock plan.

use crate::{RenderPlan, TabId};
use alloc::{format, string::String};
use prism_ui::{Key, Key as UiKey};
use prism_ui_a11y::{
    A11yNode, A11yTree, AriaState, FocusMove, FocusOrder, KeyInput, KeyboardNav, Label, NavAction,
    Role,
};

/// Result of resolving a keyboard action against the dock accessibility tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DockA11yAction {
    /// Move the platform focus ring.
    MoveFocus(FocusMove),
    /// Activate a panel/document tab.
    Activate(TabId),
    /// Dismiss the current reveal/modal context.
    Dismiss,
    /// No dock action is associated with the input.
    None,
}

/// Stable accessibility projection for a [`RenderPlan`].
#[derive(Clone, Debug)]
pub struct DockA11ySnapshot {
    /// Flat semantic tree consumed by the a11y adapter.
    pub tree: A11yTree,
    /// Computed sequential tab order.
    pub focus_order: FocusOrder,
}

impl DockA11ySnapshot {
    /// Builds stable tab, panel, document, and auto-hide button semantics.
    pub fn build(plan: &RenderPlan) -> Self {
        let mut tree = A11yTree::new();
        for tab in &plan.tabs {
            let key = key_for_tab(&tab.id);
            let label = Label::text(tab_label(&tab.id));
            tree.insert(
                A11yNode::builder(key, Role::Tab)
                    .label(label)
                    .state(AriaState::new().selected(tab.active))
                    .focusable(true)
                    .build(),
            );
        }
        for panel in &plan.panels {
            tree.insert(
                A11yNode::builder(key_for_panel(&panel.id), Role::Group)
                    .label(Label::text(panel.id.clone()))
                    .state(AriaState::new().hidden(matches!(
                        panel.visibility,
                        crate::PanelVisibility::AutoHideCollapsed
                    )))
                    .build(),
            );
        }
        for document in &plan.documents {
            tree.insert(
                A11yNode::builder(key_for_document(&document.id), Role::Group)
                    .label(Label::text(document.id.clone()))
                    .build(),
            );
        }
        for bar in &plan.autohide.bars {
            for button in &bar.buttons {
                tree.insert(
                    A11yNode::builder(key_for_autohide(&button.panel), Role::Button)
                        .label(Label::text(format!("Show {}", button.panel)))
                        .state(AriaState::new().selected(button.active))
                        .build(),
                );
            }
        }
        let focus_order = FocusOrder::compute(&tree);
        Self { tree, focus_order }
    }

    /// Resolves a semantic key press without mutating the layout.
    pub fn resolve(&self, current: &UiKey, input: KeyInput) -> DockA11yAction {
        match KeyboardNav::new().resolve(&self.tree, current, input) {
            NavAction::MoveFocus(move_) => DockA11yAction::MoveFocus(move_),
            NavAction::Dismiss => DockA11yAction::Dismiss,
            NavAction::Activate => self
                .tab_for_key(current)
                .map(DockA11yAction::Activate)
                .unwrap_or(DockA11yAction::None),
            NavAction::Toggle | NavAction::None => DockA11yAction::None,
        }
    }

    /// Returns the tab represented by a semantic key.
    pub fn tab_for_key(&self, key: &Key) -> Option<TabId> {
        self.tree.get(key)?;
        match key {
            Key::Str(value) if value.starts_with("dock:tab:panel:") => {
                Some(TabId::Panel(value[15..].to_owned()))
            }
            Key::Str(value) if value.starts_with("dock:tab:document:") => {
                Some(TabId::Document(value[18..].to_owned()))
            }
            _ => None,
        }
    }
}

fn key_for_tab(tab: &TabId) -> Key {
    match tab {
        TabId::Panel(id) => key_for_panel_tab(id),
        TabId::Document(id) => key_for_document_tab(id),
    }
}

fn key_for_panel_tab(id: &str) -> Key {
    Key::Str(format!("dock:tab:panel:{id}"))
}

fn key_for_document_tab(id: &str) -> Key {
    Key::Str(format!("dock:tab:document:{id}"))
}

fn key_for_panel(id: &str) -> Key {
    Key::Str(format!("dock:panel:{id}"))
}

fn key_for_document(id: &str) -> Key {
    Key::Str(format!("dock:document:{id}"))
}

fn key_for_autohide(id: &str) -> Key {
    Key::Str(format!("dock:autohide:{id}"))
}

fn tab_label(tab: &TabId) -> String {
    match tab {
        TabId::Panel(id) | TabId::Document(id) => id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockLayout, DockNode, Viewport};

    #[test]
    fn tabs_are_focusable_and_enter_activates_them() {
        let layout = DockLayout {
            root: DockNode::TabStack {
                tabs: vec!["one".into(), "two".into()],
                active: 0,
            },
            ..DockLayout::default()
        };
        let plan = crate::solve(&layout, Viewport::new(640, 480)).unwrap();
        let snapshot = DockA11ySnapshot::build(&plan);
        let key = key_for_panel_tab("two");
        assert!(snapshot.focus_order.position(&key).is_some());
        assert_eq!(
            snapshot.resolve(&key, KeyInput::Enter),
            DockA11yAction::Activate(TabId::Panel("two".into()))
        );
    }
}
