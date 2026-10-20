//! Loom assembly for dock workspace chrome.

use prism_editor_dock::DockContributionRegistry;
use prism_ui::Element;

use crate::support::{empty_state, label};

/// Inputs for the dock workspace Element tree.
#[derive(Clone, Debug)]
pub struct DockWorkspaceProps<'a> {
    /// Editor-to-dock contribution mapping.
    pub panels: &'a DockContributionRegistry,
    /// Stable panel ID currently shown as active.
    pub active_panel: Option<&'a str>,
    /// Materialized panel body Elements in their host order.
    pub bodies: Vec<Element>,
}

/// Builds dock tabs and a body host from the existing dock contribution map.
#[must_use]
pub fn dock_workspace(props: &DockWorkspaceProps<'_>) -> Element {
    let mut tabs = Element::box_().class("prism-editor-dock__tabs");
    for contribution in props.panels.iter() {
        let descriptor = contribution.dock();
        let mut tab = Element::box_()
            .key_str(format!("dock-tab:{}", descriptor.id))
            .class("prism-editor-dock__tab")
            .child(label(
                "prism-editor-dock__tab-label",
                descriptor.title.clone(),
            ));
        if props.active_panel == Some(descriptor.id.as_str()) {
            tab = tab.class("is-active");
        }
        if contribution.editor().singleton {
            tab = tab.class("is-singleton");
        }
        tabs = tabs.child(tab);
    }

    let body = if props.bodies.is_empty() {
        Element::box_()
            .class("prism-editor-dock__body")
            .child(empty_state(
                "prism-editor-dock__empty",
                if props.panels.is_empty() {
                    "No editor panels registered"
                } else {
                    "No panel body is active"
                },
            ))
    } else {
        Element::box_()
            .class("prism-editor-dock__body")
            .children(props.bodies.iter().cloned())
    };

    Element::custom("prism.editor.dock-workspace")
        .class("prism-editor-dock")
        .child(tabs)
        .child(body)
}
