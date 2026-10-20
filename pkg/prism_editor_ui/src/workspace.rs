//! Fixed-order composition of the complete editor surface.

use prism_ui::Element;

/// Already assembled editor regions placed into the workspace shell.
#[derive(Clone, Debug)]
pub struct EditorWorkspaceProps {
    /// Persistent shell chrome.
    pub chrome: Element,
    /// Dock tab/body host.
    pub dock: Element,
    /// Outliner region.
    pub outliner: Element,
    /// Main viewport region.
    pub viewport: Element,
    /// Inspector region.
    pub inspector: Element,
    /// Console region.
    pub console: Element,
}

/// Builds the complete editor Element tree in a stable semantic order.
#[must_use]
pub fn editor_workspace(props: &EditorWorkspaceProps) -> Element {
    let left = Element::box_()
        .class("prism-editor-workspace__left")
        .child(props.outliner.clone());
    let center = Element::box_()
        .class("prism-editor-workspace__center")
        .child(props.viewport.clone())
        .child(props.console.clone());
    let right = Element::box_()
        .class("prism-editor-workspace__right")
        .child(props.inspector.clone());
    let content = Element::box_()
        .class("prism-editor-workspace__content")
        .child(left)
        .child(center)
        .child(right);

    Element::custom("prism.editor.workspace")
        .class("prism-editor-workspace")
        .child(props.chrome.clone())
        .child(content)
        .child(props.dock.clone())
}
