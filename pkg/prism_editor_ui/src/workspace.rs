//! Fixed-order composition of the complete editor surface.

use prism_ui::Element;

use crate::action::{ActionBindings, BoundElement, UiBuildError};

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
    build_workspace(
        &props.chrome,
        &props.dock,
        &props.outliner,
        &props.viewport,
        &props.inspector,
        &props.console,
    )
}

/// Already assembled action-aware editor regions.
#[derive(Clone, Debug)]
pub struct BoundEditorWorkspaceProps {
    /// Persistent shell chrome and its bindings.
    pub chrome: BoundElement,
    /// Dock tab/body host and its bindings.
    pub dock: BoundElement,
    /// Outliner region and its bindings.
    pub outliner: BoundElement,
    /// Main viewport region and its bindings.
    pub viewport: BoundElement,
    /// Inspector region and its bindings.
    pub inspector: BoundElement,
    /// Console region and its bindings.
    pub console: BoundElement,
}

/// Builds the complete action-aware editor Element tree.
pub fn editor_workspace_bound(
    props: &BoundEditorWorkspaceProps,
) -> Result<BoundElement, UiBuildError> {
    let mut bindings = ActionBindings::new();
    bindings.extend(props.chrome.bindings())?;
    bindings.extend(props.dock.bindings())?;
    bindings.extend(props.outliner.bindings())?;
    bindings.extend(props.viewport.bindings())?;
    bindings.extend(props.inspector.bindings())?;
    bindings.extend(props.console.bindings())?;
    let element = build_workspace(
        props.chrome.element(),
        props.dock.element(),
        props.outliner.element(),
        props.viewport.element(),
        props.inspector.element(),
        props.console.element(),
    );
    Ok(BoundElement::new(element, bindings))
}

fn build_workspace(
    chrome: &Element,
    dock: &Element,
    outliner: &Element,
    viewport: &Element,
    inspector: &Element,
    console: &Element,
) -> Element {
    let left = Element::box_()
        .class("prism-editor-workspace__left")
        .child(outliner.clone());
    let center = Element::box_()
        .class("prism-editor-workspace__center")
        .child(viewport.clone())
        .child(console.clone());
    let right = Element::box_()
        .class("prism-editor-workspace__right")
        .child(inspector.clone());
    let content = Element::box_()
        .class("prism-editor-workspace__content")
        .child(left)
        .child(center)
        .child(right);

    Element::custom("prism.editor.workspace")
        .class("prism-editor-workspace")
        .child(chrome.clone())
        .child(content)
        .child(dock.clone())
}
