//! Loom assembly for a renderer-neutral viewport host.

use prism_editor_viewport::{Projection, Viewport};
use prism_ui::Element;

use crate::support::{empty_state, label};

/// Inputs for the viewport host Element tree.
#[derive(Clone, Debug)]
pub struct ViewportHostProps<'a> {
    /// Current viewport model, or `None` when no viewport is open.
    pub viewport: Option<&'a Viewport>,
    /// Overlay Elements owned by the editor host, such as gizmo guides.
    pub overlays: Vec<Element>,
}

/// Builds a renderer-independent viewport host with explicit camera/settings
/// readouts and an overlay slot.
#[must_use]
pub fn viewport_host(props: &ViewportHostProps<'_>) -> Element {
    let Some(viewport) = props.viewport else {
        return Element::custom("prism.editor.viewport-host")
            .class("prism-editor-viewport-host")
            .child(empty_state(
                "prism-editor-viewport-host__empty",
                "No viewport is open",
            ));
    };

    let camera = viewport.camera();
    let projection = match camera.projection {
        Projection::Perspective { fov_millidegrees } => {
            format!("Perspective {}mdeg", fov_millidegrees)
        }
        Projection::Orthographic { scale_milliunits } => {
            format!("Orthographic {}milli", scale_milliunits)
        }
    };
    let surface = Element::custom("prism.editor.viewport-surface")
        .class("prism-editor-viewport-host__surface")
        .children(props.overlays.iter().cloned());
    let header = Element::box_()
        .class("prism-editor-viewport-host__header")
        .child(label("prism-editor-viewport-host__title", viewport.title()))
        .child(label("prism-editor-viewport-host__projection", projection));
    let footer = Element::box_()
        .class("prism-editor-viewport-host__footer")
        .child(label(
            "prism-editor-viewport-host__generation",
            format!("Generation {}", viewport.generation()),
        ))
        .child(label(
            "prism-editor-viewport-host__scale",
            format!("Scale {}%", viewport.settings().render_scale_percent),
        ));

    Element::custom("prism.editor.viewport-host")
        .key_str(format!("viewport:{}", viewport.id().raw()))
        .class("prism-editor-viewport-host")
        .child(header)
        .child(surface)
        .child(footer)
}
