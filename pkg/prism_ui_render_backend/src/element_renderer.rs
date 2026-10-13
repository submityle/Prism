//! Retained Loom Element to draw-list host integration.

use prism_ui::layout::{AvailableSpace, Size};
use prism_ui::{Element, Ui};
use prism_ui_style::{StyleSheet, Theme};

use crate::scene::RetainedScene;
use crate::DrawList;

/// Reusable retained UI host that lowers Elements into absolute draw lists.
///
/// `update` reuses Loom's keyed retained tree, solves layout, and returns the
/// current draw list. It performs no GPU submission; a renderer chooses CPU or
/// wgpu rasterization after this boundary.
pub struct ElementRenderer {
    ui: Ui<RetainedScene>,
}

impl ElementRenderer {
    /// Creates a renderer with the supplied theme and stylesheet.
    #[must_use]
    pub fn new(theme: Theme, stylesheet: StyleSheet) -> Self {
        Self {
            ui: Ui::new(RetainedScene::new())
                .with_theme(theme)
                .with_stylesheet(stylesheet),
        }
    }

    /// Reconciles an Element tree, solves it for `size`, and returns its draw list.
    #[must_use]
    pub fn update(&mut self, element: &Element, size: Size<f32>) -> DrawList {
        self.ui.set_viewport_width(size.width);
        self.ui.update(element);
        self.ui.compute_layout(Size::new(
            AvailableSpace::Definite(size.width),
            AvailableSpace::Definite(size.height),
        ));
        self.ui.backend().to_draw_list()
    }

    /// Number of retained nodes currently mounted.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.ui.node_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::Theme;

    #[test]
    fn update_reuses_retained_tree_for_same_shape() {
        let mut renderer = ElementRenderer::new(Theme::with_default_palette(), StyleSheet::new());
        let first = renderer.update(
            &Element::box_().child(Element::text("one")),
            Size::new(160.0, 80.0),
        );
        let first_nodes = renderer.node_count();
        let second = renderer.update(
            &Element::box_().child(Element::text("two")),
            Size::new(160.0, 80.0),
        );
        assert_eq!(first.len(), second.len());
        assert!(renderer.node_count() > 0);
        assert_eq!(renderer.node_count(), first_nodes);
    }
}
