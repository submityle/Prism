//! [`Grid`] — a two-dimensional grid container.
//!
//! A grid renders a `pk-grid` box with [`Display::Grid`](prism_ui_style::Keyword::Grid)
//! and a token-backed gap. The desired `columns` count is carried in the props
//! as a backend hint: there is no column-count style prop in the kit's
//! [`StyleProp`](prism_ui_style::StyleProp) vocabulary, so a capable layout
//! backend reads `columns` to size the tracks. The control attaches only kit
//! class names; the gap resolves from theme tokens via [`crate::preset`].

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::ControlSize;
use crate::preset::StyleSheet;

/// The modifier class for a gap density step.
#[must_use]
const fn gap_class(size: ControlSize) -> &'static str {
    match size {
        ControlSize::Small => "pk-grid--gap-sm",
        ControlSize::Medium => "pk-grid--gap-md",
        ControlSize::Large => "pk-grid--gap-lg",
    }
}

/// Props for [`Grid`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GridProps {
    /// The number of columns the grid should lay out (a backend layout hint).
    pub columns: u16,
    /// The gap density between cells.
    pub gap: ControlSize,
    /// The grid cells.
    pub children: Vec<Element>,
}

impl GridProps {
    /// Creates empty grid props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the column count hint.
    #[must_use]
    pub fn columns(mut self, columns: u16) -> Self {
        self.columns = columns;
        self
    }

    /// Sets the gap density between cells.
    #[must_use]
    pub fn gap(mut self, gap: ControlSize) -> Self {
        self.gap = gap;
        self
    }

    /// Appends a cell.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the cells with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The grid control. Zero-sized; all configuration lives in [`GridProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Grid;

impl Grid {
    /// The accessibility role a grid container exposes (a generic grouping).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Grid {
    type Props = GridProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-grid")
            .class(gap_class(props.gap))
            .children(props.children.iter().cloned())
    }
}

/// Registers the `pk-grid` class family: base grid plus gap densities.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Base: a grid box. Track sizing is a backend concern driven by `columns`.
    sheet.insert(Class::new("pk-grid").with(StyleProp::Display, kw(Keyword::Grid)));

    sheet.insert(Class::new("pk-grid--gap-sm").with(StyleProp::Gap, tok("space.xs")));
    sheet.insert(Class::new("pk-grid--gap-md").with(StyleProp::Gap, tok("space.sm")));
    sheet.insert(Class::new("pk-grid--gap-lg").with(StyleProp::Gap, tok("space.lg")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: GridProps) -> Element {
        Grid.render(&props)
    }

    #[test]
    fn attaches_base_and_gap_class() {
        let el = render(GridProps::new().columns(3).gap(ControlSize::Small));
        assert_eq!(el.class_names(), ["pk-grid", "pk-grid--gap-sm"]);
    }

    #[test]
    fn cells_are_direct_children() {
        let el = render(GridProps::new().children([Element::box_(), Element::box_()]));
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn columns_is_carried_in_props() {
        let props = GridProps::new().columns(4);
        assert_eq!(props.columns, 4);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Grid::role(), Role::Group);
    }
}
