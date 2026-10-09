//! [`Collapse`] — a height/opacity reveal for show/hide content.
//!
//! Unlike the phase-driven transitions, a collapse is a simple two-state
//! toggle: `open` or closed. It attaches `pk-collapse` plus the shared
//! `is-open`/`is-closed` state class and wraps its children. The closed state
//! collapses [`Height`](prism_ui_style::StyleProp::Height) to zero and hides
//! via opacity; opening resets both. [`crate::nav`]'s Accordion builds on this.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Collapse`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CollapseProps {
    /// Whether the content is revealed (`true`) or collapsed (`false`).
    pub open: bool,
    /// Content wrapped by the collapse box.
    pub children: Vec<Element>,
}

impl CollapseProps {
    /// Creates closed props with no children.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the open/closed state.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Appends a single child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the children with the given iterator's items.
    #[must_use]
    pub fn children(mut self, children: impl IntoIterator<Item = Element>) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The collapse control. Zero-sized; all configuration lives in
/// [`CollapseProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Collapse;

impl Component for Collapse {
    type Props = CollapseProps;

    fn render(&self, props: &Self::Props) -> Element {
        let state = if props.open { "is-open" } else { "is-closed" };
        Element::box_()
            .class("pk-collapse")
            .class(state)
            .children(props.children.iter().cloned())
    }
}

/// Registers the `pk-collapse` family. The base is a visible, auto-height box;
/// `is-open` restates that explicitly, while `is-closed` collapses height to
/// zero and hides via opacity.
///
/// Note: `is-open`/`is-closed` are intentionally shared state names (per the
/// kit's state-class convention) rather than block-scoped modifiers.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(
        Class::new("pk-collapse")
            .with(StyleProp::Height, StyleValue::auto())
            .with(StyleProp::Opacity, StyleValue::number(1.0)),
    );
    sheet.insert(
        Class::new("is-open")
            .with(StyleProp::Height, StyleValue::auto())
            .with(StyleProp::Opacity, StyleValue::number(1.0)),
    );
    sheet.insert(
        Class::new("is-closed")
            .with(StyleProp::Height, StyleValue::px(0.0))
            .with(StyleProp::Opacity, StyleValue::number(0.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CollapseProps) -> Element {
        Collapse.render(&props)
    }

    #[test]
    fn open_attaches_is_open() {
        let el = render(CollapseProps::new().open(true));
        assert_eq!(el.class_names(), ["pk-collapse", "is-open"]);
    }

    #[test]
    fn closed_by_default_attaches_is_closed() {
        let el = render(CollapseProps::new());
        assert_eq!(el.class_names(), ["pk-collapse", "is-closed"]);
    }

    #[test]
    fn wraps_children() {
        let el = render(
            CollapseProps::new()
                .open(true)
                .child(Element::text("row")),
        );
        assert_eq!(el.child_elements().len(), 1);
    }
}
