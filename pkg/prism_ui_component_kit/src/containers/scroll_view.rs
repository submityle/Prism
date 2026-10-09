//! [`ScrollView`] — a scrollable container for overflowing content.
//!
//! A scroll view renders a `pk-scroll-view` flex container (vertical by
//! default, `--horizontal` for a row). The kit's
//! [`StyleProp`](prism_ui_style::StyleProp) vocabulary has no overflow
//! property, so actual clipping/scrolling is a backend concern keyed off this
//! class; the control only attaches kit class names and lays its children out
//! along the scroll axis.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`ScrollView`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ScrollViewProps {
    /// The scrollable content.
    pub children: Vec<Element>,
    /// Whether the scroll axis is horizontal (default is vertical).
    pub horizontal: bool,
}

impl ScrollViewProps {
    /// Creates empty scroll-view props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Selects a horizontal scroll axis.
    #[must_use]
    pub fn horizontal(mut self, horizontal: bool) -> Self {
        self.horizontal = horizontal;
        self
    }
}

/// The scroll-view control. Zero-sized; config lives in [`ScrollViewProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ScrollView;

impl Component for ScrollView {
    type Props = ScrollViewProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-scroll-view");
        if props.horizontal {
            el = el.class("pk-scroll-view--horizontal");
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-scroll-view` class family: base (vertical) plus the
/// horizontal modifier.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::kw;

    // Base: a vertical flex container. Overflow clipping is a backend concern.
    sheet.insert(
        Class::new("pk-scroll-view")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    // Horizontal: switch the main axis to a row.
    sheet.insert(
        Class::new("pk-scroll-view--horizontal").with(StyleProp::FlexDirection, kw(Keyword::Row)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ScrollViewProps) -> Element {
        ScrollView.render(&props)
    }

    #[test]
    fn vertical_by_default() {
        let el = render(ScrollViewProps::new());
        assert_eq!(el.class_names(), ["pk-scroll-view"]);
    }

    #[test]
    fn horizontal_adds_modifier() {
        let el = render(ScrollViewProps::new().horizontal(true));
        assert_eq!(
            el.class_names(),
            ["pk-scroll-view", "pk-scroll-view--horizontal"]
        );
    }

    #[test]
    fn children_are_direct_children() {
        let el = render(ScrollViewProps::new().children([Element::box_(), Element::box_()]));
        assert_eq!(el.child_elements().len(), 2);
    }
}
