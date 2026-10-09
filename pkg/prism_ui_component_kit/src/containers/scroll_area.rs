//! [`ScrollArea`] — a scroll container with a custom scrollbar track.
//!
//! Unlike the bare [`ScrollView`](super::scroll_view::ScrollView), a scroll
//! area draws its own scrollbar chrome: a `pk-scroll-area__track` holding a
//! `pk-scroll-area__thumb`, laid beside a `pk-scroll-area__viewport` that wraps
//! the content. The container carries `pk-scroll-area` with a `--horizontal`
//! variant for a horizontal scroll axis. The kit's
//! [`StyleProp`](prism_ui_style::StyleProp) vocabulary has no overflow
//! property, so actual clipping/scrolling and thumb sizing are backend concerns
//! keyed off these classes; the control only attaches kit class names.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`ScrollArea`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ScrollAreaProps {
    /// The scrollable content.
    pub children: Vec<Element>,
    /// Whether to render the custom scrollbar track and thumb.
    pub show_scrollbar: bool,
    /// Whether the scroll axis is horizontal (default is vertical).
    pub horizontal: bool,
}

impl ScrollAreaProps {
    /// Creates empty scroll-area props (vertical, no scrollbar chrome).
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

    /// Sets whether the scrollbar track and thumb are rendered.
    #[must_use]
    pub fn show_scrollbar(mut self, show_scrollbar: bool) -> Self {
        self.show_scrollbar = show_scrollbar;
        self
    }

    /// Selects a horizontal scroll axis.
    #[must_use]
    pub fn horizontal(mut self, horizontal: bool) -> Self {
        self.horizontal = horizontal;
        self
    }
}

/// The scroll-area control. Zero-sized; config lives in [`ScrollAreaProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ScrollArea;

impl Component for ScrollArea {
    type Props = ScrollAreaProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-scroll-area");
        if props.horizontal {
            el = el.class("pk-scroll-area--horizontal");
        }

        let viewport = Element::box_()
            .class("pk-scroll-area__viewport")
            .children(props.children.iter().cloned());
        el = el.child(viewport);

        if props.show_scrollbar {
            let thumb = Element::box_().class("pk-scroll-area__thumb");
            let track = Element::box_().class("pk-scroll-area__track").child(thumb);
            el = el.child(track);
        }

        el
    }
}

/// Registers the `pk-scroll-area` class family: container (+ horizontal
/// variant), viewport, and the scrollbar track and thumb.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a flex box stacking the viewport and the scrollbar track.
    // Clipping is a backend concern keyed off this class.
    sheet.insert(
        Class::new("pk-scroll-area")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Horizontal: stack viewport over the (now horizontal) track.
    sheet.insert(
        Class::new("pk-scroll-area--horizontal").with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    // Viewport: grows to fill, scrolls its own content column.
    sheet.insert(
        Class::new("pk-scroll-area__viewport")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    // Track: a thin capsule rail hosting the thumb.
    sheet.insert(
        Class::new("pk-scroll-area__track")
            .with(StyleProp::Width, StyleValue::px(6.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );

    // Thumb: the draggable indicator; sizing/position is a backend concern.
    sheet.insert(
        Class::new("pk-scroll-area__thumb")
            .with(StyleProp::Width, StyleValue::px(6.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ScrollAreaProps) -> Element {
        ScrollArea.render(&props)
    }

    #[test]
    fn vertical_viewport_only_by_default() {
        let el = render(ScrollAreaProps::new());
        assert_eq!(el.class_names(), ["pk-scroll-area"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-scroll-area__viewport"]);
    }

    #[test]
    fn horizontal_adds_modifier() {
        let el = render(ScrollAreaProps::new().horizontal(true));
        assert_eq!(
            el.class_names(),
            ["pk-scroll-area", "pk-scroll-area--horizontal"]
        );
    }

    #[test]
    fn scrollbar_adds_track_with_thumb_after_viewport() {
        let el = render(ScrollAreaProps::new().show_scrollbar(true));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-scroll-area__viewport"]);
        assert_eq!(kids[1].class_names(), ["pk-scroll-area__track"]);
        let thumb = kids[1].child_elements();
        assert_eq!(thumb.len(), 1);
        assert_eq!(thumb[0].class_names(), ["pk-scroll-area__thumb"]);
    }

    #[test]
    fn content_lives_in_the_viewport() {
        let el = render(ScrollAreaProps::new().children([Element::box_(), Element::box_()]));
        let viewport = &el.child_elements()[0];
        assert_eq!(viewport.child_elements().len(), 2);
    }
}
