//! [`InfiniteScroll`] — a lazily-growing list with a load sentinel.
//!
//! An infinite scroll renders a `pk-infinite-scroll` column of item elements
//! followed by a tail marker. While more items remain it emits a
//! `pk-infinite-scroll__sentinel` (gaining `--loading` while a fetch is in
//! flight); once exhausted it emits a `pk-infinite-scroll__end` cap instead.
//!
//! The control is purely presentational: detecting that the sentinel scrolled
//! into view and triggering the next page is driven by the layer above, which
//! flips [`loading`](InfiniteScrollProps::loading) and
//! [`has_more`](InfiniteScrollProps::has_more). [`LoadMore`] is a convenience
//! alias for the same control.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`InfiniteScroll`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct InfiniteScrollProps {
    /// The already-loaded items, rendered top-to-bottom.
    pub items: Vec<Element>,
    /// Whether a fetch for the next page is currently in flight.
    pub loading: bool,
    /// Whether more items remain to be loaded.
    pub has_more: bool,
}

impl InfiniteScrollProps {
    /// Creates empty infinite-scroll props (idle, exhausted).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an item.
    #[must_use]
    pub fn item(mut self, element: Element) -> Self {
        self.items.push(element);
        self
    }

    /// Replaces the items with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets whether a fetch is in flight.
    #[must_use]
    pub fn loading(mut self, loading: bool) -> Self {
        self.loading = loading;
        self
    }

    /// Sets whether more items remain.
    #[must_use]
    pub fn has_more(mut self, has_more: bool) -> Self {
        self.has_more = has_more;
        self
    }
}

/// The infinite-scroll control. Zero-sized; config lives in
/// [`InfiniteScrollProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct InfiniteScroll;

/// Alias for [`InfiniteScroll`] when read as a "load more" affordance.
pub type LoadMore = InfiniteScroll;

impl Component for InfiniteScroll {
    type Props = InfiniteScrollProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-infinite-scroll")
            .children(props.items.iter().cloned());

        if props.has_more {
            // Sentinel: the upper layer observes this to trigger the next page.
            let mut sentinel = Element::box_().class("pk-infinite-scroll__sentinel");
            if props.loading {
                sentinel = sentinel.class("pk-infinite-scroll__sentinel--loading");
            }
            el = el.child(sentinel);
        } else {
            // End cap: nothing more to load.
            el = el.child(Element::box_().class("pk-infinite-scroll__end"));
        }

        el
    }
}

/// Registers the `pk-infinite-scroll` class family: the list column, the load
/// sentinel (+ loading modifier) and the end cap.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // List: a vertical column of items with a token gap.
    sheet.insert(
        Class::new("pk-infinite-scroll")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Sentinel: a centered footer slot the backend watches for intersection.
    sheet.insert(
        Class::new("pk-infinite-scroll__sentinel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with_padding_y(tok("space.md")),
    );

    // Loading: dim the sentinel while a fetch is in flight.
    sheet.insert(
        Class::new("pk-infinite-scroll__sentinel--loading").with(StyleProp::Color, tok("color.label.secondary")),
    );

    // End cap: a muted terminal marker when the list is exhausted.
    sheet.insert(
        Class::new("pk-infinite-scroll__end")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: InfiniteScrollProps) -> Element {
        InfiniteScroll.render(&props)
    }

    #[test]
    fn exhausted_by_default_shows_end_cap() {
        let el = render(InfiniteScrollProps::new());
        assert_eq!(el.class_names(), ["pk-infinite-scroll"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-infinite-scroll__end"]);
    }

    #[test]
    fn has_more_shows_sentinel_after_items() {
        let el = render(
            InfiniteScrollProps::new()
                .items([Element::box_(), Element::box_()])
                .has_more(true),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[2].class_names(), ["pk-infinite-scroll__sentinel"]);
    }

    #[test]
    fn loading_adds_modifier_to_sentinel() {
        let el = render(InfiniteScrollProps::new().has_more(true).loading(true));
        let sentinel = &el.child_elements()[0];
        assert_eq!(
            sentinel.class_names(),
            [
                "pk-infinite-scroll__sentinel",
                "pk-infinite-scroll__sentinel--loading"
            ]
        );
    }

    #[test]
    fn loading_without_has_more_still_shows_end_cap() {
        let el = render(InfiniteScrollProps::new().loading(true));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-infinite-scroll__end"]);
    }

    #[test]
    fn load_more_alias_renders_identically() {
        let via_alias = LoadMore::default().render(&InfiniteScrollProps::new().has_more(true));
        let via_name = InfiniteScroll.render(&InfiniteScrollProps::new().has_more(true));
        assert_eq!(via_alias, via_name);
    }
}
