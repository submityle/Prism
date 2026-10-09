//! [`Accordion`] — a vertical stack of collapsible sections.
//!
//! An accordion renders a `pk-accordion` container whose children are
//! `pk-accordion__item` sections. Each item pairs a `pk-accordion__header`
//! (its title) with a `pk-accordion__panel` (its content) and carries the
//! `is-open` marker when expanded. Colors, spacing and separators resolve from
//! theme tokens via [`crate::preset`]; the control attaches only kit classes.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single accordion section: a title, a content panel, and open state.
#[derive(Clone, Debug, PartialEq)]
pub struct AccordionItem {
    /// The header title rendered in the always-visible row.
    pub title: Element,
    /// The collapsible content revealed when the item is open.
    pub content: Element,
    /// Whether the item is expanded.
    pub open: bool,
}

impl AccordionItem {
    /// Creates a collapsed item from a `title` and `content`.
    #[must_use]
    pub fn new(title: Element, content: Element) -> Self {
        Self {
            title,
            content,
            open: false,
        }
    }

    /// Sets the open state.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// Props for [`Accordion`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AccordionProps {
    /// The ordered sections, rendered top-to-bottom.
    pub items: Vec<AccordionItem>,
}

impl AccordionProps {
    /// Creates empty accordion props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a section.
    #[must_use]
    pub fn item(mut self, item: AccordionItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the sections with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = AccordionItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }
}

/// The accordion control. Zero-sized; config lives in [`AccordionProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Accordion;

impl Accordion {
    /// The accessibility role an accordion exposes (a generic grouping).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Accordion {
    type Props = AccordionProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-accordion");
        for item in &props.items {
            let mut section = Element::box_().class("pk-accordion__item");
            if item.open {
                section = section.class("is-open");
            }
            let header = item.title.clone().class("pk-accordion__header");
            let panel = item.content.clone().class("pk-accordion__panel");
            section = section.child(header).child(panel);
            el = el.child(section);
        }
        el
    }
}

/// Registers the `pk-accordion` class family: container, item, header, panel.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack with a hairline border and rounded corners.
    sheet.insert(
        Class::new("pk-accordion")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary")),
    );

    // Item: a vertical stack separated from the next by a hairline.
    sheet.insert(
        Class::new("pk-accordion__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Header: the clickable title row with headline-adjacent emphasis.
    sheet.insert(
        Class::new("pk-accordion__header")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Panel: the revealed content region with gutter padding.
    sheet.insert(
        Class::new("pk-accordion__panel")
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: AccordionProps) -> Element {
        Accordion.render(&props)
    }

    #[test]
    fn empty_accordion_has_no_items() {
        let el = render(AccordionProps::new());
        assert_eq!(el.class_names(), ["pk-accordion"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn each_item_has_header_then_panel() {
        let el = render(
            AccordionProps::new()
                .item(AccordionItem::new(Element::text("A"), Element::text("body A")))
                .item(AccordionItem::new(Element::text("B"), Element::text("body B"))),
        );
        let items = el.child_elements();
        assert_eq!(items.len(), 2);
        for item in items {
            assert_eq!(item.class_names(), ["pk-accordion__item"]);
            let kids = item.child_elements();
            assert_eq!(kids.len(), 2);
            assert!(kids[0].class_names().iter().any(|c| c == "pk-accordion__header"));
            assert!(kids[1].class_names().iter().any(|c| c == "pk-accordion__panel"));
        }
    }

    #[test]
    fn open_item_carries_marker() {
        let el = render(AccordionProps::new().item(
            AccordionItem::new(Element::text("Open"), Element::text("body")).open(true),
        ));
        let item = &el.child_elements()[0];
        assert_eq!(item.class_names(), ["pk-accordion__item", "is-open"]);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Accordion::role(), Role::Group);
    }
}
