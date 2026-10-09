//! [`Card`] — a surface that groups header/body/footer content.
//!
//! A card renders a `pk-card` surface: an optional `pk-card__header`, a
//! `pk-card__body` wrapping its children, and an optional `pk-card__footer`.
//! The `glass` flag swaps the opaque surface for the frosted
//! [`with_glass`](prism_ui_style::Class::with_glass) treatment. Every value
//! resolves from theme tokens via [`crate::preset`]; the control attaches only
//! kit class names.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Card`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CardProps {
    /// Optional header rendered above the body.
    pub header: Option<Element>,
    /// The card's body content.
    pub children: Vec<Element>,
    /// Optional footer rendered below the body.
    pub footer: Option<Element>,
    /// Whether to use the frosted-glass surface instead of the opaque one.
    pub glass: bool,
}

impl CardProps {
    /// Creates empty card props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the header element.
    #[must_use]
    pub fn header(mut self, element: Element) -> Self {
        self.header = Some(element);
        self
    }

    /// Appends a body child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the body children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Sets the footer element.
    #[must_use]
    pub fn footer(mut self, element: Element) -> Self {
        self.footer = Some(element);
        self
    }

    /// Selects the frosted-glass surface variant.
    #[must_use]
    pub fn glass(mut self, glass: bool) -> Self {
        self.glass = glass;
        self
    }
}

/// The card control. Zero-sized; all configuration lives in [`CardProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Card;

impl Card {
    /// The accessibility role a card exposes (a generic grouping).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Card {
    type Props = CardProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-card");
        if props.glass {
            el = el.class("pk-card--glass");
        }
        if let Some(header) = props.header.clone() {
            el = el.child(header.class("pk-card__header"));
        }
        let body = Element::box_()
            .class("pk-card__body")
            .children(props.children.iter().cloned());
        el = el.child(body);
        if let Some(footer) = props.footer.clone() {
            el = el.child(footer.class("pk-card__footer"));
        }
        el
    }
}

/// Registers the `pk-card` class family: surface, glass variant, header, body,
/// footer.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Surface: a vertical stack with rounded corners, a hairline border and a
    // secondary surface fill.
    sheet.insert(
        Class::new("pk-card")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.md"))
            .with_padding_x(tok("space.lg"))
            .with_padding_y(tok("space.lg"))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary")),
    );

    // Glass variant: frosted tint, lit rim, soft drop shadow.
    sheet.insert(
        Class::new("pk-card--glass")
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow")),
    );

    // Header: headline typography.
    sheet.insert(
        Class::new("pk-card__header")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: a vertical stack of the card's children.
    sheet.insert(
        Class::new("pk-card__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Footer: muted footnote typography.
    sheet.insert(
        Class::new("pk-card__footer")
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CardProps) -> Element {
        Card.render(&props)
    }

    #[test]
    fn body_only_when_no_header_or_footer() {
        let el = render(CardProps::new().child(Element::box_()));
        assert_eq!(el.class_names(), ["pk-card"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].class_names(), ["pk-card__body"]);
    }

    #[test]
    fn header_body_footer_render_in_order() {
        let el = render(
            CardProps::new()
                .header(Element::text("Title"))
                .child(Element::box_())
                .footer(Element::text("Footer")),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert!(children[0].class_names().iter().any(|c| c == "pk-card__header"));
        assert_eq!(children[1].class_names(), ["pk-card__body"]);
        assert!(children[2].class_names().iter().any(|c| c == "pk-card__footer"));
    }

    #[test]
    fn glass_adds_modifier_class() {
        let el = render(CardProps::new().glass(true));
        assert_eq!(el.class_names(), ["pk-card", "pk-card--glass"]);
    }

    #[test]
    fn body_wraps_all_children() {
        let el = render(CardProps::new().children([Element::box_(), Element::box_()]));
        let body = &el.child_elements()[0];
        assert_eq!(body.child_elements().len(), 2);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Card::role(), Role::Group);
    }
}
