//! [`HoverCard`] — a rich hover preview built on the overlay base.
//!
//! A hover card is an optional `anchor` plus a `content` element shown in a
//! floating glass surface. Unlike [`Tooltip`](crate::feedback::Tooltip) — a
//! terse text bubble — it carries arbitrary rich content (avatars, metadata,
//! actions). It composes the shared [`Popover`] base per architecture
//! invariant K7 and attaches only kit classes; [`crate::preset`] resolves every
//! value against the active theme.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`HoverCard`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HoverCardProps {
    /// The trigger element the card is positioned against.
    pub anchor: Option<Element>,
    /// The rich content shown inside the floating surface.
    pub content: Option<Element>,
    /// Whether the card is currently shown.
    pub open: bool,
}

impl HoverCardProps {
    /// Creates empty, closed hover-card props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the trigger element the card is positioned against.
    #[must_use]
    pub fn anchor(mut self, anchor: Element) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Sets the rich content shown inside the surface.
    #[must_use]
    pub fn content(mut self, content: Element) -> Self {
        self.content = Some(content);
        self
    }

    /// Sets whether the card is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The hover-card control. Zero-sized; configuration lives in [`HoverCardProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct HoverCard;

impl Component for HoverCard {
    type Props = HoverCardProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut card = Element::box_().class("pk-hover-card");
        if let Some(content) = props.content.clone() {
            card = card.child(content.class("pk-hover-card__content"));
        }

        let mut popover = PopoverProps::new().open(props.open).child(card);
        if let Some(anchor) = props.anchor.clone() {
            popover = popover.anchor(anchor);
        }
        Popover.render(&popover)
    }
}

/// Registers the `pk-hover-card` class family: the rich content panel and its
/// inner content slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Panel: a comfortably sized vertical stack of rich content.
    sheet.insert(
        Class::new("pk-hover-card")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::MinWidth, StyleValue::px(220.0))
            .with(StyleProp::MaxWidth, StyleValue::px(360.0))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Content slot: a neutral wrapper around the host-provided element.
    sheet.insert(
        Class::new("pk-hover-card__content")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: HoverCardProps) -> Element {
        HoverCard.render(&props)
    }

    fn surface(el: &Element) -> Element {
        el.child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface")
            .clone()
    }

    #[test]
    fn composes_popover_base() {
        let el = render(HoverCardProps::new());
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
    }

    #[test]
    fn open_propagates_to_popover() {
        let el = render(HoverCardProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
    }

    #[test]
    fn content_mounts_inside_the_surface() {
        let el = render(HoverCardProps::new().content(Element::text("rich")));
        let surface = surface(&el);
        let card = &surface.child_elements()[0];
        assert!(card.class_names().iter().any(|c| c == "pk-hover-card"));
        assert!(card.child_elements()[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-hover-card__content"));
    }

    #[test]
    fn anchor_is_passed_through_to_popover() {
        let el = render(HoverCardProps::new().anchor(Element::box_().class("trigger")));
        assert!(el
            .child_elements()
            .iter()
            .any(|c| c.class_names().iter().any(|n| n == "pk-popover__anchor")));
    }
}
