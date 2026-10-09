//! [`Tooltip`] — a small caption bubble shown next to wrapped content.
//!
//! A tooltip wraps its trigger `children` and, beside them, renders a caption
//! bubble composed from the shared [`Popover`](crate::feedback::Popover) base
//! (architecture invariant K7). The bubble reuses the popover glass surface and
//! only layers the `pk-tooltip` caption treatment on top; it attaches kit
//! classes and lets [`crate::preset`] resolve every value against the theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::feedback::popover::{Placement, Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`Tooltip`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TooltipProps {
    /// The caption text shown in the bubble.
    pub text: String,
    /// The trigger content the tooltip is attached to.
    pub children: Vec<Element>,
    /// Which side the bubble prefers to open toward.
    pub placement: Placement,
}

impl TooltipProps {
    /// Creates tooltip props with the given caption and the default placement.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Appends a trigger child.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Replaces the trigger content with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Sets the preferred placement.
    #[must_use]
    pub fn placement(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }
}

/// The tooltip control. Zero-sized; all configuration lives in [`TooltipProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tooltip;

impl Component for Tooltip {
    type Props = TooltipProps;

    fn render(&self, props: &Self::Props) -> Element {
        // Compose the overlay base: the caption rides on the popover surface.
        let caption = Element::text(props.text.clone()).class("pk-tooltip__caption");
        let bubble = Popover
            .render(
                &PopoverProps::new()
                    .open(true)
                    .placement(props.placement)
                    .child(caption),
            )
            .class("pk-tooltip__bubble");

        Element::box_()
            .class("pk-tooltip")
            .children(props.children.iter().cloned())
            .child(bubble)
    }
}

/// Registers the `pk-tooltip` class family: the inline trigger wrapper, the
/// composed bubble modifier and the small caption text.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Wrapper: keeps the trigger and its bubble on one inline-ish row.
    sheet.insert(
        Class::new("pk-tooltip")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Bubble: tightens the composed popover surface for a compact caption.
    sheet.insert(
        Class::new("pk-tooltip__bubble").with(StyleProp::MaxWidth, tok("space.xxxl")),
    );

    // Caption: footnote type on the glass surface.
    sheet.insert(
        Class::new("pk-tooltip__caption")
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TooltipProps) -> Element {
        Tooltip.render(&props)
    }

    #[test]
    fn wraps_trigger_then_bubble() {
        let el = render(TooltipProps::new("Help").child(Element::box_().class("trigger")));
        assert_eq!(el.class_names(), ["pk-tooltip"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "trigger"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-tooltip__bubble"));
    }

    #[test]
    fn bubble_composes_popover_base() {
        let el = render(TooltipProps::new("Help"));
        let bubble = &el.child_elements()[0];
        assert!(bubble.class_names().iter().any(|c| c == "pk-popover"));
        assert!(bubble.class_names().iter().any(|c| c == "pk-tooltip__bubble"));
    }

    #[test]
    fn caption_text_rides_on_popover_surface() {
        let el = render(TooltipProps::new("Save file"));
        let bubble = &el.child_elements()[0];
        let surface = bubble
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface");
        let caption = &surface.child_elements()[0];
        assert_eq!(caption.text_content(), Some("Save file"));
        assert!(caption.class_names().iter().any(|c| c == "pk-tooltip__caption"));
    }
}
