//! [`Popover`] — the kit's overlay base (architecture invariant K7).
//!
//! Every overlay-backed feedback control shares this one surface. A popover is
//! an optional `anchor` plus a floating glass `__surface` holding arbitrary
//! children, parameterised by a [`Placement`] hint. It carries no color or
//! shadow of its own: it attaches the `pk-popover` class family and lets
//! [`crate::preset`] resolve every value against the active theme.
//!
//! Concrete screen positioning is deliberately left to the backend; the control
//! only emits the placement intent as a modifier class so a light/dark or
//! accent change requires no change here. [`crate::feedback::Tooltip`] and
//! [`crate::feedback::Dialog`] compose this base in their own `render`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Which side of its anchor a popover prefers to open toward.
///
/// This is only an intent hint: the control emits it as a `--placement`
/// modifier and the backend resolves concrete coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Placement {
    /// Open above the anchor.
    Top,
    /// Open below the anchor (the default).
    #[default]
    Bottom,
    /// Open to the left of the anchor.
    Left,
    /// Open to the right of the anchor.
    Right,
}

impl Placement {
    /// The modifier suffix used in class names (e.g. `pk-popover--top`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Placement::Top => "top",
            Placement::Bottom => "bottom",
            Placement::Left => "left",
            Placement::Right => "right",
        }
    }
}

/// Props for [`Popover`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PopoverProps {
    /// Whether the overlay is currently shown.
    pub open: bool,
    /// Optional trigger element the surface is positioned against.
    pub anchor: Option<Element>,
    /// The floating surface's content.
    pub children: Vec<Element>,
    /// The side the surface prefers to open toward.
    pub placement: Placement,
}

impl PopoverProps {
    /// Creates empty, closed popover props with the default placement.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether the overlay is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Sets the trigger element the surface is positioned against.
    #[must_use]
    pub fn anchor(mut self, anchor: Element) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Appends a child to the floating surface.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Replaces the surface content with `children`.
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

/// The popover control. Zero-sized; all configuration lives in [`PopoverProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Popover;

impl Component for Popover {
    type Props = PopoverProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-popover", &[props.placement.suffix()]) {
            el = el.class(name);
        }
        if props.open {
            el = el.class("is-open");
        }

        if let Some(anchor) = props.anchor.clone() {
            el = el.child(anchor.class("pk-popover__anchor"));
        }

        let surface = Element::box_()
            .class("pk-popover__surface")
            .children(props.children.iter().cloned());
        el.child(surface)
    }
}

/// Registers the `pk-popover` class family: the positioning root, four
/// placement hints, the anchor slot and the frosted-glass surface.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Root: a vertical stack that gives the backend a positioning context.
    sheet.insert(
        Class::new("pk-popover")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Placement hints: a small gap toward the side the surface opens from.
    sheet.insert(Class::new("pk-popover--top").with(StyleProp::MarginBottom, tok("space.xs")));
    sheet.insert(Class::new("pk-popover--bottom").with(StyleProp::MarginTop, tok("space.xs")));
    sheet.insert(Class::new("pk-popover--left").with(StyleProp::MarginRight, tok("space.xs")));
    sheet.insert(Class::new("pk-popover--right").with(StyleProp::MarginLeft, tok("space.xs")));

    // Anchor slot: a neutral inline wrapper around the trigger.
    sheet.insert(
        Class::new("pk-popover__anchor")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center)),
    );

    // Surface: the shared frosted-glass panel every overlay reuses.
    sheet.insert(
        Class::new("pk-popover__surface")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: PopoverProps) -> Element {
        Popover.render(&props)
    }

    #[test]
    fn default_is_closed_with_bottom_placement() {
        let el = render(PopoverProps::new());
        assert_eq!(el.class_names(), ["pk-popover", "pk-popover--bottom"]);
    }

    #[test]
    fn open_adds_marker_class_and_placement_modifier() {
        let el = render(PopoverProps::new().open(true).placement(Placement::Top));
        assert_eq!(
            el.class_names(),
            ["pk-popover", "pk-popover--top", "is-open"]
        );
    }

    #[test]
    fn surface_holds_children() {
        let el = render(
            PopoverProps::new()
                .child(Element::text("a"))
                .child(Element::text("b")),
        );
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface child");
        assert_eq!(surface.child_elements().len(), 2);
    }

    #[test]
    fn anchor_precedes_surface_and_is_classed() {
        let el = render(PopoverProps::new().anchor(Element::box_().class("trigger")));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-popover__anchor"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-popover__surface"));
    }
}
