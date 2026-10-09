//! [`FloatButton`] — a floating primary action with an expandable speed dial.
//!
//! A float button is a pinned `__trigger` (holding an optional `icon`) plus an
//! `open`-gated column of `__action` controls. Unlike the other feedback
//! overlays it is **not** a popover: it is a fixed-position chrome affordance,
//! so positioning is left to the host region and expressed only through kit
//! classes. It attaches only kit classes; [`crate::preset`] resolves the glass
//! surface and accent against the active theme. The [`SpeedDial`] alias names
//! the same control.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`FloatButton`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FloatButtonProps {
    /// Optional glyph shown inside the trigger.
    pub icon: Option<Element>,
    /// The expandable speed-dial actions.
    pub actions: Vec<Element>,
    /// Whether the speed dial is expanded.
    pub open: bool,
}

impl FloatButtonProps {
    /// Creates empty, collapsed float-button props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the trigger glyph.
    #[must_use]
    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Appends a speed-dial action.
    #[must_use]
    pub fn action(mut self, action: Element) -> Self {
        self.actions.push(action);
        self
    }

    /// Replaces the speed-dial actions with `actions`.
    #[must_use]
    pub fn actions<I: IntoIterator<Item = Element>>(mut self, actions: I) -> Self {
        self.actions = actions.into_iter().collect();
        self
    }

    /// Sets whether the speed dial is expanded.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The float-button control. Zero-sized; configuration lives in
/// [`FloatButtonProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FloatButton;

/// Alias for [`FloatButton`] under its common "speed dial" name.
pub type SpeedDial = FloatButton;

impl FloatButton {
    /// The accessibility role the trigger exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Button
    }
}

impl Component for FloatButton {
    type Props = FloatButtonProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-float-button");
        if props.open {
            el = el.class("pk-float-button--open");
        }

        // Expanded actions stack above the trigger, each a smaller pill.
        for action in &props.actions {
            el = el.child(action.clone().class("pk-float-button__action"));
        }

        let mut trigger = Element::box_().class("pk-float-button__trigger");
        if let Some(icon) = props.icon.clone() {
            trigger = trigger.child(icon);
        }
        el.child(trigger)
    }
}

/// Registers the `pk-float-button` class family: the pinned stack, the
/// `--open` state, the primary trigger and the speed-dial action pills.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Stack: a bottom-anchored column; actions above, trigger below. The
    // margin nudges it off the trailing/bottom edge of its host region.
    sheet.insert(
        Class::new("pk-float-button")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::ColumnReverse))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::MarginRight, tok("space.xl"))
            .with(StyleProp::MarginBottom, tok("space.xl")),
    );

    // Open: slightly roomier spacing when the dial is expanded.
    sheet.insert(
        Class::new("pk-float-button--open").with(StyleProp::Gap, tok("space.md")),
    );

    // Trigger: the hero glass circle.
    sheet.insert(
        Class::new("pk-float-button__trigger")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(56.0))
            .with(StyleProp::Height, StyleValue::px(56.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.94))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.84)),
    );

    // Action: a smaller secondary glass pill revealed on expand.
    sheet.insert(
        Class::new("pk-float-button__action")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(44.0))
            .with(StyleProp::Height, StyleValue::px(44.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("color.fill"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 4.0, 12.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: FloatButtonProps) -> Element {
        FloatButton.render(&props)
    }

    #[test]
    fn role_is_button() {
        assert_eq!(FloatButton::role(), Role::Button);
    }

    #[test]
    fn base_has_trigger() {
        let el = render(FloatButtonProps::new());
        assert!(el.class_names().iter().any(|c| c == "pk-float-button"));
        let trigger = el.child_elements();
        let trigger = trigger.last().unwrap();
        assert!(trigger.class_names().iter().any(|c| c == "pk-float-button__trigger"));
    }

    #[test]
    fn open_adds_modifier_class() {
        let el = render(FloatButtonProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-float-button--open"));
    }

    #[test]
    fn actions_are_classed_and_precede_the_trigger() {
        let el = render(
            FloatButtonProps::new()
                .icon(Element::box_().class("plus"))
                .action(Element::box_().class("share"))
                .action(Element::box_().class("copy")),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-float-button__action"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-float-button__action"));
        assert!(kids[2].class_names().iter().any(|c| c == "pk-float-button__trigger"));
    }

    #[test]
    fn trigger_holds_the_icon() {
        let el = render(FloatButtonProps::new().icon(Element::box_().class("plus")));
        let trigger = el.child_elements();
        let trigger = trigger.last().unwrap();
        assert!(trigger.child_elements()[0].class_names().iter().any(|c| c == "plus"));
    }

    #[test]
    fn speed_dial_alias_is_the_same_control() {
        let el = SpeedDial::default().render(&FloatButtonProps::new());
        assert!(el.class_names().iter().any(|c| c == "pk-float-button"));
    }
}
