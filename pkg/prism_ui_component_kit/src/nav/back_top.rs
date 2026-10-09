//! [`BackTop`] — a floating "scroll to top" affordance.
//!
//! A back-to-top control renders a `pk-back-top` floating round button holding
//! an optional `icon`. When `visible` it gains the block-level
//! `pk-back-top--visible` modifier (fully opaque); otherwise it is faded out.
//! It exposes the [`Role::Button`](prism_ui_a11y::Role::Button) accessibility
//! role. Only kit class names are attached; whether the button is visible — and
//! the scroll-to-top action itself — is driven by the scroll layer upstream.

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Props for [`BackTop`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BackTopProps {
    /// Optional icon rendered inside the button.
    pub icon: Option<Element>,
    /// Whether the button is currently revealed.
    pub visible: bool,
}

impl BackTopProps {
    /// Creates hidden back-to-top props with no icon.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the button icon.
    #[must_use]
    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Sets the visible state.
    #[must_use]
    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }
}

/// The back-to-top control. Zero-sized; config lives in [`BackTopProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BackTop;

impl BackTop {
    /// The accessibility role the control exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Button
    }
}

impl Component for BackTop {
    type Props = BackTopProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mods: &[&str] = if props.visible { &["visible"] } else { &[] };
        let mut el = Element::box_();
        for name in classes("pk-back-top", mods) {
            el = el.class(name);
        }
        if let Some(icon) = props.icon.clone() {
            el = el.child(icon.class("pk-back-top__icon"));
        }
        el
    }
}

/// Registers the `pk-back-top` class family: the floating glass button, its
/// visible modifier and the icon slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Button: a fixed-size round glass puck, faded out until revealed.
    sheet.insert(
        Class::new("pk-back-top")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(44.0))
            .with(StyleProp::Height, StyleValue::px(44.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::Opacity, StyleValue::number(0.0)),
    );

    // Visible: fully reveal the button.
    sheet.insert(
        Class::new("pk-back-top--visible").with(StyleProp::Opacity, StyleValue::number(1.0)),
    );

    // Icon slot: never shrinks.
    sheet.insert(
        Class::new("pk-back-top__icon").with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: BackTopProps) -> Element {
        BackTop.render(&props)
    }

    #[test]
    fn hidden_has_only_base_class() {
        let el = render(BackTopProps::new());
        assert_eq!(el.class_names(), ["pk-back-top"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn visible_adds_modifier_class() {
        let el = render(BackTopProps::new().visible(true));
        assert_eq!(el.class_names(), ["pk-back-top", "pk-back-top--visible"]);
    }

    #[test]
    fn renders_icon_with_slot_class() {
        let el = render(BackTopProps::new().icon(Element::box_().class("ic")));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-back-top__icon"));
    }

    #[test]
    fn role_is_button() {
        assert_eq!(BackTop::role(), Role::Button);
    }
}
