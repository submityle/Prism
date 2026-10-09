//! [`Link`] — an inline navigational text link (a11y `Role::Link`).

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Link`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LinkProps {
    /// The visible link text.
    pub label: String,
    /// The destination (carried for the backend; not painted).
    pub href: String,
    /// Whether the link is non-interactive.
    pub disabled: bool,
}

impl LinkProps {
    /// Creates props for a link.
    #[must_use]
    pub fn new(label: impl Into<String>, href: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            href: href.into(),
            disabled: false,
        }
    }

    /// Marks the link disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The link control. Zero-sized; configuration lives in [`LinkProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Link;

impl Link {
    /// The accessibility role a link exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Link
    }
}

impl Component for Link {
    type Props = LinkProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::text(props.label.clone()).class("pk-link");
        if props.disabled {
            el = el.class("is-disabled");
        }
        el
    }
}

/// Registers the `pk-link` class.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-link")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::Color, tok("color.tint"))
            .with_state(InteractionState::Disabled, StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn renders_text_link() {
        let el = Link.render(&LinkProps::new("Home", "/home"));
        assert_eq!(el.kind(), &ElementKind::Text);
        assert_eq!(el.class_names(), ["pk-link"]);
    }

    #[test]
    fn disabled_adds_marker() {
        let el = Link.render(&LinkProps::new("Home", "/home").disabled(true));
        assert_eq!(el.class_names(), ["pk-link", "is-disabled"]);
    }

    #[test]
    fn register_adds_class() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-link").is_some());
    }
}
