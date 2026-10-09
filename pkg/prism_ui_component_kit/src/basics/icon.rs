//! [`Icon`] — a glyph placeholder sized by [`ControlSize`].
//!
//! The kit does not ship a glyph set; `Icon` renders a named
//! `Element::custom("pk-icon")` carrying the glyph name so a backend icon font
//! or SVG sprite can resolve it. Size resolves from a token.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, ControlSize};
use crate::preset::StyleSheet;

/// Props for [`Icon`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct IconProps {
    /// The glyph name (resolved by the backend sprite/font).
    pub name: String,
    /// The density step controlling the box size.
    pub size: ControlSize,
}

impl IconProps {
    /// Creates props for a named icon.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Sets the size step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }
}

/// The icon control. Zero-sized; all configuration lives in [`IconProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Icon;

impl Component for Icon {
    type Props = IconProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::custom("pk-icon");
        for name in classes("pk-icon", &[props.size.suffix()]) {
            el = el.class(name);
        }
        el.child(Element::text(props.name.clone()).class("pk-icon__name"))
    }
}

/// Registers the `pk-icon` family: a base plus a per-size square.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(Class::new("pk-icon").with(StyleProp::Color, tok("color.label")));
    for (size, px) in [
        (ControlSize::Small, 16.0),
        (ControlSize::Medium, 20.0),
        (ControlSize::Large, 28.0),
    ] {
        let mut name = String::from("pk-icon--");
        name.push_str(size.suffix());
        sheet.insert(
            Class::new(name)
                .with(StyleProp::Width, StyleValue::px(px))
                .with(StyleProp::Height, StyleValue::px(px)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn renders_custom_element_with_size_class() {
        let el = Icon.render(&IconProps::new("gear").size(ControlSize::Large));
        assert_eq!(el.kind(), &ElementKind::Custom(String::from("pk-icon")));
        assert_eq!(el.class_names(), ["pk-icon", "pk-icon--lg"]);
    }

    #[test]
    fn register_adds_all_sizes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-icon", "pk-icon--sm", "pk-icon--md", "pk-icon--lg"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
