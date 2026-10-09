//! [`Divider`] — a hairline separator, horizontal or vertical.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The orientation of a [`Divider`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DividerOrientation {
    /// A full-width horizontal rule.
    #[default]
    Horizontal,
    /// A full-height vertical rule.
    Vertical,
}

impl DividerOrientation {
    /// The modifier suffix (e.g. `pk-divider--vertical`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            DividerOrientation::Horizontal => "horizontal",
            DividerOrientation::Vertical => "vertical",
        }
    }
}

/// Props for [`Divider`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DividerProps {
    /// Horizontal (default) or vertical.
    pub orientation: DividerOrientation,
}

impl DividerProps {
    /// Creates props for a horizontal divider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the orientation.
    #[must_use]
    pub fn orientation(mut self, orientation: DividerOrientation) -> Self {
        self.orientation = orientation;
        self
    }
}

/// The divider control. Zero-sized; all configuration lives in [`DividerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Divider;

impl Divider {
    /// The accessibility role a divider exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Presentation
    }
}

impl Component for Divider {
    type Props = DividerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut suffix = String::from("pk-divider--");
        suffix.push_str(props.orientation.suffix());
        Element::box_().class("pk-divider").class(suffix)
    }
}

/// Registers the `pk-divider` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-divider").with(StyleProp::BackgroundColor, tok("color.separator")),
    );
    sheet.insert(
        Class::new("pk-divider--horizontal")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(1.0)),
    );
    sheet.insert(
        Class::new("pk-divider--vertical")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with(StyleProp::Height, StyleValue::percent(100.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_orientation_class() {
        let el = Divider.render(&DividerProps::new().orientation(DividerOrientation::Vertical));
        assert_eq!(el.class_names(), ["pk-divider", "pk-divider--vertical"]);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-divider", "pk-divider--horizontal", "pk-divider--vertical"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
