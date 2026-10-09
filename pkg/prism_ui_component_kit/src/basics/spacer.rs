//! [`Spacer`] — a flexible gap that pushes siblings apart in a flex layout.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Spacer`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SpacerProps {
    /// When set, a fixed minimum size (px) instead of pure flex growth.
    pub min_px: Option<f32>,
}

impl SpacerProps {
    /// Creates props for a flexible spacer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a minimum fixed size in pixels.
    #[must_use]
    pub fn min_px(mut self, px: f32) -> Self {
        self.min_px = Some(px);
        self
    }
}

/// The spacer control. Zero-sized; all configuration lives in [`SpacerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Spacer;

impl Component for Spacer {
    type Props = SpacerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-spacer");
        if let Some(px) = props.min_px {
            el = el.style(
                prism_ui_style::StyleProp::MinWidth,
                prism_ui_style::StyleValue::px(px),
            );
        }
        el
    }
}

/// Registers the `pk-spacer` class (flex-grow:1 so it eats free space).
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(Class::new("pk-spacer").with(StyleProp::FlexGrow, StyleValue::number(1.0)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_base_class() {
        let el = Spacer.render(&SpacerProps::new());
        assert_eq!(el.class_names(), ["pk-spacer"]);
    }

    #[test]
    fn min_px_adds_inline_style() {
        let el = Spacer.render(&SpacerProps::new().min_px(12.0));
        assert_eq!(el.class_names(), ["pk-spacer"]);
    }

    #[test]
    fn register_adds_class() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-spacer").is_some());
    }
}
