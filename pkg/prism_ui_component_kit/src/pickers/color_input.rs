//! [`ColorInput`] — a textual color-value field (`#RRGGBB`).
//!
//! This is the text entry point for a color, **not** a panel: it composes a
//! `pk-color-input` row from a `pk-color-input__swatch` preview box and a
//! `pk-color-input__text` field element. When `with_picker` is set it also
//! emits a `pk-color-input__trigger` box that a host can wire to a
//! [`super::color_picker`] popover. The preview chip is tinted by the current
//! value as *data*; no color literal is baked into `render`.

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::StyleValue;

use crate::preset::StyleSheet;

/// Props for [`ColorInput`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ColorInputProps {
    /// The current color value shown in the preview chip. `None` leaves the
    /// chip neutral.
    pub value: Option<StyleValue>,
    /// Whether to emit the trailing picker-trigger affordance.
    pub with_picker: bool,
}

impl ColorInputProps {
    /// Creates default input props (no value, no trigger).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current color value.
    #[must_use]
    pub fn value(mut self, value: StyleValue) -> Self {
        self.value = Some(value);
        self
    }

    /// Enables or disables the trailing picker trigger.
    #[must_use]
    pub fn with_picker(mut self, with_picker: bool) -> Self {
        self.with_picker = with_picker;
        self
    }
}

/// The color-input control. Zero-sized; config lives in [`ColorInputProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorInput;

impl ColorInput {
    /// The accessibility role a textual color field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for ColorInput {
    type Props = ColorInputProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::StyleProp;

        let mut preview = Element::box_().class("pk-color-input__swatch");
        if let Some(value) = &props.value {
            preview = preview.style(StyleProp::BackgroundColor, value.clone());
        }
        let field = Element::custom("input").class("pk-color-input__text");

        let mut el = Element::box_()
            .class("pk-color-input")
            .child(preview)
            .child(field);
        if props.with_picker {
            el = el.child(Element::box_().class("pk-color-input__trigger"));
        }
        el
    }
}

/// Registers the `pk-color-input` class family: the row, the preview chip, the
/// text field and the optional picker trigger.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Row: a glass field holding chip + text + optional trigger.
    sheet.insert(
        Class::new("pk-color-input")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with_padding_x(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("color.fill"), Some(tok("glass.highlight")))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // Preview chip: a small square showing the resolved color (fill inline).
    sheet.insert(
        Class::new("pk-color-input__swatch")
            .with(StyleProp::Width, StyleValue::px(18.0))
            .with(StyleProp::Height, StyleValue::px(18.0))
            .with(StyleProp::MinWidth, StyleValue::px(18.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Text: the hex/token field, filling the remaining width.
    sheet.insert(
        Class::new("pk-color-input__text")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Trigger: a compact chip opening the swatch/wheel popover.
    sheet.insert(
        Class::new("pk-color-input__trigger")
            .with(StyleProp::Width, StyleValue::px(18.0))
            .with(StyleProp::Height, StyleValue::px(18.0))
            .with(StyleProp::MinWidth, StyleValue::px(18.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use prism_ui::ElementKind;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: ColorInputProps) -> Element {
        ColorInput.render(&props)
    }

    #[test]
    fn row_is_chip_then_text() {
        let el = render(ColorInputProps::new());
        assert_eq!(el.class_names(), ["pk-color-input"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-color-input__swatch"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-color-input__text"));
        assert_eq!(kids[1].kind(), &ElementKind::Custom("input".into()));
    }

    #[test]
    fn preview_is_tinted_by_the_value() {
        let el = render(ColorInputProps::new().value(StyleValue::token("color.purple")));
        let chip = &el.child_elements()[0];
        let bg = chip
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::BackgroundColor)
            .map(|(_, v)| v.clone());
        assert_eq!(bg, Some(StyleValue::token("color.purple")));
    }

    #[test]
    fn trigger_is_opt_in() {
        let plain = render(ColorInputProps::new());
        assert_eq!(plain.child_elements().len(), 2);
        let with = render(ColorInputProps::new().with_picker(true));
        let names: Vec<_> = with
            .child_elements()
            .iter()
            .flat_map(|c| c.class_names().iter().cloned())
            .collect();
        assert!(names.iter().any(|n| n == "pk-color-input__trigger"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(ColorInput::role(), Role::Textbox);
    }
}
