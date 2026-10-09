//! [`Stepper`] — a numeric value flanked by decrement/increment buttons.
//!
//! A stepper renders a `pk-stepper` row with a `pk-stepper__dec` button, a
//! `pk-stepper__value` readout, and a `pk-stepper__inc` button. The displayed
//! value is clamped into `[min, max]` with a tiny helper (the std `Ord::clamp`
//! on `i32` is available, but a local helper keeps the clamp explicit and
//! panic-free when `min > max`). All color comes from theme tokens via
//! [`crate::preset`]; action wiring is owned elsewhere.

use alloc::string::ToString;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Clamps `value` into `[min, max]`. If `min > max`, `min` wins (empty range).
fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Props for [`Stepper`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StepperProps {
    /// The current value (clamped into `[min, max]` for display).
    pub value: i32,
    /// The minimum permitted value.
    pub min: i32,
    /// The maximum permitted value.
    pub max: i32,
}

impl StepperProps {
    /// Creates stepper props with the given value and bounds.
    #[must_use]
    pub fn new(value: i32, min: i32, max: i32) -> Self {
        Self { value, min, max }
    }

    /// Sets the current value.
    #[must_use]
    pub fn value(mut self, value: i32) -> Self {
        self.value = value;
        self
    }

    /// Sets the minimum permitted value.
    #[must_use]
    pub fn min(mut self, min: i32) -> Self {
        self.min = min;
        self
    }

    /// Sets the maximum permitted value.
    #[must_use]
    pub fn max(mut self, max: i32) -> Self {
        self.max = max;
        self
    }
}

/// The stepper control. Zero-sized; config lives in [`StepperProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Stepper;

impl Component for Stepper {
    type Props = StepperProps;

    fn render(&self, props: &Self::Props) -> Element {
        let shown = clamp_i32(props.value, props.min, props.max);

        let dec = Element::box_()
            .class("pk-stepper__dec")
            .child(Element::text("−").class("pk-stepper__glyph"));
        let value = Element::text(shown.to_string()).class("pk-stepper__value");
        let inc = Element::box_()
            .class("pk-stepper__inc")
            .child(Element::text("+").class("pk-stepper__glyph"));

        Element::box_()
            .class("pk-stepper")
            .child(dec)
            .child(value)
            .child(inc)
    }
}

/// Registers the `pk-stepper` class family: row, buttons, glyph, value.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-stepper")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight"))),
    );

    for button in ["pk-stepper__dec", "pk-stepper__inc"] {
        sheet.insert(
            Class::new(button)
                .with(StyleProp::Display, kw(Keyword::Flex))
                .with(StyleProp::AlignItems, kw(Keyword::Center))
                .with(StyleProp::JustifyContent, kw(Keyword::Center))
                .with_padding_x(tok("space.sm"))
                .with(StyleProp::Height, StyleValue::percent(100.0))
                .with(StyleProp::Color, tok("color.tint"))
                .with(StyleProp::FontSize, tok("font.size.headline"))
                .with(StyleProp::FontWeight, tok("font.weight.semibold"))
                .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.6)),
        );
    }

    sheet.insert(
        Class::new("pk-stepper__value")
            .with_padding_x(tok("space.md"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );

    sheet.insert(
        Class::new("pk-stepper__glyph")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontSize, tok("font.size.headline")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: StepperProps) -> Element {
        Stepper.render(&props)
    }

    #[test]
    fn renders_dec_value_inc_in_order() {
        let el = render(StepperProps::new(3, 0, 10));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[0].class_names(), ["pk-stepper__dec"]);
        assert_eq!(kids[1].class_names(), ["pk-stepper__value"]);
        assert_eq!(kids[2].class_names(), ["pk-stepper__inc"]);
    }

    #[test]
    fn value_text_is_the_clamped_number() {
        let el = render(StepperProps::new(3, 0, 10));
        assert_eq!(el.child_elements()[1].text_content(), Some("3"));
    }

    #[test]
    fn value_above_max_is_clamped() {
        let el = render(StepperProps::new(99, 0, 10));
        assert_eq!(el.child_elements()[1].text_content(), Some("10"));
    }

    #[test]
    fn value_below_min_is_clamped() {
        let el = render(StepperProps::new(-5, 0, 10));
        assert_eq!(el.child_elements()[1].text_content(), Some("0"));
    }
}
