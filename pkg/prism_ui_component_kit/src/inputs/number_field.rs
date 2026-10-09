//! [`NumberField`] — a numeric text input flanked by increment/decrement steps.
//!
//! A number field renders a `pk-number-field` row pairing a free-text
//! `__input` (which exposes [`Role::Textbox`], so unlike [`crate::inputs::Stepper`]
//! the value can be typed directly) with a `__controls` stack holding `__inc`
//! and `__dec` buttons. The displayed value is clamped into the optional
//! `[min, max]` bounds with a hand-rolled `no_std` helper; all color comes from
//! theme tokens via [`crate::preset`].

use alloc::string::{String, ToString};

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Clamps `value` into the optional `[min, max]` bounds. Absent bounds impose
/// no limit; an inverted range lets `min` win, so the result is never a panic.
fn clamp_opt(value: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let mut v = value;
    if let Some(mx) = max {
        v = v.min(mx);
    }
    if let Some(mn) = min {
        v = v.max(mn);
    }
    v
}

/// Props for [`NumberField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NumberFieldProps {
    /// The current value. `None` renders an empty input.
    pub value: Option<f32>,
    /// Optional lower bound used to clamp the displayed value.
    pub min: Option<f32>,
    /// Optional upper bound used to clamp the displayed value.
    pub max: Option<f32>,
    /// Optional step size (carried for callers; not rendered as structure).
    pub step: Option<f32>,
    /// Whether the field is non-interactive.
    pub disabled: bool,
}

impl NumberFieldProps {
    /// Creates empty number-field props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current value.
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = Some(value);
        self
    }

    /// Sets the lower bound.
    #[must_use]
    pub fn min(mut self, min: f32) -> Self {
        self.min = Some(min);
        self
    }

    /// Sets the upper bound.
    #[must_use]
    pub fn max(mut self, max: f32) -> Self {
        self.max = Some(max);
        self
    }

    /// Sets the step size.
    #[must_use]
    pub fn step(mut self, step: f32) -> Self {
        self.step = Some(step);
        self
    }

    /// Marks the field disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The number-field control. Zero-sized; config lives in [`NumberFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NumberField;

impl NumberField {
    /// The accessibility role the editable input exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for NumberField {
    type Props = NumberFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-number-field");
        if props.disabled {
            el = el.class("pk-number-field--disabled");
        }

        let text: String = match props.value {
            Some(v) => clamp_opt(v, props.min, props.max).to_string(),
            None => String::new(),
        };
        let input = Element::text(text).class("pk-number-field__input");

        let inc = Element::box_()
            .class("pk-number-field__inc")
            .child(Element::text("+").class("pk-number-field__glyph"));
        let dec = Element::box_()
            .class("pk-number-field__dec")
            .child(Element::text("−").class("pk-number-field__glyph"));
        let controls = Element::box_()
            .class("pk-number-field__controls")
            .child(inc)
            .child(dec);

        el.child(input).child(controls)
    }
}

/// Registers the `pk-number-field` class family: row, disabled modifier,
/// editable input, controls stack, step buttons and their glyphs.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-number-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint")),
    );
    sheet.insert(
        Class::new("pk-number-field--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-number-field__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-number-field__controls")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    for button in ["pk-number-field__inc", "pk-number-field__dec"] {
        sheet.insert(
            Class::new(button)
                .with(StyleProp::Display, kw(Keyword::Flex))
                .with(StyleProp::AlignItems, kw(Keyword::Center))
                .with(StyleProp::JustifyContent, kw(Keyword::Center))
                .with_padding_x(tok("space.xxs"))
                .with(StyleProp::Color, tok("color.tint"))
                .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.6)),
        );
    }

    sheet.insert(
        Class::new("pk-number-field__glyph")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: NumberFieldProps) -> Element {
        NumberField.render(&props)
    }

    #[test]
    fn renders_input_then_controls() {
        let el = render(NumberFieldProps::new().value(3.0));
        assert_eq!(el.class_names(), ["pk-number-field"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-number-field__input"]);
        assert_eq!(kids[1].class_names(), ["pk-number-field__controls"]);
    }

    #[test]
    fn controls_hold_inc_then_dec() {
        let el = render(NumberFieldProps::new().value(1.0));
        let controls = &el.child_elements()[1];
        let kids = controls.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-number-field__inc"]);
        assert_eq!(kids[1].class_names(), ["pk-number-field__dec"]);
    }

    #[test]
    fn value_above_max_is_clamped() {
        let el = render(NumberFieldProps::new().value(99.0).min(0.0).max(10.0));
        assert_eq!(el.child_elements()[0].text_content(), Some("10"));
    }

    #[test]
    fn value_below_min_is_clamped() {
        let el = render(NumberFieldProps::new().value(-7.0).min(0.0).max(10.0));
        assert_eq!(el.child_elements()[0].text_content(), Some("0"));
    }

    #[test]
    fn none_renders_empty_input() {
        let el = render(NumberFieldProps::new());
        assert_eq!(el.child_elements()[0].text_content(), Some(""));
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(NumberFieldProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-number-field--disabled"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(NumberField::role(), Role::Textbox);
    }
}
