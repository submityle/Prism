//! [`Slider`] — a horizontal track showing a `0.0..=1.0` value.
//!
//! A slider renders a `pk-slider` track containing a `pk-slider__fill` whose
//! width is the value as a percentage, and a `pk-slider__thumb`. The fill width
//! is an inline, data-derived percentage length (never a color/shadow literal);
//! all color and sizing come from theme tokens via [`crate::preset`]. The value
//! is clamped to `0.0..=1.0` with a tiny helper (the std `f32::clamp` is not
//! available under `no_std`).

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Clamps `value` into the inclusive range `0.0..=1.0`.
///
/// A hand-rolled replacement for the std-only `f32::clamp`. `NaN` collapses to
/// `0.0` because neither comparison holds for it.
fn clamp01(value: f32) -> f32 {
    if value > 1.0 {
        1.0
    } else if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Props for [`Slider`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SliderProps {
    /// The current value, in `0.0..=1.0`. Values outside are clamped.
    pub value: f32,
    /// Whether the slider is non-interactive.
    pub disabled: bool,
}

impl SliderProps {
    /// Creates default (zero, enabled) slider props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current value (clamped to `0.0..=1.0` at render time).
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    /// Marks the slider disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The slider control. Zero-sized; config lives in [`SliderProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Slider;

impl Component for Slider {
    type Props = SliderProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-slider");
        if props.disabled {
            el = el.class("is-disabled");
        }

        let pct = clamp01(props.value) * 100.0;
        let fill = Element::box_()
            .class("pk-slider__fill")
            .style(StyleProp::Width, StyleValue::percent(pct));
        let thumb = Element::box_().class("pk-slider__thumb");

        el.child(fill).child(thumb)
    }
}

/// Registers the `pk-slider` class family: track, fill, thumb.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Track: a thin capsule rail; the fill overlays its leading portion.
    sheet.insert(
        Class::new("pk-slider")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(4.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // Fill: accent portion from the start to the value (width set inline).
    sheet.insert(
        Class::new("pk-slider__fill")
            .with(StyleProp::Height, StyleValue::px(4.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    // Thumb: a fixed white knob with a soft shadow, never shrinks.
    sheet.insert(
        Class::new("pk-slider__thumb")
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::MinWidth, StyleValue::px(20.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: SliderProps) -> Element {
        Slider.render(&props)
    }

    fn fill_width(el: &Element) -> Option<StyleValue> {
        el.child_elements()[0]
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::Width)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn track_has_fill_then_thumb() {
        let el = render(SliderProps::new().value(0.5));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-slider__fill"]);
        assert_eq!(kids[1].class_names(), ["pk-slider__thumb"]);
    }

    #[test]
    fn fill_width_is_value_percent() {
        let el = render(SliderProps::new().value(0.25));
        assert_eq!(fill_width(&el), Some(StyleValue::Length(Length::Percent(25.0))));
    }

    #[test]
    fn value_is_clamped_to_unit_range() {
        let hi = render(SliderProps::new().value(4.0));
        assert_eq!(fill_width(&hi), Some(StyleValue::Length(Length::Percent(100.0))));
        let lo = render(SliderProps::new().value(-2.0));
        assert_eq!(fill_width(&lo), Some(StyleValue::Length(Length::Percent(0.0))));
    }

    #[test]
    fn disabled_adds_marker() {
        let el = render(SliderProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
    }
}
