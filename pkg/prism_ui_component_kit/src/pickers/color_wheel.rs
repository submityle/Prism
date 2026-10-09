//! [`ColorWheel`] — a continuous HSV color ring.
//!
//! Unlike [`super::color_picker`]'s discrete swatch grid, the wheel is a
//! continuous hue/saturation surface. It renders a `pk-color-wheel` box holding
//! a `pk-color-wheel__ring` and a draggable `pk-color-wheel__thumb`; an optional
//! `pk-color-wheel__alpha` track appears when alpha editing is enabled. Concrete
//! thumb placement on the ring is a backend concern — the control only emits the
//! classed boxes and paints the thumb with the current value as *data*.

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::StyleValue;

use crate::preset::StyleSheet;

/// Props for [`ColorWheel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ColorWheelProps {
    /// The current color, used to tint the thumb. `None` leaves the thumb
    /// neutral.
    pub value: Option<StyleValue>,
    /// Whether to render the trailing alpha track.
    pub show_alpha: bool,
}

impl ColorWheelProps {
    /// Creates default wheel props (no value, no alpha track).
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

    /// Enables or disables the alpha track.
    #[must_use]
    pub fn show_alpha(mut self, show_alpha: bool) -> Self {
        self.show_alpha = show_alpha;
        self
    }
}

/// The color-wheel control. Zero-sized; config lives in [`ColorWheelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorWheel;

impl ColorWheel {
    /// The accessibility role a continuous color surface exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for ColorWheel {
    type Props = ColorWheelProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::StyleProp;

        let ring = Element::box_().class("pk-color-wheel__ring");
        let mut thumb = Element::box_().class("pk-color-wheel__thumb");
        if let Some(value) = &props.value {
            thumb = thumb.style(StyleProp::BackgroundColor, value.clone());
        }
        let mut el = Element::box_()
            .class("pk-color-wheel")
            .child(ring)
            .child(thumb);
        if props.show_alpha {
            el = el.child(Element::box_().class("pk-color-wheel__alpha"));
        }
        el
    }
}

/// Registers the `pk-color-wheel` class family: the frame, the hue ring, the
/// thumb knob and the optional alpha track.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Frame: a square column stacking the ring over the alpha track.
    sheet.insert(
        Class::new("pk-color-wheel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Width, StyleValue::px(200.0)),
    );

    // Ring: the circular hue/saturation surface.
    sheet.insert(
        Class::new("pk-color-wheel__ring")
            .with(StyleProp::Width, StyleValue::px(200.0))
            .with(StyleProp::Height, StyleValue::px(200.0))
            .with(StyleProp::MinWidth, StyleValue::px(200.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Thumb: a small white knob with a soft shadow (fill overridden inline).
    sheet.insert(
        Class::new("pk-color-wheel__thumb")
            .with(StyleProp::Width, StyleValue::px(16.0))
            .with(StyleProp::Height, StyleValue::px(16.0))
            .with(StyleProp::MinWidth, StyleValue::px(16.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(2.0))
            .with(StyleProp::BorderColor, StyleValue::rgba8(255, 255, 255, 255))
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );

    // Alpha: a thin capsule track for opacity.
    sheet.insert(
        Class::new("pk-color-wheel__alpha")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: ColorWheelProps) -> Element {
        ColorWheel.render(&props)
    }

    #[test]
    fn ring_precedes_thumb() {
        let el = render(ColorWheelProps::new());
        assert_eq!(el.class_names(), ["pk-color-wheel"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-color-wheel__ring"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-color-wheel__thumb"));
    }

    #[test]
    fn thumb_is_tinted_by_the_value() {
        let el = render(ColorWheelProps::new().value(StyleValue::token("color.green")));
        let thumb = &el.child_elements()[1];
        let bg = thumb
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::BackgroundColor)
            .map(|(_, v)| v.clone());
        assert_eq!(bg, Some(StyleValue::token("color.green")));
    }

    #[test]
    fn alpha_track_is_opt_in() {
        let without = render(ColorWheelProps::new());
        assert_eq!(without.child_elements().len(), 2);
        let with = render(ColorWheelProps::new().show_alpha(true));
        assert!(with
            .child_elements()
            .iter()
            .any(|c| c.class_names().iter().any(|n| n == "pk-color-wheel__alpha")));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(ColorWheel::role(), Role::Group);
    }
}
