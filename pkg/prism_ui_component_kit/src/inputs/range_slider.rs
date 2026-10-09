//! [`RangeSlider`] — a two-ended interval selector over a horizontal track.
//!
//! A range slider renders a `pk-range-slider` root wrapping a `__track`. Inside
//! the track a `__range` paints the selected segment: its leading offset and
//! width are inline, data-derived percentage lengths (never color/shadow
//! literals), and two `__thumb`s (`--low` and `--high`) mark the handles. The
//! low/high values are normalised into `[min, max]` and clamped with hand-rolled
//! `no_std` helpers; all color and sizing come from theme tokens via
//! [`crate::preset`].

use prism_ui::Element;
use prism_ui_a11y::Role;
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

/// Normalises `value` within `[min, max]` into a `0.0..=100.0` percentage.
///
/// An empty or inverted span (`max <= min`) collapses to `0.0` so the control
/// never divides by zero or emits a negative length.
fn norm_pct(value: f32, min: f32, max: f32) -> f32 {
    let span = max - min;
    if span <= 0.0 {
        return 0.0;
    }
    clamp01((value - min) / span) * 100.0
}

/// Props for [`RangeSlider`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RangeSliderProps {
    /// The low end of the selected interval (normalised into `[min, max]`).
    pub low: f32,
    /// The high end of the selected interval (normalised into `[min, max]`).
    pub high: f32,
    /// The lower bound of the track.
    pub min: f32,
    /// The upper bound of the track.
    pub max: f32,
    /// Whether the slider is non-interactive.
    pub disabled: bool,
}

impl RangeSliderProps {
    /// Creates default (zero span, enabled) range-slider props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the low end of the interval.
    #[must_use]
    pub fn low(mut self, low: f32) -> Self {
        self.low = low;
        self
    }

    /// Sets the high end of the interval.
    #[must_use]
    pub fn high(mut self, high: f32) -> Self {
        self.high = high;
        self
    }

    /// Sets the lower bound of the track.
    #[must_use]
    pub fn min(mut self, min: f32) -> Self {
        self.min = min;
        self
    }

    /// Sets the upper bound of the track.
    #[must_use]
    pub fn max(mut self, max: f32) -> Self {
        self.max = max;
        self
    }

    /// Marks the slider disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The range-slider control. Zero-sized; config lives in [`RangeSliderProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RangeSlider;

impl RangeSlider {
    /// The accessibility role a range slider exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for RangeSlider {
    type Props = RangeSliderProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-range-slider");
        if props.disabled {
            el = el.class("pk-range-slider--disabled");
        }

        // Normalise both ends and order them so the painted segment is never
        // negative, regardless of how low/high were supplied.
        let a = norm_pct(props.low, props.min, props.max);
        let b = norm_pct(props.high, props.min, props.max);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let width = hi - lo;

        let range = Element::box_()
            .class("pk-range-slider__range")
            .style(StyleProp::MarginLeft, StyleValue::percent(lo))
            .style(StyleProp::Width, StyleValue::percent(width));

        let thumb_low = Element::box_()
            .class("pk-range-slider__thumb")
            .class("pk-range-slider__thumb--low");
        let thumb_high = Element::box_()
            .class("pk-range-slider__thumb")
            .class("pk-range-slider__thumb--high");

        let track = Element::box_()
            .class("pk-range-slider__track")
            .child(range)
            .child(thumb_low)
            .child(thumb_high);

        el.child(track)
    }
}

/// Registers the `pk-range-slider` class family: root, disabled modifier,
/// track, selected range, and the two thumbs.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-range-slider")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::percent(100.0)),
    );
    sheet.insert(
        Class::new("pk-range-slider--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // Track: a thin capsule rail that the range overlays.
    sheet.insert(
        Class::new("pk-range-slider__track")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(4.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Range: accent segment between the two handles (offset/width set inline).
    sheet.insert(
        Class::new("pk-range-slider__range")
            .with(StyleProp::Height, StyleValue::px(4.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    // Thumbs: fixed white knobs with a soft shadow; they never shrink.
    sheet.insert(
        Class::new("pk-range-slider__thumb")
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::MinWidth, StyleValue::px(20.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );
    sheet.insert(Class::new("pk-range-slider__thumb--low"));
    sheet.insert(Class::new("pk-range-slider__thumb--high"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: RangeSliderProps) -> Element {
        RangeSlider.render(&props)
    }

    fn range_el(el: &Element) -> Element {
        el.child_elements()[0].child_elements()[0].clone()
    }

    fn inline(el: &Element, prop: StyleProp) -> Option<StyleValue> {
        el.inline_pairs()
            .iter()
            .find(|(p, _)| *p == prop)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn root_has_track_with_range_and_two_thumbs() {
        let el = render(RangeSliderProps::new().min(0.0).max(10.0).low(2.0).high(8.0));
        assert_eq!(el.class_names(), ["pk-range-slider"]);
        let track = &el.child_elements()[0];
        assert_eq!(track.class_names(), ["pk-range-slider__track"]);
        let kids = track.child_elements();
        assert_eq!(kids.len(), 3);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-range-slider__range"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-range-slider__thumb--low"));
        assert!(kids[2].class_names().iter().any(|c| c == "pk-range-slider__thumb--high"));
    }

    #[test]
    fn range_offset_and_width_are_data_driven() {
        // Powers-of-two bounds keep the normalised percentages exact in f32.
        let el = render(RangeSliderProps::new().min(0.0).max(8.0).low(2.0).high(6.0));
        let range = range_el(&el);
        assert_eq!(inline(&range, StyleProp::MarginLeft), Some(StyleValue::Length(Length::Percent(25.0))));
        assert_eq!(inline(&range, StyleProp::Width), Some(StyleValue::Length(Length::Percent(50.0))));
    }

    #[test]
    fn inverted_ends_are_ordered_so_width_is_non_negative() {
        let el = render(RangeSliderProps::new().min(0.0).max(8.0).low(6.0).high(2.0));
        let range = range_el(&el);
        assert_eq!(inline(&range, StyleProp::MarginLeft), Some(StyleValue::Length(Length::Percent(25.0))));
        assert_eq!(inline(&range, StyleProp::Width), Some(StyleValue::Length(Length::Percent(50.0))));
    }

    #[test]
    fn out_of_range_ends_clamp_to_track_bounds() {
        let el = render(RangeSliderProps::new().min(0.0).max(10.0).low(-5.0).high(50.0));
        let range = range_el(&el);
        assert_eq!(inline(&range, StyleProp::MarginLeft), Some(StyleValue::Length(Length::Percent(0.0))));
        assert_eq!(inline(&range, StyleProp::Width), Some(StyleValue::Length(Length::Percent(100.0))));
    }

    #[test]
    fn empty_span_collapses_to_zero_without_panicking() {
        let el = render(RangeSliderProps::new().min(5.0).max(5.0).low(5.0).high(5.0));
        let range = range_el(&el);
        assert_eq!(inline(&range, StyleProp::MarginLeft), Some(StyleValue::Length(Length::Percent(0.0))));
        assert_eq!(inline(&range, StyleProp::Width), Some(StyleValue::Length(Length::Percent(0.0))));
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(RangeSliderProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-range-slider--disabled"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(RangeSlider::role(), Role::Group);
    }
}
