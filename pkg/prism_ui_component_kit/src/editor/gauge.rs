//! [`Gauge`] — a radial gauge shell with a needle and a value readout.
//!
//! A gauge renders a round `pk-gauge` dial carrying a `pk-gauge__needle` marker
//! and a `pk-gauge__value` text readout. The driving value is clamped to
//! `0.0..=1.0` with a tiny helper (std `f32::clamp` is unavailable under
//! `no_std`). Only kit class names are attached; surface, needle and type
//! resolve from theme tokens via [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Clamps `value` into the inclusive range `0.0..=1.0`.
///
/// Hand-rolled replacement for the std-only `f32::clamp`; `NaN` collapses to
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

/// Props for [`Gauge`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GaugeProps {
    /// The needle position, in `0.0..=1.0`. Values outside are clamped.
    pub value: f32,
    /// Optional readout text shown in the `__value` slot (e.g. `"72%"`).
    pub text: Option<String>,
}

impl GaugeProps {
    /// Creates gauge props at `value` (clamped) with no readout text.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            value: clamp01(value),
            ..Self::default()
        }
    }

    /// Sets the needle position (clamped to `0.0..=1.0`).
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = clamp01(value);
        self
    }

    /// Sets the readout text shown at the gauge center.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// The clamped value, for backends that orient the needle.
    #[must_use]
    pub fn normalized(&self) -> f32 {
        clamp01(self.value)
    }
}

/// The gauge control. Zero-sized; all configuration lives in [`GaugeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Gauge;

impl Gauge {
    /// The accessibility role a gauge exposes (a read-only image).
    #[must_use]
    pub const fn role() -> Role {
        Role::Image
    }
}

impl Component for Gauge {
    type Props = GaugeProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-gauge")
            .child(Element::box_().class("pk-gauge__needle"));
        if let Some(text) = props.text.clone() {
            el = el.child(Element::text(text).class("pk-gauge__value"));
        }
        el
    }
}

/// Registers the `pk-gauge` family: dial shell, needle, value readout.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-gauge")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::Width, StyleValue::px(96.0))
            .with(StyleProp::Height, StyleValue::px(96.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary")),
    );

    sheet.insert(
        Class::new("pk-gauge__needle")
            .with(StyleProp::Width, StyleValue::px(3.0))
            .with(StyleProp::Height, StyleValue::px(36.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-gauge__value")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.title3"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: GaugeProps) -> Element {
        Gauge.render(&props)
    }

    #[test]
    fn renders_needle_without_text_by_default() {
        let el = render(GaugeProps::new(0.5));
        assert_eq!(el.class_names(), ["pk-gauge"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class_names(), ["pk-gauge__needle"]);
    }

    #[test]
    fn renders_value_text_when_set() {
        let el = render(GaugeProps::new(0.72).text("72%"));
        let value = &el.child_elements()[1];
        assert_eq!(value.class_names(), ["pk-gauge__value"]);
        assert_eq!(value.text_content(), Some("72%"));
    }

    #[test]
    fn value_is_clamped_to_unit_range() {
        assert!((GaugeProps::new(5.0).normalized() - 1.0).abs() < f32::EPSILON);
        assert!((GaugeProps::new(-2.0).normalized() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn role_is_image() {
        assert_eq!(Gauge::role(), Role::Image);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-gauge", "pk-gauge__needle", "pk-gauge__value"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
