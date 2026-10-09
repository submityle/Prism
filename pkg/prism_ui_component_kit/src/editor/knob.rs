//! [`Knob`] / [`Dial`] — a round value knob with a position indicator.
//!
//! A knob renders a circular `pk-knob` body sized by [`ControlSize`] with a
//! `pk-knob__indicator` box marking the current position. The value is clamped
//! to `0.0..=1.0` with a tiny helper (std `f32::clamp` is unavailable under
//! `no_std`). The knob carries only kit class names; surface, size and color
//! resolve from theme tokens via [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::{classes, ControlSize};
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

/// Props for [`Knob`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct KnobProps {
    /// The current value, in `0.0..=1.0`. Values outside are clamped.
    pub value: f32,
    /// The density step controlling the knob's diameter.
    pub size: ControlSize,
}

impl KnobProps {
    /// Creates knob props at `value` (clamped) with the default size.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            value: clamp01(value),
            ..Self::default()
        }
    }

    /// Sets the current value (clamped to `0.0..=1.0`).
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = clamp01(value);
        self
    }

    /// Sets the density step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }

    /// The clamped value, for backends that interpret the indicator position.
    #[must_use]
    pub fn normalized(&self) -> f32 {
        clamp01(self.value)
    }
}

/// The knob control. Zero-sized; all configuration lives in [`KnobProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Knob;

/// Alias matching the `Dial` naming used by the design doc (section 12).
pub type Dial = Knob;

impl Knob {
    /// The accessibility role a knob exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Knob {
    type Props = KnobProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-knob", &[props.size.suffix()]) {
            el = el.class(name);
        }
        el.child(Element::box_().class("pk-knob__indicator"))
    }
}

/// Registers the `pk-knob` family: a base circle, per-size diameters, indicator.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a centered circle with a glass surface.
    sheet.insert(
        Class::new("pk-knob")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.fill"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 2.0, 6.0, tok("glass.shadow")),
    );

    for (size, diameter) in [
        (ControlSize::Small, 28.0),
        (ControlSize::Medium, 40.0),
        (ControlSize::Large, 56.0),
    ] {
        let mut name = String::from("pk-knob--");
        name.push_str(size.suffix());
        sheet.insert(
            Class::new(name)
                .with(StyleProp::Width, StyleValue::px(diameter))
                .with(StyleProp::Height, StyleValue::px(diameter))
                .with(StyleProp::MinWidth, StyleValue::px(diameter)),
        );
    }

    // Indicator: a small pointer dot painted with the accent tint.
    sheet.insert(
        Class::new("pk-knob__indicator")
            .with(StyleProp::Width, StyleValue::px(4.0))
            .with(StyleProp::Height, StyleValue::px(10.0))
            .with(StyleProp::MarginTop, tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: KnobProps) -> Element {
        Knob.render(&props)
    }

    #[test]
    fn attaches_base_and_size_classes_in_order() {
        let el = render(KnobProps::new(0.5).size(ControlSize::Large));
        assert_eq!(el.class_names(), ["pk-knob", "pk-knob--lg"]);
    }

    #[test]
    fn renders_single_indicator_child() {
        let el = render(KnobProps::new(0.5));
        let children = el.child_elements();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class_names(), ["pk-knob__indicator"]);
    }

    #[test]
    fn value_is_clamped_to_unit_range() {
        assert!((KnobProps::new(2.0).normalized() - 1.0).abs() < f32::EPSILON);
        assert!((KnobProps::new(-1.0).normalized() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Knob::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-knob",
            "pk-knob--sm",
            "pk-knob--md",
            "pk-knob--lg",
            "pk-knob__indicator",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
