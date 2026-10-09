//! [`Spinner`] — an indeterminate activity ring.
//!
//! A spinner is just a [`ControlSize`]; it renders a `pk-spinner` box holding a
//! single `pk-spinner__ring`. The spin itself is a backend concern signalled by
//! the ring class, so this control attaches only kit classes and lets
//! [`crate::preset`] size the ring against the theme.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, ControlSize};
use crate::preset::StyleSheet;

/// Props for [`Spinner`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SpinnerProps {
    /// The ring's density step.
    pub size: ControlSize,
}

impl SpinnerProps {
    /// Creates spinner props with the default (medium) size.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the density step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }
}

/// The spinner control. Zero-sized; all configuration lives in [`SpinnerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Spinner;

impl Component for Spinner {
    type Props = SpinnerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-spinner", &[props.size.suffix()]) {
            el = el.class(name);
        }
        el.child(Element::box_().class("pk-spinner__ring"))
    }
}

/// Registers the `pk-spinner` class family: the centring base, three sizes and
/// the ring that the backend animates.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a centred box.
    sheet.insert(
        Class::new("pk-spinner")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center)),
    );

    // Sizes set the ring's square footprint.
    sheet.insert(
        Class::new("pk-spinner--sm")
            .with(StyleProp::Width, StyleValue::px(16.0))
            .with(StyleProp::Height, StyleValue::px(16.0)),
    );
    sheet.insert(
        Class::new("pk-spinner--md")
            .with(StyleProp::Width, StyleValue::px(24.0))
            .with(StyleProp::Height, StyleValue::px(24.0)),
    );
    sheet.insert(
        Class::new("pk-spinner--lg")
            .with(StyleProp::Width, StyleValue::px(36.0))
            .with(StyleProp::Height, StyleValue::px(36.0)),
    );

    // Ring: a full-size circle with an accent rim the backend rotates.
    sheet.insert(
        Class::new("pk-spinner__ring")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::percent(100.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(2.0))
            .with(StyleProp::BorderColor, tok("color.tint")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SpinnerProps) -> Element {
        Spinner.render(&props)
    }

    #[test]
    fn attaches_base_and_size_classes() {
        let el = render(SpinnerProps::new().size(ControlSize::Large));
        assert_eq!(el.class_names(), ["pk-spinner", "pk-spinner--lg"]);
    }

    #[test]
    fn default_size_is_medium() {
        let el = render(SpinnerProps::new());
        assert_eq!(el.class_names(), ["pk-spinner", "pk-spinner--md"]);
    }

    #[test]
    fn holds_a_single_ring() {
        let el = render(SpinnerProps::new());
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-spinner__ring"));
    }
}
