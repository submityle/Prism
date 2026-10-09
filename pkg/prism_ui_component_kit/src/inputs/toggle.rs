//! [`Toggle`] — a two-state on/off switch.
//!
//! A toggle renders a `pk-toggle` track containing a single
//! `pk-toggle__thumb`. The on-state is expressed by the bare `is-on` marker on
//! the track, which [`crate::preset`] styles to recolor the track and push the
//! thumb to the trailing edge (via `justify-content`, since this layer has no
//! transforms). All color comes from theme tokens; the control attaches only
//! class names. Interaction/animation state is owned by `prism_ui_form`.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Toggle`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToggleProps {
    /// Whether the switch is in the on position.
    pub on: bool,
    /// Whether the switch is non-interactive.
    pub disabled: bool,
}

impl ToggleProps {
    /// Creates default (off, enabled) toggle props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the on/off position.
    #[must_use]
    pub fn on(mut self, on: bool) -> Self {
        self.on = on;
        self
    }

    /// Marks the switch disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The toggle control. Zero-sized; config lives in [`ToggleProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Toggle;

impl Component for Toggle {
    type Props = ToggleProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-toggle");
        if props.on {
            el = el.class("is-on");
        }
        if props.disabled {
            el = el.class("is-disabled");
        }
        el.child(Element::box_().class("pk-toggle__thumb"))
    }
}

/// Registers the `pk-toggle` class family: track, on-state marker, and thumb.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Track: a capsule rail, thumb packed to the start (off position).
    sheet.insert(
        Class::new("pk-toggle")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Start))
            .with(StyleProp::Width, StyleValue::px(44.0))
            .with(StyleProp::Height, StyleValue::px(26.0))
            .with_padding_x(StyleValue::px(2.0))
            .with_padding_y(StyleValue::px(2.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // On: accent track, thumb pushed to the trailing edge.
    sheet.insert(
        Class::new("is-on")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::JustifyContent, kw(Keyword::End)),
    );

    // Thumb: a fixed white knob with a soft shadow.
    sheet.insert(
        Class::new("pk-toggle__thumb")
            .with(StyleProp::Width, StyleValue::px(22.0))
            .with(StyleProp::Height, StyleValue::px(22.0))
            .with(StyleProp::MinWidth, StyleValue::px(22.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ToggleProps) -> Element {
        Toggle.render(&props)
    }

    #[test]
    fn off_toggle_has_only_base_and_a_thumb() {
        let el = render(ToggleProps::new());
        assert_eq!(el.class_names(), ["pk-toggle"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-toggle__thumb"]);
    }

    #[test]
    fn on_toggle_adds_is_on_marker() {
        let el = render(ToggleProps::new().on(true));
        assert_eq!(el.class_names(), ["pk-toggle", "is-on"]);
    }

    #[test]
    fn disabled_adds_marker() {
        let el = render(ToggleProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
    }
}
