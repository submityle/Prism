//! [`ProgressBar`] — a determinate completion track.
//!
//! A progress bar is a `value` in `0.0..=1.0` plus a semantic [`Tone`]. It
//! renders a `pk-progress` track holding a `pk-progress__fill` whose width is
//! the clamped percentage. The width is the one legitimately dynamic value and
//! rides as an inline length; every color comes from a tone class so
//! [`crate::preset`] owns the palette.

use prism_ui::Element;
use prism_ui_component::Component;

use alloc::string::String;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`ProgressBar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ProgressBarProps {
    /// Completion fraction in `0.0..=1.0` (clamped at render time).
    pub value: f32,
    /// The semantic tone selecting the fill color.
    pub tone: Tone,
}

impl ProgressBarProps {
    /// Creates progress props at `value` with the default (accent) tone.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            value,
            ..Self::default()
        }
    }

    /// Sets the completion fraction.
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// Clamps `value` into `0.0..=1.0` without relying on `f32::clamp` (keeps the
/// control trivially `no_std` and NaN-safe: a non-ordered value falls to 0).
#[must_use]
fn clamp01(value: f32) -> f32 {
    if value > 1.0 {
        1.0
    } else if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// The progress-bar control. Zero-sized; config lives in [`ProgressBarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ProgressBar;

impl Component for ProgressBar {
    type Props = ProgressBarProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_();
        for name in classes("pk-progress", &[props.tone.suffix()]) {
            el = el.class(name);
        }

        let fill = Element::box_()
            .class("pk-progress__fill")
            // The only inline value: a dynamic, continuous width.
            .style(StyleProp::Width, StyleValue::percent(clamp01(props.value) * 100.0));
        el.child(fill)
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-progress--success`.
fn tone_class(block: &str, tone: Tone) -> String {
    let mut name = String::with_capacity(block.len() + 2 + tone.suffix().len());
    name.push_str(block);
    name.push_str("--");
    name.push_str(tone.suffix());
    name
}

/// Every tone, for exhaustive style registration.
const TONES: [Tone; 5] = [
    Tone::Accent,
    Tone::Neutral,
    Tone::Success,
    Tone::Warning,
    Tone::Danger,
];

/// Registers the `pk-progress` class family: the track base, the fill, and the
/// per-tone fill color (attached to the track so the cascade reaches the fill).
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    // Track: a thin rounded rail on a quiet fill.
    sheet.insert(
        Class::new("pk-progress")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(6.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Fill: full height, rounded, default accent (overridden per tone below).
    sheet.insert(
        Class::new("pk-progress__fill")
            .with(StyleProp::Height, StyleValue::percent(100.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    // Tones recolor the fill via the track modifier.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-progress", tone))
                .with(StyleProp::BackgroundColor, tok(tone.color_token())),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: ProgressBarProps) -> Element {
        ProgressBar.render(&props)
    }

    fn fill_width(el: &Element) -> f32 {
        let fill = &el.child_elements()[0];
        for (prop, value) in fill.inline_pairs() {
            if *prop == StyleProp::Width
                && let StyleValue::Length(Length::Percent(p)) = value
            {
                return *p;
            }
        }
        panic!("fill has no percent width");
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(ProgressBarProps::new(0.5).tone(Tone::Warning));
        assert_eq!(el.class_names(), ["pk-progress", "pk-progress--warning"]);
    }

    #[test]
    fn width_tracks_value_as_percent() {
        let el = render(ProgressBarProps::new(0.25));
        assert!((fill_width(&el) - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn value_is_clamped_to_unit_range() {
        assert!((fill_width(&render(ProgressBarProps::new(2.0))) - 100.0).abs() < f32::EPSILON);
        assert!((fill_width(&render(ProgressBarProps::new(-1.0))) - 0.0).abs() < f32::EPSILON);
    }
}
