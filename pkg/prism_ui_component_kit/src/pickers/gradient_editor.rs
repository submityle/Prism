//! [`GradientEditor`] — a gradient preview bar with draggable stops.
//!
//! The editor renders a `pk-gradient-editor` frame holding a
//! `pk-gradient-editor__preview` bar and a `pk-gradient-editor__stops` track of
//! `pk-gradient-editor__stop` markers. Each stop is a `(position, color)` pair:
//! its `0.0..=1.0` position maps to an inline left offset (data-derived, like
//! the slider's fill width) and its color tints the marker inline. No color or
//! offset literal is hard-coded in `render`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::StyleValue;

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

/// Props for [`GradientEditor`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GradientEditorProps {
    /// The gradient stops as `(position, color)` pairs. Position is `0.0..=1.0`
    /// along the bar; color is a concrete color or token reference.
    pub stops: Vec<(f32, StyleValue)>,
}

impl GradientEditorProps {
    /// Creates empty editor props (no stops).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a stop at `position` with `color`.
    #[must_use]
    pub fn stop(mut self, position: f32, color: StyleValue) -> Self {
        self.stops.push((position, color));
        self
    }

    /// Replaces the stops with `stops`.
    #[must_use]
    pub fn stops<I: IntoIterator<Item = (f32, StyleValue)>>(mut self, stops: I) -> Self {
        self.stops = stops.into_iter().collect();
        self
    }
}

/// The gradient-editor control. Zero-sized; config lives in
/// [`GradientEditorProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct GradientEditor;

impl GradientEditor {
    /// The accessibility role a gradient editor exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for GradientEditor {
    type Props = GradientEditorProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::StyleProp;

        let preview = Element::box_().class("pk-gradient-editor__preview");
        let mut track = Element::box_().class("pk-gradient-editor__stops");
        for stop in &props.stops {
            let left = clamp01(stop.0) * 100.0;
            let marker = Element::box_()
                .class("pk-gradient-editor__stop")
                .style(StyleProp::MarginLeft, StyleValue::percent(left))
                .style(StyleProp::BackgroundColor, stop.1.clone());
            track = track.child(marker);
        }
        Element::box_()
            .class("pk-gradient-editor")
            .child(preview)
            .child(track)
    }
}

/// Registers the `pk-gradient-editor` class family: the frame, the preview bar,
/// the stops track and a stop marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Frame: a column stacking the preview over its stop track.
    sheet.insert(
        Class::new("pk-gradient-editor")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    // Preview: a wide capsule bar previewing the ramp.
    sheet.insert(
        Class::new("pk-gradient-editor__preview")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(24.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Stops track: the row the markers sit along.
    sheet.insert(
        Class::new("pk-gradient-editor__stops")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Height, StyleValue::px(16.0)),
    );

    // Stop: a small round handle (fill + offset set inline per stop).
    sheet.insert(
        Class::new("pk-gradient-editor__stop")
            .with(StyleProp::Width, StyleValue::px(12.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::MinWidth, StyleValue::px(12.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(2.0))
            .with(StyleProp::BorderColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: GradientEditorProps) -> Element {
        GradientEditor.render(&props)
    }

    #[test]
    fn preview_precedes_stops_track() {
        let el = render(GradientEditorProps::new());
        assert_eq!(el.class_names(), ["pk-gradient-editor"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-gradient-editor__preview"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-gradient-editor__stops"));
    }

    #[test]
    fn one_marker_per_stop_with_inline_offset_and_color() {
        let el = render(
            GradientEditorProps::new()
                .stop(0.0, StyleValue::token("color.red"))
                .stop(1.0, StyleValue::token("color.blue")),
        );
        let track = &el.child_elements()[1];
        let markers = track.child_elements();
        assert_eq!(markers.len(), 2);
        let left = markers[1]
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::MarginLeft)
            .map(|(_, v)| v.clone());
        assert_eq!(left, Some(StyleValue::Length(Length::Percent(100.0))));
        let color = markers[0]
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::BackgroundColor)
            .map(|(_, v)| v.clone());
        assert_eq!(color, Some(StyleValue::token("color.red")));
    }

    #[test]
    fn stop_position_is_clamped() {
        let el = render(GradientEditorProps::new().stop(4.0, StyleValue::token("color.red")));
        let marker = &el.child_elements()[1].child_elements()[0];
        let left = marker
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::MarginLeft)
            .map(|(_, v)| v.clone());
        assert_eq!(left, Some(StyleValue::Length(Length::Percent(100.0))));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(GradientEditor::role(), Role::Group);
    }
}
