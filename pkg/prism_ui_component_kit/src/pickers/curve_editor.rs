//! [`CurveEditor`] — an easing/animation curve canvas with control points.
//!
//! The editor renders a `pk-curve-editor` canvas holding one
//! `pk-curve-editor__point` knob per control point. Each knob is absolutely
//! positioned within the canvas: a point is an `(x, y)` pair in the unit
//! square `0.0..=1.0` where `x` maps to an inline `left` percentage and `y` to
//! an inline `top` percentage, with `y` inverted so a higher value sits
//! visually higher on the canvas. The knob's static class carries symmetric
//! negative margins so the anchor lands on the knob's center rather than its
//! top-left corner. No color or offset literal is hard-coded in `render`.

use alloc::vec::Vec;

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

/// Props for [`CurveEditor`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CurveEditorProps {
    /// The curve's control points as `(x, y)` pairs, each in `0.0..=1.0`. `x`
    /// runs left-to-right; `y` runs bottom-to-top (so `1.0` is the top edge).
    pub points: Vec<(f32, f32)>,
}

impl CurveEditorProps {
    /// Creates empty editor props (no control points).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a control point at `(x, y)`.
    #[must_use]
    pub fn point(mut self, x: f32, y: f32) -> Self {
        self.points.push((x, y));
        self
    }

    /// Replaces the control points with `points`.
    #[must_use]
    pub fn points<I: IntoIterator<Item = (f32, f32)>>(mut self, points: I) -> Self {
        self.points = points.into_iter().collect();
        self
    }
}

/// The curve-editor control. Zero-sized; config lives in [`CurveEditorProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CurveEditor;

impl CurveEditor {
    /// The accessibility role a curve editor exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for CurveEditor {
    type Props = CurveEditorProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut canvas = Element::box_().class("pk-curve-editor");
        for point in &props.points {
            let left = clamp01(point.0) * 100.0;
            // Invert y: a value of 1.0 sits at the top (0% from the top edge).
            let top = (1.0 - clamp01(point.1)) * 100.0;
            let knob = Element::box_()
                .class("pk-curve-editor__point")
                .style(StyleProp::Left, StyleValue::percent(left))
                .style(StyleProp::Top, StyleValue::percent(top));
            canvas = canvas.child(knob);
        }
        canvas
    }
}

/// Registers the `pk-curve-editor` class family: the canvas frame and a
/// control-point knob.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::tok;

    // Canvas: a tall framed box the curve is drawn within.
    sheet.insert(
        Class::new("pk-curve-editor")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(160.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Point: a small round handle absolutely positioned on the canvas. The
    // `left`/`top` anchors are set inline per point; the symmetric -6px margins
    // (half the 12px knob) recenter the anchor on the knob instead of its
    // top-left corner.
    sheet.insert(
        Class::new("pk-curve-editor__point")
            .with(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
            .with(StyleProp::Width, StyleValue::px(12.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::MarginLeft, StyleValue::px(-6.0))
            .with(StyleProp::MarginTop, StyleValue::px(-6.0))
            .with(StyleProp::BackgroundColor, tok("color.tint"))
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

    fn render(props: CurveEditorProps) -> Element {
        CurveEditor.render(&props)
    }

    fn inline(el: &Element, prop: StyleProp) -> Option<StyleValue> {
        el.inline_pairs()
            .iter()
            .find(|(p, _)| *p == prop)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn empty_editor_is_a_bare_canvas() {
        let el = render(CurveEditorProps::new());
        assert_eq!(el.class_names(), ["pk-curve-editor"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn one_knob_per_point_with_inverted_inline_offsets() {
        let el = render(CurveEditorProps::new().point(0.0, 1.0).point(1.0, 0.0));
        let knobs = el.child_elements();
        assert_eq!(knobs.len(), 2);
        assert!(knobs
            .iter()
            .all(|k| k.class_names().iter().any(|c| c == "pk-curve-editor__point")));
        // First point: x=0 -> left 0%, y=1 -> top 0% (top edge).
        assert_eq!(
            inline(&knobs[0], StyleProp::Left),
            Some(StyleValue::Length(Length::Percent(0.0)))
        );
        assert_eq!(
            inline(&knobs[0], StyleProp::Top),
            Some(StyleValue::Length(Length::Percent(0.0)))
        );
        // Second point: x=1 -> left 100%, y=0 -> top 100% (bottom edge).
        assert_eq!(
            inline(&knobs[1], StyleProp::Left),
            Some(StyleValue::Length(Length::Percent(100.0)))
        );
        assert_eq!(
            inline(&knobs[1], StyleProp::Top),
            Some(StyleValue::Length(Length::Percent(100.0)))
        );
    }

    #[test]
    fn offsets_are_clamped() {
        let el = render(CurveEditorProps::new().point(4.0, -2.0));
        let knob = &el.child_elements()[0];
        assert_eq!(
            inline(knob, StyleProp::Left),
            Some(StyleValue::Length(Length::Percent(100.0)))
        );
        // y clamps to 0 -> top 100%.
        assert_eq!(
            inline(knob, StyleProp::Top),
            Some(StyleValue::Length(Length::Percent(100.0)))
        );
    }

    #[test]
    fn role_is_group() {
        assert_eq!(CurveEditor::role(), Role::Group);
    }
}
