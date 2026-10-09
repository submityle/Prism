//! [`EdgeView`] — the visual stand-in for a graph edge.
//!
//! The style layer has no path/curve primitive and no `transform`, so this
//! control cannot draw the real connector. Instead it emits a single
//! `pk-node-edge` box sized to the axis-aligned bounding region between the two
//! endpoints and offset into place with inline left/top margins. It is a
//! *semantic* placeholder: the render layer is expected to draw the actual
//! Bézier/spline on top; this box only records "an edge spans here" so layout
//! and hit-testing have something to anchor to.

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Returns the absolute value of `x` without relying on the std-only
/// `f32::abs` (this crate is `no_std`).
#[inline]
fn absf(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Returns the smaller of two `f32`s (treating ties as `a`).
#[inline]
fn minf(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

/// Returns the larger of two `f32`s (treating ties as `a`).
#[inline]
fn maxf(a: f32, b: f32) -> f32 {
    if b > a {
        b
    } else {
        a
    }
}

/// Props for [`EdgeView`]: the start and end points in canvas pixels.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct EdgeViewProps {
    /// The edge's start point as `(x, y)` in logical pixels.
    pub from: (f32, f32),
    /// The edge's end point as `(x, y)` in logical pixels.
    pub to: (f32, f32),
}

impl EdgeViewProps {
    /// Creates props spanning `from` to `to`.
    #[must_use]
    pub fn new(from: (f32, f32), to: (f32, f32)) -> Self {
        Self { from, to }
    }
}

/// The edge view control. Zero-sized; configuration lives in [`EdgeViewProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeView;

impl EdgeView {
    /// Edges are decorative; they carry no interactive role.
    #[must_use]
    pub const fn role() -> Role {
        Role::Presentation
    }
}

impl Component for EdgeView {
    type Props = EdgeViewProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{Length, StyleProp, StyleValue};

        let left = minf(props.from.0, props.to.0);
        let top = minf(props.from.1, props.to.1);
        // Keep at least a 1px extent so a perfectly horizontal/vertical edge is
        // still a visible hairline.
        let width = maxf(absf(props.to.0 - props.from.0), 1.0);
        let height = maxf(absf(props.to.1 - props.from.1), 1.0);

        Element::box_()
            .class("pk-node-edge")
            .style(StyleProp::MarginLeft, StyleValue::Length(Length::Px(left)))
            .style(StyleProp::MarginTop, StyleValue::Length(Length::Px(top)))
            .style(StyleProp::Width, StyleValue::Length(Length::Px(width)))
            .style(StyleProp::Height, StyleValue::Length(Length::Px(height)))
    }
}

/// Registers the `pk-node-edge` class: a faint, rounded hairline box that the
/// render layer paints the real connector over.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-node-edge")
            .with(StyleProp::BackgroundColor, tok("color.separator"))
            .with(StyleProp::BorderRadius, tok("radius.xs"))
            .with(StyleProp::Opacity, StyleValue::number(0.6)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn find(el: &Element, prop: StyleProp) -> Option<StyleValue> {
        el.inline_pairs()
            .iter()
            .find(|(p, _)| *p == prop)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn attaches_edge_class() {
        let el = EdgeView.render(&EdgeViewProps::new((0.0, 0.0), (10.0, 10.0)));
        assert_eq!(el.class_names(), ["pk-node-edge"]);
    }

    #[test]
    fn bounding_box_offsets_and_sizes_from_endpoints() {
        let el = EdgeView.render(&EdgeViewProps::new((30.0, 40.0), (10.0, 100.0)));
        assert_eq!(find(&el, StyleProp::MarginLeft), Some(StyleValue::Length(Length::Px(10.0))));
        assert_eq!(find(&el, StyleProp::MarginTop), Some(StyleValue::Length(Length::Px(40.0))));
        assert_eq!(find(&el, StyleProp::Width), Some(StyleValue::Length(Length::Px(20.0))));
        assert_eq!(find(&el, StyleProp::Height), Some(StyleValue::Length(Length::Px(60.0))));
    }

    #[test]
    fn degenerate_edge_keeps_hairline_extent() {
        let el = EdgeView.render(&EdgeViewProps::new((5.0, 5.0), (5.0, 5.0)));
        assert_eq!(find(&el, StyleProp::Width), Some(StyleValue::Length(Length::Px(1.0))));
        assert_eq!(find(&el, StyleProp::Height), Some(StyleValue::Length(Length::Px(1.0))));
    }

    #[test]
    fn role_is_presentation() {
        assert_eq!(EdgeView::role(), Role::Presentation);
    }
}
