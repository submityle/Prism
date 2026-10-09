//! [`SplitView`] — two panes separated by a draggable divider.
//!
//! A split view renders a `pk-split-view` flex container (`--horizontal` for
//! side-by-side panes, `--vertical` for stacked ones). It lays out a primary
//! `pk-split-view__pane`, a `pk-split-view__divider` (carrying `--resizable`
//! when the split can be dragged) and a secondary `pk-split-view__pane`. The
//! primary pane receives an inline `FlexBasis` percentage derived from `ratio`
//! (clamped to `0..=1`); this is the one sanctioned data-driven inline style.
//! Actual drag handling is a backend concern keyed off `--resizable`.

use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_style::{StyleProp, StyleValue};

use crate::preset::StyleSheet;

/// Clamps `x` into the inclusive `0.0..=1.0` range (no `std` dependency).
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The axis along which a [`SplitView`] divides its two panes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SplitOrientation {
    /// Panes sit side-by-side, split along the horizontal axis (the default).
    #[default]
    Horizontal,
    /// Panes stack top-to-bottom, split along the vertical axis.
    Vertical,
}

impl SplitOrientation {
    /// The modifier class for this orientation.
    #[must_use]
    const fn class(self) -> &'static str {
        match self {
            SplitOrientation::Horizontal => "pk-split-view--horizontal",
            SplitOrientation::Vertical => "pk-split-view--vertical",
        }
    }
}

/// Props for [`SplitView`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SplitViewProps {
    /// The primary (first) pane's content.
    pub primary: Option<Element>,
    /// The secondary (second) pane's content.
    pub secondary: Option<Element>,
    /// The axis along which the view is split.
    pub orientation: SplitOrientation,
    /// The primary pane's size fraction, clamped to `0.0..=1.0`.
    pub ratio: f32,
    /// Whether the divider can be dragged to resize the panes.
    pub resizable: bool,
}

impl SplitViewProps {
    /// Creates empty split-view props (horizontal, non-resizable).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the primary pane's content.
    #[must_use]
    pub fn primary(mut self, element: Element) -> Self {
        self.primary = Some(element);
        self
    }

    /// Sets the secondary pane's content.
    #[must_use]
    pub fn secondary(mut self, element: Element) -> Self {
        self.secondary = Some(element);
        self
    }

    /// Sets the split orientation.
    #[must_use]
    pub fn orientation(mut self, orientation: SplitOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Sets the primary pane's size fraction (clamped on render to `0.0..=1.0`).
    #[must_use]
    pub fn ratio(mut self, ratio: f32) -> Self {
        self.ratio = ratio;
        self
    }

    /// Sets whether the divider is draggable.
    #[must_use]
    pub fn resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }
}

/// The split-view control. Zero-sized; config lives in [`SplitViewProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitView;

impl Component for SplitView {
    type Props = SplitViewProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-split-view")
            .class(props.orientation.class());

        // Primary pane: data-driven FlexBasis percentage from the clamped ratio.
        let basis = clamp01(props.ratio) * 100.0;
        let mut primary = Element::box_()
            .class("pk-split-view__pane")
            .style(StyleProp::FlexBasis, StyleValue::percent(basis));
        if let Some(child) = props.primary.clone() {
            primary = primary.child(child);
        }
        el = el.child(primary);

        // Divider: carries the resizable affordance marker when draggable.
        let mut divider = Element::box_().class("pk-split-view__divider");
        if props.resizable {
            divider = divider.class("pk-split-view__divider--resizable");
        }
        el = el.child(divider);

        // Secondary pane: grows to fill the remaining space.
        let mut secondary = Element::box_().class("pk-split-view__pane");
        if let Some(child) = props.secondary.clone() {
            secondary = secondary.child(child);
        }
        el = el.child(secondary);

        el
    }
}

/// Registers the `pk-split-view` class family: container (+ orientations),
/// panes, and the divider (+ resizable modifier).
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword};

    use crate::preset::{kw, tok};

    // Container: a flex box; orientation picks the main axis.
    sheet.insert(Class::new("pk-split-view").with(StyleProp::Display, kw(Keyword::Flex)));

    sheet.insert(
        Class::new("pk-split-view--horizontal").with(StyleProp::FlexDirection, kw(Keyword::Row)),
    );
    sheet.insert(
        Class::new("pk-split-view--vertical").with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    // Pane: fills available space; the primary pane's basis is set inline.
    sheet.insert(
        Class::new("pk-split-view__pane")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexShrink, StyleValue::number(1.0)),
    );

    // Divider: a hairline separator between panes.
    sheet.insert(
        Class::new("pk-split-view__divider")
            .with(StyleProp::FlexGrow, StyleValue::number(0.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BackgroundColor, tok("color.separator")),
    );

    // Resizable: a thicker, interactive grab target.
    sheet.insert(
        Class::new("pk-split-view__divider--resizable")
            .with(StyleProp::BackgroundColor, tok("color.separator.opaque")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SplitViewProps) -> Element {
        SplitView.render(&props)
    }

    #[test]
    fn horizontal_by_default_with_three_children() {
        let el = render(SplitViewProps::new());
        assert_eq!(
            el.class_names(),
            ["pk-split-view", "pk-split-view--horizontal"]
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[0].class_names(), ["pk-split-view__pane"]);
        assert_eq!(kids[1].class_names(), ["pk-split-view__divider"]);
        assert_eq!(kids[2].class_names(), ["pk-split-view__pane"]);
    }

    #[test]
    fn vertical_orientation_modifier() {
        let el = render(SplitViewProps::new().orientation(SplitOrientation::Vertical));
        assert_eq!(
            el.class_names(),
            ["pk-split-view", "pk-split-view--vertical"]
        );
    }

    #[test]
    fn resizable_marks_the_divider() {
        let el = render(SplitViewProps::new().resizable(true));
        let divider = &el.child_elements()[1];
        assert_eq!(
            divider.class_names(),
            ["pk-split-view__divider", "pk-split-view__divider--resizable"]
        );
    }

    #[test]
    fn ratio_sets_primary_pane_flex_basis_percent() {
        use prism_ui_style::Length;

        let el = render(SplitViewProps::new().ratio(0.3));
        let primary = &el.child_elements()[0];
        let pairs = primary.inline_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, StyleProp::FlexBasis);
        match pairs[0].1 {
            StyleValue::Length(Length::Percent(p)) => assert!((p - 30.0).abs() < 1e-3),
            ref other => panic!("expected a percent length, got {other:?}"),
        }
    }

    #[test]
    fn ratio_is_clamped_to_unit_range() {
        let hi = render(SplitViewProps::new().ratio(5.0));
        assert_eq!(
            hi.child_elements()[0].inline_pairs(),
            [(StyleProp::FlexBasis, StyleValue::percent(100.0))]
        );
        let lo = render(SplitViewProps::new().ratio(-1.0));
        assert_eq!(
            lo.child_elements()[0].inline_pairs(),
            [(StyleProp::FlexBasis, StyleValue::percent(0.0))]
        );
    }

    #[test]
    fn panes_wrap_their_content() {
        let el = render(
            SplitViewProps::new()
                .primary(Element::box_().class("a"))
                .secondary(Element::box_().class("b")),
        );
        let kids = el.child_elements();
        assert_eq!(kids[0].child_elements().len(), 1);
        assert_eq!(kids[2].child_elements().len(), 1);
    }
}
