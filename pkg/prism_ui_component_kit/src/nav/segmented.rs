//! [`SegmentedControl`] — a capsule of mutually-exclusive segments.
//!
//! A segmented control renders a `pk-segmented` capsule track containing one
//! `pk-segmented__segment` per label; the segment at the zero-based `selected`
//! index gains the shared `is-selected` state class. Only kit class names are
//! attached; surface, radius and typography resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`SegmentedControl`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SegmentedControlProps {
    /// The segment labels, rendered left-to-right.
    pub segments: Vec<String>,
    /// The zero-based index of the selected segment.
    pub selected: usize,
}

impl SegmentedControlProps {
    /// Creates empty segmented-control props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a segment label.
    #[must_use]
    pub fn segment(mut self, label: impl Into<String>) -> Self {
        self.segments.push(label.into());
        self
    }

    /// Replaces the segment labels with `segments`.
    #[must_use]
    pub fn segments<I, S>(mut self, segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.segments = segments.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the selected segment index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }
}

/// The segmented control. Zero-sized; config lives in
/// [`SegmentedControlProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentedControl;

impl Component for SegmentedControl {
    type Props = SegmentedControlProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-segmented");
        for (index, label) in props.segments.iter().enumerate() {
            let mut segment = Element::box_()
                .class("pk-segmented__segment")
                .child(Element::text(label.clone()).class("pk-segmented__label"));
            if index == props.selected {
                segment = segment.class("is-selected");
            }
            el = el.child(segment);
        }
        el
    }
}

/// Registers the `pk-segmented` class family: the capsule track, the segments
/// and the shared selected state.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Track: a neutral capsule rail holding equal-weight segments.
    sheet.insert(
        Class::new("pk-segmented")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_x(tok("space.xxs"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Segment: an equal-share capsule cell with centered body typography.
    sheet.insert(
        Class::new("pk-segmented__segment")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.subheadline")),
    );

    // Label: inherits the segment color.
    sheet.insert(
        Class::new("pk-segmented__label").with(StyleProp::Color, tok("color.label")),
    );

    // Shared selected state: raised surface pill with accent label.
    sheet.insert(
        Class::new("is-selected")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SegmentedControlProps) -> Element {
        SegmentedControl.render(&props)
    }

    #[test]
    fn empty_control_has_no_segments() {
        let el = render(SegmentedControlProps::new());
        assert_eq!(el.class_names(), ["pk-segmented"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn selected_segment_gets_is_selected() {
        let el = render(
            SegmentedControlProps::new()
                .segments(["Day", "Week", "Month"])
                .selected(2),
        );
        let segs = el.child_elements();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].class_names(), ["pk-segmented__segment"]);
        assert_eq!(segs[2].class_names(), ["pk-segmented__segment", "is-selected"]);
    }

    #[test]
    fn segment_renders_label_text() {
        let el = render(SegmentedControlProps::new().segment("Day"));
        let seg = &el.child_elements()[0];
        assert_eq!(seg.child_elements()[0].text_content(), Some("Day"));
    }
}
