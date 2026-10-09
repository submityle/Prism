//! [`Skeleton`] — shimmering placeholder lines shown while content loads.
//!
//! A skeleton is a count of `lines`; it renders a `pk-skeleton` stack of
//! `pk-skeleton__line` placeholders. The shimmer is a backend concern signalled
//! by the line class, so this control attaches only kit classes and lets
//! [`crate::preset`] resolve sizing against the theme.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Skeleton`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SkeletonProps {
    /// How many placeholder lines to render.
    pub lines: u16,
}

impl SkeletonProps {
    /// Creates skeleton props for `lines` placeholder rows.
    #[must_use]
    pub fn new(lines: u16) -> Self {
        Self { lines }
    }

    /// Sets the placeholder line count.
    #[must_use]
    pub fn lines(mut self, lines: u16) -> Self {
        self.lines = lines;
        self
    }
}

/// The skeleton control. Zero-sized; all configuration lives in [`SkeletonProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Skeleton;

impl Component for Skeleton {
    type Props = SkeletonProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-skeleton");
        for _ in 0..props.lines {
            el = el.child(Element::box_().class("pk-skeleton__line"));
        }
        el
    }
}

/// Registers the `pk-skeleton` class family: the stacking base and the single
/// shimmering line placeholder.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a vertical stack of placeholder rows.
    sheet.insert(
        Class::new("pk-skeleton")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Line: a full-width rounded bar on a quiet fill the backend shimmers.
    sheet.insert(
        Class::new("pk-skeleton__line")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SkeletonProps) -> Element {
        Skeleton.render(&props)
    }

    #[test]
    fn empty_skeleton_has_no_lines() {
        let el = render(SkeletonProps::new(0));
        assert_eq!(el.class_names(), ["pk-skeleton"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn renders_requested_line_count() {
        let el = render(SkeletonProps::new(3));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        for line in kids {
            assert_eq!(line.class_names(), ["pk-skeleton__line"]);
        }
    }
}
