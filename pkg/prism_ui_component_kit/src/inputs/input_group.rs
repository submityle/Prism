//! [`InputGroup`] — a horizontal row that wraps an input with optional
//! prefix/suffix addons.
//!
//! An input group renders a `pk-input-group` flex row stitching an optional
//! prefix addon, the input element and an optional suffix addon into a single
//! segmented control. It is a pure layout aggregator: callers pass fully-formed
//! [`Element`]s for each slot and this control only positions them and attaches
//! the `pk-input-group` class family. The addons carry neutral fill tokens while
//! the input slot grows to fill the remaining width. All color comes from theme
//! tokens via [`crate::preset`].

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`InputGroup`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct InputGroupProps {
    /// The leading addon element, rendered before the input.
    pub prefix: Option<Element>,
    /// The input element, rendered in the middle and allowed to grow.
    pub input: Option<Element>,
    /// The trailing addon element, rendered after the input.
    pub suffix: Option<Element>,
}

impl InputGroupProps {
    /// Creates empty input-group props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the leading prefix addon element.
    #[must_use]
    pub fn prefix(mut self, element: Element) -> Self {
        self.prefix = Some(element);
        self
    }

    /// Sets the input element.
    #[must_use]
    pub fn input(mut self, element: Element) -> Self {
        self.input = Some(element);
        self
    }

    /// Sets the trailing suffix addon element.
    #[must_use]
    pub fn suffix(mut self, element: Element) -> Self {
        self.suffix = Some(element);
        self
    }
}

/// The input-group control. Zero-sized; config lives in [`InputGroupProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct InputGroup;

impl Component for InputGroup {
    type Props = InputGroupProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-input-group");

        if let Some(prefix) = props.prefix.clone() {
            el = el.child(
                prefix
                    .class("pk-input-group__addon")
                    .class("pk-input-group__addon--prefix"),
            );
        }
        if let Some(input) = props.input.clone() {
            el = el.child(input.class("pk-input-group__input"));
        }
        if let Some(suffix) = props.suffix.clone() {
            el = el.child(
                suffix
                    .class("pk-input-group__addon")
                    .class("pk-input-group__addon--suffix"),
            );
        }
        el
    }
}

/// Registers the `pk-input-group` class family: the horizontal row, its
/// neutral-fill prefix/suffix addons and the growing input slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-input-group")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Stretch))
            .with(StyleProp::Gap, StyleValue::px(0.0))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    sheet.insert(
        Class::new("pk-input-group__addon")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_padding_x(tok("space.sm")),
    );

    sheet.insert(Class::new("pk-input-group__addon--prefix"));
    sheet.insert(Class::new("pk-input-group__addon--suffix"));

    sheet.insert(
        Class::new("pk-input-group__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexShrink, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_prefix_input_suffix_in_order() {
        let el = InputGroup.render(
            &InputGroupProps::new()
                .prefix(Element::box_())
                .input(Element::box_())
                .suffix(Element::box_()),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert!(kids[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-input-group__addon--prefix"));
        assert!(kids[1]
            .class_names()
            .iter()
            .any(|c| c == "pk-input-group__input"));
        assert!(kids[2]
            .class_names()
            .iter()
            .any(|c| c == "pk-input-group__addon--suffix"));
    }

    #[test]
    fn omitting_slots_omits_children() {
        let el = InputGroup.render(&InputGroupProps::new().input(Element::box_()));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-input-group__input"));
    }

    #[test]
    fn root_carries_group_class() {
        let el = InputGroup.render(&InputGroupProps::new());
        assert!(el.class_names().iter().any(|c| c == "pk-input-group"));
        assert!(el.child_elements().is_empty());
    }
}
