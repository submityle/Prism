//! [`Blockquote`] — a quoted passage with an accent rule.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Blockquote`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BlockquoteProps {
    /// The quoted content elements.
    pub children: Vec<Element>,
    /// Optional attribution/citation line.
    pub cite: Option<String>,
}

impl BlockquoteProps {
    /// Creates empty blockquote props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a child element.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Sets the attribution line.
    #[must_use]
    pub fn cite(mut self, cite: impl Into<String>) -> Self {
        self.cite = Some(cite.into());
        self
    }
}

/// The blockquote control. Zero-sized; configuration lives in
/// [`BlockquoteProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Blockquote;

impl Component for Blockquote {
    type Props = BlockquoteProps;

    fn render(&self, props: &Self::Props) -> Element {
        let body = Element::box_()
            .class("pk-blockquote__body")
            .children(props.children.iter().cloned());
        let mut el = Element::box_().class("pk-blockquote").child(body);
        if let Some(cite) = &props.cite {
            el = el.child(Element::text(cite.clone()).class("pk-blockquote__cite"));
        }
        el
    }
}

/// Registers the `pk-blockquote` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-blockquote")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BorderColor, tok("color.tint"))
            .with(StyleProp::BorderWidth, StyleValue::px(3.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm")),
    );
    sheet.insert(
        Class::new("pk-blockquote__body")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-blockquote__cite")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cite_adds_second_child() {
        let el = Blockquote.render(
            &BlockquoteProps::new()
                .child(Element::text("quote"))
                .cite("— Someone"),
        );
        assert_eq!(el.class_names(), ["pk-blockquote"]);
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-blockquote", "pk-blockquote__body", "pk-blockquote__cite"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
