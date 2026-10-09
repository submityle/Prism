//! [`Watermark`] — a repeated label laid over arbitrary content.
//!
//! A watermark stamps a faint, repeating text mark across a region (think
//! "CONFIDENTIAL" tiled behind a document preview). The control renders a
//! `pk-watermark` box containing the protected content children plus a single
//! `pk-watermark__mark` text node that carries the mark string.
//!
//! The control is data-only: it does not tile or rotate the mark itself. It
//! records the mark text and the `pk-watermark__mark` class so a backend (or a
//! future tiling pass) can repeat and composite it over the content. The mark
//! is emitted *after* the content children so it paints on top (paint order is
//! sibling order; the kit exposes no `z-index`).

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Watermark`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WatermarkProps {
    /// The mark text repeated across the overlay.
    pub text: String,
    /// The protected content the mark is laid over.
    pub children: Vec<Element>,
}

impl WatermarkProps {
    /// Creates watermark props for the given mark text with no content yet.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Sets the mark text.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Appends a single content child beneath the mark.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many content children beneath the mark.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }
}

/// The watermark control. Zero-sized; config lives in [`WatermarkProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Watermark;

impl Component for Watermark {
    type Props = WatermarkProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-watermark")
            .children(props.children.iter().cloned())
            .child(Element::text(props.text.clone()).class("pk-watermark__mark"))
    }
}

/// Registers the `pk-watermark` family: the container is a plain block; the
/// mark is faint secondary-label text. Tiling, rotation and compositing are a
/// backend concern — the kit only records the token-backed appearance of a
/// single mark instance.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-watermark").with(StyleProp::Display, kw(Keyword::Block)),
    );

    sheet.insert(
        Class::new("pk-watermark__mark")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Opacity, StyleValue::number(0.12)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_ui::ElementKind;

    fn render(props: WatermarkProps) -> Element {
        Watermark.render(&props)
    }

    #[test]
    fn attaches_watermark_class_to_box() {
        let el = render(WatermarkProps::new("DRAFT"));
        assert_eq!(el.kind(), &ElementKind::Box);
        assert_eq!(el.class_names(), ["pk-watermark"]);
    }

    #[test]
    fn mark_is_last_child_and_carries_text() {
        let el = render(
            WatermarkProps::new("CONFIDENTIAL").children(vec![Element::box_().class("doc")]),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["doc"]);
        let mark = &kids[1];
        assert_eq!(mark.kind(), &ElementKind::Text);
        assert_eq!(mark.class_names(), ["pk-watermark__mark"]);
        assert_eq!(mark.text_content(), Some("CONFIDENTIAL"));
    }

    #[test]
    fn register_adds_both_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-watermark").is_some());
        assert!(sheet.get("pk-watermark__mark").is_some());
    }
}
