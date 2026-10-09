//! [`Section`] — a titled grouping of content.
//!
//! A section renders an optional `header` element above a body that wraps its
//! `children`. It is a pure layout primitive: it supplies spacing and
//! typographic rhythm through theme tokens and attaches only `pk-section`
//! class names.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Section`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SectionProps {
    /// Optional header rendered above the body (e.g. a title row).
    pub header: Option<Element>,
    /// The section's body content.
    pub children: Vec<Element>,
}

impl SectionProps {
    /// Creates empty section props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the header element.
    #[must_use]
    pub fn header(mut self, element: Element) -> Self {
        self.header = Some(element);
        self
    }

    /// Appends a body child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the body children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The section control. Zero-sized; all configuration lives in [`SectionProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Section;

impl Component for Section {
    type Props = SectionProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-section");
        if let Some(header) = props.header.clone() {
            el = el.child(header.class("pk-section__header"));
        }
        let body = Element::box_()
            .class("pk-section__body")
            .children(props.children.iter().cloned());
        el = el.child(body);
        el
    }
}

/// Registers the `pk-section` class family: container, header, body.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Container: a vertical stack separating header from body.
    sheet.insert(
        Class::new("pk-section")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Header: headline typography.
    sheet.insert(
        Class::new("pk-section__header")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: a vertical stack of the section's children.
    sheet.insert(
        Class::new("pk-section__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SectionProps) -> Element {
        Section.render(&props)
    }

    #[test]
    fn without_header_only_body_child() {
        let el = render(SectionProps::new().child(Element::box_().class("row")));
        assert_eq!(el.class_names(), ["pk-section"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].class_names(), ["pk-section__body"]);
    }

    #[test]
    fn header_precedes_body() {
        let el = render(
            SectionProps::new()
                .header(Element::text("Title"))
                .child(Element::box_()),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert!(children[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-section__header"));
        assert_eq!(children[1].class_names(), ["pk-section__body"]);
    }

    #[test]
    fn body_wraps_all_children() {
        let el = render(
            SectionProps::new().children([Element::box_(), Element::box_(), Element::box_()]),
        );
        let body = &el.child_elements()[0];
        assert_eq!(body.child_elements().len(), 3);
    }
}
