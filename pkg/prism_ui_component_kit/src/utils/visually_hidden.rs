//! [`VisuallyHidden`] — keep text in the tree, out of sight.
//!
//! Screen-reader-only content: a label, instruction or live-region update that
//! must exist for assistive technology but must not take visible space. The
//! control wraps its children in a box carrying the `pk-visually-hidden` class;
//! [`register_styles`] clamps that box to a 1px square at zero opacity, so the
//! text stays in the accessibility tree (and in document order) while reading
//! as invisible on screen.
//!
//! This is deliberately *not* `Display::None` or an empty element — removing
//! the node would also remove it from the accessibility tree, defeating the
//! purpose. The subtree is preserved as data; only its painted size is clamped.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`VisuallyHidden`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VisuallyHiddenProps {
    /// The accessibility-only content kept in the tree but clamped off-screen.
    pub children: Vec<Element>,
}

impl VisuallyHiddenProps {
    /// Creates empty visually-hidden props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a single child to the hidden subtree.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many children to the hidden subtree.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }
}

/// The visually-hidden control. Zero-sized; config lives in
/// [`VisuallyHiddenProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct VisuallyHidden;

impl Component for VisuallyHidden {
    type Props = VisuallyHiddenProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-visually-hidden")
            .children(props.children.iter().cloned())
    }
}

/// Registers `pk-visually-hidden`: a 1px square at zero opacity with its
/// padding collapsed. The box stays in layout and in the accessibility tree
/// but paints as nothing. (No `overflow`/`clip` style prop exists in the kit,
/// so size + opacity are the available levers; a backend may additionally honor
/// the tiny clamped box as a clip hint.)
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(
        Class::new("pk-visually-hidden")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with(StyleProp::Height, StyleValue::px(1.0))
            .with(StyleProp::MaxWidth, StyleValue::px(1.0))
            .with(StyleProp::MaxHeight, StyleValue::px(1.0))
            .with_padding_x(StyleValue::px(0.0))
            .with_padding_y(StyleValue::px(0.0))
            .with(StyleProp::Opacity, StyleValue::number(0.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_ui::ElementKind;

    fn render(props: VisuallyHiddenProps) -> Element {
        VisuallyHidden.render(&props)
    }

    #[test]
    fn attaches_visually_hidden_class() {
        let el = render(VisuallyHiddenProps::new());
        assert_eq!(el.class_names(), ["pk-visually-hidden"]);
    }

    #[test]
    fn is_a_box_that_keeps_children() {
        let el = render(
            VisuallyHiddenProps::new().children(vec![Element::text("skip to content")]),
        );
        assert_eq!(el.kind(), &ElementKind::Box);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].text_content(), Some("skip to content"));
    }

    #[test]
    fn register_clamps_size_and_opacity() {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        let class = sheet.get("pk-visually-hidden").expect("class registered");
        assert_eq!(class.base.get(&StyleProp::Width), Some(&StyleValue::px(1.0)));
        assert_eq!(class.base.get(&StyleProp::Height), Some(&StyleValue::px(1.0)));
        assert_eq!(
            class.base.get(&StyleProp::Opacity),
            Some(&StyleValue::number(0.0))
        );
    }
}
