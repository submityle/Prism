//! [`Code`] — inline or block monospaced code.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Code`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CodeProps {
    /// The source text.
    pub source: String,
    /// Whether to render as a multi-line block instead of inline.
    pub block: bool,
}

impl CodeProps {
    /// Creates props for inline code.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Self::default()
        }
    }

    /// Renders as a block instead of inline.
    #[must_use]
    pub fn block(mut self, block: bool) -> Self {
        self.block = block;
        self
    }
}

/// The code control. Zero-sized; configuration lives in [`CodeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Code;

impl Component for Code {
    type Props = CodeProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-code");
        if props.block {
            el = el.class("pk-code--block");
        } else {
            el = el.class("pk-code--inline");
        }
        el.child(Element::text(props.source.clone()).class("pk-code__source"))
    }
}

/// Registers the `pk-code` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-code")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::BorderRadius, tok("radius.sm")),
    );
    sheet.insert(
        Class::new("pk-code--inline")
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xxs")),
    );
    sheet.insert(
        Class::new("pk-code--block")
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_by_default() {
        let el = Code.render(&CodeProps::new("let x = 1;"));
        assert_eq!(el.class_names(), ["pk-code", "pk-code--inline"]);
    }

    #[test]
    fn block_variant() {
        let el = Code.render(&CodeProps::new("fn main() {}").block(true));
        assert_eq!(el.class_names(), ["pk-code", "pk-code--block"]);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-code", "pk-code--inline", "pk-code--block"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
