//! [`Heading`] — a titled text run with a semantic level (1–6).
//!
//! The level selects a `font.size.*`/`font.weight.*` token pair in the preset
//! layer; the control itself only attaches `pk-heading` + `pk-heading--l{n}`.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The heading level, 1 (most prominent) through 6.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum HeadingLevel {
    /// Large title (`font.size.large-title`).
    #[default]
    L1,
    /// Title 1.
    L2,
    /// Title 2.
    L3,
    /// Title 3.
    L4,
    /// Headline.
    L5,
    /// Subheadline.
    L6,
}

impl HeadingLevel {
    /// The modifier suffix used in class names (e.g. `pk-heading--l1`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            HeadingLevel::L1 => "l1",
            HeadingLevel::L2 => "l2",
            HeadingLevel::L3 => "l3",
            HeadingLevel::L4 => "l4",
            HeadingLevel::L5 => "l5",
            HeadingLevel::L6 => "l6",
        }
    }

    /// The `font.size.*` token this level resolves against.
    #[must_use]
    pub const fn size_token(self) -> &'static str {
        match self {
            HeadingLevel::L1 => "font.size.large-title",
            HeadingLevel::L2 => "font.size.title1",
            HeadingLevel::L3 => "font.size.title2",
            HeadingLevel::L4 => "font.size.title3",
            HeadingLevel::L5 => "font.size.headline",
            HeadingLevel::L6 => "font.size.subheadline",
        }
    }
}

/// Props for [`Heading`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HeadingProps {
    /// The heading text.
    pub content: String,
    /// The semantic level.
    pub level: HeadingLevel,
}

impl HeadingProps {
    /// Creates props for a level-1 heading.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Sets the heading level.
    #[must_use]
    pub fn level(mut self, level: HeadingLevel) -> Self {
        self.level = level;
        self
    }
}

/// The heading control. Zero-sized; all configuration lives in [`HeadingProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Heading;

impl Component for Heading {
    type Props = HeadingProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut level_class = String::from("pk-heading--");
        level_class.push_str(props.level.suffix());
        Element::text(props.content.clone())
            .class("pk-heading")
            .class(level_class)
    }
}

/// Registers the `pk-heading` family: a base plus one class per level.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-heading")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontWeight, tok("font.weight.bold")),
    );
    for level in [
        HeadingLevel::L1,
        HeadingLevel::L2,
        HeadingLevel::L3,
        HeadingLevel::L4,
        HeadingLevel::L5,
        HeadingLevel::L6,
    ] {
        let mut name = String::from("pk-heading--");
        name.push_str(level.suffix());
        sheet.insert(Class::new(name).with(StyleProp::FontSize, tok(level.size_token())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn renders_level_class() {
        let el = Heading.render(&HeadingProps::new("Title").level(HeadingLevel::L3));
        assert_eq!(el.kind(), &ElementKind::Text);
        assert_eq!(el.class_names(), ["pk-heading", "pk-heading--l3"]);
    }

    #[test]
    fn register_adds_all_levels() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-heading", "pk-heading--l1", "pk-heading--l6"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
